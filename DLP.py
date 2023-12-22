import os
import math
import csv
import pickle
from urllib import request
import scipy.stats as st

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import argparse
import tifgsm
import difgsm
import mifgsm

device = torch.device("cuda:0")

def load_ground_truth(fname):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    df = pd.read_csv(fname)
    for _, row in df.iterrows():
        image_id_list.append( row['ImageId'] )
        label_ori_list.append( int(row['TrueLabel']) - 1 )
        label_tar_list.append( int(row['TargetClass']) - 1 )
    gt = pickle.load(request.urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))
    return image_id_list,label_ori_list,label_tar_list, gt

## simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

def fgsm(model_gen, model_pred, ids, origins, targets, gt, trn, norm, args):
    preds_ls = []
    labels_ls =[]
    origin_ls = []
    
    batch_size = args.batch_size
    img_size = args.img_size
    input_path = args.input_path
    if args.mode == "fgsm":
        max_iterations = 1
    else:
        max_iterations = args.max_iterations
    if args.mode == "mi-fgsm":
        decay = args.decay
    lr = args.lr/255
    epsilon = args.epsilon/255
    epochs = int(np.ceil(len(ids) / batch_size))
    for k in tqdm(range(epochs), total=epochs):
        batch_size_cur = min(batch_size, len(ids) - k * batch_size)
        X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
        delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
        if args.mode == "mi-fgsm":
            momentum = torch.zeros_like(X_ori).detach().to(device)
        for i in range(batch_size_cur):
            X_ori[i] = trn(Image.open(input_path + ids[k * batch_size + i] + '.png'))
        ori_idx = origins[k * batch_size:k * batch_size + batch_size_cur]
        labels = torch.tensor(targets[k * batch_size:k * batch_size + batch_size_cur]).to(device)

        prev = float('inf')
        for t in range(max_iterations):
            logits = model_gen(norm(X_ori + delta))
            loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
            loss.backward()
            grad_c = delta.grad.clone()
            if args.mode == "mi-fgsm":
                grad_c = grad_c + momentum * decay
                momentum = grad_c
            delta.grad.zero_()
            delta.data = delta.data - lr * torch.sign(grad_c)
            delta.data = delta.data.clamp(-epsilon / 255,epsilon / 255)
            delta.data = ((X_ori + delta.data).clamp(0,1)) - X_ori

        X_pur = norm(X_ori + delta)
        preds = torch.argmax(model_pred(X_pur), dim=1)

        preds_ls.append(preds.cpu().numpy())
        labels_ls.append(labels.cpu().numpy())
        origin_ls.append(ori_idx)
    
    #viz(X_ori.cpu().detach(), X_pur.cpu().detach(), ori_idx, labels.cpu().numpy(), gt, preds.cpu().numpy())

    return preds_ls, labels_ls, origin_ls

def ti_fgsm(model_gen, model_pred, ids, origins, targets, gt, trn, norm, args):
    preds_ls = []
    labels_ls =[]
    origin_ls = []
    
    batch_size = args.batch_size
    img_size = args.img_size
    input_path = args.input_path
    kernel_name=args.kernel_name
    epsilon = args.epsilon/255
    lr = args.lr/255
    max_iterations = args.max_iterations
    decay=args.decay
    epochs = int(np.ceil(len(ids) / batch_size))
    attack = tifgsm.TIFGSM(model_gen, eps =epsilon, alpha=lr, steps=max_iterations, decay=decay, kernel_name=kernel_name)

    torch.cuda.empty_cache()
    for k in tqdm(range(epochs), total=epochs):
        batch_size_cur = min(batch_size, len(ids) - k * batch_size)
        X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
        for i in range(batch_size_cur):
            X_ori[i] = trn(Image.open(input_path + ids[k * batch_size + i] + '.png'))
        ori_idx = origins[k * batch_size:k * batch_size + batch_size_cur]
        labels = torch.tensor(targets[k * batch_size:k * batch_size + batch_size_cur]).to(device)

        
        X_pur = attack(norm(X_ori),labels)
        preds = torch.argmax(model_pred(norm(X_pur)), dim=1)
        
        preds_ls.append(preds.cpu().numpy())
        labels_ls.append(labels.cpu().numpy())
        origin_ls.append(ori_idx)
    
    #viz(X_ori.cpu().detach(), X_pur.cpu().detach(), ori_idx, labels.cpu().numpy(), gt, preds.cpu().numpy())

    return preds_ls, labels_ls, origin_ls

def di_fgsm(model_gen, model_pred, ids, origins, targets, gt, trn, norm, args):
    preds_ls = []
    labels_ls =[]
    origin_ls = []
    
    batch_size = args.batch_size
    img_size = args.img_size
    input_path = args.input_path
    epsilon = args.epsilon/255
    lr = args.lr/255
    max_iterations = args.max_iterations
    decay=args.decay
    epochs = int(np.ceil(len(ids) / batch_size))
    attack = difgsm.DIFGSM(model_gen, eps =epsilon, alpha=lr, steps=max_iterations, decay=decay)

    torch.cuda.empty_cache()
    for k in tqdm(range(epochs), total=epochs):
        batch_size_cur = min(batch_size, len(ids) - k * batch_size)
        X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
        for i in range(batch_size_cur):
            X_ori[i] = trn(Image.open(input_path + ids[k * batch_size + i] + '.png'))
        ori_idx = origins[k * batch_size:k * batch_size + batch_size_cur]
        labels = torch.tensor(targets[k * batch_size:k * batch_size + batch_size_cur]).to(device)

        
        X_pur = attack(X_ori,labels)
        preds = torch.argmax(model_pred(norm(X_pur)), dim=1)
        
        preds_ls.append(preds.cpu().numpy())
        labels_ls.append(labels.cpu().numpy())
        origin_ls.append(ori_idx)
    
    #viz(X_ori.cpu().detach(), X_pur.cpu().detach(), ori_idx, labels.cpu().numpy(), gt, preds.cpu().numpy())

    return preds_ls, labels_ls, origin_ls

def viz(img_A, img_B, origins, labels, gt, preds):
    for img_a, img_b, origin, label, pred in zip(img_A, img_B, origins, labels, preds):
        img_a = img_a.permute(1, 2, 0)
        img_b = img_b.permute(1, 2, 0)

        fig, (axA, axB) = plt.subplots(1, 2, figsize=(10,3))
        axA.imshow(img_a)
        axA.set_title("True label: " + gt[origin])
        axB.imshow(img_b)
        axB.set_title("Target: " + gt[label])

        result = 'Failed' if pred != label else 'Success'
        caption = f'Pred: {gt[pred]} -> {result}'
        fig.text(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=12)

        plt.show()

def main(args):
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trn = transforms.Compose([transforms.ToTensor(),])
    ids, origins, targets, gt = load_ground_truth('train.csv')

    resnet = models.resnet50(weights="IMAGENET1K_V1").eval()
    vgg = models.vgg16_bn(weights="IMAGENET1K_V1").eval()

    for param in resnet.parameters():
        param.requires_grad = False
    for param in vgg.parameters():
        param.requires_grad = False

    resnet.to(device)
    vgg.to(device)

    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    if args.mode == "fgsm" or args.mode == "i-fgsm" or args.mode == "mi-fgsm":
        preds_ls, labels_ls, origin_ls = fgsm(resnet, vgg, ids, origins, targets, gt, trn, norm, args)
    elif args.mode == "ti-fgsm":
        preds_ls, labels_ls, origin_ls = ti_fgsm(resnet, vgg, ids, origins, targets, gt, trn, norm, args)
    elif args.mode == "di-fgsm":
        preds_ls, labels_ls, origin_ls = di_fgsm(resnet, vgg, ids, origins, targets, gt, trn, norm, args)
    df = pd.DataFrame({
    'origin': [a for b in origin_ls for a in b],
    'pred': [a for b in preds_ls for a in b],
    'label': [a for b in labels_ls for a in b]
    })
    # df.head()
    print(accuracy_score(df['label'], df['pred']))

    # print(f'{accuracy_score(df["label"], df["pred"]):.8f}')

    # df.to_csv(f'logs/{args.mode}/{args.epsilon}_{args.max_iterations}_submission.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="di-fgsm", type=str)
    # parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--img_size", default=299, type=int)
    parser.add_argument("--input_path", default='/home/jrespect/jLim/images/', type=str)
    parser.add_argument("--max_iterations", default=300, type=int)
    parser.add_argument("--lr", default=2, type=float)
    parser.add_argument("--epsilon", default=255, type=int)
    parser.add_argument("--decay", default=0.0, type=float)
    parser.add_argument("--kernel_name", default="gaussian", type=str)

    args = parser.parse_args()
    main(args)