import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torchvision.utils import save_image

from pathlib import Path
import sys
from IPython.display import display

import pydicom
import glob

import time
import datetime 
from tqdm import tqdm 
import shutil

from copy import deepcopy
from torchsummary import summary
from torchmetrics import JaccardIndex

from data_processing import get_dataLoader

def plot_training_curve(train_losses=None, valid_losses=None, train_acc=None, val_acc=None):
    if train_losses != None:
        plt.title("Train Loss")
        plt.plot(range(len(train_losses)), train_losses, label="Train")
        plt.xlabel("iters")
        plt.ylabel("Loss")
        plt.legend(loc='best')
        plt.show()

    if valid_losses != None:
        plt.title("Validation Loss")
        plt.plot(range(len(valid_losses)), valid_losses, label="Validation")
        plt.xlabel("iters")
        plt.ylabel("Loss")
        plt.legend(loc='best')
        plt.show()

    if train_acc != None:
        plt.title("Train Accuracy")
        plt.plot(range(len(train_acc)), train_acc, label="Train")
        plt.xlabel("iters")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.show()

    if val_acc != None:
        plt.title("Validation Accuracy")
        plt.plot(range(len(val_acc)), val_acc, label="Validation")
        plt.xlabel("iters")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.show()    


def IoU(pred, target):
    jaccard = JaccardIndex(num_classes=2)
    if torch.cuda.is_available():
        jaccard.to('cuda')
    return jaccard(pred, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)
    print(str(info)+"\n")
    


def train_network(net, epochs, batch_size, learning_rate, dl_train=None, dl_val=None):
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_fn = FocalLoss(alpha=0.25)
    optimizer= torch.optim.Adam(net.parameters(),lr = learning_rate)   

    dl_train, dl_val = get_dataLoader(batch_size=batch_size)

    #early_stopping
    monitor="val_acc"
    patience=5
    mode="max"

    history = {}


    for epoch in range(1, epochs+1):
        printlog("Epoch {0} / {1}".format(epoch, epochs))

        # 1，train -------------------------------------------------  
        net.train()
        
        total_loss,step = 0,0
        total_acc = []
        train_loss_list = []
        val_loss = []
        
        loop = tqdm(enumerate(dl_train), total =len(dl_train))
        
        for i, batch in loop: 
            
            features,labels = batch
            if torch.cuda.is_available():
                features = features.to(dev)
                labels = labels.to(dev)

            #forward
            preds = net(features.float())
            
            loss = loss_fn(preds.float(),labels.float())
            
            #backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
                
            #metrics
            total_acc.append(IoU(preds.int(), labels.int()))

            total_loss += loss.item()
            train_loss_list.append(loss.item())

            
            _img = features[0]
            _label = labels[0]
            _out = preds[0]

            _img = np.squeeze(_img, axis=0)
            _img = np.squeeze(_img, axis=0)
            _label = np.squeeze(_label, axis=0)
            _label = np.squeeze(_label, axis=0)
            _out = np.squeeze(_out, axis=0)
            _out = np.squeeze(_out, axis=0)

            img = torch.stack([_img, _label, _out], dim=0)
            t_path = '/content/temp_out'
            save_image(_img.int(), f"{t_path}/img{i}.png")
            save_image(_label.int(), f"{t_path}/label{i}.png")
            save_image(_out, f"{t_path}/out{i}.png")
            
            step+=1
        print(f"Epoch {epoch} acc = {sum(total_acc) / len(total_acc)}")
        print(f"Epoch {epoch} train_loss = {sum(train_loss_list) / len(train_loss_list)}")

        # 2，validate -------------------------------------------------
        net.eval()
        
        total_loss,step = 0,0
        total_val_acc = []
        loop = tqdm(enumerate(dl_val), total =len(dl_val))
        
        with torch.no_grad():
            for i, batch in loop: 

                features,labels = batch
                if torch.cuda.is_available():
                    features = features.cuda()
                    labels = labels.cuda()                
                #forward
                preds = net(features.float())
                
                loss = loss_fn(preds.float(), labels.float())

                #metrics
                total_val_acc.append(IoU(preds.int(), labels.int()))

                total_loss += loss.item()
                val_loss.append(loss.item())

                step+=1
        print(f"Epoch {epoch} val_acc = {sum(total_val_acc) / len(total_val_acc)}")
        print(f"Epoch {epoch} val_loss = {sum(val_loss) / len(val_loss)}")
        

        # 3，early-stopping -------------------------------------------------
        # arr_scores = history[monitor]
        # best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
        # if best_score_idx==len(arr_scores)-1:
        #     torch.save(net.state_dict(),ckpt_path)
        #     print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
        #         arr_scores[best_score_idx]),file=sys.stderr)
        # if len(arr_scores)-best_score_idx>patience:
        #     print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
        #         monitor,patience),file=sys.stderr)
        #     break 
        # net.load_state_dict(torch.load(ckpt_path))
        
    # dfhistory = pd.DataFrame(history)
    plot_training_curve(train_loss_list, val_loss)
