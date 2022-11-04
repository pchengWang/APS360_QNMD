import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from pathlib import Path
from tqdm import tqdm
import sys

import pydicom
import glob

from copy import deepcopy

# helper to transform rle to mask img
from mask_functions import rle2mask
from skimage.color import label2rgb

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def load_image():

    path_to_module = '/content/drive/MyDrive/APS360_Project/APS360_Segmentation_Project/'
    sys.path.append(path_to_module)

    work_folder = '/content/drive/MyDrive/APS360_Project'
    work_folder = Path(work_folder)
    train_data_path = Path('/content/drive/MyDrive/APS360_Project/APS360_Segmentation_Project/export_sample_data')

    train_rle_path = work_folder / 'train-rle.csv'
    train_rle = pd.read_csv(train_rle_path)

    dicom_file_path = glob.glob(f'{train_data_path}/*')
    image_height = 1024
    image_width = 1024
    # channel_num = 1
    num_img = len(dicom_file_path)

    dcm_img = []
    masks = []
    img_id = []

    count = 0
    for idx in tqdm(range(len(dicom_file_path))):
        if count == num_img:
            break

        img = plt.imread(dicom_file_path[idx])

        cur_img_id = os.path.basename(dicom_file_path[idx])
        id_len = len(cur_img_id)
        cur_img_id = cur_img_id[:id_len - 4]

        mask = train_rle.loc[train_rle['ImageId'] == cur_img_id][' EncodedPixels'] # find the crossponding rle encoding

        # filter out the outliers due to missing labels
        if mask.shape[0] != 1:
            continue

        mask_val = mask.values[0]

        if(mask_val != ' -1'):
            masks.append(np.expand_dims(rle2mask(mask_val, image_height, image_width).T, axis=0)) # convert rle to Pneumothorax image
            dcm_img.append(np.expand_dims(img, axis=0)) # expand dimensions to fit array
            img_id.append(cur_img_id) # consider ds as a large dictionary['key']
            count += 1
        else:
            masks[count] = np.zeros((1024, 1024, 1)) # empty image
            dcm_img[count] = np.expand_dims(img, axis=2) # expand dimensions to fit array
            img_id.append(cur_img_id) # consider ds as a large dictionary['key']
            count += 1
    return dcm_img, masks


def image_transform(train_data, val_data):
    transforms = T.Compose([
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        T.RandomHorizontalFlip(),
        T.RandomHorizontalFlip(),
        T.RandomRotation()
        ])
    
    train_set = ImageFolder(train_data, transform=transforms)
    val_set = ImageFolder(val_data, transform=transforms)



def get_combo_dataset(dcm_img, masks, train_num, val_num):
    combo_data = []
    for i in range(len(dcm_img)):
        combo_data.append([dcm_img[i], masks[i]])


    train_data, val_data = torch.utils.data.random_split(combo_data, [train_num,  val_num], generator=torch.Generator().manual_seed(42))
    return train_data, val_data


def get_dataLoader(batch_size):
    dcm_img, masks = load_image()
    train_set, val_set = get_combo_dataset(dcm_img, masks, len(dcm_img) * 0.6, len(dcm_img)*0.4)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=True)

    return train_loader, val_loader