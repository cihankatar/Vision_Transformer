##IMPORT 

import numpy as np
from tqdm import tqdm, trange
import os
from glob import glob
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image


class KVasir_dataset(Dataset):
    def __init__(self,train_path,mask_path,image_dir_list,mask_dir_list):
        super().__init__()
        self.train_path      = train_path
        self.image_dir_list  = image_dir_list
        self.mask_path       = mask_path
        self.mask_dir_list   = mask_dir_list

        self.images=[]
        self.masks=[]

    def getitems(self):        
        for idx in range(len(self.image_dir_list)):
            #if 'jpg' in self.image_dir_list:
            image_dir = os.path.join(self.train_path,self.image_dir_list[idx])
            image=Image.open(image_dir)
            image=np.array(image)
            image = np.transpose(image, (2, 0, 1))
            image=torch.from_numpy(image)
            self.images.append(image)
        
        for idx in range(len(self.image_dir_list)):
            #if 'jpg' in self.image_dir_list:
            mask_dir = os.path.join(self.mask_path,self.mask_dir_list[idx])
            mask=Image.open(mask_dir)
            mask=np.array(mask)
            mask = np.transpose(mask, (2, 0, 1))
            mask=torch.from_numpy(mask)
            self.masks.append(mask)

        return self.images, self.masks
    
    def len(self):
        return len(self.image_dir_list)

def main():
    train_im_path = "train/images"
    images_dir_list = os.listdir(train_im_path) 
    train_mask_path = "train/masks"
    mask_dir_list = os.listdir(train_mask_path) 

    data=KVasir_dataset(train_im_path,train_mask_path,images_dir_list,mask_dir_list)
    train_dataset = list(data.getitems())
    train_dataset = next(iter(train_dataset))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=2
    )

    for batch in train_loader:
        x,y=batch

if __name__ == '__main__':
    main()