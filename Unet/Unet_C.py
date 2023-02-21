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
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    
    def __init__(self, images_path, masks_path):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        return image, mask
        

'''
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
            image=np.array(image,dtype=float)
            image = np.transpose(image, (2, 0, 1))
            image=torch.from_numpy(image)
            self.images.append(image)
        
        for idx in range(len(self.image_dir_list)):
            #if 'jpg' in self.image_dir_list:
            mask_dir = os.path.join(self.mask_path,self.mask_dir_list[idx])
            mask=Image.open(mask_dir)
            mask=np.array(mask,dtype=float)
            mask = np.transpose(mask, (2, 0, 1))
            mask=torch.from_numpy(mask)
            self.masks.append(mask)

        return self.images, self.masks
    
    def len(self):
        return len(self.image_dir_list)'''

def main():
    train_im_path = "c:\\Users\\cihan.katar\\Desktop\\Git_Repo\\Vision_Transformer\\Unet\\train\images"
    images_dir_list = os.listdir(train_im_path) 
    train_mask_path = "c:\\Users\\cihan.katar\\Desktop\\Git_Repo\\Vision_Transformer\\Unet\\train\masks"
    mask_dir_list = os.listdir(train_mask_path) 
    data=DriveDataset(train_im_path,train_mask_path)
    #data=KVasir_dataset(train_im_path,train_mask_path,images_dir_list,mask_dir_list)
    #train_dataset = data.getitems()
    #train_dataset = next(iter(train_dataset))

    train_loader = DataLoader(
        dataset=data,
        batch_size=10,
        shuffle=True,
        num_workers=2
    )

    for batch in train_loader:
        x,y=batch

if __name__ == '__main__':
    main()





