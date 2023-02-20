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


'''
image=Image.open("train/images/cju0qkwl35piu0993l0dewei2.jpg")
image.show()
print(type(image))
image=np.array(image)
image=torch.from_numpy(image)
print(image.shape)
'''

cwd = os.getcwd() 
print("Current working directory:", cwd) 


class dataset(Dataset):
    def __init__(self,path,image_dir_list):
        super().__init__()
        self.path       = path
        self.image_dir_list  = image_dir_list
    
    def __getitem__(self, idx):        

        if 'jpg' in self.image_dir_list:
            image_dir = os.path.join(path,self.image_dir_list[idx])
            image=Image.open(self.image_dir)
            image=np.array(image)
            image=torch.from_numpy(image)
            return image

path = "train/images"
images_dir_list = os.listdir(path) 
print("Files and directories in '", path, "' :")  
print('type:{},lengt:{}'.format(type(images_dir_list),len(images_dir_list)))



im=[]

for idx in range(len(images_dir_list)):
    im.append(dataset(path,images_dir_list[idx]))




train_dataset = getimage(path)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2
)