##IMPORT 

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class KVasir_dataset(Dataset):
    def __init__(self,train_path,mask_path,image_dir_list,mask_dir_list):
        super().__init__()
        self.train_path      = train_path
        self.image_dir_list  = image_dir_list
        self.mask_path       = mask_path
        self.mask_dir_list   = mask_dir_list

    def __len__(self):
        return len(self.image_dir_list)
    
    def __getitem__(self,index):        
        #for idx in range(len(self.image_dir_list)):
            #if 'jpg' in self.image_dir_list:
            image_dir = os.path.join(self.train_path,self.image_dir_list[index])
            image=Image.open(image_dir)
            image=np.array(image,dtype=float)
            image = np.transpose(image, (2, 0, 1))
            image=torch.from_numpy(image)
            #self.images.append(image)
        
        #for idx in range(len(self.image_dir_list)):
            #if 'jpg' in self.image_dir_list:
            mask_dir = os.path.join(self.mask_path,self.mask_dir_list[index])
            mask=Image.open(mask_dir)
            mask=np.array(mask,dtype=float)
            mask = np.transpose(mask, (2, 0, 1))
            mask=torch.from_numpy(mask)
            #self.masks.append(mask)
            return image, mask
    
def main():

    train_im_path = "train/images"
    images_dir_list = os.listdir(train_im_path) 
    train_mask_path = "train/masks"
    mask_dir_list = os.listdir(train_mask_path) 

    data=KVasir_dataset(train_im_path,images_dir_list,train_mask_path,mask_dir_list)

    data=KVasir_dataset(train_im_path,train_mask_path,images_dir_list,mask_dir_list)

    train_loader = DataLoader(
        dataset=data,
        batch_size=2,
        shuffle=True,
        num_workers=2 )

    images,masks=next(iter(train_loader))
    print(images,masks)


if __name__ == '__main__':
    main()

