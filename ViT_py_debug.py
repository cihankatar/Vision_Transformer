

import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
import matplotlib.pyplot as plt
from patchify import patchify

##PATCHFY FUNCTION & POSITIONAL_EMB

def patching_func(images,patch_size):

    n,c,h_iamge,w_image=images.shape
    n_patch=0
    n_patch_per_img = int((h_iamge/patch_size)**2)
    patch_vector=int(patch_size**2)
    images=images.squeeze()

    total_patches = np.zeros((n,n_patch_per_img,patch_vector))
 
    for idx,image in enumerate(images):
        patching = patchify(image, (patch_size,patch_size),step=4) # split image into 7x7 small 4x4 patches.patchify(image, (3, 3), step=1)
        for i in range(patching.shape[0]):
            for j in range (patching.shape[1]):
                single_patch=patching[i,j,:,:]
                total_patches[idx,n_patch,:]=single_patch.flatten()
                n_patch=n_patch+1
                if n_patch==49:
                    n_patch=0
    return total_patches

#IMPORT DATA
# 
transform = ToTensor()
train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
test_set = MNIST(root='./../datasets', train=False, download=False, transform=transform)

train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

#DATA TO PATCH
for batch in train_loader:
    images, targets = batch 

#plt.imshow(images[0].squeeze())      #1x28x28--> 28x28
images=images.numpy()

print(images.shape)
print(patching_func(images,4).shape)
patches=torch.tensor(patching_func(images,4))
print(patches.shape)
print(patches.float().dtype)

token_dim=8
patch_size=4

class_token = nn.Parameter(torch.rand(96,1, token_dim))
linear_map=nn.Linear(patch_size**2,token_dim)

#print(class_token.shape)
#print(patches.dtype)
#print(patches.shape)

linear_emb = linear_map(patches.float())
#print(linear_emb.shape)

#print(class_token.shape)

tokens = torch.cat((class_token,linear_emb),dim=1)
#print(tokens.shape)

n,number_tokens,patch_size = tokens.shape
result = torch.zeros(n,number_tokens,patch_size)
print('result: ',result.shape)

n_heads    = 2

q_layers   = nn.ModuleList([nn.Linear(token_dim,token_dim) for _ in range(n_heads)])
k_layers   = nn.ModuleList([nn.Linear(token_dim,token_dim) for _ in range(n_heads)])
v_layers   = nn.ModuleList([nn.Linear(token_dim,token_dim) for _ in range(n_heads)])
softmax    = nn.Softmax()

n,number_tokens,patch_size = tokens.shape
result = torch.zeros(n,number_tokens*n_heads,patch_size)

for idx,token in enumerate(tokens):   # 128 batch. each of 50x8, token size : 50x8   --> 50x8

    concat      = torch.zeros(n_heads,number_tokens,patch_size)

    for head in range(n_heads):        # number of heads : 2
        q_linear = q_layers[head]      # linear (8x8)  == 50x8 --> 50x8
        k_linear = k_layers[head]
        v_linear = v_layers[head]

        q  = q_linear(token)
        k  = k_linear(token)
        v  = v_linear(token)

        mat_mul = (torch.matmul(q, k.T)) / ((number_tokens-1)**0.5)   # 50x8 x 8x50 = 50x50 
        attention_mask  = softmax(mat_mul)
        attention       = torch.matmul(attention_mask,v)
        concat[head,:,:]= attention
    result[idx,:,:] = torch.flatten(input=concat, start_dim=0, end_dim=1)

print(result.shape)
print(concat.shape)