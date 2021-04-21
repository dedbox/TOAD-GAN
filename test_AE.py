# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 01:16:10 2021

@author: eesha


"""
from __future__ import print_function
import os

from PIL import Image, ImageOps, ImageEnhance
import torch


from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose
from keras.datasets import mnist

from mario.tokens import REPLACE_TOKENS as MARIO_REPLACE_TOKENS
from mario.level_image_gen import LevelImageGen as MarioLevelGen
from mario.special_mario_downsampling import special_mario_downsampling
from mario.level_utils import read_level, read_level_from_file
from config import get_arguments, post_config
from loguru import logger
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from mario.level_utils import one_hot_to_ascii_level, group_to_token, token_to_group, read_level
from mario.level_image_gen import LevelImageGen

sprite_path = r'D:\UMich\Courses\EECS 545\project\TOAD-GAN-master\TOAD-GAN-master\mario\sprites'
mariosheet = Image.open(os.path.join(sprite_path, 'smallmariosheet.png'))
print(mariosheet)
enemysheet = Image.open(os.path.join(sprite_path, 'enemysheet.png'))
itemsheet = Image.open(os.path.join(sprite_path, 'itemsheet.png'))
mapsheet = Image.open(os.path.join(sprite_path, 'mapsheet.png'))

sprite_dict = dict()

# Mario Sheet
sprite_dict['M'] = mariosheet.crop((4*16, 0, 5*16, 16))

# Enemy Sheet
enemy_names = ['r', 'k', 'g', 'y', 'wings', '*', 'plant']
for i, e in enumerate(enemy_names):
    sprite_dict[e] = enemysheet.crop((0, i*2*16, 16, (i+1)*2*16))

sprite_dict['E'] = enemysheet.crop((16, 2*2*16, 2*16, 3*2*16))  # Set generic enemy to second goomba sprite
sprite_dict['plant'] = enemysheet.crop((16, (len(enemy_names)-1)*2*16, 2*16, len(enemy_names)*2*16))

# Item Sheet
sprite_dict['shroom'] = itemsheet.crop((0, 0, 16, 16))
sprite_dict['flower'] = itemsheet.crop((16, 0, 2*16, 16))
sprite_dict['flower2'] = itemsheet.crop((0, 16, 16, 2*16))
sprite_dict['1up'] = itemsheet.crop((16, 16, 2*16, 2*16))

enemy_names = ['r', 'k', 'g', 'y', 'wings', '*', 'plant']
for i, e in enumerate(enemy_names):
    print(i,e)


# Logger init
logger.remove()
logger.add(sys.stdout, colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                    + "<level>{level}</level> | "
                    + "<light-black>{file.path}:{line}</light-black> | "
                    + "{message}")

# Parse arguments
opt = get_arguments().parse_args()
opt = post_config(opt)
# print(opt)

# Init game specific inputs
replace_tokens = {}
sprite_path = opt.game + '/sprites'
if opt.game == 'mario':
    opt.ImgGen = MarioLevelGen(sprite_path)
    replace_tokens = MARIO_REPLACE_TOKENS
    downsample = special_mario_downsampling
elif opt.game == 'mariokart':
    opt.ImgGen = MariokartLevelGen(sprite_path)
    replace_tokens = MARIOKART_REPLACE_TOKENS
    downsample = special_mariokart_downsampling
else:
    NameError("name of --game not recognized. Supported: mario, mariokart")


real = read_level(opt, None, replace_tokens).to(opt.device)
real = real.cpu()
print(type(real))
print(real.shape)
# exit()
plt.figure(1)
n = 12
for i in range(n):
    ax = plt.subplot(n,1,i+1)
    plt.imshow(real[0,i,:,:])

plt.show(block = False) #12 patches corresponding to the 12 tokens

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv2d(12, 16, 3)  
        self.conv2 = nn.Conv2d(16, 4, 3)
        # self.pool = nn.MaxPool2d(2, 2)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 3)
        self.t_conv2 = nn.ConvTranspose2d(16, 12, 3)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = self.pool(x)
        x = F.relu(self.conv2(x))
        # x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
              
        return x
    
#Instantiate the model
model = ConvAutoencoder()
print(model)

#Loss function
criterion = nn.BCELoss()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

device = get_device()
print(device)
model.to(device)

#Epochs
n_epochs = 700

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0

    #Training
    images = real
        
    images = images.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    # print(outputs.shape)
    loss = criterion(outputs, images)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()*images.size(0)
          
    train_loss = train_loss/1
    if((epoch % 100)==0):
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
print('training complete')
    
#     #Batch of test images
# dataiter = iter(test_loader)
# images, labels = dataiter.next()


# images = images.numpy()

# output = output.view(batch_size, 3, 32, 32)
# output = output.detach().numpy()

#Original Images
print("Original Images")
plt.figure(2)
n = 12
for i in range(n):
    ax = plt.subplot(n,1,i+1)
    plt.imshow(real[0,i,:,:])
plt.show(block = False)    
   

#Sample outputs
images = real
images = images.to(device)
output = model(images)
output = output.cpu()
# output = output.detach().numpy()
# print(output[0,2,:,0:10])
#Reconstructed Images
# print('Reconstructed Images')
# plt.figure(3)
# n = 12
# for i in range(n):
#     ax = plt.subplot(n,1,i+1)
#     plt.imshow(output[0,i,:,:])
# plt.show()

ind = torch.argmax(output, dim = 1)
hotenc = torch.zeros_like(output)

for x in range(hotenc.shape[2]):
    for y in range(hotenc.shape[3]):
        hotenc[0,ind[0,x,y],x,y] = 1

ascii_gen = one_hot_to_ascii_level(hotenc, opt.token_list)
ascii_real = one_hot_to_ascii_level(real, opt.token_list)

gen_level = opt.ImgGen.render(ascii_gen)
real_level = opt.ImgGen.render(ascii_real)

gen_level.save(rf"Gen_Levels\{n_epochs}.png")
# real_level.save(r"Gen_Levels\real.png")
# # y = torch.randn(5,5)
# # print(y)
# print(y.shape)

# z = y.unfold(0,3,2)
# print(z)
# print(z.shape)

