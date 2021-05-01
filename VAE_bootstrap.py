from __future__ import print_function
import os

from PIL import Image, ImageOps, ImageEnhance
import torch

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
from torch.nn.functional import interpolate
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from mario.level_utils import one_hot_to_ascii_level, group_to_token, token_to_group, read_level, load_level_from_text
from mario.level_image_gen import LevelImageGen

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class VAE(nn.Module):
    def __init__(self, x):
        super(VAE, self).__init__()
        self.h = x.shape[2]
        self.w = x.shape[3]
        self.fc1 = nn.Conv2d(len(opt.token_list), 12, 3)
        self.fc2 = nn.Conv2d(12, 4, 3)
        self.fc21 = nn.Linear(4*(self.h-4)*(self.w-4), opt.latent_dim) #mu
        self.fc22 = nn.Linear(4*(self.h-4)*(self.w-4), opt.latent_dim) #var
        self.fc31 = nn.Linear(opt.latent_dim, 4*(self.h-4)*(self.w-4))
        self.fc32 = nn.Linear(opt.latent_dim, 4*(self.h-4)*(self.w-4))
        self.fc3 = nn.ConvTranspose2d(4, 12, 3)
        self.fc4 = nn.ConvTranspose2d(12, len(opt.token_list), 3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h2 = torch.flatten(h2)
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = self.fc31(z)
        z = torch.reshape(z,(1,4,(self.h-4),(self.w-4)))
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(1, -1, patch_height, patch_width), reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(x, epoch):
    model.train()
    train_loss = 0
    losses = []
    # for batch_idx, (data, _) in enumerate(train_loader):
            # monitor training loss
    train_loss = 0.0

    #Training
    for i in range(real.shape[0]):
        data = x[i:i+1,:,:,:]            
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        losses.append(loss.item())
        optimizer.step()
    return losses, train_loss, mu, logvar

def generate(n,h,w):
    for j in range(n):
        
        with torch.no_grad():
            samples = [torch.normal(mu,logvar).to(device)]
            samples = torch.tensor([model.decode(sample).cpu().numpy() for sample in samples])
            samples = samples.reshape((1,len(opt.token_list),h,w))
         
        ind = torch.argmax(samples, dim = 1)
        hotenc = torch.zeros_like(samples)
        
        for x in range(hotenc.shape[2]):
            for y in range(hotenc.shape[3]):
                hotenc[0,ind[0,x,y],x,y] = 1
        
        ascii_gen = one_hot_to_ascii_level(hotenc, opt.token_list)
        # ascii_real = one_hot_to_ascii_level(real[9:10,:,:,:], opt.token_list)
        if not os.path.exists(f'{opt.out_dir}/txt'):
            os.makedirs(f'{opt.out_dir}/txt')
        with open(f"{opt.out_dir}/txt/run{j}.txt", 'w') as file:
            for x in ascii_gen:
                file.write(f'{x}')
        
        gen_level = opt.ImgGen.render(ascii_gen)
        # real_level = opt.ImgGen.render(ascii_real)
        
        if n>1:
            if not os.path.exists(f'{opt.out_dir}/img'):
                os.makedirs(f'{opt.out_dir}/img')
            gen_level.save(rf"{opt.out_dir}/img/run{j}.png")
        else:
            return samples
        # real_level.save(rf"VAE_Gen_levels\real.png")

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
device = torch.device("cuda")
# print(opaat)

# Init game specific inputs
replace_tokens = {}
sprite_path = opt.game + '/sprites'
if opt.game == 'mario':
    opt.ImgGen = MarioLevelGen(sprite_path)
    replace_tokens = MARIO_REPLACE_TOKENS
    downsample = special_mario_downsampling
    
# Read Tokens from File
# style = opt.style
# with open(f'tokens_{style}.txt', 'r') as file:
#     tokens = file.read()
#     tokens = list(tokens) 
lev_list = ['1-1','1-2','1-3','2-1','3-1','3-3','4-1','4-2','5-1','5-3','6-1','6-2','6-3','7-1','8-1']
opt.input_dir = 'input'

for j in lev_list:
    opt.input_name = f'lvl_{j}.txt'
    opt.out_dir = f'VAE_Gen_Levels/bootstrap/lvl_{j}'
    real = read_level(opt, None, replace_tokens).to(opt.device)
    levels = {}
    
    levels['1'] = real
    
    for scale in opt.scales:
        levels[f'{scale}'] = downsample(1, [[scale, scale]], real, opt.token_list)[0]
        
    opt.scales = [0.5, 0.75, 0.88, 1]  
    
    # Train Modelf
    losses = []
    avg_losses = []
    
    epochs = 2000
    
    for sc in opt.scales:
        model = VAE(levels[f'{sc}']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        print(f'Training Scale: {sc}')
        for epoch in range(0, epochs):
            if sc == 0.5:
                loss, avg_loss, mu, logvar = train(levels[f'{sc}'], epoch)
            else:
                gen_upsamp = interpolate(gen, levels[f'{sc}'].shape[-2:], mode="bilinear",align_corners=False).to(opt.device)
                in_lvl = gen_upsamp + levels[f'{sc}']
                in_lvl = in_lvl/torch.max(in_lvl)
                loss, avg_loss, mu, logvar = train(in_lvl, epoch)
            losses.append(loss)
            avg_losses.append(avg_loss)
            if((epoch+1)%100==0):
                print('====> Epoch: {} Average loss: {:.4f}'.format(epoch+1, avg_loss))
        _,_,h,w = levels[f'{sc}'].shape
        gen = generate(1,h,w)
            
            
    plt.figure()
    plt.plot(avg_losses)
    
    # Generate Samples
    n = 50   
    generate(n,h,w)
    