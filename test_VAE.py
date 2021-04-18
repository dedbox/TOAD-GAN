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

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
print(real.shape)
# exit()
plt.figure(1)
n = 12
for i in range(n):
    ax = plt.subplot(n,1,i+1)
    plt.imshow(real[0,i,:,:])

# plt.show() #12 patches corresponding to the 12 tokens
patch_height = 16
patch_width = 202
patch_size = patch_height * patch_width

hidden_layer1_size = opt.hidden_dim
hidden_layer2_size = opt.latent_dim

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Conv2d(12, 16, 3)
        self.fc2 = nn.Conv2d(16, 4, 3)
        self.fc21 = nn.Linear(4*12*198, hidden_layer2_size) #mu
        self.fc22 = nn.Linear(4*12*198, hidden_layer2_size) #var
        # self.fc21 = nn.Conv2d(16, 4, 3)
        # self.fc22 = nn.Conv2d(16, 4, 3)
        self.fc31 = nn.Linear(hidden_layer2_size,4*12*198)
        self.fc32 = nn.Linear(hidden_layer2_size,4*12*198)
        self.fc3 = nn.ConvTranspose2d(4, 16, 3)
        self.fc4 = nn.ConvTranspose2d(16, 12, 3)

        # self.fc1 = nn.Linear(patch_size, hidden_layer1_size)
        # self.fc21 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        # self.fc22 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        # self.fc3 = nn.Linear(hidden_layer2_size, hidden_layer1_size)
        # self.fc4 = nn.Linear(hidden_layer1_size, patch_size)

    def encode(self, x):
        # print("000", x.shape)
        # print("111", self.fc1(x.reshape((1, 1, 7, 7))).shape)
        # print("222", F.relu(self.fc1(x.reshape((1, 1, 7, 7)))).shape)
        # h1 = F.relu(self.fc1(x.reshape((1, 1, 7, 7))))
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h2 = torch.flatten(h2)
        print(h2.shape)
        # print("333", self.fc21(h1).shape)
        # print("444", self.fc22(h1).shape)
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        # print("555", mu.shape, std.shape, eps.shape)
        return mu + eps*std

    def decode(self, z):
        # print("666", z.shape)
        # print("777", self.fc3(z).shape)
        # print("888", self.fc3(z).shape)
        # print("999", F.relu(self.fc3(z)).shape)
        z = self.fc31(z)
        z = torch.reshape(z,(1,4,12,198))
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(model)

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



def train(epoch):
    model.train()
    train_loss = 0
    losses = []
    # for batch_idx, (data, _) in enumerate(train_loader):
            # monitor training loss
    train_loss = 0.0

    #Training
    images = real
        
    data = images.to(device)
    # data = data.to(device)
    optimizer.zero_grad()
    print('line165=',data.shape)
    recon_batch, mu, logvar = model(data)
    print('mu size=' , mu.shape)
    loss = loss_function(recon_batch, data, mu, logvar)
    loss.backward()
    train_loss += loss.item()
    losses.append(loss.item())
    optimizer.step()
    # if batch_idx % log_interval == 0:
    # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #     epoch, batch_idx, len(patches),
    #     100. * batch_idx / len(patches),
    #     loss.item()))
        # print('mu = ',mu)
        # print('====> Epoch: {} Average loss: {:.4f}'.format(
        #   epoch, train_loss / len(patches)))

    return losses, train_loss, mu, logvar


losses = []
avg_losses = []
epochs = 400
# if __name__ == "__main__":
for epoch in range(1, epochs + 1):
    loss, avg_loss, mu, logvar = train(epoch)
    losses.append(loss)
    avg_losses.append(avg_loss)
    print('mu = ',mu.shape)
    print('var = ',logvar.shape)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
                                                         epoch, avg_loss))
    
plt.figure(4)
plt.plot(avg_losses)

    # test(epoch)
with torch.no_grad():
        # sample = torch.randn(1, 1, hidden_layer2_size).to(device)
        # print("BEFORE", sample.shape)
        # print("AFTER", sample.shape)
        # save_image(sample.view(1, 1, patch_height, patch_width),
        #             'results/sample_' + str(epoch) + '.png')

    K = 1
    cols = int(np.ceil(np.sqrt(K)))
    rows = K // cols + 1
    # print('mu=',mu)
    # print('var =', logvar)
    # samples_gen = torch.normal(mu,logvar)
    # print('gen =',samples_gen)
    samples = [torch.normal(mu,logvar).to(device) for _ in range(K)]
    # print('samles=',samples)
    samples = np.array([model.decode(sample).cpu().numpy() for sample in samples])
    
    # fig = plt.figure(figsize=(cols * 1.1, rows * 1.4))
    # grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=(0.1, 0.4))
    
    # # fig.suptitle(f'PCA Top {K} of {N} ({int(100*precision)}% total variance)')
    # for k, ax, img in zip(range(10), grid, samples):
    #    # plt.clf()
    #     ax.set_title(f'{k+1}')
    #     plt.imshow(img.reshape((patch_height, patch_width)), cmap='magma')
    # # for i in range(K, rows * cols):
    # #     grid[i].set_visible(False)
    # fig.tight_layout()
    # # plt.savefig(f'pca_{opt.input_name[:-4]}.png', bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    
    # if opt.vae_show:
    #     # fig = plt.figure(figsize=(cols * 1.1, rows * 1.4))
    #     # grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=(0.1, 0.4))
    #     #Reconstructed Images
    #     print(len(samples))
    #     samples = samples.reshape((1,12,16,202))
    #     print('Reconstructed Images')
    #     plt.figure(3)
    #     n = 12
    #     for i in range(n):
    #         ax = plt.subplot(n,1,i+1)
    #         plt.imshow(samples[0,i,:,:])
    if opt.vae_show:
        # fig = plt.figure(figsize=(cols * 1.1, rows * 1.4))
        # grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=(0.1, 0.4))
        #Reconstructed Images
        print(len(samples))
        samples = samples.reshape((1,12,16,202))
        print('Reconstructed Images')
        plt.figure(3)
        n = 12
        for i in range(n):
            ax = plt.subplot(n,1,i+1)
            plt.imshow(samples[0,i,:,:])
            # fig.suptitle(f'VAE {K} Random Samples, H={opt.hidden_dim}, L={opt.latent_dim}')
            # for k, ax, img in zip(range(K), grid, samples):
            #     ax.axis('off')
            #     ax.set_title(f'{k+1}')
            #     ax.imshow(img.reshape((12,16,202)), cmap='magma')
            # for i in range(K, rows * cols):
            #     grid[i].set_visible(False)
            # fig.tight_layout()
            # if opt.vae_save:
            #     plt.savefig(f'vae-{opt.input_name[:-4]}-{opt.hidden_dim}-{opt.latent_dim}-{epoch}.png', bbox_inches='tight', pad_inches=0.1)
            # else:
            #     plt.show()

if opt.vae_save:
    np.savez(f"vae-{opt.input_name[:-4]}-{opt.hidden_dim}-{opt.latent_dim}", loss=np.array(losses), avg_loss=np.array(avg_losses))


ind = np.argmax(samples, axis = 1)
hotenc = np.zeros_like(samples)

for x in range(hotenc.shape[2]):
    for y in range(hotenc.shape[3]):
        hotenc[0,ind[0,x,y],x,y] = 1

ascii_gen = one_hot_to_ascii_level(hotenc, opt.token_list)
ascii_real = one_hot_to_ascii_level(real, opt.token_list)

gen_level = opt.ImgGen.render(ascii_gen)
real_level = opt.ImgGen.render(ascii_real)

gen_level.save(rf"VAE_Gen_levels\{epochs}.png")
real_level.save(rf"VAE_Gen_levels\real.png")