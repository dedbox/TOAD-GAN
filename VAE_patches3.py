import sys
from loguru import logger
from config import get_arguments, post_config

from mario.tokens import REPLACE_TOKENS as MARIO_REPLACE_TOKENS
from mario.level_image_gen import LevelImageGen as MarioLevelGen
from mario.special_mario_downsampling import special_mario_downsampling
from mario.level_utils import one_hot_to_ascii_level, group_to_token, token_to_group, read_level

import torch
import torch.nn.functional as F
from torch import nn, optim

import numpy as np

import matplotlib.pyplot as plt

from ToadVAE import LevelPatchesDataset, ToadPatchVAE, ToadLevelVAE


################################################################################
# TOAD-GAN

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

################################################################################
# MarioVAE

N, C, H_real, W_real = real.shape
print(f'real {real.shape}')

# architecture
PATCH_SIZE = 7
LATENT_SIZE = 7

# training
BATCH_SIZE = 1
EPOCHS = 200
# MAX_LOSS = 300
LOG_INTERVAL = 10

# output
NUM_SAMPLES = 10

# adapted from
# https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/

# load the data
dataset = LevelPatchesDataset(real, patch_size=(PATCH_SIZE, PATCH_SIZE))
patches = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True)

# model = ToadPatchVAE(C, PATCH_SIZE, LATENT_SIZE).cuda()
model = ToadLevelVAE(C, PATCH_SIZE, LATENT_SIZE).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(model)


def loss_function(recon_x, x, mu, logvar):
    '''Reconstruction + KL divergence losses summed over all elements and batch.

    see Appendix B from VAE paper:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114

    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    '''
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(patches):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch % LOG_INTERVAL == 0:
        print('====> Epoch: {} Batch: {} Loss: {:.4f} {:.4f} {:.4f}'.format(
            epoch, batch_idx, loss.item(), bce.item(), kld.item()))
    return train_loss, mu, logvar


epochs = 0
# while True:
for _ in range(EPOCHS):
    epochs += 1
    loss, mu, logvar = train(epochs)
    # if loss <= MAX_LOSS:
    #     break

with torch.no_grad():
    samples = [torch.normal(mu, logvar.exp().sqrt()) for _ in range(NUM_SAMPLES)]
    samples = [model.decode(sample) for sample in samples]


def one_hot(x):
    x = F.one_hot(x, C).cuda()
    x = torch.transpose(x, 2, 3)
    x = torch.transpose(x, 1, 2)
    return x


ind = [torch.argmax(sample, dim=1) for sample in samples]
hotenc = [one_hot(x) for x in ind]

ascii_gens = [one_hot_to_ascii_level(x, opt.token_list) for x in hotenc]
ascii_real = one_hot_to_ascii_level(real, opt.token_list)

gen_levels = [opt.ImgGen.render(x) for x in ascii_gens]
real_level = opt.ImgGen.render(ascii_real)

for i, gen_level in enumerate(gen_levels):
    gen_level.save(rf"VAE_Gen_patches\{EPOCHS}-{i}.png")
