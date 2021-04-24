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

################################################################################
# TOAD-GAN

# Logger init
# logger.remove()
# logger.add(sys.stdout, colorize=True,
#            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
#            + "<level>{level}</level> | "
#            + "<light-black>{file.path}:{line}</light-black> | "
#            + "{message}")

# Parse arguments
opt = get_arguments().parse_args()
opt = post_config(opt)
# device = torch.device("cuda")

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


def load(level_name):
    opt.input_name = level_name
    real = read_level(opt, None, replace_tokens).to(opt.device)
    # Remove the sky layer
    real1 = real[:, :2]
    real2 = real[:, 3:]
    real = torch.cat((real1, real2), dim=1)
    return real

################################################################################
# VAE


class VAE(nn.Module):
    def __init__(self, C, H, W, HIDDEN_SIZE, LATENT_SIZE):
        super(VAE, self).__init__()

        self.shape = (C, H, W)

        self.fc1 = nn.Conv2d(C, 16, 3)
        self.fc2 = nn.Conv2d(16, 4, 3)
        self.fc21 = nn.Linear(HIDDEN_SIZE, LATENT_SIZE)  # mu
        self.fc22 = nn.Linear(HIDDEN_SIZE, LATENT_SIZE)  # var
        self.fc31 = nn.Linear(LATENT_SIZE, HIDDEN_SIZE)
        self.fc32 = nn.Linear(LATENT_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.ConvTranspose2d(4, 16, 3)
        self.fc4 = nn.ConvTranspose2d(16, C, 3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h2f = torch.flatten(h2)
        return self.fc21(h2f), self.fc22(h2f)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        C, H, W = self.shape
        h31 = self.fc31(z)
        h31 = torch.reshape(h31, (1, 4, H-4, W-4))
        h3 = F.relu(self.fc3(h31))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


BATCH_SIZE = 1
EPOCHS = 300
LOG_INTERVAL = 10


def run(real):
    N, C, H, W = real.shape
    print(f'real.shape={real.shape}')

    INPUT_SIZE = H * W
    HIDDEN_SIZE = 4*(H-4)*(W-4)  # opt.hidden_dim
    LATENT_SIZE = 1  # opt.latent_dim

    NUM_SAMPLES = 10

    model = VAE(C, H, W, HIDDEN_SIZE, LATENT_SIZE).cuda()
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
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(real)
        loss, bce, kld = loss_function(recon_batch, real, mu, logvar)
        loss.backward()
        train_loss = loss.item()
        optimizer.step()
        if epoch % LOG_INTERVAL == 0:
            print('====> Epoch: {} Loss: {:.4f} {:.4f} {:.4f}'.format(
                epoch, loss.item(), bce.item(), kld.item()))
        return loss, mu, logvar

    for epoch in range(1, EPOCHS + 1):
        loss, mu, logvar = train(epoch)

    return mu, logvar


models = []
for level_name in ['lvl_1-1.txt', 'lvl_1-2.txt', 'lvl_1-3.txt']:
    real = load(level_name)
    level_models = []
    for n in range(20):
        mu, logvar = run(real)
        mu = mu.detach().cpu().numpy()
        logvar = logvar.detach().cpu().numpy()
        level_models.append(np.array([mu, logvar]))
    models.append(np.array(level_models))
models = np.array(models)

print("DONE!")
print(models.shape)

np.save(f'vae-models-nosky-L1-{EPOCHS}', models)
