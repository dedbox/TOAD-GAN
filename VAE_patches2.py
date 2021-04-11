from __future__ import print_function



from mario.tokens import REPLACE_TOKENS as MARIO_REPLACE_TOKENS
from mario.level_image_gen import LevelImageGen as MarioLevelGen
from mario.special_mario_downsampling import special_mario_downsampling
from mario.level_utils import read_level, read_level_from_file
from config import get_arguments, post_config
from loguru import logger
import sys

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

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

skymap = real[:, opt.token_list.index('-')]
B, H, W = skymap.shape
skymap_patches = skymap.unfold(1, 7, 1).unfold(2, 7, 1).reshape(1, H - 7 + 1, W - 7 + 1, 7*7)

patches = skymap_patches.cpu().detach().numpy()[0]
h, w, D = patches.shape
N = h * w
# patches = patches.reshape((N, 7, 7))[:, :, :, np.newaxis]
# patches = patches.reshape(())

patches = skymap_patches.reshape((N, 7, 7, 1))

# print("AAA", real.shape, skymap_patches.shape, patches.shape)
# exit()

################################################################################

# import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


# parser = argparse.ArgumentParser(description='VAE MNIST Example')
# parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()

# torch.manual_seed(args.seed)

log_interval = 100
epochs = 10

device = torch.device("cuda")

# kwargs = {'num_workers': 1, 'pin_memory': True}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

patch_height = 7
patch_width = 7
patch_size = patch_height * patch_width
hidden_layer1_size = 30
hidden_layer2_size = 7

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # self.fc1 = nn.Conv2d(1, 1, 3)
        # self.fc21 = nn.Conv2d(1, 1, 2)
        # self.fc22 = nn.Conv2d(1, 1, 2)
        # self.fc3 = nn.ConvTranspose2d(1, 1, 2)
        # self.fc4 = nn.ConvTranspose2d(1, 1, 3)

        self.fc1 = nn.Linear(patch_size, hidden_layer1_size)
        self.fc21 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.fc22 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.fc3 = nn.Linear(hidden_layer2_size, hidden_layer1_size)
        self.fc4 = nn.Linear(hidden_layer1_size, patch_size)

    def encode(self, x):
        # print("000", x.shape)
        # print("111", self.fc1(x.reshape((1, 1, 7, 7))).shape)
        # print("222", F.relu(self.fc1(x.reshape((1, 1, 7, 7)))).shape)
        # h1 = F.relu(self.fc1(x.reshape((1, 1, 7, 7))))
        h1 = F.relu(self.fc1(x))
        # print("333", self.fc21(h1).shape)
        # print("444", self.fc22(h1).shape)
        return self.fc21(h1), self.fc22(h1)

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
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, patch_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(1, -1, patch_height, patch_width), reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, patch_height * patch_width), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    # for batch_idx, (data, _) in enumerate(train_loader):
    for batch_idx, data in enumerate(patches):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(patches),
                100. * batch_idx / len(patches),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(patches)))


# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         # for i, (data, _) in enumerate(test_loader):
#         for i, data in enumerate(patches):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), patch_height)
#                 comparison = torch.cat([data[:n],
#                                       recon_batch.view(1, patch_height, patch_width)[:n]])
#                 save_image(comparison.cpu(),
#                          'results/reconstruction_' + str(epoch) + '.png', nrow=n)

#     test_loss /= D
#     print('====> Test set loss: {:.4f}'.format(test_loss))

# if __name__ == "__main__":
for epoch in range(1, epochs + 1):
    train(epoch)
    # test(epoch)
    with torch.no_grad():
        # sample = torch.randn(1, 1, hidden_layer2_size).to(device)
        # print("BEFORE", sample.shape)
        # print("AFTER", sample.shape)
        # save_image(sample.view(1, 1, patch_height, patch_width),
        #             'results/sample_' + str(epoch) + '.png')

        K = 25
        cols = int(np.ceil(np.sqrt(K)))
        rows = K // cols + 1

        samples = [torch.randn(1, 1, hidden_layer2_size).to(device) for _ in range(K)]
        samples = [model.decode(sample).cpu() for sample in samples]

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

        fig = plt.figure(figsize=(cols * 1.1, rows * 1.4))
        grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=(0.1, 0.4))

        # fig.suptitle(f'PCA Top {K} of {N} ({int(100*precision)}% total variance)')
        for k, ax, img in zip(range(K), grid, samples):
            ax.axis('off')
            ax.set_title(f'{k+1}')
            ax.imshow(img.reshape((7, 7)), cmap='magma')
        for i in range(K, rows * cols):
            grid[i].set_visible(False)
        fig.tight_layout()
        # plt.savefig(f'pca_{opt.input_name[:-4]}.png', bbox_inches='tight', pad_inches=0.1)
        plt.show()
