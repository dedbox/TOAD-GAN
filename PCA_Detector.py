import argparse
import random

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.decomposition import PCA
from scipy import signal

import cv2

from mario.tokens import REPLACE_TOKENS
from mario.level_image_gen import LevelImageGen
from mario.special_mario_downsampling import special_mario_downsampling

################################################################################
# PyTorch

# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device:', device, end=' ')
if device.type == 'cuda':
    print(f'({torch.cuda.get_device_name(0)})')
    print('Memory Allocated:',
          round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Memory Cached:   ',
          round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')
else:
    print()

################################################################################
# TOAD-GAN

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--not_cuda", help="disables cuda",
                    action="store_true", default=0)
parser.add_argument("--seed", help="manual seed", type=int)
parser.add_argument("--input-dir", help="input image dir", default="input")
parser.add_argument("--input-name", help="input image name",
                    default="lvl_1-1.txt")
parser.add_argument("--patch-width", help="horizontal patch dimension",
                    default=7)
parser.add_argument("--patch-height", help="vertical patch dimension",
                    default=7)
parser.add_argument("--kernel-width", help="horizontal kernel dimension",
                    default=7)
parser.add_argument("--kernel-height", help="vertical kernel dimension",
                    default=7)
parser.add_argument("--conv-layers", help="number of convolutional layers",
                    default=1)
opt = parser.parse_args()


def set_seed(seed=0):
    """ Set the seed for all possible sources of randomness to allow for reproduceability. """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


# configure default state
opt.device = "cpu" if opt.not_cuda else device
if torch.cuda.is_available() and opt.not_cuda:
    print("WARNING: CUDA device is present but disabled.")
if opt.seed is None:
    opt.seed = random.randint(1, 10000)
print("Random Seed: ", opt.seed)
set_seed(opt.seed)
opt.ImgGen = LevelImageGen('mario/sprites')


def read_level(opt):
    """ Wrapper function for read_level_from_file using namespace opt. Updates parameters for opt."""
    level, uniques = read_level_from_file(opt.input_dir, opt.input_name)
    opt.token_list = uniques
    print("Tokens in {}/{}: {}".format(
        opt.input_dir,
        opt.input_name,
        ' '.join(opt.token_list)))
    return level


def read_level_from_file(input_dir, input_name):
    """ Returns a full token level tensor from a .txt file. Also returns the unique tokens found in this level.
    Token. """
    txt_level = load_level_from_text("%s/%s" % (input_dir, input_name))
    uniques = set()
    for line in txt_level:
        for token in line:
            # if token != "\n" and token != "M" and token != "F":
            if token != "\n" and token not in REPLACE_TOKENS.items():
                uniques.add(token)
    uniques = list(uniques)
    uniques.sort()  # necessary! otherwise we won't know the token order later
    oh_level = ascii_to_one_hot_level(txt_level, uniques)
    return oh_level.unsqueeze(dim=0), uniques


def load_level_from_text(path_to_level_txt):
    """ Loads an ascii level from a text file. """
    with open(path_to_level_txt, "r") as f:
        ascii_level = []
        for line in f:
            for token, replacement in REPLACE_TOKENS.items():
                line = line.replace(token, replacement)
            ascii_level.append(line)
    return ascii_level


def ascii_to_one_hot_level(level, tokens):
    """ Converts an ascii level to a full token level tensor. """
    oh_level = torch.zeros((len(tokens), len(level), len(level[-1])))
    for i in range(len(level)):
        for j in range(len(level[-1])):
            token = level[i][j]
            if token in tokens and token != "\n":
                oh_level[tokens.index(token), i, j] = 1
    return oh_level


def one_hot_to_ascii_level(level, tokens):
    """ Converts a full token level tensor to an ascii level. """
    ascii_level = []
    for i in range(level.shape[2]):
        line = ""
        for j in range(level.shape[3]):
            line += tokens[level[:, :, i, j].argmax()]
        if i < level.shape[2] - 1:
            line += "\n"
        ascii_level.append(line)
    return ascii_level

################################################################################
# PCA_CNN


class PCA_Detector:
    def __init__(self, kernels, shape):
        self.kernels = kernels.to(opt.device)

        K, H_kernel, W_kernel = kernels.shape
        self.kernel_dims = (H_kernel, W_kernel)

        N, C, H, W = shape
        self.input_dims = shape

        H_conv = H - H_kernel + 1
        W_conv = W - W_kernel + 1
        H_pool = int((H_conv - H_kernel + 0.5) // H_kernel + 1)
        W_pool = int((W_conv - W_kernel + 0.5) // W_kernel + 1)
        self.output_dims = (H_pool, W_pool)

    def __call__(self, x):
        N, C, H, W = x.shape
        H_kernel, W_kernel = self.kernel_dims
        h1 = [F.relu(F.conv2d(x, kernel.view(1, 1, H_kernel, W_kernel)))
              for kernel in self.kernels]
        h2 = [F.avg_pool2d(h, (H_kernel, W_kernel)) for h in h1]
        h3 = torch.cat(h2, dim=2)
        return h3


class LevelPatchesDataset(torch.utils.data.Dataset):
    """Mario level patches dataset."""

    def __init__(self, level, patch_size):
        H_patch, W_patch = patch_size
        N_level, C, H_level, W_level = level.shape
        assert N_level == 1

        # extract patches
        H_grid = H_level - H_patch + 1
        W_grid = W_level - W_patch + 1
        patches = level.unfold(2, H_patch, 1).unfold(3, W_patch, 1)
        patches = patches.transpose(1, 2).transpose(2, 3)
        patches = patches.reshape(
            N_level * H_grid * W_grid, C, H_patch, W_patch)
        self.level = level
        self.patches = patches.unsqueeze(1)

        # create normalization transforms
        mean = patches.mean()
        std = patches.std()
        self.normalize = torchvision.transforms.Normalize(
            mean.tolist(), std.tolist())
        self.unnormalize = torchvision.transforms.Normalize(
            (-mean / std).tolist(), (1.0 / std).tolist())

    def __len__(self):
        return len(self.level)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.patches[index]


input_names = ['1-1', '1-2', '1-3', '2-1', '3-1', '3-3', '4-1',
               '4-2', '5-1', '5-3', '6-1', '6-2', '6-3', '7-1', '8-1']

# Load all levels
reals = {}
normalize = {}
unnormalize = {}
for input_name in input_names:
    # Load a level
    opt.input_name = f'lvl_{input_name}.txt'
    real = read_level(opt).to(opt.device)
    N, C, H, W = real.shape
    print("real", real.shape)

    # Remove the sky
    sky_index = opt.token_list.index('-')
    real = torch.cat((real[:, :sky_index], real[:, sky_index+1:]), dim=1)
    C -= 1

    # Undo one-hot encoding
    real = real.argmax(dim=1).unsqueeze(1).float()

    # Define normalization transforms
    std, mean = torch.std_mean(real)
    normalize[input_name] = lambda x: (x - mean) / std
    unnormalize[input_name] = lambda x: x * std + mean

    # Normalize the input
    reals[input_name] = normalize[input_name](real)

# Build a PCA_Detector for each level
detectors = {}
for input_name in input_names:
    real = reals[input_name]
    N, C, H, W = real.shape

    # Extract patches
    H_patch, W_patch = (7, 7)
    H_grid = H - H_patch + 1
    W_grid = W - W_patch + 1
    patches = real.unfold(2, H_patch, 1).unfold(3, W_patch, 1)
    patches = patches.transpose(1, 2).transpose(2, 3)
    patches = patches.reshape(H_grid * W_grid, H_patch * W_patch)
    patches = patches.detach().cpu().numpy()

    # Extract all eigenpatches
    principals = PCA().fit_transform(patches.T).T.reshape(-1, H_patch, W_patch)
    principals = torch.tensor(principals).to(device)

    # Build an detector with the eigenpatches
    detector = PCA_Detector(principals, real.shape)
    print("detector", detector,
          "input_dims",  np.array(detector.input_dims),
          "output_dims", np.array(detector.output_dims))
    detectors[input_name] = detector


def divergence(a, b):
    H_a, W_a = a.shape[2:]
    H_b, W_b = b.shape[2:]
    H = max(H_a, H_b)
    W = max(W_a, W_b)
    A = F.pad(a, (0, W - W_a, 0, H - H_a))
    B = F.pad(b, (0, W - W_b, 0, H - H_b))
    return torch.linalg.norm(B - A).item()


N = len(input_names)

# Compute divergence between input pairs
data = np.zeros((N, N))
for i, input_name1 in enumerate(input_names):
    for j, input_name2 in enumerate(input_names):
        a = detectors[input_name1](reals[input_name1])
        b = detectors[input_name1](reals[input_name2])
        data[i, j] = divergence(a, b)

plt.imshow(data, interpolation='nearest', extent=[0, N, 0, N])
# plt.axis(False)
plt.xticks([i + 0.5 for i in range(N)], input_names)
plt.yticks([i + 0.5 for i in range(N)], reversed(input_names))
plt.show()
plt.savefig()

exit()

y = model(real0).to(device)
y = F.interpolate(y, (y.shape[2], W))
z1 = torch.sum(y, dim=2)
z1 = (z1 - z1.mean()) / z1.std()
z2 = torch.sum(y)

# Undo one-hot encoding
# y = y.argmax(dim=1).unsqueeze(1).float()

x = real0[0, 0].detach().cpu().numpy()
y = y[0, 0].detach().cpu().numpy()
z1 = z1[0].detach().cpu().numpy()
z2 = z2.item()

print("x", x.shape, "mean", x.mean(), "std", x.std())
print(x)

print("z1", z1.shape, "mean", z1.mean(), "std", z1.std())
print(z1)

print("z2", z2)

fig = plt.figure(figsize=(W, H + len(principals)))
grid = ImageGrid(fig, 111, nrows_ncols=(3, 1), axes_pad=(0.1, 0.1))
for ax, img, cmap in zip(grid, [x, z1, y], ['tab20b', 'magma', 'magma']):
    ax.axis('off')
    ax.imshow(img, cmap=cmap)
fig.tight_layout()
plt.show()

exit()


# Apply top-K eigenpatches as kernels
real1 = real0.detach().cpu().numpy()[0, 0]

K = 9

detectors = []
for k in range(K):
    detector = signal.convolve2d(
        real1, principals[k], mode='same', boundary='symm')
    detector = torch.tensor(detector).to(opt.device).reshape(1, 1, H, W)
    detector = F.avg_pool2d(detector, 2)
    detector = F.interpolate(detector, (H, W))
    detector = F.relu(detector)
    detector = detector[0, 0].detach().cpu().numpy()
    detector = signal.convolve2d(
        real1, principals[k], mode='same', boundary='symm')
    detector = torch.tensor(detector).to(opt.device).reshape(1, 1, H, W)
    detector = F.avg_pool2d(detector, 4)
    detector = F.interpolate(detector, (H, W))
    detector = F.relu(detector)
    # detector = detector.mean(dim=2).unsqueeze(2)
    # detector = detector.repeat(1, 1, H, 1)
    detector = detector[0, 0].detach().cpu().numpy()
    detectors.append(detector)

tmp = np.array(detectors).sum(axis=0)
# tmp = np.tile(tmp, (H, 1))

rows, cols = (2, 1)

fig = plt.figure(figsize=(cols * 1.1, rows * 1.1))
grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=(0.1, 0.4))

for k, ax, img, cmap in zip(range(2), grid, [real1, tmp], ['tab20b', 'magma']):
    ax.axis('off')
    ax.imshow(img, cmap=cmap)
fig.tight_layout()
plt.show()
# exit()

# Plot eignepatches

rows, cols = (1, K)

fig = plt.figure(figsize=(cols * 1.1, rows * 1.4))
grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=(0.1, 0.4))

# fig.suptitle(f'VAE {K} Random Samples, H={opt.hidden_dim}, L={opt.latent_dim}')
for k, ax, img in zip(range(K), grid, principals):
    ax.axis('off')
    # ax.set_title(f'{k+1}')
    ax.imshow(img, cmap='magma')
# for i in range(K, rows * cols):
#     grid[i].set_visible(False)
fig.tight_layout()
# if opt.vae_save:
#     plt.savefig(f'vae-{opt.input_name[:-4]}-{opt.hidden_dim}-{opt.latent_dim}-{epoch}.png', bbox_inches='tight', pad_inches=0.1)
# else:
plt.show(block=False)

# Plot the outputs

# rows, cols = (int((K+1) / 2 + 0.5), 2)
rows, cols = (K+1, 1)

fig = plt.figure(figsize=(cols * 1.1, rows * 1.1))
grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=(0.1, 0.4))

# fig.suptitle(f'VAE {K} Random Samples, H={opt.hidden_dim}, L={opt.latent_dim}')
for k, ax, img, cmap in zip(range(K+1), grid, [real1] + detectors, ['tab20b'] + ['magma'] * K):
    ax.axis('off')
    # ax.set_title(f'{k+1}')
    ax.imshow(img, cmap=cmap)
# for i in range(K, rows * cols):
#     grid[i].set_visible(False)
fig.tight_layout()
# if opt.vae_save:
#     plt.savefig(f'vae-{opt.input_name[:-4]}-{opt.hidden_dim}-{opt.latent_dim}-{epoch}.png', bbox_inches='tight', pad_inches=0.1)
# else:
plt.show()

exit()

# # Cut the level up into overlapping patches
# dataset = LevelPatchesDataset(real0, (opt.kernel_height, opt.kernel_width))
# patches = torch.utils.data.DataLoader(
#     dataset, batch_size=1, shuffle=True)
# print(next(iter(dataset)))
# exit()

# # Build the model
# model = PlayCNN(real).to(opt.device)
# print(model)

# # Configure the learning process
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# # Train the model
# model.train()

# losses = []
# for epoch in range(500):
#     optimizer.zero_grad()
#     y = model.forward(real)
#     loss = criterion(y, real)
#     loss.backward()
#     optimizer.step()
#     losses.append(loss.item())
#     print("loss", loss.item())

# plt.title('Loss')
# plt.xlabel('iteration')
# plt.plot(losses)
# plt.show()

# # Use the model
# model.eval()

# pred = model(real).detach().cpu().numpy()
# #  F.one_hot(model(real).argmax(dim=1)).transpose(2, 3).transpose(1, 2)

# # render the real level
# ascii_real = one_hot_to_ascii_level(real, opt.token_list)
# real_level = opt.ImgGen.render(ascii_real)
# real_rgb = np.array(real_level) / 255

# # render the predicted level
# ascii_pred = one_hot_to_ascii_level(pred, opt.token_list)
# pred_level = opt.ImgGen.render(ascii_pred)
# pred_rgb = np.array(pred_level)

# # # render the final visualization
# # plt.figure(figsize=(real.shape[2], real.shape[3]))
# # plt.imshow(real_rgb)
# # plt.axis('off')
# # plt.tight_layout()
# # plt.show()

# # render the final visualization
# plt.figure(figsize=(pred.shape[2], pred.shape[3]))
# plt.imshow(pred_rgb)
# plt.axis('off')
# plt.tight_layout()
# plt.show()