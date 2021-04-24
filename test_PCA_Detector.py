import argparse
import random

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from mario.tokens import REPLACE_TOKENS
from mario.level_image_gen import LevelImageGen
from mario.special_mario_downsampling import special_mario_downsampling

from PCA_Detector import PCA_Detector, difference, divergence

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
# PCA_Detector


input_names = ['1-1', '1-2', '1-3', '2-1', '3-1', '3-3', '4-1',
               '4-2', '5-1', '5-3', '6-1', '6-2', '6-3', '7-1', '8-1']

# # render the real levels
# for input_name in input_names:
#     opt.input_name = f'lvl_{input_name}.txt'
#     real = read_level(opt).to(opt.device)
#     ascii_real = one_hot_to_ascii_level(real, opt.token_list)
#     real_level = opt.ImgGen.render(ascii_real)
#     real_level.save(f'output/lvl_{input_name}.png', format='png')


def preprocess(level):
    # remove the sky layer
    sky_index = opt.token_list.index('-')
    before_sky = level[:, :sky_index]
    after_sky = level[:, sky_index+1:]
    level = torch.cat((before_sky, after_sky), dim=1)
    # Undo one-hot encoding
    level = level.argmax(dim=1).unsqueeze(1).float()
    return level


# Load all levels
reals = {}
for input_name in input_names:
    opt.input_name = f'lvl_{input_name}.txt'
    real = read_level(opt).to(opt.device)
    reals[input_name] = preprocess(real)

# Build a PCA_Detector for each level
detectors = {}
for input_name in input_names:
    real = reals[input_name]
    detectors[input_name] = PCA_Detector(opt, reals[input_name], (7, 7))

# Compute pairwise divergence
N = len(input_names)
data = np.zeros((N, N))
for i, input_name1 in enumerate(input_names):
    for j, input_name2 in enumerate(input_names):
        a = detectors[input_name1](reals[input_name1])
        b = detectors[input_name1](reals[input_name2])
        # data[i, j] = np.exp(divergence(a, b)) - 1
        data[i, j] = divergence(a, b)

plt.imshow(data, interpolation='nearest', extent=[0, 2*N, 0, 2*N])
plt.xticks([2*i + 1 for i in range(N)], input_names, rotation=45)
plt.yticks([2*i + 1 for i in range(N)], reversed(input_names))
# plt.savefig(f'pca-detector-2d-intensities.png',
#             bbox_inches='tight', pad_inches=0.1)
plt.show()

# visualize detector outputs
for input_name1 in input_names:
    detector = detectors[input_name1]
    for input_name2 in input_names:
        x1 = reals[input_name1]
        x2 = reals[input_name2]
        y = detector(x2)
        w = detector(reals[input_name1])
        _, y, w = difference(y, w)
        d = divergence(w, y)
        dw = torch.abs(y - w)
        z = torch.sum(y, dim=2)
        z = (z - z.mean()) / z.std()

        H = max(x1.shape[2], x2.shape[2])
        W = max(x1.shape[3], x2.shape[3])

        y = F.interpolate(y, (y.shape[2], W))
        dw = F.interpolate(dw, (dw.shape[2], W))

        x1 = x1[0, 0].detach().cpu().numpy()
        x2 = x2[0, 0].detach().cpu().numpy()
        y = y[0, 0].detach().cpu().numpy()
        z = z[0].detach().cpu().numpy()
        dw = dw[0, 0].detach().cpu().numpy()

        fig = plt.figure(figsize=(W / 10, 10 + 0.2 * H))
        grid = ImageGrid(fig, 111, nrows_ncols=(5, 1), axes_pad=(0.1, 0.1))
        fig.suptitle(
            f'divergence({input_name1}, {input_name2}) = {d:.2f}', fontsize=30)
        for ax, img, cmap in zip(grid, [x2, z, y, x1, dw], ['tab20b', 'magma', 'viridis', 'tab20b', 'magma']):
            ax.axis('off')
            ax.imshow(img, cmap=cmap)
        fig.tight_layout()
        plt.subplots_adjust(top=0.92)
        # plt.savefig(rf'PCA_Detector_output\detector_{input_name1}_level_{input_name2}.png',
        #             bbox_inches='tight', pad_inches=0.1)
        plt.show()
        plt.close()
        # exit()
