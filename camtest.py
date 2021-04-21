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
from mpl_toolkits.axes_grid1 import ImageGrid

from ToadVAE import LevelPatchesDataset, ToadPatchVAE, ToadLevelVAE

import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

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
# ToadCam


class ToadConv(nn.Module):
    def __init__(self, level):
        super(ToadConv, self).__init__()

        N, C, H, W = level.shape
        self.C = C

        H_conv1 = H - 7
        W_conv1 = W - 7

        H_pool1 = int(H_conv1 / 2 + 0.5)
        W_pool1 = int(W_conv1 / 2 + 0.5)

        H_conv2 = H_pool1 - 2
        W_conv2 = W_pool1 - 2

        H_pool2 = int(H_conv2 / 2 + 0.5)
        W_pool2 = int(W_conv2 / 2 + 0.5)

        self.internal_sizes = (
            (H, W),
            (H_pool1, W_pool1),
            (H_pool2, W_pool2))

        # Encoder
        self.conv1 = nn.Conv2d(C, 16, 7)
        self.conv2 = nn.Conv2d(16, 3, 2)
        self.fc3 = nn.Linear(3 * H_pool2 * W_pool2, 10)

        # Decoder
        self.fc4 = nn.Linear(10, 3 * H_pool2 * W_pool2)
        self.conv5 = nn.ConvTranspose2d(3, 16, 2)
        self.conv6 = nn.ConvTranspose2d(16, C, 7)

    def encode(self, x):
        h1 = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        h2 = F.avg_pool2d(F.relu(self.conv2(h1)), 2)
        h3 = self.fc3(h2.flatten())
        return h3

    def decode(self, z):
        (H, W), (H_pool1, W_pool1), (H_pool2, W_pool2) = self.internal_sizes
        h4 = self.fc4(z).reshape(1, 3, H_pool2, W_pool2)
        h5 = F.interpolate(F.relu(self.conv5(h4)), (H_pool1, W_pool1))
        h6 = F.interpolate(F.relu(self.conv6(h5)), (H, W))
        h6 = torch.sigmoid(h6)
        h7 = F.one_hot(torch.argmax(h6, dim=1), self.C)
        h7 = torch.transpose(h7, 2, 3)
        h7 = torch.transpose(h7, 1, 2)
        return h6

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        return y


N, C, H, W = real.shape
print("real", real.shape)

model = ToadConv(real).cuda()
print(model)

cam1 = GradCAM(model=model, target_layer=model.conv1, use_cuda=True)
cam2 = GradCAM(model=model, target_layer=model.conv2, use_cuda=True)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
for epoch in range(200):
    optimizer.zero_grad()
    y = model.forward(real)
    loss = criterion(y, real)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    print("loss", loss.item())

# render the real level
ascii_real = one_hot_to_ascii_level(real, opt.token_list)
real_level = opt.ImgGen.render(ascii_real)
real_rgb = np.array(real_level) / 255
input_tensor = preprocess_image(real_rgb, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

# render cam overlays
cam_images1 = []
cam_images2 = []
for i, _ in enumerate(opt.token_list):
    grayscale_cam1 = cam1(input_tensor=real, target_category=i)
    grayscale_cam2 = cam2(input_tensor=real, target_category=i)
    grayscale_cam1 = cv2.resize(grayscale_cam1, dsize=(16*W, 16*H))
    grayscale_cam2 = cv2.resize(grayscale_cam2, dsize=(16*W, 16*H))
    cam_images1.append(show_cam_on_image(real_rgb, grayscale_cam1))
    cam_images2.append(show_cam_on_image(real_rgb, grayscale_cam2))

# cv2.imwrite('camtest.png', cam_image)

token_names = {
    '!': 'coin [?]',
    '#': 'pyramid',
    '-': 'sky',
    '1': 'invis. 1 up',
    'L': '1 up',
    '@': 'special [?]',
    'Q': 'coin [?]',
    'C': 'coin brick',
    'S': 'normal brick',
    'U': 'mushroom brick',
    'X': 'ground',
    'E': 'goomba',
    'g': 'goomba',
    'k': 'green koopa',
    't': 'pipe',
    '%': 'platform',
    '|': 'platform bg',
    'R': 'winged red koopa',
    'o': 'coin',
    'r': 'red koopa',
    'T': 'plant pipe'
}


_, axs = plt.subplots(C, 1, figsize=(12, 12))
axs = axs.flatten()
for i, img, ax in zip(range(C), cam_images1, axs):
    ax.axis('off')
    ax.text(-0.1, 0.5, token_names[opt.token_list[i]], rotation=0, verticalalignment='center', horizontalalignment='right', transform=ax.transAxes)
    ax.imshow(img)
plt.suptitle('conv1')
plt.show()

_, axs = plt.subplots(C, 1, figsize=(12, 12))
axs = axs.flatten()
for i, img, ax in zip(range(C), cam_images1, axs):
    ax.axis('off')
    ax.text(-0.1, 0.5, token_names[opt.token_list[i]], rotation=0, verticalalignment='center', horizontalalignment='right', transform=ax.transAxes)
    ax.imshow(img)
plt.suptitle('conv2')
plt.show()


# rows = int(C / 2 + 0.5)
# cols = 2

# fig = plt.figure()
# grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols))

# fig.suptitle(f'CAM1')
# for i, ax, img in zip(range(C), grid, cam_images1):
#     ax.axis('off')
#     # ax.set_title(f'{opt.token_list[i]}')
#     ax.imshow(img)
# # for i in range(K, rows * cols):
# #     grid[i].set_visible(False)
# # fig.tight_layout()
# # if opt.vae_save:
# #     plt.savefig(f'vae-{opt.input_name[:-4]}-{opt.hidden_dim}-{opt.latent_dim}-{epoch}.png', bbox_inches='tight', pad_inches=0.1)
# # else:
# plt.show()

# # plt.imshow(cam_image, cmap=magma)
# # plt.show()
