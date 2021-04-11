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

np.set_printoptions(threshold=sys.maxsize)

print("XXX", real.shape)
print(np.array2string(real.cpu().numpy().astype(int)[0, 2], max_line_width=np.inf))
exit()



skymap = real[:, opt.token_list.index('-')]
B, H, W = skymap.shape
skymap_patches = skymap.unfold(1, 7, 1).unfold(2, 7, 1).reshape(1, H - 7 + 1, W - 7 + 1, 7*7)

patches = skymap_patches.cpu().detach().numpy()[0]
h, w, D = patches.shape
N = h * w
patches = patches.reshape((N, D)).T

scaler = StandardScaler()
scaler.fit(patches)
patches = scaler.transform(patches)

precision = 0.99

pca = PCA(n_components=precision)
principal_components = pca.fit_transform(patches).T

K, D = principal_components.shape
cols = int(np.ceil(np.sqrt(K)))
rows = K // cols + 1

print(f'K = {K}, rows = {rows}, cols = {cols}')

fig = plt.figure(figsize=(cols * 1.1, rows * 1.4))
grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=(0.1, 0.4))

fig.suptitle(f'PCA Top {K} of {N} ({int(100*precision)}% total variance)')
for k, ax, img in zip(range(K), grid, principal_components.reshape((K, 7, 7))):
    ax.axis('off')
    ax.set_title(f'{k+1}')
    ax.imshow(img, cmap='magma')
for i in range(K, rows * cols):
    grid[i].set_visible(False)
fig.tight_layout()
plt.savefig(f'pca_{opt.input_name[:-4]}.png', bbox_inches='tight', pad_inches=0.1)
# plt.show()
