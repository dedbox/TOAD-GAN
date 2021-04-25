import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def unify_shapes(a, b):
    H_a, W_a = a.shape[2:]
    H_b, W_b = b.shape[2:]
    H = max(H_a, H_b)
    W = max(W_a, W_b)
    A = F.pad(a, (0, W - W_a, 0, H - H_a))
    B = F.pad(b, (0, W - W_b, 0, H - H_b))
    return A, B


def divergence(a, b):
    A, B = unify_shapes(a, b)
    return torch.linalg.norm(A - B).item() / A.shape[-2]


class PCA_Detector:
    def __init__(self, opt, name, level, kernel_dims):
        self.opt = opt
        self.name = name
        self.level = level

        N, C, H, W = level.shape
        H_kernel, W_kernel = kernel_dims
        self.kernel_dims = (H_kernel, W_kernel)

        # Define normalization transforms
        std, mean = torch.std_mean(level)
        self.normalize = lambda x: (x - mean) / std
        self.unnormalize = lambda x: x * std + mean

        # Normalize the input
        level = self.normalize(level)

        # Extract eigenpatches
        H_grid = H - H_kernel + 1
        W_grid = W - W_kernel + 1
        patches = level.unfold(2, H_kernel, 1).unfold(3, W_kernel, 1)
        patches = patches.transpose(1, 2).transpose(2, 3)
        patches = patches.reshape(H_grid * W_grid, H_kernel * W_kernel)
        patches = patches.detach().cpu().numpy()
        kernels = PCA().fit_transform(patches.T).T.reshape(-1, H_kernel, W_kernel)
        kernels = torch.tensor(kernels).to(opt.device)
        self.kernels = kernels
        print("kernels", len(kernels))

        # Setup convolution and pooling
        H_conv = H - H_kernel + 1
        W_conv = W - W_kernel + 1
        H_pool = H_kernel if H_kernel < H_conv else H_conv
        W_pool = W_kernel if W_kernel < W_conv else W_conv
        self.output_dims = (H_pool, W_pool)

    def __call__(self, x):
        N, C, H, W = x.shape
        H_kernel, W_kernel = self.kernel_dims
        H_pool, W_pool = self.output_dims

        # print("BBB", (H, W), (H_kernel, W_kernel), (H_pool, W_pool))

        # Normalize the input
        x = self.normalize(x)

        # apply each kernel separately
        h1 = [F.relu(F.conv2d(x, kernel.view(1, 1, H_kernel, W_kernel)))
              for kernel in self.kernels]
        # print("H1", torch.cat(h1, dim=2).shape)
        h2 = [F.avg_pool2d(h, (H_pool, W_pool)) for h in h1]
        # print("H2", torch.cat(h2, dim=2).shape)

        # concatenate the results
        h3 = torch.cat(h2, dim=2)

        return h3

    def visualize(self, name, level, fname=None):
        # inputs
        x_base = self.level
        x_test = level
        # outputs
        x = self(x_base)
        y = self(x_test)
        # print("111", y.shape)
        x, y = unify_shapes(x, y)
        # print("222", y.shape)
        z = torch.sum(y, dim=2).unsqueeze(2)
        z = (z - z.mean()) / z.std()
        # divergence
        div = divergence(x, y)
        dx = torch.abs(y - x)
        # dimensions of plot
        H = max(x_base.shape[2], x_test.shape[2])
        W = max(x_base.shape[3], x_test.shape[3])
        # resize outputs
        y = F.interpolate(y, (y.shape[2], W))
        # print("333", y.shape)
        z = F.interpolate(z, (z.shape[2], W))
        dx = F.interpolate(dx, (dx.shape[2], W))
        # convert to numpy
        x_base = x_base[0, 0].detach().cpu().numpy()
        x_test = x_test[0, 0].detach().cpu().numpy()
        y = y[0, 0].detach().cpu().numpy()
        # print("444", y.shape)
        z = z[0, 0].detach().cpu().numpy()
        dx = dx[0, 0].detach().cpu().numpy()
        # generate plot
        fig = plt.figure(figsize=(W / 10, 10 + 0.2 * H))
        grid = ImageGrid(fig, 111, nrows_ncols=(5, 1), axes_pad=(0.1, 0.1))
        fig.suptitle(
            f'divergence({self.name}, {name}) = {div:.3f}', fontsize=30)
        # print("x_test", x_test.shape)
        # print("z", z.shape)
        # print("y", y.shape)
        # print("x_base", x_base.shape)
        # print("dx", dx.shape)
        for ax, img, cmap in zip(grid, [x_test, z, y, x_base, dx], ['tab20b', 'magma', 'viridis', 'tab20b', 'magma']):
            ax.axis('off')
            ax.imshow(img, cmap=cmap)
        fig.tight_layout()
        plt.subplots_adjust(top=0.92)
        # show or save
        if fname is None:
            plt.show()
        else:
            plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
        plt.close()
