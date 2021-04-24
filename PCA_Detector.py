import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


def difference(a, b):
    H_a, W_a = a.shape[2:]
    H_b, W_b = b.shape[2:]
    H = max(H_a, H_b)
    W = max(W_a, W_b)
    A = F.pad(a, (0, W - W_a, 0, H - H_a))
    B = F.pad(b, (0, W - W_b, 0, H - H_b))
    return A - B, A, B


def divergence(a, b):
    diff, _, _ = difference(a, b)
    return torch.log(1 + torch.linalg.norm(diff)).item()


class PCA_Detector:
    def __init__(self, opt, level, kernel_dims):
        self.opt = opt

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
        H_pool = int((H_conv - H_kernel + 0.5) // H_kernel + 1)
        W_pool = int((W_conv - W_kernel + 0.5) // W_kernel + 1)
        self.output_dims = (H_pool, W_pool)

    def __call__(self, x):
        N, C, H, W = x.shape
        H_kernel, W_kernel = self.kernel_dims
        H_pool, W_pool = self.output_dims
        if H_pool < 1:
            H_pool = 1
        if W_pool < 1:
            W_pool = 1

        # print("BBB", (H, W), (H_kernel, W_kernel), (H_pool, W_pool))

        # Normalize the input
        x = self.normalize(x)

        # apply each kernel separately
        h1 = [F.relu(F.conv2d(x, kernel.view(1, 1, H_kernel, W_kernel)))
              for kernel in self.kernels]
        # print("H1", torch.stack(h1, dim=2).shape)
        h2 = [F.avg_pool2d(h, (H_pool, W_pool)) for h in h1]
        # print("H2", torch.cat(h2, dim=2).shape)

        # concatenate the results
        h3 = torch.cat(h2, dim=2)

        return h3
