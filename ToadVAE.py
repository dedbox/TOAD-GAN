import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision


class LevelPatchesDataset(torch.utils.data.Dataset):
    """Mario level patches dataset."""

    def __init__(self, level, patch_size, transform=None):
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
        self.patches = patches

        # setup normalize transform
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


class ToadLevelVAE(nn.Module):
    def __init__(self, C, patch_size, latent_dims):
        super(ToadLevelVAE, self).__init__()

        self.patch_size = patch_size
        self.latent_dims = latent_dims
        self.hidden_dims = 4 * (patch_size - 4) * (patch_size - 4)
        self.input_dims = C * patch_size * patch_size

        # Encoder
        self.fc1 = nn.Conv2d(C, 16, 3)
        self.fc2 = nn.Conv2d(16, 4, 3)
        self.fc31 = nn.Linear(self.hidden_dims, self.latent_dims)  # mu
        self.fc32 = nn.Linear(self.hidden_dims, self.latent_dims)  # logvar

        # decoder
        self.fc41 = nn.Linear(self.latent_dims, self.hidden_dims)
        self.fc42 = nn.Linear(self.latent_dims, self.hidden_dims)
        self.fc5 = nn.ConvTranspose2d(4, 16, 3)
        self.fc6 = nn.ConvTranspose2d(16, C, 3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h2_flat = torch.flatten(h2)
        h31 = self.fc31(h2_flat)
        h32 = self.fc32(h2_flat)
        return h31, h32

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h41_flat = self.fc41(z)
        h41 = torch.reshape(
            h41_flat, (1, 4, self.patch_size - 4, self.patch_size - 4))
        h5 = F.relu(self.fc5(h41))
        h6 = torch.sigmoid(self.fc6(h5))
        return h6

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ToadPatchVAE(nn.Module):
    def __init__(self, C, patch_size, latent_dims):
        super(ToadPatchVAE, self).__init__()

        self.C = C
        self.patch_size = patch_size
        self.latent_dims = latent_dims
        # self.hidden_dims = 4 * (patch_size - 4) * (patch_size - 4)
        self.hidden_dims = patch_size * patch_size
        self.input_dims = C * patch_size * patch_size

        # Encoder
        self.fc1 = nn.Linear(self.input_dims, self.hidden_dims)
        self.fc21 = nn.Linear(self.hidden_dims, self.latent_dims)  # mu
        self.fc22 = nn.Linear(self.hidden_dims, self.latent_dims)  # logvar

        # decoder
        self.fc31 = nn.Linear(self.latent_dims, self.hidden_dims)
        self.fc32 = nn.Linear(self.latent_dims, self.hidden_dims)
        self.fc4 = nn.Linear(self.hidden_dims, self.input_dims)

    def encode(self, x):
        x_flat = torch.flatten(x)
        h1 = F.relu(self.fc1(x_flat))
        h21 = self.fc21(h1)
        h22 = self.fc22(h1)
        return h21, h22

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h31 = self.fc31(z)
        h4 = F.relu(self.fc4(h31))
        h4_flat = torch.reshape(
            h4, (1, self.C, self.patch_size, self.patch_size))
        return torch.sigmoid(h4_flat)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
