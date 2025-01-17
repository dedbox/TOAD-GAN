import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import interpolate
from loguru import logger
from tqdm import tqdm

import wandb

from draw_concat import draw_concat
from generate_noise import generate_spatial_noise
from mario.level_utils import group_to_token, one_hot_to_ascii_level, token_to_group
from mario.tokens import TOKEN_GROUPS as MARIO_TOKEN_GROUPS
from mariokart.tokens import TOKEN_GROUPS as MARIOKART_TOKEN_GROUPS
from models import calc_gradient_penalty, save_networks

import numpy as np
import matplotlib.pyplot as plt

from PCA_Detector import PCA_Detector, unify_shapes, divergence

from captum.attr import LayerGradCam, LayerAttribution  # LayerConductance
from captum.attr import visualization as viz


def update_noise_amplitude(z_prev, real, opt):
    """ Update the amplitude of the noise for the current scale according to the previous noise map. """
    RMSE = torch.sqrt(F.mse_loss(real, z_prev))
    return opt.noise_update * RMSE


def preprocess(opt, level, keepSky=False):
    # remove the sky layer
    if not keepSky:
        sky_index = opt.token_list.index('-')
        before_sky = level[:, :sky_index]
        after_sky = level[:, sky_index+1:]
        level = torch.cat((before_sky, after_sky), dim=1)
    # Undo one-hot encoding
    level = level.argmax(dim=1).unsqueeze(1).float()
    return level


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]) / 255


def train_single_scale(D, G, reals, generators, noise_maps, input_from_prev_scale, noise_amplitudes, opt):
    """ Train one scale. D and G are the current discriminator and generator, reals are the scaled versions of the
    original level, generators and noise_maps contain information from previous scales and will receive information in
    this scale, input_from_previous_scale holds the noise map and images from the previous scale, noise_amplitudes hold
    the amplitudes for the noise in all the scales. opt is a namespace that holds all necessary parameters. """
    current_scale = len(generators)
    real = reals[current_scale]

    keepSky=False
    kernel_dims = (2, 2)

    # Initialize real detector
    real0 = preprocess(opt, real, keepSky)
    N, C, H, W = real0.shape

    scale = opt.scales[current_scale] if current_scale < len(opt.scales) else 1

    if opt.cgan:
        detector = PCA_Detector(opt, 'real', real0, kernel_dims)
        real_detection_map = detector(real0)
        detection_scale = 0.1
        real_detection_map *= detection_scale
        real1 = torch.cat([real, F.interpolate(real_detection_map, (H, W))], dim=1)
        divergences = []
    else:
        real1 = real

    if opt.game == 'mario':
        token_group = MARIO_TOKEN_GROUPS
    else:  # if opt.game == 'mariokart':
        token_group = MARIOKART_TOKEN_GROUPS

    nzx = real.shape[2]  # Noise size x
    nzy = real.shape[3]  # Noise size y

    padsize = int(1 * opt.num_layer)  # As kernel size is always 3 currently, padsize goes up by one per layer

    if not opt.pad_with_noise:
        pad_noise = nn.ZeroPad2d(padsize)
        pad_image = nn.ZeroPad2d(padsize)
    else:
        pad_noise = nn.ReflectionPad2d(padsize)
        pad_image = nn.ReflectionPad2d(padsize)

    # setup optimizer
    optimizerD = optim.Adam(D.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600, 2500], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600, 2500], gamma=opt.gamma)

    if current_scale == 0:  # Generate new noise
        z_opt = generate_spatial_noise([1, opt.nc_current, nzx, nzy], device=opt.device)
        z_opt = pad_noise(z_opt)
    else:  # Add noise to previous output
        z_opt = torch.zeros([1, opt.nc_current, nzx, nzy]).to(opt.device)
        z_opt = pad_noise(z_opt)

    logger.info("Training at scale {}", current_scale)
    for epoch in tqdm(range(opt.niter)):
        step = current_scale * opt.niter + epoch
        noise_ = generate_spatial_noise([1, opt.nc_current, nzx, nzy], device=opt.device)
        noise_ = pad_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            D.zero_grad()

            output = D(real1).to(opt.device)

            errD_real = -output.mean()
            errD_real.backward(retain_graph=True)

            # train with fake
            if (j == 0) & (epoch == 0):
                if current_scale == 0:  # If we are in the lowest scale, noise is generated from scratch
                    prev = torch.zeros(1, opt.nc_current, nzx, nzy).to(opt.device)
                    input_from_prev_scale = prev
                    prev = pad_image(prev)
                    z_prev = torch.zeros(1, opt.nc_current, nzx, nzy).to(opt.device)
                    z_prev = pad_noise(z_prev)
                    opt.noise_amp = 1
                else:  # First step in NOT the lowest scale
                    # We need to adapt our inputs from the previous scale and add noise to it
                    prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                       "rand", pad_noise, pad_image, opt)

                    # For the seeding experiment, we need to transform from token_groups to the actual token
                    if current_scale == (opt.token_insert + 1):
                        prev = group_to_token(prev, opt.token_list, token_group)

                    prev = interpolate(prev, real1.shape[-2:], mode="bilinear", align_corners=False)
                    prev = pad_image(prev)
                    z_prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                         "rec", pad_noise, pad_image, opt)

                    # For the seeding experiment, we need to transform from token_groups to the actual token
                    if current_scale == (opt.token_insert + 1):
                        z_prev = group_to_token(z_prev, opt.token_list, token_group)

                    z_prev = interpolate(z_prev, real1.shape[-2:], mode="bilinear", align_corners=False)
                    opt.noise_amp = update_noise_amplitude(z_prev, real1[:, :-1], opt)
                    z_prev = pad_image(z_prev)
            else:  # Any other step
                prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                   "rand", pad_noise, pad_image, opt)

                # For the seeding experiment, we need to transform from token_groups to the actual token
                if current_scale == (opt.token_insert + 1):
                    prev = group_to_token(prev, opt.token_list, token_group)

                prev = interpolate(prev, real1.shape[-2:], mode="bilinear", align_corners=False)
                prev = pad_image(prev)

            # After creating our correct noise input, we feed it to the generator:
            noise = opt.noise_amp * noise_ + prev
            fake = G(noise.detach(), prev, temperature=1 if current_scale != opt.token_insert else 1)

            fake0 = preprocess(opt, fake, keepSky)
            if opt.cgan:
                Nf, Cf, Hf, Wf = fake0.shape
                fake_detection_map = detector(fake0) * detection_scale
                fake1 = torch.cat([fake, F.interpolate(fake_detection_map, (Hf, Wf))], dim=1)
            else:
                fake1 = fake

            # Then run the result through the discriminator
            output = D(fake1.detach())
            errD_fake = output.mean()

            # Backpropagation
            errD_fake.backward(retain_graph=False)

            # Gradient Penalty
            gradient_penalty = calc_gradient_penalty(D, real1, fake1, opt.lambda_grad, opt.device)
            gradient_penalty.backward(retain_graph=False)

            # Logging:
            if step % 10 == 0:
                wandb.log({f"D(G(z))@{current_scale}": errD_fake.item(),
                           f"D(x)@{current_scale}": -errD_real.item(),
                           f"gradient_penalty@{current_scale}": gradient_penalty.item()
                           },
                          step=step, sync=False)
            optimizerD.step()

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            G.zero_grad()
            fake = G(noise.detach(), prev.detach(), temperature=1 if current_scale != opt.token_insert else 1)

            fake0 = preprocess(opt, fake, keepSky)
            Nf, Cf, Hf, Wf = fake0.shape

            if opt.cgan:
                fake_detection_map = detector(fake0) * detection_scale
                fake1 = torch.cat([fake, F.interpolate(fake_detection_map, (Hf, Wf))], dim=1)
            else:
                fake1 = fake

            output = D(fake1)

            errG = -output.mean()
            errG.backward(retain_graph=False)
            if opt.alpha != 0:  # i. e. we are trying to find an exact recreation of our input in the lat space
                Z_opt = opt.noise_amp * z_opt + z_prev
                G_rec = G(Z_opt.detach(), z_prev, temperature=1 if current_scale != opt.token_insert else 1)
                rec_loss = opt.alpha * F.mse_loss(G_rec, real)
                if opt.cgan:
                    div = divergence(real_detection_map, preprocess(opt, G_rec, keepSky))
                    rec_loss += div
                rec_loss.backward(retain_graph=False)  # TODO: Check for unexpected argument retain_graph=True
                rec_loss = rec_loss.detach()
            else:  # We are not trying to find an exact recreation
                rec_loss = torch.zeros([])
                Z_opt = z_opt

            optimizerG.step()

        # More Logging:
        div = divergence(real_detection_map, preprocess(opt, fake, keepSky))
        divergences.append(div)
        # logger.info("divergence(fake) = {}", div)
        if step % 10 == 0:
            wandb.log({f"noise_amplitude@{current_scale}": opt.noise_amp,
                       f"rec_loss@{current_scale}": rec_loss.item()},
                      step=step, sync=False, commit=True)

        # Rendering and logging images of levels
        if epoch % 500 == 0 or epoch == (opt.niter - 1):
            if opt.token_insert >= 0 and opt.nc_current == len(token_group):
                token_list = [list(group.keys())[0] for group in token_group]
            else:
                token_list = opt.token_list

            img = opt.ImgGen.render(one_hot_to_ascii_level(fake1[:, :-1].detach(), token_list))
            img2 = opt.ImgGen.render(one_hot_to_ascii_level(
                G(Z_opt.detach(), z_prev, temperature=1 if current_scale != opt.token_insert else 1).detach(),
                token_list))
            real_scaled = one_hot_to_ascii_level(real1[:, :-1].detach(), token_list)
            img3 = opt.ImgGen.render(real_scaled)
            wandb.log({f"G(z)@{current_scale}": wandb.Image(img),
                       f"G(z_opt)@{current_scale}": wandb.Image(img2),
                       f"real@{current_scale}": wandb.Image(img3)},
                      sync=False, commit=False)

            real_scaled_path = os.path.join(wandb.run.dir, f"real@{current_scale}.txt")
            with open(real_scaled_path, "w") as f:
                f.writelines(real_scaled)
            wandb.save(real_scaled_path)

        # Learning Rate scheduler step
        schedulerD.step()
        schedulerG.step()

    if opt.cgan:
        div = divergence(real_detection_map, preprocess(opt, z_opt, keepSky))
        divergences.append(div)

    # visualization config
    folder_name = 'gradcam'
    level_name = opt.input_name.rsplit(".", 1)[0].split("_", 1)[1]

    # GradCAM on D
    camD = LayerGradCam(D, D.tail)
    real0 = one_hot_to_ascii_level(real, opt.token_list)
    real0 = opt.ImgGen.render(real0)
    real0 = np.array(real0)
    attr = camD.attribute(real1, target=(0, 0, 0), relu_attributions=True)
    attr = LayerAttribution.interpolate(attr, (real0.shape[0], real0.shape[1]), 'bilinear')
    attr = attr.permute(2, 3, 1, 0).squeeze(3)
    attr = attr.detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1)
    fig.figsize = (10, 1)
    ax.imshow(rgb2gray(real0), cmap='gray', vmin=0, vmax=1)
    im = ax.imshow(attr, cmap='jet', alpha=0.5)
    ax.axis('off')
    fig.colorbar(im, ax=ax, location='bottom', shrink=0.85)
    plt.suptitle(f'cGAN {level_name} D(x)@{current_scale} ({step})')
    plt.savefig(rf'{folder_name}\{level_name}_D_{current_scale}_{step}.png',
                bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close()

    # GradCAM on G
    token_names = {
        'M': 'Mario start',
        'F': 'Mario finish',
        'y': 'spiky',
        'Y': 'winged spiky',
        'k': 'green koopa',
        'K': 'winged green koopa',
        '!': 'coin [?]',
        '#': 'pyramid',
        '-': 'sky',
        '1': 'invis. 1 up',
        '2': 'invis. coin',
        'L': '1 up',
        '?': 'special [?]',
        '@': 'special [?]',
        'Q': 'coin [?]',
        '!': 'coin [?]',
        'C': 'coin brick',
        'S': 'normal brick',
        'U': 'mushroom brick',
        'X': 'ground',
        'E': 'goomba',
        'g': 'goomba',
        'k': 'green koopa',
        '%': 'platform',
        '|': 'platform bg',
        'r': 'red koopa',
        'R': 'winged red koopa',
        'o': 'coin',
        't': 'pipe',
        'T': 'plant pipe',
        '*': 'bullet bill',
        '<': 'pipe top left',
        '>': 'pipe top right',
        '[': 'pipe left',
        ']': 'pipe right',
        'B': 'bullet bill head',
        'b': 'bullet bill body',
        'D': 'used block',
    }
    def wrappedG(z):
        return G(z, z_opt)
    camG = LayerGradCam(wrappedG, G.tail[0])
    z_cam = generate_spatial_noise([1, opt.nc_current, nzx, nzy], device=opt.device)
    z_cam = pad_noise(z_cam)
    attrs = []
    for i in range(opt.nc_current):
        attr = camG.attribute(z_cam, target=(i, 0, 0), relu_attributions=True)
        attr = LayerAttribution.interpolate(attr, (real0.shape[0], real0.shape[1]), 'bilinear')
        attr = attr.permute(2, 3, 1, 0).squeeze(3)
        attr = attr.detach().cpu().numpy()
        attrs.append(attr)
    fig, axs = plt.subplots(opt.nc_current, 1)
    fig.figsize = (10, opt.nc_current)
    for i in range(opt.nc_current):
        axs[i].axis('off')
        axs[i].text(-0.1, 0.5, token_names[opt.token_list[i]],
                    rotation=0,
                    verticalalignment='center',
                    horizontalalignment='right',
                    transform=axs[i].transAxes)
        axs[i].imshow(rgb2gray(real0), cmap='gray', vmin=0, vmax=1)
        im = axs[i].imshow(attrs[i], cmap='jet', alpha=0.5)
    fig.colorbar(im, ax=axs, shrink=0.85)
    plt.suptitle(f'cGAN {level_name} G(z)@{current_scale} ({step})')
    plt.savefig(rf'{folder_name}\{level_name}_G_{current_scale}_{step}.png',
                bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close()
    
    # Save networks
    torch.save(z_opt, "%s/z_opt.pth" % opt.outf)
    save_networks(G, D, z_opt, opt)
    wandb.save(opt.outf)
    return z_opt, input_from_prev_scale, G, divergences
