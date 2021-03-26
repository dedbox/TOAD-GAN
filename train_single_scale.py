import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import interpolate
from loguru import logger
from tqdm import tqdm

# import wandb

from draw_concat import draw_concat
from generate_noise import generate_spatial_noise
from mario.level_utils import group_to_token, one_hot_to_ascii_level, token_to_group
from mario.tokens import TOKEN_GROUPS as MARIO_TOKEN_GROUPS
from mariokart.tokens import TOKEN_GROUPS as MARIOKART_TOKEN_GROUPS
from models import calc_gradient_penalty, save_networks


import matplotlib.pyplot as plt


def update_noise_amplitude(z_prev, real, opt):
    """ Update the amplitude of the noise for the current scale according to the previous noise map. """
    RMSE = torch.sqrt(F.mse_loss(real, z_prev))
    return opt.noise_update * RMSE


def train_single_scale(D, G, reals, generators, noise_maps, input_from_prev_scale, noise_amplitudes, opt):
    """ Train one scale. D and G are the current discriminator and generator, reals are the scaled versions of the
    original level, generators and noise_maps contain information from previous scales and will receive information in
    this scale, input_from_previous_scale holds the noise map and images from the previous scale, noise_amplitudes hold
    the amplitudes for the noise in all the scales. opt is a namespace that holds all necessary parameters. """
    current_scale = len(generators)
    real = reals[current_scale]

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

            output = D(real).to(opt.device)

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

                    prev = interpolate(prev, real.shape[-2:], mode="bilinear", align_corners=False)
                    prev = pad_image(prev)
                    z_prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                         "rec", pad_noise, pad_image, opt)

                    # For the seeding experiment, we need to transform from token_groups to the actual token
                    if current_scale == (opt.token_insert + 1):
                        z_prev = group_to_token(z_prev, opt.token_list, token_group)

                    z_prev = interpolate(z_prev, real.shape[-2:], mode="bilinear", align_corners=False)
                    opt.noise_amp = update_noise_amplitude(z_prev, real, opt)
                    z_prev = pad_image(z_prev)
            else:  # Any other step
                prev = draw_concat(generators, noise_maps, reals, noise_amplitudes, input_from_prev_scale,
                                   "rand", pad_noise, pad_image, opt)

                # For the seeding experiment, we need to transform from token_groups to the actual token
                if current_scale == (opt.token_insert + 1):
                    prev = group_to_token(prev, opt.token_list, token_group)

                prev = interpolate(prev, real.shape[-2:], mode="bilinear", align_corners=False)
                prev = pad_image(prev)

            # After creating our correct noise input, we feed it to the generator:
            noise = opt.noise_amp * noise_ + prev
            fake = G(noise.detach(), prev, temperature=1 if current_scale != opt.token_insert else 1)

            # Then run the result through the discriminator
            output = D(fake.detach())
            errD_fake = output.mean()

            # Backpropagation
            errD_fake.backward(retain_graph=False)

            # Gradient Penalty
            gradient_penalty = calc_gradient_penalty(D, real, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward(retain_graph=False)

            # blur kernel:
            # weights = torch.ones((1, 1, 7, 7)).cuda().repeat(1, opt.nc_current, 1, 1) / 49

            # low-gap kernel:
            gap_weights = torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
            ).cuda()
            gap_weights = F.normalize(gap_weights)
            gap_weights=gap_weights.view(1, 1, 7, 7).repeat(1, opt.nc_current, 1, 1)

            # discard top row
            gap2_weights = torch.tensor(
                [[0.5, 0.5],
                 [1.0, 1.0]]
            ).cuda()
            gap2_weights=F.normalize(gap2_weights)
            gap2_weights = gap2_weights.view(1, 1, 2, 2)

            fake_copy = fake.detach().clone()
            for n in range(opt.nc_current):
                if n != 2:
                    fake_copy[:, n, :, :] = 0.

            gap_detector=F.conv2d(fake_copy, gap_weights)
            gap_detector = F.batch_norm(gap_detector, None, None, training=True)
            gap2_detector = F.leaky_relu(gap_detector, 0.2)
            gap2_detector = F.conv2d(gap_detector, gap2_weights)
            gap2_detector = F.batch_norm(gap2_detector, None, None, training=True)
            gap2_detector = F.leaky_relu(gap2_detector, 0.2)
            gap2_detector = F.pad(gap2_detector, (5, 4, 4, 3), "constant", 0.)

            # right-wall kernel:
            wall_weights = torch.tensor(
                [[1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]
            ).cuda()
            wall_weights = F.normalize(wall_weights)
            wall_weights=wall_weights.view(1, 1, 7, 7).repeat(1, opt.nc_current, 1, 1)

            # sum both rows
            wall2_weights = torch.tensor(
                [[1.0, 1.0],
                 [1.0, 1.0]]
            ).cuda()
            wall2_weights=F.normalize(wall2_weights)
            wall2_weights = wall2_weights.view(1, 1, 2, 2)

            fake_copy = fake.detach().clone()
            for n in range(opt.nc_current):
                if n != 2:
                    fake_copy[:, n, :, :] = 0.

            wall_detector=F.conv2d(fake_copy, wall_weights)
            wall_detector = F.batch_norm(wall_detector, None, None, training=True)
            wall2_detector = F.leaky_relu(wall_detector, 0.2)
            wall2_detector = F.conv2d(wall_detector, wall2_weights)
            wall2_detector = F.batch_norm(wall2_detector, None, None, training=True)
            wall2_detector = F.leaky_relu(wall2_detector, 0.2)
            wall2_detector = F.pad(wall2_detector, (5, 4, 4, 3), "constant", 0.)

            # left-leaning kernel:
            lean_weights = torch.tensor(
                [[1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.1, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.1, 1.0, 0.1, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.1, 1.0, 0.1, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.1, 1.0, 0.1, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.1],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0]]
            ).cuda()
            lean_weights = F.normalize(lean_weights)
            lean_weights=lean_weights.view(1, 1, 7, 7).repeat(1, opt.nc_current, 1, 1)

            # sum the left-leaning diagonal
            lean2_weights = torch.tensor(
                [[1.0, 0.0],
                 [0.0, 1.0]]
            ).cuda()
            lean2_weights=F.normalize(lean2_weights)
            lean2_weights = lean2_weights.view(1, 1, 2, 2)

            fake_copy = fake.detach().clone()
            for n in range(opt.nc_current):
                if n != 2:
                    fake_copy[:, n, :, :] = 0.

            lean_detector=F.conv2d(fake_copy, lean_weights)
            lean_detector = F.batch_norm(lean_detector, None, None, training=True)
            lean2_detector = F.leaky_relu(lean_detector, 0.2)
            lean2_detector = F.conv2d(lean_detector, lean2_weights)
            lean2_detector = F.batch_norm(lean2_detector, None, None, training=True)
            lean2_detector = F.leaky_relu(lean2_detector, 0.2)
            lean2_detector = F.pad(lean2_detector, (5, 4, 4, 3), "constant", 0.)


            token_names={
                '!': 'coin [?]',
                '#': 'pyramid',
                '-': 'sky',
                '1': 'invis. 1 up',
                '@': 'special [?]',
                'C': 'coin brick',
                'S': 'normal brick',
                'U': 'musrhoom brick',
                'X': 'ground',
                'g': 'goomba',
                'k': 'green koopa',
                't': 'pipe',
            }

            # setup plots
            np_output = fake.cpu().detach().numpy()
            fig, axs=plt.subplots(opt.nc_current + 3)
            fig.suptitle(f"Scale {current_scale} Epoch {epoch}")

            # plot level channels
            for n in range(opt.nc_current):
                axs[n].get_xaxis().set_visible(False)
                axs[n].get_yaxis().set_visible(False)
                # axs[n].set_yticks([])
                # axs[n].set_yticklabels([])
                # axs[n].set_ylabel(token_names[opt.token_list[n]], rotation=45)
                axs[n].text(-0.1, 0.5, token_names[opt.token_list[n]], rotation=0, verticalalignment='center', horizontalalignment='right', transform=axs[n].transAxes)
                axs[n].imshow(np_output[0, n])
            
            # plot gap detector
            axs[opt.nc_current].get_xaxis().set_visible(False)
            axs[opt.nc_current].get_yaxis().set_visible(False)
            axs[opt.nc_current].text(-0.1, 0.5, 'detect gap', rotation=0, verticalalignment='center', horizontalalignment='right', transform=axs[opt.nc_current].transAxes)
            axs[opt.nc_current].imshow(gap2_detector[0, 0].cpu().detach().numpy())
            
            # plot wall detector
            axs[opt.nc_current + 1].get_xaxis().set_visible(False)
            axs[opt.nc_current + 1].get_yaxis().set_visible(False)
            axs[opt.nc_current + 1].text(-0.1, 0.5, 'detect wall', rotation=0, verticalalignment='center', horizontalalignment='right', transform=axs[opt.nc_current + 1].transAxes)
            axs[opt.nc_current + 1].imshow(wall2_detector[0, 0].cpu().detach().numpy())
            
            # plot lean detector
            axs[opt.nc_current + 2].get_xaxis().set_visible(False)
            axs[opt.nc_current + 2].get_yaxis().set_visible(False)
            axs[opt.nc_current + 2].text(-0.1, 0.5, 'detect lean', rotation=0, verticalalignment='center', horizontalalignment='right', transform=axs[opt.nc_current + 2].transAxes)
            axs[opt.nc_current + 2].imshow(lean2_detector[0, 0].cpu().detach().numpy())
            
            # finish plots
            fig.savefig(f"figs/figure_{current_scale}_{epoch}.png")
            plt.close(fig)

            # Logging:
            if step % 10 == 0:
                pass
                # wandb.log({f"D(G(z))@{current_scale}": errD_fake.item(),
                #            f"D(x)@{current_scale}": -errD_real.item(),
                #            f"gradient_penalty@{current_scale}": gradient_penalty.item()
                #            },
                #           step=step, sync=False)
            optimizerD.step()

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            G.zero_grad()
            fake = G(noise.detach(), prev.detach(), temperature=1 if current_scale != opt.token_insert else 1)
            output = D(fake)

            errG = -output.mean()
            errG.backward(retain_graph=False)
            if opt.alpha != 0:  # i. e. we are trying to find an exact recreation of our input in the lat space
                Z_opt = opt.noise_amp * z_opt + z_prev
                G_rec = G(Z_opt.detach(), z_prev, temperature=1 if current_scale != opt.token_insert else 1)
                rec_loss = opt.alpha * F.mse_loss(G_rec, real)
                rec_loss.backward(retain_graph=False)  # TODO: Check for unexpected argument retain_graph=True
                rec_loss = rec_loss.detach()
            else:  # We are not trying to find an exact recreation
                rec_loss = torch.zeros([])
                Z_opt = z_opt

            optimizerG.step()

        # More Logging:
        if step % 10 == 0:
            pass
            # wandb.log({f"noise_amplitude@{current_scale}": opt.noise_amp,
            #            f"rec_loss@{current_scale}": rec_loss.item()},
            #           step=step, sync=False, commit=True)

        # Rendering and logging images of levels
        if epoch % 500 == 0 or epoch == (opt.niter - 1):
            if opt.token_insert >= 0 and opt.nc_current == len(token_group):
                token_list = [list(group.keys())[0] for group in token_group]
            else:
                token_list = opt.token_list

            img = opt.ImgGen.render(one_hot_to_ascii_level(fake.detach(), token_list))
            img2 = opt.ImgGen.render(one_hot_to_ascii_level(
                G(Z_opt.detach(), z_prev, temperature=1 if current_scale != opt.token_insert else 1).detach(),
                token_list))
            real_scaled = one_hot_to_ascii_level(real.detach(), token_list)
            img3 = opt.ImgGen.render(real_scaled)
            # wandb.log({f"G(z)@{current_scale}": wandb.Image(img),
            #            f"G(z_opt)@{current_scale}": wandb.Image(img2),
            #            f"real@{current_scale}": wandb.Image(img3)},
            #           sync=False, commit=False)

            # real_scaled_path = os.path.join(wandb.run.dir, f"real@{current_scale}.txt")
            # with open(real_scaled_path, "w") as f:
            #     f.writelines(real_scaled)
            # wandb.save(real_scaled_path)

        # Learning Rate scheduler step
        schedulerD.step()
        schedulerG.step()

    # Save networks
    torch.save(z_opt, "%s/z_opt.pth" % opt.outf)
    save_networks(G, D, z_opt, opt)
    # wandb.save(opt.outf)
    return z_opt, input_from_prev_scale, G
