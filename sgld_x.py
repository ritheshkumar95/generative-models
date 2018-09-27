import argparse
import torch
from tqdm import tqdm
import os
import sys
from torchvision.utils import save_image
from modules import Discriminator, Generator
from imageio import imread, mimwrite
import numpy as np
from scipy.misc import imsave


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_points', type=int, default=100)
    parser.add_argument('--n_steps', type=int, default=4)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)

    parser.add_argument('--sigma', type=float, default=.01)
    args = parser.parse_args()
    return args


args = parse_args()


netG = Generator(args.z_dim, args.dim).cuda()
netD = Discriminator(args.dim).cuda()

netG.eval()
netD.eval()

netD.load_state_dict(torch.load('models/ebm_netE_CelebA.pt'))
netG.load_state_dict(torch.load('models/ebm_netG_CelebA.pt'))

images = []
z = torch.randn(args.n_points, args.z_dim).cuda()
x = netG(z)

for i in range(1, 1 + args.n_steps):
    x.requires_grad_(True)
    e_x = netD(x).squeeze()

    score = torch.autograd.grad(
        outputs=e_x, inputs=x,
        grad_outputs=torch.ones_like(e_x),
        create_graph=True, retain_graph=False, only_inputs=True
    )[0]

    # magnitude = score.view(score.size(0), -1).norm(2, dim=-1)
    # direction = score / magnitude[:, None, None, None]

    noise = torch.randn_like(x) * np.sqrt(args.sigma * 2)
    x_prop = (x - args.sigma * score + noise).detach()

    e_x_prop = netD(x_prop).squeeze()

    ratio = (-e_x_prop + e_x).exp().clamp(max=1)
    rnd_u = torch.rand(ratio.shape).cuda()
    mask = (rnd_u < ratio).float()[:, None, None, None]
    x = (x_prop * mask + x * (1 - mask)).detach()
    print("Energy: %f" % e_x.mean().item())

    save_image(
        x, 'mcmc_samples/image_%05d.png' % i,
        normalize=True, nrow=int(args.n_points ** .5)
    )
    images.append(imread('mcmc_samples/image_%05d.png' % i))

mimwrite('mcmc.gif', images)
