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
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--n_points', type=int, default=100)
    parser.add_argument('--n_steps', type=int, default=4)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)

    parser.add_argument('--sigma', type=float, default=.01)
    parser.add_argument('--temp', type=float, default=.01)
    args = parser.parse_args()
    return args


args = parse_args()


netG = Generator(args.z_dim, args.dim).cuda()
netD = Discriminator(args.dim).cuda()

netG.eval()
netD.eval()

netD.load_state_dict(torch.load('models/ebm_netE_CelebA.pt'))
netG.load_state_dict(torch.load('models/ebm_netG_CelebA.pt'))

z = torch.randn(args.n_points, args.z_dim).cuda()
x = netG(z)

images = []
img_tensors = []

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
for i in range(1, 1 + args.n_steps):
    img_tensors.append(x.detach().cpu())
    save_image(
        x, 'mcmc_samples/image_%05d.png' % i,
        normalize=True, nrow=int(args.n_points ** .5)
    )
    images.append(imread('mcmc_samples/image_%05d.png' % i))

    z.requires_grad_(True)
    x = netG(z)
    e_x = netD(x).squeeze() * args.temp

    score = torch.autograd.grad(
        outputs=e_x, inputs=z,
        grad_outputs=torch.ones_like(e_x),
        create_graph=True, retain_graph=False, only_inputs=True
    )[0]

    noise = torch.randn_like(z) * np.sqrt(args.sigma * 2)
    z_prop = (z - args.sigma * score + noise).detach()

    x_prop = netG(z_prop)
    e_x_prop = netD(x_prop).squeeze() * args.temp

    ratio = (-e_x_prop + e_x).exp().clamp(max=1)
    rnd_u = torch.rand(ratio.shape).cuda()
    mask = (rnd_u < ratio).float()[:, None]
    z = (z_prop * mask + z * (1 - mask)).detach()

    print("Ratio: %f Energy: %f" % (ratio.mean().item(), e_x.mean().item()))
    x = netG(z)


mimwrite('mcmc.gif', images)
images = torch.stack([x for i, x in enumerate(img_tensors) if i % 2 == 0], 1)
images = images.view(-1, 3, 64, 64)
save_image(images, './large_celeba.png', normalize=True, nrow=images.size(0)//2)
