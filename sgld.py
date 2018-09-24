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
    parser.add_argument('--v', action='store_true')
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

netD.load_state_dict(torch.load('models/ebm_netE_CIFAR10.pt'))
netG.load_state_dict(torch.load('models/ebm_netG_CIFAR10.pt'))

images = []
for j in tqdm(range(50)):
    z = torch.randn(args.n_points, args.z_dim).cuda()
    for i in range(1, 1 + args.n_steps):
        z.requires_grad_(True)
        x = netG(z)
        e_x = netD(x)

        score = torch.autograd.grad(
            outputs=e_x, inputs=z,
            grad_outputs=torch.ones_like(e_x),
            create_graph=True, retain_graph=False, only_inputs=True
        )[0]

        # magnitude = score.view(score.size(0), -1).norm(2, dim=-1)
        # direction = score / magnitude[:, None, None, None]

        noise = torch.randn_like(z) * np.sqrt(args.sigma * 2)
        z = (z - args.sigma * score + noise).detach()

        # print("Energy: %f" % e_x.mean().item())

        if args.v:
            save_image(x, 'mcmc_samples/image_%05d.png' % i, normalize=True, nrow=10)
            images.append(imread('mcmc_samples/image_%05d.png' % i))

    x = netG(z)
    images.append(x.detach().cpu().numpy())
    if args.v:
        # mimwrite('mcmc.gif', images)
        break


if not args.v:
    all_samples = np.concatenate(images, axis=0)
    all_samples = (((all_samples * .5) + .5) * 255).astype('int32')

    for i in tqdm(range(all_samples.shape[0])):
        imsave('cifar_samples/image_%d.png' % i, all_samples[i].transpose(1, 2, 0))

    from inception_score import get_inception_score
    images = np.concatenate(images, axis=0)
    print(get_inception_score(images))
