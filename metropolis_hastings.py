import argparse
import torch
from tqdm import tqdm

from torchvision.utils import save_image
from modules import Discriminator, Generator
from imageio import imread, mimwrite


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_stack', type=int, default=1)
    parser.add_argument('--n_points', type=int, default=64)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)

    parser.add_argument('--sigma', type=float, default=.1)
    args = parser.parse_args()
    return args


args = parse_args()

netG = Generator(args.z_dim, args.dim).cuda()
netE = Discriminator(args.dim).cuda()

netE.load_state_dict(torch.load('models/ebm_netE_CIFAR10.pt'))
netG.load_state_dict(torch.load('models/ebm_netG_CIFAR10.pt'))

z = torch.zeros(args.n_points, args.z_dim).cuda()

images = []
for i in tqdm(range(1, 251)):
    x = netG(z)
    # x = x * (1 - x_mask) + orig_x * x_mask

    z_prop = z + torch.normal(0, torch.ones_like(z) * args.sigma).cuda()
    x_prop = netG(z_prop)
    # x_prop = x_prop * (1 - x_mask) + orig_x * x_mask

    e_x_prop = netE(x_prop).squeeze()
    e_x = netE(x).squeeze()
    ratio = (-e_x_prop + e_x).exp().clamp(max=1)
    # mask = torch.bernoulli(ratio)[:, None]
    rnd_u = torch.rand(ratio.shape).cuda()
    mask = (rnd_u < ratio).float()[:, None]

    z = z_prop * mask + z * (1 - mask)
    print("Ratio: ", mask.mean().item())

    if i % 1 == 0:
        save_image(x, 'mcmc_samples/image_%05d.png' % i, normalize=True)
        images.append(imread('mcmc_samples/image_%05d.png' % i))

mimwrite('mcmc.gif', images)
