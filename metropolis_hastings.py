import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from tqdm import tqdm

from plot_utils import visualize_energy, sample
from modules import MLP_Discriminator, MLP_Generator
from imageio import imread, mimwrite


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--n_points', type=int, default=1000)

    parser.add_argument('--sigma', type=float, default=.1)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--z_dim', type=int, default=2)
    parser.add_argument('--dim', type=int, default=512)
    args = parser.parse_args()
    return args


args = parse_args()

netE = MLP_Discriminator(args.input_dim, args.dim).cuda()
netG = MLP_Generator(args.input_dim, args.z_dim, args.dim).cuda()
netE.load_state_dict(torch.load('toy_models/ebm_netE_%s.pt' % args.dataset))
netG.load_state_dict(torch.load('toy_models/ebm_netG_%s.pt' % args.dataset))

log_Z = sample(netE, netG, args)
visualize_energy(log_Z, netE, args, 500)

# x = torch.zeros(args.n_points, 2).cuda()
# x.data.uniform_(-2, 2)

z = torch.zeros(args.n_points, args.z_dim).cuda()

images = []
for i in tqdm(range(1, 251)):
    x = netG(z)

    z_prop = z + torch.normal(0, torch.ones_like(z) * args.sigma).cuda()
    x_prop = netG(z_prop)

    e_x_prop = netE(x_prop).squeeze()
    e_x = netE(x).squeeze()
    ratio = (-e_x_prop + e_x).exp().clamp(max=1)
    mask = torch.bernoulli(ratio)[:, None]

    z = z_prop * mask + z * (1 - mask)
    print("Ratio: ", mask.mean().item())

    if i % 1 == 0:
        img = x.detach().cpu().numpy()
        plt.clf()
        plt.scatter(img[:, 0], img[:, 1])
        plt.title("Iter %d" % i)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.savefig('mcmc_samples/image_%05d.png' % i)
        images.append(imread('mcmc_samples/image_%05d.png' % i))

mimwrite('mcmc.gif', images)
