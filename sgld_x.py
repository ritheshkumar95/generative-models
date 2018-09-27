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
    parser.add_argument('--n_steps', type=int, default=100)

    parser.add_argument('--temp', type=float, default=1)
    parser.add_argument('--sigma', type=float, default=.01)
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

z = torch.randn(args.n_points, args.z_dim).cuda()
x = netG(z)

images = []
for i in range(1, 1 + args.n_steps):
    x.requires_grad_(True)
    e_x = netE(x).squeeze() * args.temp

    score = torch.autograd.grad(
        outputs=e_x, inputs=x,
        grad_outputs=torch.ones_like(e_x),
        only_inputs=True
    )[0]

    noise = torch.randn_like(x) * np.sqrt(args.sigma * 2)
    x_prop = (x - args.sigma * score + noise).detach()
    x.data.clamp_(-2, 2)
    # x_prop = (x + noise).detach()

    e_x_prop = netE(x_prop).squeeze() * args.temp

    ratio = (-e_x_prop + e_x).exp().clamp(max=1)
    rnd_u = torch.rand(ratio.shape).cuda()
    mask = (rnd_u <= ratio).float()[:, None]
    x = (x_prop * mask + x * (1 - mask)).detach()
    print("Energy: %f" % e_x.mean().item())

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
