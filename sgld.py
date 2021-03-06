import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from tqdm import tqdm

from modules import MLP_Discriminator, MLP_Generator
from data import inf_train_gen
from imageio import imread, mimwrite


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--n_points', type=int, default=1000)

    parser.add_argument('--alpha', type=float, default=.01)

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

z = torch.randn(args.n_points, args.z_dim).cuda()

images = []
for i in tqdm(range(1, 251)):
    z.requires_grad_(True)
    x = netG(z)
    e_x = netE(x)
    score = torch.autograd.grad(
        outputs=e_x, inputs=z,
        grad_outputs=torch.ones_like(e_x),
        create_graph=True, retain_graph=False, only_inputs=True
    )[0]

    noise = torch.randn_like(z) * np.sqrt(args.alpha * 2)
    z = (z - args.alpha * score + noise).detach()

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
