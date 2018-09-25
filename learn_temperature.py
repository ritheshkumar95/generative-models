import numpy as np
import time
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from plot_utils import visualize_energy, sample
from modules import MLP_Generator, MLP_Discriminator, MLP_Classifier
from data import inf_train_gen


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--n_points', type=int, default=10 ** 4)
    parser.add_argument('--sigma', type=float, default=.1)

    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--z_dim', type=int, default=2)
    parser.add_argument('--dim', type=int, default=512)
    args = parser.parse_args()
    return args


args = parse_args()
itr = inf_train_gen(args.dataset, args.n_points)

netG = MLP_Generator(args.input_dim, args.z_dim, args.dim).cuda()
netE = MLP_Discriminator(args.input_dim, args.dim).cuda()

netE.load_state_dict(torch.load('toy_models/ebm_netE_%s.pt' % args.dataset))
netG.load_state_dict(torch.load('toy_models/ebm_netG_%s.pt' % args.dataset))

x_real = torch.from_numpy(itr.__next__()).cuda()
x_real.requires_grad_(True)

eps = torch.normal(0, torch.ones_like(x_real) * args.sigma).cuda()
x = x_real + eps
e_x = netE(x)
score = torch.autograd.grad(
    outputs=e_x, inputs=x,
    grad_outputs=torch.ones_like(e_x),
    only_inputs=True
)[0]


numerator = (score * eps).sum()
denominator = args.sigma * score.pow(2).sum()
beta = (numerator / denominator).item()
print(beta)


log_Z = sample(netE, netG, args, beta)
visualize_energy(log_Z, netE, args, 500, beta=beta)
