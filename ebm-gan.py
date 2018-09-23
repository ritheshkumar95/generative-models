import numpy as np
import time
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from plot_utils import visualize_energy, sample
from modules import MLP_Generator, MLP_Discriminator, MLP_Classifier
from modules import calc_penalty
from data import inf_train_gen


def train_generator(netG, netD, optimizerG, optimizerD, g_costs, lamda):
    netG.zero_grad()
    netD.zero_grad()

    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x_fake = netG(z)
    D_fake = netE(x_fake)
    D_fake = D_fake.mean()
    D_fake.backward(retain_graph=True)

    ##########################################
    # DeepInfoMAX for MI estimation
    ##########################################
    label = torch.zeros(2 * args.batch_size).cuda()
    label[:args.batch_size].data.fill_(1)

    z_bar = z[torch.randperm(args.batch_size)]
    joint = torch.cat([x_fake, z], -1)
    marginal = torch.cat([x_fake, z_bar], -1)
    mi_estimate = nn.BCEWithLogitsLoss()(
        netD(torch.cat([joint, marginal], 0)).squeeze(),
        label
    )
    (lamda * mi_estimate).backward()

    optimizerG.step()
    if optimizerD is not None:
        optimizerD.step()

    g_costs.append(
        [D_fake.item(), mi_estimate.item()]
    )


def train_energy(netG, netE, optimizerE, d_costs):
    x_real = torch.from_numpy(itr.__next__()).cuda()

    netE.zero_grad()
    D_real = netE(x_real)
    D_real = D_real.mean()
    D_real.backward()

    # train with fake
    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x_fake = netG(z).detach()
    D_fake = netE(x_fake)
    D_fake = D_fake.mean()
    (-D_fake).backward()

    penalty = calc_penalty(netE, x_real, args.lamda)
    penalty.backward()

    optimizerE.step()
    d_costs.append(
        [D_real.item(), D_fake.item(), penalty.item()]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--iters', type=int, default=100000)
    parser.add_argument('--critic_iters', type=int, default=5)
    parser.add_argument('--lamda', type=float, default=.1)
    parser.add_argument('--entropy_coeff', type=float, default=1.)

    parser.add_argument('--n_points', type=int, default=10 ** 3)

    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--z_dim', type=int, default=2)
    parser.add_argument('--dim', type=int, default=512)
    args = parser.parse_args()
    return args


args = parse_args()
itr = inf_train_gen(args.dataset, args.batch_size)

#####################
# Dump Original Data
#####################
orig_data = inf_train_gen(args.dataset, args.n_points).__next__()
plt.clf()
plt.scatter(orig_data[:, 0], orig_data[:, 1])
plt.savefig('orig_samples_%s.png' % args.dataset)

netG_le = MLP_Generator(args.input_dim, args.z_dim, args.dim).cuda()
netG_he = MLP_Generator(args.input_dim, args.z_dim, args.dim).cuda()
netE = MLP_Discriminator(args.input_dim, args.dim).cuda()
netD = MLP_Classifier(args.input_dim, args.z_dim, args.dim).cuda()

optimizerE = torch.optim.Adam(netE.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG_le = torch.optim.Adam(netG_le.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG_he = torch.optim.Adam(netG_he.parameters(), lr=1e-4, betas=(0.5, 0.9))

schedule = np.linspace(1., 0.001, 25000).tolist() + [.001] * (args.iters - 25000)

start_time = time.time()
d_costs = []
g_le_costs = []
g_he_costs = []
for iters in range(args.iters):
    train_generator(netG_he, netD, optimizerG_he, optimizerD, g_he_costs, 1.)
    train_generator(netG_le, netD, optimizerG_le, None, g_le_costs, schedule[iters])

    for i in range(args.critic_iters):
        train_energy(netG_he, netE, optimizerE, d_costs)

    if iters % 100 == 0:
        print('Train Iter: {}/{} ({:.0f}%)\t'
              'Ent Coeff: {:.3f} D_costs: {} G_le_costs: {} G_he_costs: {} Time: {:5.3f}'.format(
               iters, args.iters, (100. * iters) / args.iters,
               schedule[iters],
               np.asarray(d_costs)[-100:].mean(0),
               np.asarray(g_le_costs)[-100:].mean(0),
               np.asarray(g_he_costs)[-100:].mean(0),
               (time.time() - start_time) / 100
              ))
        log_Z = sample(netE, netG_le, args)
        visualize_energy(log_Z, netE, args, 500)

        start_time = time.time()

    if iters % 1000 == 0:
        torch.save(netG_le.state_dict(), 'toy_models/ebm_netG_%s.pt' % args.dataset)
        torch.save(netE.state_dict(), 'toy_models/ebm_netE_%s.pt' % args.dataset)
