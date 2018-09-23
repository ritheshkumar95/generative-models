import numpy as np
import time
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image

from modules import Generator, Discriminator, Classifier
from modules import calc_penalty
from data import inf_train_gen
from eval import ModeCollapseEval


def sample(netG_le, netG_he, batch_size=32):
    z = torch.randn(batch_size, args.z_dim).cuda()
    x_fake_le = netG_le(z).detach().cpu()
    x_fake_he = netG_he(z).detach().cpu()
    x_fake = torch.cat([x_fake_le, x_fake_he], 0)
    save_image(
        x_fake[:, :3], 'samples/ebm_MNIST_%d.png' % args.n_stack,
        nrow=8, normalize=True
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_stack', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iters', type=int, default=100000)
    parser.add_argument('--critic_iters', type=int, default=5)
    parser.add_argument('--lamda', type=float, default=10)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)
    args = parser.parse_args()
    return args


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
    concat_x = torch.cat([x_fake, x_fake], 0)
    concat_z = torch.cat([z, z_bar], 0)
    mi_estimate = nn.BCEWithLogitsLoss()(
        netD(concat_x, concat_z).squeeze(),
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
    x_real = itr.__next__().cuda()

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


args = parse_args()
itr = inf_train_gen(args.batch_size, n_stack=args.n_stack)

#####################
# Dump Original Data
#####################
orig_data = itr.__next__()
save_image(
    orig_data[:, :3], 'samples/orig_MNIST_%d.png' % args.n_stack,
    nrow=8, normalize=True
)

netG_le = Generator(args.n_stack, args.z_dim, args.dim).cuda()
netG_he = Generator(args.n_stack, args.z_dim, args.dim).cuda()
netE = Discriminator(args.n_stack, args.dim).cuda()
netD = Classifier(args.n_stack, args.z_dim, args.dim).cuda()
evals = ModeCollapseEval(args.n_stack, args.z_dim)

optimizerG_le = torch.optim.Adam(netG_le.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG_he = torch.optim.Adam(netG_he.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerE = torch.optim.Adam(netE.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

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

        netG_le.eval()
        netG_he.eval()
        sample(netG_le, netG_he)
        netG_le.train()
        netG_he.train()
        start_time = time.time()

    if iters % 1000 == 0 and args.n_stack <= 3:
        netG_le.eval()
        print("-" * 100)
        evals.count_modes(netG_le)
        print("-" * 100)
        netG_le.train()

        torch.save(netG_le.state_dict(), 'models/ebm_netG_MNIST_%d.pt' % args.n_stack)
        torch.save(netE.state_dict(), 'models/ebm_netE_MNIST_%d.pt' % args.n_stack)
