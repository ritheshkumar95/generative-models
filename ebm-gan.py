import numpy as np
import time
import argparse

import torch
from torchvision.utils import save_image

from modules import Generator, Discriminator, calc_gradient_penalty
from data import inf_train_gen


def sample(netG, batch_size=64):
    z = torch.randn(batch_size, args.z_dim).cuda()
    x_fake = netG(z).detach().cpu()
    save_image(x_fake, 'samples.png', nrow=8, normalize=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iters', type=int, default=200000)
    parser.add_argument('--critic_iters', type=int, default=5)
    parser.add_argument('--lamda', type=float, default=10)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=64)
    args = parser.parse_args()
    return args


args = parse_args()
itr = inf_train_gen(args.dataset, args.batch_size)

#####################
# Dump Original Data
#####################
orig_data = inf_train_gen(args.dataset, args.batch_size).__next__()
save_image(orig_data, 'orig_images.png', nrow=8, normalize=True)

netG = Generator(args.z_dim, args.dim).cuda()
netD = Discriminator(args.dim).cuda()

optimizerG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.9), amsgrad=True)
optimizerD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.9), amsgrad=True)

one = torch.tensor(1., dtype=torch.float32).cuda()
mone = one * -1

for iters in range(args.iters):
    wass_dist = []
    g_costs = []

    netG.zero_grad()
    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x_fake = netG(z)
    D_fake = netD(x_fake)
    D_fake = D_fake.mean()
    D_fake.backward(mone)
    G_cost = -D_fake
    optimizerG.step()
    g_costs.append(G_cost.item())

    for i in range(args.critic_iters):
        x_real = itr.__next__().cuda()

        start_time = time.time()
        netD.zero_grad()
        D_real = netD(x_real)
        D_real = D_real.mean()
        D_real.backward(mone)

        # train with fake
        z = torch.randn(args.batch_size, args.z_dim).cuda()
        x_fake = netG(z).detach()
        D_fake = netD(x_fake)
        D_fake = D_fake.mean()
        D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(
            netD, x_real.data, x_fake.data, lamda=args.lamda
        )
        gradient_penalty.backward()

        Wasserstein_D = D_real - D_fake
        optimizerD.step()
        wass_dist.append(Wasserstein_D.item())

    if iters % 50 == 0:
        print('Train Iter: {}/{} ({:.0f}%)\t'
              'Wass_D: {:5.3f} G_cost: {:5.3f} Time: {:5.3f}'.format(
               iters, args.iters,
               (100. * iters) / args.iters,
               np.mean(wass_dist), np.mean(g_costs),
               time.time() - start_time
              ))
        start_time = time.time()

        sample(netG)
