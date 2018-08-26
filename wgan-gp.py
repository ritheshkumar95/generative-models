import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import torch

from modules import MLP_Generator, MLP_Discriminator, calc_gradient_penalty
from data import SwissRoll


def sample(netG, n_points=10 ** 3):
    z = torch.randn(args.n_data, args.z_dim).cuda()
    x_fake = netG(z).detach().cpu().numpy()
    plt.clf()
    plt.scatter(x_fake[:, 0], x_fake[:, 1])
    plt.savefig('samples.png')


def visualize_energy(netE, n_points=100):
    x = np.linspace(-1, 1, n_points)
    y = np.linspace(-1, 1, n_points)
    grid = np.asarray(np.meshgrid(x, y)).transpose(1, 2, 0).reshape((-1, 2))
    grid = torch.from_numpy(grid).float().cuda()
    energies = netE(grid).detach().cpu().numpy()
    e_grid = energies.reshape((n_points, n_points))

    plt.clf()
    plt.imshow(e_grid, origin='lower')
    plt.savefig('energies.png')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_data', type=int, default=10 ** 4)
    parser.add_argument('--std_data', type=float, default=.25)

    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--z_dim', type=int, default=2)
    parser.add_argument('--dim', type=int, default=32)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()
    return args


args = parse_args()
dataset = SwissRoll(args.n_data, args.std_data)

netG = MLP_Generator(args.input_dim, args.z_dim, args.dim).cuda()
netD = MLP_Discriminator(args.input_dim, args.dim).cuda()

optimizerG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.9), amsgrad=True)
optimizerD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.9), amsgrad=True)

one = torch.tensor(1., dtype=torch.float32).cuda()
mone = one * -1

for epoch in range(args.epochs):
    wass_dist = []
    g_costs = []
    itr = dataset.create_iterator(args.batch_size)

    start_time = time.time()
    for i, x_real in enumerate(itr):
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
            netD, x_real.data, x_fake.data
        )
        gradient_penalty.backward()

        Wasserstein_D = D_real - D_fake
        optimizerD.step()
        wass_dist.append(Wasserstein_D.item())

        if i % 6 == 0:
            netG.zero_grad()
            z = torch.randn(args.batch_size, args.z_dim).cuda()
            x_fake = netG(z)
            D_fake = netD(x_fake)
            D_fake = D_fake.mean()
            D_fake.backward(mone)
            G_cost = -D_fake
            optimizerG.step()

            g_costs.append(G_cost.item())

        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Wass_D: {:5.3f} G_cost: {:5.3f} Time: {:5.3f}'.format(
                    epoch, i * args.batch_size, args.n_data,
                    (100. * i * args.batch_size) / args.n_data,
                    np.mean(wass_dist), np.mean(g_costs),
                    time.time() - start_time
            ))
            start_time = time.time()

    sample(netG)
    visualize_energy(netD)
