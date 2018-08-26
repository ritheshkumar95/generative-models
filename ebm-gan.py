import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import torch
import torch.nn as nn

from modules import MLP_Generator, MLP_Discriminator, MLP_Classifier
from data import SwissRoll


def sample(netG, n_points=10 ** 3):
    z = torch.randn(args.n_data, args.z_dim).cuda()
    x_fake = netG(z).detach().cpu().numpy()

    plt.clf()
    plt.scatter(x_fake[:, 0], x_fake[:, 1])
    plt.savefig('ebm_samples.png')


def visualize_energy(netE, n_points=500):
    x = np.linspace(-1, 1, n_points)
    y = np.linspace(-1, 1, n_points)
    grid = np.asarray(np.meshgrid(x, y)).transpose(1, 2, 0).reshape((-1, 2))
    grid = torch.from_numpy(grid).float().cuda()
    energies = netE(grid).detach().cpu().numpy()
    e_grid = energies.reshape((n_points, n_points))

    plt.clf()
    plt.imshow(e_grid, origin='lower')
    plt.colorbar()
    plt.savefig('ebm_energies.png')


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
netE = MLP_Discriminator(args.input_dim, args.dim).cuda()
netD = MLP_Classifier(args.input_dim, args.z_dim, args.dim).cuda()

optimizerG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.9), amsgrad=True)
optimizerE = torch.optim.Adam(netE.parameters(), lr=2e-4, betas=(0.5, 0.9), amsgrad=True)
optimizerD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.9), amsgrad=True)


one = torch.tensor(1., dtype=torch.float32).cuda()
mone = one * -1
label = torch.ones(128).float().cuda()
label[64:].zero_()
shuf_idxs = np.arange(args.batch_size)

for epoch in range(args.epochs):
    costs = []
    itr = dataset.create_iterator(args.batch_size)

    start_time = time.time()
    for i, x_real in enumerate(itr):
        netE.zero_grad()
        D_real = netE(x_real)
        D_real = D_real.mean()
        D_real.backward(one)

        # train with fake
        z = torch.randn(args.batch_size, args.z_dim).cuda()
        x_fake = netG(z).detach()
        D_fake = netE(x_fake)
        D_fake = D_fake.mean()
        D_fake.backward(mone)

        optimizerE.step()

        if i % 6 == 0:
            netG.zero_grad()
            netD.zero_grad()

            z = torch.randn(args.batch_size, args.z_dim).cuda()
            x_fake = netG(z)
            D_fake = netE(x_fake)
            D_fake = D_fake.mean()
            D_fake.backward(one, retain_graph=True)

            z_bar = z.clone()[torch.randperm(z.size(0))]
            # z_bar = torch.randn(args.batch_size, args.z_dim).cuda()
            orig_x_z = torch.cat([x_fake, z], -1)
            shuf_x_z = torch.cat([x_fake, z_bar], -1)
            concat_x_z = torch.cat([orig_x_z, shuf_x_z], 0)

            logits = netD(concat_x_z)
            dim_estimate = nn.BCEWithLogitsLoss()(logits.squeeze(), label)
            dim_estimate.backward()

            optimizerG.step()
            optimizerD.step()

            costs.append(
                [D_real.item(), D_fake.item(), dim_estimate.item()]
            )

        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'costs: {} Time: {:5.3f}'.format(
                    epoch, i * args.batch_size, args.n_data,
                    (100. * i * args.batch_size) / args.n_data,
                    np.asarray(costs).mean(0), time.time() - start_time
            ))
            start_time = time.time()

    sample(netG, args.n_data)
    visualize_energy(netE)