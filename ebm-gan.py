import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import torch
import torch.nn as nn

from modules import MLP_Generator, MLP_Discriminator, MLP_Classifier, calc_gradient_penalty
from data import inf_train_gen


def sample(netG, n_points=10 ** 3):
    z = torch.randn(n_points, args.z_dim).cuda()
    x_fake = netG(z).detach().cpu().numpy()
    plt.clf()
    plt.scatter(x_fake[:, 0], x_fake[:, 1])
    plt.savefig('ebm_samples.png')


def visualize_energy(netE, n_points=100):
    x = np.linspace(-2, 2, n_points)
    y = np.linspace(-2, 2, n_points)
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
    parser.add_argument('--dataset', required=True)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--iters', type=int, default=100000)
    parser.add_argument('--critic_iters', type=int, default=5)

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
plt.savefig('orig_samples.png')

netG = MLP_Generator(args.input_dim, args.z_dim, args.dim).cuda()
netE = MLP_Discriminator(args.input_dim, args.dim).cuda()
netD = MLP_Classifier(args.input_dim, args.z_dim, args.dim).cuda()

optimizerG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.9), amsgrad=True)
optimizerE = torch.optim.Adam(netE.parameters(), lr=2e-4, betas=(0.5, 0.9), amsgrad=True)
optimizerD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.9), amsgrad=True)


one = torch.tensor(1., dtype=torch.float32).cuda()
mone = one * -1
label = torch.ones(2 * args.batch_size).float().cuda()
label[args.batch_size:].zero_()
shuf_idxs = np.arange(args.batch_size)

start_time = time.time()
for iters in range(args.iters):
    d_costs = []
    g_costs = []

    netG.zero_grad()
    netD.zero_grad()

    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x_fake = netG(z)
    D_fake = netE(x_fake)
    D_fake = D_fake.mean()
    D_fake.backward(one, retain_graph=True)

    z_bar = z.clone()[torch.randperm(z.size(0))]
    orig_x_z = torch.cat([x_fake, z], -1)
    shuf_x_z = torch.cat([x_fake, z_bar], -1)
    concat_x_z = torch.cat([orig_x_z, shuf_x_z], 0)

    logits = netD(concat_x_z)
    dim_estimate = nn.BCEWithLogitsLoss()(logits.squeeze(), label)
    dim_estimate.backward()

    optimizerG.step()
    optimizerD.step()

    g_costs.append(
        [D_fake.item(), dim_estimate.item()]
    )

    for i in range(args.critic_iters):
        x_real = torch.from_numpy(itr.__next__()).cuda()

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

        gradient_penalty = calc_gradient_penalty(
            netE, x_real.data, x_fake.data
        )
        gradient_penalty.backward()

        optimizerE.step()
        d_costs.append(
            [D_real.item(), D_fake.item()]
        )

    if iters % 100 == 0:
        print('Train Iter: {}/{} ({:.0f}%)\t'
              'D_costs: {} G_costs: {} Time: {:5.3f}'.format(
               iters, args.iters, (100. * iters) / args.iters,
               np.asarray(d_costs)[-100:].mean(0),
               np.asarray(g_costs)[-100:].mean(0),
               time.time() - start_time
              ))
        sample(netG, args.n_points)
        visualize_energy(netE, 500)

        start_time = time.time()
