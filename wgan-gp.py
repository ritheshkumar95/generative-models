import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import torch

from modules import MLP_Generator, MLP_Discriminator, calc_gradient_penalty
from data import inf_train_gen


def log_sum_exp(vec):
    max_val = vec.max()[0]
    return max_val + (vec - max_val).exp().sum().log()


def sample(netE, netG, n_points):
    z = torch.randn(n_points, args.z_dim).cuda()
    x_fake = netG(z)
    neg_e_x = netE(x_fake)
    log_z = log_sum_exp(neg_e_x.squeeze()).item()

    x_fake = x_fake.detach().cpu().numpy()
    plt.clf()
    plt.scatter(x_fake[:, 0], x_fake[:, 1])
    plt.savefig('toy_samples/wgan-gp_samples_%s.png' % args.dataset)
    return log_z


def visualize_energy(log_z, netE, n_points=100):
    x = np.linspace(-2, 2, n_points)
    y = np.linspace(-2, 2, n_points)
    grid = np.asarray(np.meshgrid(x, y)).transpose(1, 2, 0).reshape((-1, 2))
    grid = torch.from_numpy(grid).float().cuda()
    energies = netE(grid).detach().cpu().numpy()
    e_grid = -energies.reshape((n_points, n_points))
    p_grid = np.exp(- e_grid - log_z)

    plt.clf()
    plt.imshow(e_grid, origin='lower')
    plt.colorbar()
    plt.savefig('toy_samples/wgan-gp_energies_%s.png' % args.dataset)

    plt.clf()
    plt.imshow(p_grid, origin='lower')
    plt.colorbar()
    plt.savefig('toy_samples/wgan-gp_densities_%s.png' % args.dataset)


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
plt.savefig('orig_samples_%s.png' % args.dataset)

netG = MLP_Generator(args.input_dim, args.z_dim, args.dim).cuda()
netD = MLP_Discriminator(args.input_dim, args.dim).cuda()

optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9), amsgrad=True)
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9), amsgrad=True)

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
        x_real = torch.from_numpy(itr.__next__()).cuda()

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
            netD, x_real.data, x_fake.data
        )
        gradient_penalty.backward()

        Wasserstein_D = D_real - D_fake
        optimizerD.step()
        wass_dist.append(Wasserstein_D.item())

    if iters % 100 == 0:
        print('Train Iter: {}/{} ({:.0f}%)\t'
              'Wass_D: {:5.3f} G_cost: {:5.3f} Time: {:5.3f}'.format(
               iters, args.iters,
               (100. * iters) / args.iters,
               np.mean(wass_dist), np.mean(g_costs),
               time.time() - start_time
              ))
        start_time = time.time()

        log_z = sample(netD, netG, args.n_points)
        visualize_energy(log_z, netD)

    if iters % 1000 == 0:
        torch.save(
            netG.state_dict(),
            'toy_models/wgan-gp_netG_%s.pt' % args.dataset
        )
        torch.save(
            netD.state_dict(),
            'toy_models/wgan-gp_netE_%s.pt' % args.dataset
        )
