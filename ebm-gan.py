import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import torch
import torch.nn as nn

from modules import MLP_Generator, MLP_Discriminator, MLP_Classifier
from modules import calc_reconstruction
from data import inf_train_gen


def log_sum_exp(vec):
    max_val = vec.max()[0]
    return max_val + (vec - max_val).exp().sum().log()


def sample(netE, netG, n_points=10 ** 3):
    z = torch.randn(n_points, args.z_dim).cuda()
    x_fake = netG(z).detach()
    Z = log_sum_exp(-netE(x_fake).squeeze()).exp().item()

    x_fake = x_fake.cpu().numpy()
    plt.clf()
    plt.scatter(x_fake[:, 0], x_fake[:, 1])
    plt.savefig('toy_samples/ebm_samples_%s.png' % args.dataset)
    return Z


def visualize_energy(Z, netE, n_points=100):
    x = np.linspace(-2, 2, n_points)
    y = np.linspace(-2, 2, n_points)
    grid = np.asarray(np.meshgrid(x, y)).transpose(1, 2, 0).reshape((-1, 2))
    grid = torch.from_numpy(grid).float().cuda()
    energies = netE(grid).detach().cpu().numpy()
    e_grid = energies.reshape((n_points, n_points))
    p_grid = np.exp(-e_grid) / Z

    plt.clf()
    plt.imshow(e_grid, origin='lower')
    plt.colorbar()
    plt.savefig('toy_samples/ebm_energies_%s.png' % args.dataset)

    plt.clf()
    plt.imshow(p_grid, origin='lower')
    plt.colorbar()
    plt.savefig('toy_samples/ebm_densities_%s.png' % args.dataset)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--iters', type=int, default=100000)
    parser.add_argument('--critic_iters', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=.01)
    parser.add_argument('--lamda', type=float, default=1.)
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

netG = MLP_Generator(args.input_dim, args.z_dim, args.dim).cuda()
netE = MLP_Discriminator(args.input_dim, args.dim).cuda()
netD = MLP_Classifier(args.input_dim, args.z_dim, args.dim).cuda()

optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerE = torch.optim.Adam(netE.parameters(), lr=1e-4, betas=(0.5, 0.9))

label = torch.ones(2 * args.batch_size).float().cuda()
label[args.batch_size:].zero_()

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
    (args.entropy_coeff * D_fake).backward(retain_graph=True)

    x = netD(x_fake)
    scores = (z[:, None] * x[None]).sum(-1)
    mi_estimate = args.entropy_coeff * nn.CrossEntropyLoss()(
        scores,
        torch.arange(args.batch_size, dtype=torch.int64).cuda()
    )
    mi_estimate.backward()

    optimizerG.step()
    optimizerD.step()

    g_costs.append(
        [D_fake.item(), mi_estimate.item()]
    )

    for i in range(args.critic_iters):
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

        data = torch.cat([x_real, x_fake], 0)
        score_matching_loss = args.lamda * nn.MSELoss()(
            calc_reconstruction(netE, data, args.sigma),
            data
        )
        score_matching_loss.backward()

        optimizerE.step()
        d_costs.append(
            [D_real.item(), D_fake.item(), score_matching_loss.item()]
        )

    if iters % 100 == 0:
        print('Train Iter: {}/{} ({:.0f}%)\t'
              'D_costs: {} G_costs: {} Time: {:5.3f}'.format(
               iters, args.iters, (100. * iters) / args.iters,
               np.asarray(d_costs)[-100:].mean(0),
               np.asarray(g_costs)[-100:].mean(0),
               (time.time() - start_time) / 100
              ))
        Z = sample(netE, netG, args.n_points)
        visualize_energy(Z, netE, 500)

        start_time = time.time()

    if iters % 1000 == 0:
        torch.save(netG.state_dict(), 'toy_models/ebm_netG_%s.pt' % args.dataset)
        torch.save(netE.state_dict(), 'toy_models/ebm_netE_%s.pt' % args.dataset)
