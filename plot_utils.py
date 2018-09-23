import torch
import numpy as np
import matplotlib.pyplot as plt


def log_sum_exp(vec):
    max_val = vec.max()[0]
    return max_val + (vec - max_val).exp().sum().log()


def sample(netE, netG, args, beta=1.):
    z = torch.randn(args.n_points, args.z_dim).cuda()
    x_fake = netG(z).detach()
    log_Z = log_sum_exp(
        -netE(x_fake).squeeze() * beta
    ).item()

    x_fake = x_fake.cpu().numpy()
    plt.clf()
    plt.scatter(x_fake[:, 0], x_fake[:, 1])
    plt.savefig('toy_samples/ebm_samples_%s.png' % args.dataset)
    return log_Z


def visualize_energy(log_Z, netE, args, n_points=100, beta=1.):
    x = np.linspace(-2, 2, n_points)
    y = np.linspace(-2, 2, n_points)
    grid = np.asarray(np.meshgrid(x, y)).transpose(1, 2, 0).reshape((-1, 2))
    grid = torch.from_numpy(grid).float().cuda()
    energies = netE(grid).detach().cpu().numpy() * beta
    e_grid = energies.reshape((n_points, n_points))
    p_grid = np.exp(- e_grid - log_Z)

    plt.clf()
    plt.imshow(e_grid, origin='lower')
    plt.colorbar()
    plt.savefig('toy_samples/ebm_energies_%s.png' % args.dataset)

    plt.clf()
    plt.imshow(p_grid, origin='lower')
    plt.colorbar()
    plt.savefig('toy_samples/ebm_densities_%s.png' % args.dataset)
