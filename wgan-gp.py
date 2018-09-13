import numpy as np
import time
import argparse

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

from modules import Generator, Discriminator, calc_gradient_penalty


def inf_train_gen(batch_size):
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            '../data/CIFAR10', train=True, download=True,
            transform=transf
        ), batch_size=64, drop_last=True
    )
    while True:
        for img, labels in loader:
            yield img


def sample(netG, batch_size=64):
    z = torch.randn(batch_size, args.z_dim).cuda()
    x_fake = netG(z).detach().cpu()
    save_image(x_fake, 'samples/wgan-gp-fast_%s.png' % args.dataset, normalize=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iters', type=int, default=200000)
    parser.add_argument('--critic_iters', type=int, default=5)
    parser.add_argument('--lamda', type=float, default=10)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=64)
    args = parser.parse_args()
    return args


args = parse_args()
args.dataset = 'CIFAR10'
itr = inf_train_gen(args.batch_size)

#####################
# Dump Original Data
#####################
orig_data = inf_train_gen(args.batch_size).__next__()
save_image(orig_data, 'samples/orig_%s.png' % args.dataset, normalize=True)

netG = Generator(args.z_dim, args.dim).cuda()
netD = Discriminator(args.dim).cuda()
print(netG)
print(netD)

optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9), amsgrad=True)
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9), amsgrad=True)

one = torch.tensor(1., dtype=torch.float32).cuda()
mone = one * -1

wass_dist = []
d_costs = []
start_time = time.time()
for iters in range(1, args.iters + 1):
    netG.zero_grad()
    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x_fake = netG(z)
    D_fake = netD(x_fake)
    D_fake = D_fake.mean()
    D_fake.backward(mone)
    G_cost = -D_fake
    optimizerG.step()

    for i in range(args.critic_iters):
        x_real = itr.__next__().cuda()

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
        d_costs.append([D_real.item(), D_fake.item()])

    if iters % 100 == 0:
        print('Train Iter: {}/{} ({:.0f}%)\t'
              'Wass_D: {:5.3f} D_costs: {} Time: {:5.3f}'.format(
               iters, args.iters, (100. * iters) / args.iters,
               np.mean(wass_dist), np.asarray(d_costs).mean(0),
               (time.time() - start_time) / 100
              ))

        netG.eval()
        sample(netG)
        netG.train()

        torch.save(netG.state_dict(), 'models/wgan-gp-fast_%s.pt' % args.dataset)
        g_costs = []
        wass_dist = []
        start_time = time.time()
