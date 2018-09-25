import numpy as np
import time
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import datasets, transforms

from modules import Generator, Discriminator, Classifier
from modules import calc_penalty
from eval import tf_inception_score


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
    save_image(x_fake, 'samples/ebm_%s.png' % args.dataset, normalize=True)


def sample_negatives(n_steps):
    z = torch.randn(args.batch_size, args.z_dim).cuda()

    for i in range(n_steps):
        z.requires_grad_(True)
        x = netG(z)
        e_x = netE(x)

        score = torch.autograd.grad(
            outputs=e_x, inputs=z,
            grad_outputs=torch.ones_like(e_x),
            only_inputs=True
        )[0]

        noise = torch.randn_like(z) * np.sqrt(args.alpha * 2)
        z = (z - args.alpha * score + noise).detach()
    return z


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iters', type=int, default=100000)

    parser.add_argument('--mcmc_iters', type=int, default=0)
    parser.add_argument('--critic_iters', type=int, default=1)
    parser.add_argument('--generator_iters', type=int, default=5)
    parser.add_argument('--lamda', type=float, default=10)
    parser.add_argument('--alpha', type=float, default=.01)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)
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
netE = Discriminator(args.dim).cuda()
netD = Classifier(args.z_dim, args.dim).cuda()
print(netG)
print(netE)
print(netD)

optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999), amsgrad=True)
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999), amsgrad=True)
optimizerE = torch.optim.Adam(netE.parameters(), lr=1e-4, betas=(0.5, 0.999), amsgrad=True)

start_time = time.time()
d_costs = []
g_costs = []
for iters in range(args.iters):
    netG.zero_grad()
    netD.zero_grad()

    for i in range(args.generator_iters):
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
        mi_estimate.backward()

        optimizerG.step()
        optimizerD.step()

        g_costs.append(
            [D_fake.item(), mi_estimate.item()]
        )

    for i in range(args.critic_iters):
        x_real = itr.__next__().cuda()

        netE.zero_grad()
        D_real = netE(x_real)
        D_real = D_real.mean()
        D_real.backward()

        # train with fake
        z = sample_negatives(args.mcmc_iters)
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

    if iters % 100 == 0:
        print('Train Iter: {}/{} ({:.0f}%)\t'
              'D_costs: {} G_costs: {} Time: {:5.3f}'.format(
               iters, args.iters, (100. * iters) / args.iters,
               np.asarray(d_costs).mean(0),
               np.asarray(g_costs).mean(0),
               (time.time() - start_time) / 100
              ))

        netG.eval()
        sample(netG)
        netG.train()

        torch.save(netG.state_dict(), 'models/ebm_netG_%s.pt' % args.dataset)
        torch.save(netE.state_dict(), 'models/ebm_netE_%s.pt' % args.dataset)
        d_costs = []
        g_costs = []
        start_time = time.time()

    if iters % 1000 == 0:
        start = time.time()
        netG.eval()
        mean, std = tf_inception_score(netG)
        print("-" * 100)
        print("Inception Score: mean = {} std = {} time: {:5.3f}".format(
            mean, std, time.time()-start
        ))
        print("-" * 100)
        netG.train()
