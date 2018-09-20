import numpy as np
import time
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision import datasets, transforms

from modules import Generator, Discriminator, Classifier
from modules import calc_reconstruction
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
    save_image(x_fake, 'samples/ebm-fast_%s.png' % args.dataset, normalize=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iters', type=int, default=100000)
    parser.add_argument('--critic_iters', type=int, default=5)
    parser.add_argument('--sigma', type=float, required=True)
    parser.add_argument('--lamda', type=float, default=1)
    parser.add_argument('--entropy_coeff', type=float, default=1)

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

optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerE = torch.optim.Adam(netE.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.tensor(1., dtype=torch.float32).cuda()
mone = one * -1

label = torch.ones(2 * args.batch_size).float().cuda()
label[args.batch_size:].zero_()

start_time = time.time()
d_costs = []
g_costs = []
for iters in range(1, args.iters):
    netG.zero_grad()
    netD.zero_grad()

    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x_fake = netG(z)
    D_fake = netE(x_fake)
    D_fake = D_fake.mean()
    D_fake.backward(one, retain_graph=True)

    # z_bar = z.clone()[torch.randperm(z.size(0))]
    # concat_x = torch.cat([x_fake, x_fake], 0)
    # concat_z = torch.cat([z, z_bar], 0)

    # logits = netD(concat_x, concat_z)
    # dim_estimate = nn.BCEWithLogitsLoss()(logits.squeeze(), label)
    # dim_estimate.backward()

    x = netD(x_fake)
    score = (z[:, None] * x[None]).sum(-1)
    mi_estimate = args.entropy_coeff * nn.CrossEntropyLoss()(
        score,
        torch.arange(args.batch_size, dtype=torch.int64).cuda()
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
        D_real.backward(one)

        # train with fake
        z = torch.randn(args.batch_size, args.z_dim).cuda()
        x_fake = netG(z).detach()
        D_fake = netE(x_fake)
        D_fake = D_fake.mean()
        D_fake.backward(mone)

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
               np.asarray(d_costs).mean(0),
               np.asarray(g_costs).mean(0),
               (time.time() - start_time) / 100
              ))

        netG.eval()
        sample(netG)
        netG.train()

        torch.save(netG.state_dict(), 'models/ebm-fast_%s.pt' % args.dataset)
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
