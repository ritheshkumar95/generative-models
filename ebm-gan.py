import numpy as np
import time
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image

from modules import Generator, Discriminator, Classifier
from modules import calc_reconstruction
from data import inf_train_gen
from eval import ModeCollapseEval


def sample(netG, batch_size=64):
    z = torch.randn(batch_size, args.z_dim).cuda()
    x_fake = netG(z).detach().cpu()
    save_image(
        x_fake[:, :3], 'samples/ebm_MNIST_%d.png' % args.n_stack,
        nrow=8, normalize=True
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_stack', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iters', type=int, default=100000)
    parser.add_argument('--critic_iters', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=.01)
    parser.add_argument('--lamda', type=float, default=1)
    parser.add_argument('--entropy_coeff', type=float, default=1)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)
    args = parser.parse_args()
    return args


args = parse_args()
itr = inf_train_gen(args.batch_size, n_stack=args.n_stack)

#####################
# Dump Original Data
#####################
orig_data = itr.__next__()
save_image(
    orig_data[:, :3], 'samples/orig_MNIST_%d.png' % args.n_stack,
    nrow=8, normalize=True
)

netG = Generator(args.n_stack, args.z_dim, args.dim).cuda()
netE = Discriminator(args.n_stack, args.dim).cuda()
netD = Classifier(args.n_stack, args.z_dim, args.dim).cuda()
evals = ModeCollapseEval(args.n_stack, args.z_dim)

optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerE = torch.optim.Adam(netE.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.tensor(1., dtype=torch.float32).cuda()
mone = one * -1

label = torch.ones(2 * args.batch_size).float().cuda()
label[args.batch_size:].zero_()

start_time = time.time()
d_costs = []
g_costs = []
for iters in range(args.iters):
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
        sample(netG)
        torch.save(netG.state_dict(), 'models/ebm_MNIST_%d.pt' % args.n_stack)
        d_costs = []
        g_costs = []
        start_time = time.time()

    if iters % 1000 == 0:
        netG.eval()
        print("-" * 100)
        evals.count_modes(netG)
        print("-" * 100)
        netG.train()
