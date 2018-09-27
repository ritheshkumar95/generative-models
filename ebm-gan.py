import numpy as np
import time
import argparse
from collections import deque

import torch
import torch.nn as nn
from torchvision.utils import save_image

from modules import Generator, Discriminator, Classifier
from modules import calc_penalty
from data import MNIST
from utils.evaluations import do_prc
import anomaly_data.mnist as data	


def inf_train_gen(label, batch_size):
    trainx, trainy = data.get_train(label, True)
    while True:
        for i in range(0, trainx.shape[0], batch_size):
            x = torch.from_numpy(trainx[i:i + batch_size]).cuda()
            yield x.squeeze()[:, None]


def test_gen(label, batch_size):
    testx, testy = data.get_test(label, True)
    for i in range(0, testx.shape[0], batch_size):
        x = torch.from_numpy(testx[i:i + batch_size]).cuda()
        y = torch.from_numpy(testy[i:i + batch_size]).cuda()
        yield x.squeeze()[:, None], y


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


def calc_scores(netE):
    # itr = dataset.test_gen(args.batch_size)
    itr = test_gen(args.label, args.batch_size)
    scores = []
    gts = []
    for i, (img, labels) in enumerate(itr):
        img.requires_grad_(True)
        e_x = netE(img)

        score = torch.autograd.grad(
            outputs=e_x, inputs=img,
            grad_outputs=torch.ones_like(e_x),
            only_inputs=True
        )[0]
        score = score.view(score.size(0), -1).norm(2, dim=-1) ** 2
        scores += score.detach().cpu().tolist()
        gts += labels.cpu().tolist()

    prc_auc = do_prc(scores, gts)
    print('PRC AUC = %f' % prc_auc)
    return prc_auc


def sample(netG, batch_size=64):
    z = torch.randn(batch_size, args.z_dim).cuda()
    x_fake = netG(z).detach().cpu()
    save_image(
        x_fake, 'samples/ebm_MNIST.png',
        nrow=8, normalize=True
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iters', type=int, default=20000)

    parser.add_argument('--mcmc_iters', type=int, default=0)
    parser.add_argument('--critic_iters', type=int, default=5)
    parser.add_argument('--generator_iters', type=int, default=1)
    parser.add_argument('--lamda', type=float, default=10)
    parser.add_argument('--alpha', type=float, default=.01)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)
    args = parser.parse_args()
    return args


args = parse_args()
# dataset = MNIST(args.label)
# itr = dataset.inf_train_gen(args.batch_size)
itr = inf_train_gen(args.label, args.batch_size)

#####################
# Dump Original Data
#####################
orig_data = itr.__next__()
save_image(
    orig_data, 'samples/orig_MNIST.png',
    nrow=8, normalize=True
)

metrics = deque(maxlen=10)
netG = Generator(1, args.z_dim, args.dim).cuda()
netE = Discriminator(1, args.dim).cuda()
netD = Classifier(1, args.z_dim, args.dim).cuda()

optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerE = torch.optim.Adam(netE.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

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
        x_real = itr.__next__()
        while x_real.dim() != 4:
            print("Skipping bad batch!")
            x_real = itr.__next__()

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
        sample(netG)
        torch.save(netG.state_dict(), 'anomaly_models/ebm_MNIST_netG.pt')
        torch.save(netE.state_dict(), 'anomaly_models/ebm_MNIST_netE.pt')
        d_costs = []
        g_costs = []
        start_time = time.time()

    if iters % 500 == 0:
        auc = calc_scores(netE)
        metrics.append(auc)
        print("PRC AUC = mean: {} std: {}".format(
            np.mean(metrics), np.std(metrics)
        ))
