import numpy as np
import time
import argparse

import torch
import torch.nn as nn
from torchvision.utils import save_image

from modules import Generator, Discriminator, Classifier
from modules import calc_reconstruction
import data.mnist as data
from utils.evaluations import do_prc


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


def calc_scores(netE):
    itr = test_gen(args.label, args.batch_size)
    scores = []
    gts = []
    for i, (img, labels) in enumerate(itr):
        scores += (-netE(img).squeeze()).detach().cpu().tolist()
        gts += labels.cpu().tolist()

    prc_auc = do_prc(scores, gts)
    print('PRC AUC = %f' % prc_auc)


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
itr = inf_train_gen(args.label, args.batch_size)

#####################
# Dump Original Data
#####################
orig_data = itr.__next__()
save_image(
    orig_data, 'samples/orig_MNIST.png',
    nrow=8, normalize=True
)

netG = Generator(1, args.z_dim, args.dim).cuda()
netE = Discriminator(1, args.dim).cuda()
netD = Classifier(1, args.z_dim, args.dim).cuda()

optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerE = torch.optim.Adam(netE.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.tensor(1., dtype=torch.float32).cuda()
mone = one * -1

start_time = time.time()
d_costs = []
g_costs = []
for iters in range(1, args.iters + 1):
    netG.zero_grad()
    netD.zero_grad()

    z = torch.randn(args.batch_size, args.z_dim).cuda()
    x_fake = netG(z)
    D_fake = netE(x_fake)
    D_fake = D_fake.mean()
    D_fake.backward(one, retain_graph=True)

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
        x_real = itr.__next__()

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

        score_matching_loss = args.lamda * nn.MSELoss()(
            calc_reconstruction(netE, x_fake, args.sigma),
            x_fake
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
        torch.save(netG.state_dict(), 'anomaly_models/ebm_MNIST_netG.pt')
        torch.save(netE.state_dict(), 'anomaly_models/ebm_MNIST_netE.pt')
        d_costs = []
        g_costs = []
        start_time = time.time()

    if iters % 500 == 0:
        calc_scores(netE)
