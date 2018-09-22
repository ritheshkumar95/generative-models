<<<<<<< HEAD
=======
import matplotlib.pyplot as plt
import numpy as np
>>>>>>> f24f6df57437b0f914259ceda0b069d136e0530f
import argparse
import torch
from tqdm import tqdm

<<<<<<< HEAD
from torchvision.utils import save_image
from modules import Discriminator, Generator
=======
from modules import Discriminator, Generator
from torchvision.utils import save_image
>>>>>>> f24f6df57437b0f914259ceda0b069d136e0530f
from imageio import imread, mimwrite


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_stack', type=int, default=1)
<<<<<<< HEAD
    parser.add_argument('--n_points', type=int, default=64)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)

    parser.add_argument('--lamda1', type=float, default=.001)
    parser.add_argument('--lamda2', type=float, default=.01)
=======
    parser.add_argument('--n_points', type=int, default=100)

    parser.add_argument('--lamda1', type=float, default=.01)
    parser.add_argument('--lamda2', type=float, default=.01)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)
>>>>>>> f24f6df57437b0f914259ceda0b069d136e0530f
    args = parser.parse_args()
    return args


args = parse_args()

<<<<<<< HEAD

netG = Generator(args.n_stack, args.z_dim, args.dim).cuda()
netD = Discriminator(args.n_stack, args.dim).cuda()

netD.load_state_dict(torch.load('models/wgan-gp_netD_MNIST_%d.pt' % args.n_stack))
netG.load_state_dict(torch.load('models/wgan-gp_netG_MNIST_%d.pt' % args.n_stack))

# x = torch.zeros(args.n_points, args.n_stack, 28, 28).cuda()
# x.data.uniform_(-1, 1)

z = torch.randn(args.n_points, args.z_dim).cuda()

images = []
for i in tqdm(range(1, 101)):
    z.requires_grad_(True)
    x = netG(z)
    # x.requires_grad_(True)
    e_x = -netD(x)

    score = torch.autograd.grad(
        outputs=e_x, inputs=z,
=======
netG = Generator(args.n_stack, args.z_dim, args.dim).cuda()
netE = Discriminator(args.n_stack, args.dim).cuda()
netE.load_state_dict(torch.load('models/ebm_netE_MNIST_%d.pt' % args.n_stack))
netG.load_state_dict(torch.load('models/ebm_netG_MNIST_%d.pt' % args.n_stack))
netG.eval()

z = torch.randn(args.n_points, args.z_dim).cuda()
x = netG(z)

images = []
for i in tqdm(range(1, 101)):
    x.requires_grad_(True)
    e_x = netE(x)
    score = torch.autograd.grad(
        outputs=e_x, inputs=x,
>>>>>>> f24f6df57437b0f914259ceda0b069d136e0530f
        grad_outputs=torch.ones_like(e_x),
        create_graph=True, retain_graph=False, only_inputs=True
    )[0]

<<<<<<< HEAD
    # magnitude = score.view(score.size(0), -1).norm(2, dim=-1)
    # direction = score / magnitude[:, None, None, None]

    noise = torch.normal(0, torch.ones_like(z) * args.lamda2).cuda()
    # x = (x - args.lamda1 * score + noise).detach()
    z = (z - args.lamda1 * score + noise).detach()

    if i % 1 == 0:
        save_image(x, 'mcmc_samples/image_%05d.png' % i, normalize=True)
=======
    noise = torch.normal(0, torch.ones_like(x) * args.lamda2).cuda()
    x = (x - args.lamda1 * score + noise).detach()

    if i % 1 == 0:
        save_image(x, 'mcmc_samples/image_%05d.png' % i, nrow=10, normalize=True)
>>>>>>> f24f6df57437b0f914259ceda0b069d136e0530f
        images.append(imread('mcmc_samples/image_%05d.png' % i))

mimwrite('mcmc.gif', images)
