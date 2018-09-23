import argparse
import torch
from tqdm import tqdm

from torchvision.utils import save_image
from modules import Discriminator, Generator
from imageio import imread, mimwrite


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_points', type=int, default=64)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)

    parser.add_argument('--lamda1', type=float, default=.001)
    parser.add_argument('--lamda2', type=float, default=.01)
    args = parser.parse_args()
    return args


args = parse_args()


netG = Generator(args.z_dim, args.dim).cuda()
netD = Discriminator(args.dim).cuda()

netD.load_state_dict(torch.load('models/ebm_netE_CIFAR10.pt'))
netG.load_state_dict(torch.load('models/ebm_netG_CIFAR10.pt'))

# x = torch.zeros(args.n_points, args.n_stack, 28, 28).cuda()
# x.data.uniform_(-1, 1)

z = torch.randn(args.n_points, args.z_dim).cuda()

images = []
for i in tqdm(range(1, 101)):
    z.requires_grad_(True)
    x = netG(z)
    # x.requires_grad_(True)
    e_x = netD(x)

    score = torch.autograd.grad(
        outputs=e_x, inputs=z,
        grad_outputs=torch.ones_like(e_x),
        create_graph=True, retain_graph=False, only_inputs=True
    )[0]

    # magnitude = score.view(score.size(0), -1).norm(2, dim=-1)
    # direction = score / magnitude[:, None, None, None]

    # noise = torch.normal(0, torch.ones_like(x) * args.lamda2).cuda()
    # x = (x - args.lamda1 * score + noise).detach()

    noise = torch.normal(0, torch.ones_like(z) * args.lamda2).cuda()
    z = (z - args.lamda1 * score + noise).detach()

    print("Energy: %f" % e_x.mean().item())

    if i % 1 == 0:
        save_image(x, 'mcmc_samples/image_%05d.png' % i, normalize=True)
        images.append(imread('mcmc_samples/image_%05d.png' % i))

mimwrite('mcmc.gif', images)
