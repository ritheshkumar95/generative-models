import argparse
import torch
from tqdm import tqdm

from data import inf_train_gen
from torchvision.utils import save_image
from modules import Discriminator, Generator
from imageio import imread, mimwrite


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_stack', type=int, default=1)
    parser.add_argument('--n_points', type=int, default=64)

    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)

    parser.add_argument('--lamda1', type=float, default=.01)
    parser.add_argument('--lamda2', type=float, default=.01)
    args = parser.parse_args()
    return args


args = parse_args()

itr = inf_train_gen(args.n_points, n_stack=args.n_stack)
netG = Generator(args.n_stack, args.z_dim, args.dim).cuda()
netE = Discriminator(args.n_stack, args.dim).cuda()

netE.load_state_dict(torch.load('models/ebm_netE_MNIST_%d.pt' % args.n_stack))
netG.load_state_dict(torch.load('models/ebm_netG_he_MNIST_%d.pt' % args.n_stack))

orig_x = itr.__next__().cuda()
mask = torch.zeros_like(orig_x)
mask[:, :, :14].data.fill_(1)

z = torch.randn(args.n_points, args.z_dim).cuda()

images = []
for i in tqdm(range(1, 101)):
    z.requires_grad_(True)
    x = netG(z)
    x = x * (1 - mask) + orig_x * mask
    e_x = netE(x)

    score = torch.autograd.grad(
        outputs=e_x, inputs=z,
        grad_outputs=torch.ones_like(e_x),
        create_graph=True, retain_graph=False, only_inputs=True
    )[0]

    # magnitude = score.view(score.size(0), -1).norm(2, dim=-1)
    # direction = score / magnitude[:, None, None, None]

    noise = torch.normal(0, torch.ones_like(z) * args.lamda2).cuda()
    # x = (x - args.lamda1 * score + noise).detach()
    z = (z - args.lamda1 * score + noise).detach()

    print("Energy: %f" % e_x.mean().item())

    if i % 1 == 0:
        save_image(x, 'mcmc_samples/image_%05d.png' % i, normalize=True)
        images.append(imread('mcmc_samples/image_%05d.png' % i))

mimwrite('mcmc.gif', images)
