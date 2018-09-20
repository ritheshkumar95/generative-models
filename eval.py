import argparse
from tqdm import tqdm
import numpy as np
from math import log

import torch
import torch.nn.functional as F

from modules import Generator
from classifier import Net


def KLD(p, q):
    if 0 in q:
        raise ValueError
    return sum(_p * log(_p/_q) for (_p, _q) in zip(p, q) if _p != 0)


class ModeCollapseEval(object):
    def __init__(self, n_stack, z_dim):
        self.classifier = Net().cuda()
        self.classifier.load_state_dict(torch.load('models/pretrained_classifier.pt'))
        self.n_stack = n_stack
        self.z_dim = z_dim

    def count_modes(self, netG):
        counts = np.zeros([10] * self.n_stack)
        for i in tqdm(range(1000)):
            with torch.no_grad():
                z = torch.randn(100, self.z_dim).cuda()
                x_fake = netG(z) * .5 + .5
                x_fake = x_fake.view(-1, 1, 28, 28)
                classes = F.softmax(self.classifier(x_fake), -1).max(1)[1]
                classes = classes.view(100, self.n_stack).cpu().numpy()

                for line in classes:
                    counts[tuple(line)] += 1


        n_modes = 10 ** self.n_stack
        true_data = np.ones(n_modes) / float(n_modes)
        print("No. of modes captured: ", len(np.where(counts > 0)[0]))
        counts = counts.flatten() / counts.sum()
        print('Reverse KL: ', KLD(counts, true_data))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', required=True)

    parser.add_argument('--n_stack', type=int, default=3)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    netG = Generator(args.n_stack, args.z_dim, args.dim).cuda()
    netG.load_state_dict(torch.load(args.load_path))
    netG.eval()

    evals = ModeCollapseEval(args.n_stack, args.z_dim)
    evals.count_modes(netG)

