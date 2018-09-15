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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', required=True)

    parser.add_argument('--n_stack', type=int, default=3)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=512)
    args = parser.parse_args()
    return args


args = parse_args()
netG = Generator(args.n_stack, args.z_dim, args.dim).cuda()
netG.load_state_dict(torch.load(args.load_path))
netG.eval()

classifier = Net().cuda()
classifier.load_state_dict(torch.load('models/pretrained_classifier.pt'))

counts = np.zeros((10, 10, 10))
for i in tqdm(range(260)):
    with torch.no_grad():
        z = torch.randn(100, args.z_dim).cuda()
        x_fake = netG(z) * .5 + .5
        x_fake = x_fake.view(-1, 1, 28, 28)
        classes = F.softmax(classifier(x_fake), -1).max(1)[1]
        classes = classes.view(100, 3).cpu().numpy()

        for line in classes:
            counts[line[0], line[1], line[2]] += 1


true_data = np.ones(1000) / 1000.
print("No. of modes captured: ", len(np.where(counts > 0)[0]))
counts = counts.flatten() / counts.sum()
print('Reverse KL: ', KLD(counts, true_data))
