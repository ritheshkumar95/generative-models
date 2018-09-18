import torch
import numpy as np
from modules import Generator
from tqdm import tqdm
import sys


def tf_inception_score(netG):
    from inception_score import get_inception_score
    all_samples = []
    for i in tqdm(range(50)):
        samples_100 = torch.randn(100, 128).cuda()
        all_samples.append(
            netG(samples_100).detach().cpu().numpy()
        )

    all_samples = np.concatenate(all_samples, axis=0)
    return get_inception_score(all_samples)


def pytorch_inception_score(netG):
    from inception_score_pytorch import inception_score
    all_samples = []
    for i in tqdm(range(50)):
        samples_100 = torch.randn(100, 128).cuda()
        all_samples.append(
            netG(samples_100).detach().cpu()
        )

    all_samples = torch.cat(all_samples, 0)
    return inception_score(all_samples, resize=True, splits=10)


if __name__ == '__main__':
    netG = Generator(dim=128).cuda()
    netG.eval()
    netG.load_state_dict(torch.load(sys.argv[1]))
    print(tf_inception_score(netG))
