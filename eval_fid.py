import torch
import numpy as np
from modules import Generator
from tqdm import tqdm
import sys
import os
from scipy.misc import imsave


if __name__ == '__main__':
    netG = Generator().cuda()
    netG.eval()
    netG.load_state_dict(torch.load(sys.argv[1]))

    all_samples = []
    for i in tqdm(range(50)):
        samples_100 = torch.randn(100, 128).cuda()
        all_samples.append(
            netG(samples_100).detach().cpu().numpy()
        )

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = (((all_samples * .5) + .5) * 255).astype('int32')

    for i in tqdm(range(all_samples.shape[0])):
        imsave(os.path.join(sys.argv[2], 'image_%d.png' % i), all_samples[i].transpose(1, 2, 0))
