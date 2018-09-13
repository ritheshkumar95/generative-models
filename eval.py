import torch
import numpy as np
from modules import Generator
from tqdm import tqdm


def compute_inception_score(netG):
    from inception_score import get_inception_score
    all_samples = []
    for i in tqdm(range(50)):
        samples_100 = torch.randn(100, 128).cuda()
        all_samples.append(
            netG(samples_100).detach().cpu().numpy()
        )

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = (((all_samples * .5) + .5) * 255).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return get_inception_score(list(all_samples))


netG = Generator(dim=128).cuda()
netG.eval()
netG.load_state_dict(torch.load('models/ebm_CIFAR10.pt'))
# netG.load_state_dict(torch.load('models/wgan-gp_CIFAR10.pt'))
print(compute_inception_score(netG))
