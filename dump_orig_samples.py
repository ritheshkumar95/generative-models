import torch
from torchvision import transforms, datasets
from scipy.misc import imsave
import os


def inf_train_gen(batch_size):
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            '../data/CIFAR10', train=True, download=True,
            transform=transf
        ), batch_size=batch_size, drop_last=True
    )
    while True:
        for img, labels in loader:
            yield img


os.system('mkdir -p true_samples/')
itr = inf_train_gen(100)
for i in range(50):
    x = itr.__next__()
    x = (((x * .5) + .5) * 255).long().cpu().numpy()
    for j in range(100):
        imsave('true_samples/image_%d.png' % (i * 100 + j), x[j].transpose(1, 2, 0))

