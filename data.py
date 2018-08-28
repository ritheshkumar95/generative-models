import torch
from torchvision import datasets, transforms


def inf_train_gen(dataset, batch_size):
    loader = torch.utils.data.DataLoader(
        eval('datasets.%s' % dataset)(
            '../data/%s' % dataset, train=True, download=True,
            transform=transforms.ToTensor()
        ), batch_size=64, drop_last=True
    )
    while True:
        for img, labels in loader:
            yield img
