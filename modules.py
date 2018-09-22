import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def calc_gradient_penalty(netD, real_data, fake_data, lamda=.1):
    alpha = torch.rand_like(real_data)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamda
    return gradient_penalty


def calc_reconstruction(netE, data, sigma):
    data.requires_grad_(True)
    # noisy_data = data + torch.normal(0, torch.ones_like(data) * sigma)
    noisy_data = data
    energy = netE(noisy_data)
    score = torch.autograd.grad(
        outputs=energy, inputs=noisy_data,
        grad_outputs=torch.ones_like(energy),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return noisy_data - sigma * score


class MLP_Generator(nn.Module):
    def __init__(self, output_dim, z_dim, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, output_dim)
        )

    def forward(self, z):
        return self.main(z)


class MLP_Discriminator(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, 1)
        )

    def forward(self, z):
        return self.main(z)


class MLP_Classifier(nn.Module):
    def __init__(self, input_dim, z_dim, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, z_dim)
        )

    def forward(self, x):
        return self.main(x)
