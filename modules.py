import torch
import torch.nn as nn


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
    noisy_data = data + torch.normal(0, torch.ones_like(data) * sigma)
    energy = netE(noisy_data)
    score = torch.autograd.grad(
        outputs=energy, inputs=noisy_data,
        grad_outputs=torch.ones_like(energy),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return noisy_data - sigma * score


class Generator(nn.Module):
    def __init__(self, input_dim, z_dim=128, dim=512):
        super(Generator, self).__init__()
        self.expand = nn.Linear(z_dim, 2 * 2 * dim)
        self.main = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim // 2, 5, 2, 2, output_padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 2, dim // 4, 5, 2, 2),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 4, dim // 8, 5, 2, 2, output_padding=1),
            nn.BatchNorm2d(dim // 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 8, input_dim, 5, 2, 2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.expand(z).view(z.size(0), -1, 2, 2)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, dim=512):
        super(Discriminator, self).__init__()
        self.expand = nn.Linear(2 * 2 * dim, 1)
        self.main = nn.Sequential(
            nn.Conv2d(input_dim, dim // 8, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 8, dim // 4, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 4, dim // 2, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 2, dim, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        out = self.main(x).view(x.size(0), -1)
        return self.expand(out)


class Classifier(nn.Module):
    def __init__(self, input_dim=1, z_dim=128, dim=512):
        super(Classifier, self).__init__()
        self.expand = nn.Linear(2 * 2 * dim, z_dim)
        self.main = nn.Sequential(
            nn.Conv2d(input_dim, dim // 8, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 8, dim // 4, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 4, dim // 2, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 2, dim, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        out = self.main(x).view(x.size(0), -1)
        return self.expand(out)


if __name__ == '__main__':
    # netD = Discriminator().cuda()
    # print(netD(torch.randn(64, 1, 28, 28).cuda()).size())
    netG = Generator(1).cuda()
    netD = Discriminator(1).cuda()
    netC = Classifier().cuda()

    z = torch.randn(64, 128).cuda()
    x = netG(z)
    print(x.shape)

    logits = netD(x)
    print(logits.shape)

    mi = netC(x, z)
    print(mi.shape)
