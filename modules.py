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


def calc_penalty(netE, data, lamda):
    data.requires_grad_(True)
    energy = netE(data)
    score = torch.autograd.grad(
        outputs=energy, inputs=data,
        grad_outputs=torch.ones_like(energy),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return (score.norm(2, dim=1) ** 2).mean() * lamda


class Generator(nn.Module):
    def __init__(self, z_dim=128, dim=512):
        super(Generator, self).__init__()
        self.expand = nn.Linear(z_dim, 4 * 4 * dim)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, 4, 2, 1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 2, dim // 4, 4, 2, 1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 4, dim // 8, 4, 2, 1),
            nn.BatchNorm2d(dim // 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 8, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.expand(z).view(z.size(0), -1, 4, 4)
        return self.main(out)


class Discriminator(nn.Module):
    def __init__(self, dim=512):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, dim // 8, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 8, dim // 8, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 8, dim // 4, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 4, dim // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2, dim // 2, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2, dim, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.expand = nn.Linear(4 * 4 * dim, 1)

    def forward(self, x):
        out = self.main(x).view(x.size(0), -1)
        return self.expand(out)


class Classifier(nn.Module):
    def __init__(self, z_dim=128, dim=512):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, dim // 8, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 8, dim // 8, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 8, dim // 4, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 4, dim // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2, dim // 2, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2, dim, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.expand = nn.Linear(4 * 4 * dim, z_dim)

    def forward(self, x):
        out = self.main(x).view(x.size(0), -1)
        return self.expand(out)


if __name__ == '__main__':
    # netD = Discriminator().cuda()
    # print(netD(torch.randn(64, 1, 28, 28).cuda()).size())
    netG = Generator().cuda()
    netD = Discriminator().cuda()
    netC = Classifier().cuda()

    z = torch.randn(64, 128).cuda()
    x = netG(z)
    logits = netD(x)
    mi = netC(x)

    print(x.shape)
    print(logits.shape)
    print(mi.shape)
