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
    energy = netE(data)
    score = torch.autograd.grad(
        outputs=energy, inputs=data,
        grad_outputs=torch.ones_like(energy),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return data - (sigma ** 2) * score


class Generator(nn.Module):
    def __init__(self, z_dim=128, dim=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, dim * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim * 4, dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim * 2, dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z[:, :, None, None])


class Discriminator(nn.Module):
    def __init__(self, dim=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim * 2, dim * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim * 4, 1, 4, 1, 0)
        )

    def forward(self, x):
        return self.main(x)


class Classifier(nn.Module):
    def __init__(self, z_dim=128, dim=64):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim * 2, dim * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim * 4, z_dim, 4, 1, 0)
        )
        self.mlp = nn.Sequential(
            nn.Linear(z_dim * 2, dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim * 4, 1)
        )

    def forward(self, x, z):
        out = self.main(x).squeeze()
        out = torch.cat([out, z], -1)
        return self.mlp(out)


if __name__ == '__main__':
    # netD = Discriminator().cuda()
    # print(netD(torch.randn(64, 1, 28, 28).cuda()).size())
    netG = Generator().cuda()
    netD = Discriminator().cuda()
    netC = Classifier().cuda()

    z = torch.randn(64, 128).cuda()
    x = netG(z)
    logits = netD(x)
    mi = netC(x, z.squeeze())

    print(x.shape)
    print(logits.shape)
    print(mi.shape)
