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
        self.preprocess = nn.Sequential(
            nn.Linear(z_dim, 4 * 4 * 4 * dim),
            nn.ReLU(True)
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 5),
            nn.ReLU(True)
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 5),
            nn.ReLU(True)
        )
        self.deconv_out = nn.Sequential(
            nn.ConvTranspose2d(dim, 1, 8, stride=2),
            nn.Sigmoid()
        )
        self.dim = dim

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.dim, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, dim=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, dim, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(dim, 2 * dim, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2 * dim, 4 * dim, 5, stride=2, padding=2),
            nn.ReLU(True)
        )
        self.output = nn.Linear(4 * 4 * 4 * dim, 1)
        self.dim = dim

    def forward(self, input):
        out = self.main(input)
        out = out.view(-1, 4 * 4 * 4 * self.dim)
        out = self.output(out)
        return out.view(-1)


class Classifier(nn.Module):
    def __init__(self, z_dim=128, dim=64):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, dim, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(dim, 2 * dim, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2 * dim, 4 * dim, 5, stride=2, padding=2),
            nn.ReLU(True)
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(4 * 4 * 4 * dim, z_dim),
            nn.ReLU(True)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(z_dim * 2, dim),
            nn.ReLU(True),
            nn.Linear(dim, 1)
        )
        self.dim = dim

    def forward(self, x, z):
        out = self.main(x)
        out = out.view(-1, 4 * 4 * 4 * self.dim)
        out = self.mlp1(out)
        out = torch.cat([out, z], -1)
        return self.mlp2(out)


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
