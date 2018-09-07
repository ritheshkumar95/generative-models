import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_discounted_returns(rewards, gamma=0.95):
    returns = []
    G_t = 0
    for r_t in reversed(list(rewards)):
        G_t = gamma * G_t + r_t
        returns.append(G_t)
    returns = list(reversed(returns))
    return returns


def reinforce(rewards, log_probs, optimizer, gamma):
    device = log_probs[0].device
    returns = torch.tensor(
        compute_discounted_returns(rewards, gamma)
    ).to(device)

    eps = np.finfo(np.float32).eps.item()
    returns = (returns - returns.mean()) / (returns.std() + eps)

    policy_loss = 0
    for log_prob, g_t in zip(log_probs, returns):
        policy_loss += -log_prob * g_t

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()


def reinforce_w_baseline(rewards, log_probs, values, optimizer, gamma):
    returns = torch.tensor(
        compute_discounted_returns(rewards, gamma)
    )

    returns = returns - torch.cat(values).detach()  # G_t - b_t
    eps = np.finfo(np.float32).eps.item()
    returns = (returns - returns.mean()) / (returns.std() + eps)

    policy_loss = 0
    for log_prob, g_t in zip(log_probs, returns):
        policy_loss += -log_prob * g_t

    value_fn_loss = F.smooth_l1_loss(
        torch.cat(values),
        torch.tensor(returns, dtype=torch.float32)
    )

    optimizer.zero_grad()
    (policy_loss + value_fn_loss).backward()
    optimizer.step()


def reinforce_curiosity(reward, log_probs, optimizer):
    policy_loss = (-torch.stack(log_probs) * reward).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()


def actor_critic(rewards, log_probs, values, optimizer, gamma):
    returns = torch.tensor(
        compute_discounted_returns(rewards, gamma)
    )

    eps = np.finfo(np.float32).eps.item()
    returns = (returns - returns.mean()) / (returns.std() + eps)

    policy_loss = 0
    for log_prob, g_t, b_t in zip(log_probs, returns, values):
        reward = g_t - b_t.item()
        policy_loss += -log_prob * reward

    value_fn_loss = F.smooth_l1_loss(
        torch.cat(values),
        torch.tensor(returns, dtype=torch.float32)
    )

    optimizer.zero_grad()
    (policy_loss + value_fn_loss).backward()
    optimizer.step()


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


def train_generative_model(
    states, networks, optimizers,
    args, g_costs, d_costs
):
    batch_size = min(len(states), args.batch_size)
    label = torch.ones(2 * batch_size).float().cuda()
    label[batch_size:].zero_()

    netG, netE, netD = networks
    optimizerG, optimizerE, optimizerD = optimizers

    ###################################
    # Train GAN Generator to match EBM
    ###################################
    netG.zero_grad()
    netD.zero_grad()

    z = torch.randn(batch_size, args.z_dim).cuda()
    x_fake = netG(z)
    D_fake = netE(x_fake)
    D_fake = D_fake.mean()
    D_fake.backward(retain_graph=True)

    ###################################
    # Get MI estimate using DeepInfoMax
    ###################################
    z_bar = z.clone()[torch.randperm(z.size(0))]
    orig_x_z = torch.cat([x_fake, z], -1)
    shuf_x_z = torch.cat([x_fake, z_bar], -1)
    concat_x_z = torch.cat([orig_x_z, shuf_x_z], 0)

    logits = netD(concat_x_z)
    dim_estimate = nn.BCEWithLogitsLoss()(logits.squeeze(), label)
    dim_estimate.backward()

    optimizerG.step()
    optimizerD.step()

    g_costs.append(
        [D_fake.item(), dim_estimate.item()]
    )

    ##########################
    # Train Energy Based Model
    ##########################
    for i in range(args.critic_iters):
        x_real = torch.tensor(states, dtype=torch.float32).cuda()

        netE.zero_grad()
        D_real = netE(x_real)
        D_real = D_real.mean()
        D_real.backward()

        # train with fake
        z = torch.randn(batch_size, args.z_dim).cuda()
        x_fake = netG(z).detach()
        D_fake = netE(x_fake)
        D_fake = D_fake.mean()
        (-D_fake).backward()

        gradient_penalty = calc_gradient_penalty(
            netE, x_real.data, x_fake.data
        )
        gradient_penalty.backward()

        optimizerE.step()
        d_costs.append(
            [D_real.item(), D_fake.item()]
        )

    return g_costs, d_costs
