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


def reinforce_w_baseline(rewards, log_probs, entropies, values, optimizer, gamma):
    returns = torch.tensor(
        compute_discounted_returns(rewards, gamma)
    )

    returns = returns - torch.cat(values).detach()  # G_t - b_t
    eps = np.finfo(np.float32).eps.item()
    returns = (returns - returns.mean()) / (returns.std() + eps)

    policy_loss = 0
    for log_prob, g_t in zip(log_probs, returns):
        policy_loss += -log_prob * g_t

    policy_loss -= 0.01 * torch.stack(entropies).sum()

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


def a2c(rewards, log_probs, entropies, values, policy, optimizer, gamma):
    R = values[-1].detach().numpy()
    policy_loss = 0
    value_loss = 0
    gae = torch.tensor([1., ])
    for i in reversed(range(len(rewards))):
        R = gamma * R + rewards[i]
        advantage = torch.tensor([R, ]) - values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        # Generalized Advantage Estimataion
        delta_t = rewards[i] + gamma * values[i + 1].data - values[i].data
        gae = gae * gamma + delta_t

        policy_loss = policy_loss - log_probs[i] * gae - 0.01 * entropies[i]

    optimizer.zero_grad()
    (policy_loss + 0.5 * value_loss).backward()
    torch.nn.utils.clip_grad_norm(policy.parameters(), 40)
    optimizer.step()


def calc_reconstruction(netE, data, sigma):
    data.requires_grad_(True)
    energy = netE(data)
    score = torch.autograd.grad(
        outputs=energy, inputs=data,
        grad_outputs=torch.ones_like(energy),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return data - (sigma ** 2) * score


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

        data = torch.cat([x_real, x_fake], 0)
        score_matching_loss = nn.MSELoss()(
            calc_reconstruction(netE, data, args.sigma),
            data
        )
        score_matching_loss.backward()

        optimizerE.step()
        d_costs.append(
            [D_real.item(), D_fake.item(), score_matching_loss.item()]
        )

    return g_costs, d_costs
