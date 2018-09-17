import argparse
import gym
from itertools import count
import torch
import numpy as np
import matplotlib.pyplot as plt

from gridworld import GridWorld
from modules import PolicyWithValueFn, ExperienceReplay
from algorithms import a2c, reinforce_w_baseline, train_generative_model
from modules import MLP_Generator, MLP_Discriminator, MLP_Classifier


def visualize_energy(netE, n_points=100):
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    grid = np.asarray(np.meshgrid(x, y)).transpose(1, 2, 0).reshape((-1, 2))
    grid = torch.from_numpy(grid).float().cuda()
    energies = netE(grid).detach().cpu().numpy()
    e_grid = energies.reshape((n_points, n_points))

    plt.clf()
    plt.imshow(e_grid.T, origin='upper')
    plt.colorbar()
    plt.savefig('env_density.png')

    plt.clf()
    plt.imshow(np.sign(e_grid.T), origin='upper')
    plt.colorbar()
    plt.savefig('env_visitation.png')

    plt.clf()
    plt.imshow(env.counts / env.counts.sum(), origin='upper')
    plt.colorbar()
    plt.savefig('true_density.png')


parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=128,
                    help='GAN dimensionality (default: 128)')
parser.add_argument('--z_dim', type=int, default=2,
                    help='GAN z_dimensionality (default: 2)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size to train GAN (default: 256)')
parser.add_argument('--buffer_size', type=int, default=10000,
                    help='Replay Buffer size (default: 1000)')
parser.add_argument('--critic_iters', type=int, default=5,
                    help='GAN no. of critic iterations (default: 5)')
parser.add_argument('--sigma', type=float, default=.1,
                    help='score matching parameter')

parser.add_argument('--rl_dim', type=int, default=128,
                    help='RL model dimensionality (default: 128)')
parser.add_argument('--gamma', type=float, default=0.95,
                    help='discount factor (default: 0.95)')
parser.add_argument('--seed', type=int, default=543,
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10,
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = GridWorld(21, state_config='2D', goal=(14, 18))
# env = gym.make('CartPole-v1')
# env.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

n_actions = env.action_shape[0]
n_input = env.state_shape[0]
# n_actions = env.action_space.n
# n_input = env.observation_space.shape[0]

#######################
# Neural Network Policy
#######################
policy = PolicyWithValueFn(n_input, n_actions, args.dim)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4, amsgrad=True)

########################################
# Generative Model for states { p(s_t) }
########################################
replay = ExperienceReplay(args.buffer_size)
netG = MLP_Generator(n_input, args.z_dim, args.dim).cuda()
netE = MLP_Discriminator(n_input, args.dim).cuda()
netD = MLP_Classifier(n_input, args.z_dim, args.dim).cuda()

optimizerG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.9), amsgrad=True)
optimizerE = torch.optim.Adam(netE.parameters(), lr=2e-4, betas=(0.5, 0.9), amsgrad=True)
optimizerD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.9), amsgrad=True)


def test(n_episodes=20, n_steps=200):
    env.counts = env.counts * 0
    for i in range(n_episodes):
        state = env.reset()
        for t in range(n_steps):
            _, _, action, _ = policy.select_action(
                torch.from_numpy(state).float()
            )
            state, reward, done = env.step(action)
            if done:
                break

    visualize_energy(netE, n_points=21)


def train():
    running_reward = 10
    g_costs = []
    d_costs = []
    for i_episode in count(1):
        state = env.reset()

        epi_rewards = []
        epi_entropies = []
        epi_log_probs = []
        epi_values = []
        g_t = 0

        for t in range(200):  # Don't infinite loop while learning
            value, log_prob, action, entropy = policy.select_action(
                torch.from_numpy(state).float()
            )
            # state, reward, done = env.step(action)
            state, reward, done = env.step(np.random.choice(4))
            g_t += reward
            # reward = reward + \
            #     .1 * netE(torch.from_numpy(state).float().cuda()).item()
            reward = 10 * netE(torch.from_numpy(state).float().cuda()).item()

            if args.render:
                env.render()

            epi_rewards.append(reward)
            epi_log_probs.append(log_prob)
            epi_values.append(value)
            epi_entropies.append(entropy)
            replay.add(state)

            if done:
                break

        running_reward = running_reward * 0.99 + g_t * 0.01

        for i in range(10):
            train_generative_model(
                replay.sample(256),
                (netG, netE, netD), (optimizerG, optimizerE, optimizerD),
                args, g_costs, d_costs
            )

        if done:
            epi_values.append(torch.tensor([0., ]))
        else:
            epi_values.append(policy(
                torch.from_numpy(state).float()
            )[1])

        mi_est = - 1 * g_costs[-1][1]
        epi_rewards = [x + mi_est for x in epi_rewards]
        a2c(epi_rewards, epi_log_probs, epi_entropies, epi_values, policy, optimizer, args.gamma)

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\t'
                  'Average length: {:.2f}\t D_costs: {}\t G_costs: {}\t'.format(
                    i_episode, t, running_reward,
                    np.asarray(d_costs)[-args.log_interval:].mean(0),
                    np.asarray(g_costs)[-args.log_interval:].mean(0)
                    ))
            test()

        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break


if __name__ == '__main__':
    train()
