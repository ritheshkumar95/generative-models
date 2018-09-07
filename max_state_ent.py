import argparse
import gym
from itertools import count
import torch
import numpy as np

from modules import Policy, ExperienceReplay
from algorithms import reinforce, reinforce_curiosity, train_generative_model
from modules import MLP_Generator, MLP_Discriminator, MLP_Classifier


parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=128,
                    help='GAN dimensionality (default: 128)')
parser.add_argument('--z_dim', type=int, default=2,
                    help='GAN z_dimensionality (default: 2)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size to train GAN (default: 256)')
parser.add_argument('--buffer_size', type=int, default=1000,
                    help='Replay Buffer size (default: 1000)')
parser.add_argument('--critic_iters', type=int, default=5,
                    help='GAN no. of critic iterations (default: 5)')

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


env = gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

n_actions = env.action_space.n
n_input = env.observation_space.shape[0]

#######################
# Neural Network Policy
#######################
policy = Policy(n_input, n_actions, args.dim).cuda()
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2, amsgrad=True)

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


def main():
    running_reward = 10
    g_costs = []
    d_costs = []
    for i_episode in count(1):
        state = env.reset()

        epi_rewards = []
        epi_log_probs = []
        g_t = 0

        for t in range(10000):  # Don't infinite loop while learning
            log_prob, action = policy.select_action(
                torch.from_numpy(state).float().cuda()
            )
            state, reward, done, _ = env.step(action)
            g_t += reward

            if args.render:
                env.render()

            epi_rewards.append(reward)
            epi_log_probs.append(log_prob)
            replay.add(state)

            if done:
                break

        running_reward = running_reward * 0.99 + g_t * 0.01

        train_generative_model(
            replay.sample(args.batch_size),
            (netG, netE, netD), (optimizerG, optimizerE, optimizerD),
            args, g_costs, d_costs
        )

        mi_est = - 1 * g_costs[-1][1]
        epi_rewards = [x + mi_est for x in epi_rewards]
        reinforce(epi_rewards, epi_log_probs, optimizer, args.gamma)
        # reinforce_curiosity(mi_est, epi_log_probs, optimizer)

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\t'
                  'Average length: {:.2f}\t D_costs: {}\t G_costs: {}\t'.format(
                    i_episode, t, running_reward,
                    np.asarray(d_costs)[-args.log_interval:].mean(0),
                    np.asarray(g_costs)[-args.log_interval:].mean(0)
                    ))

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
