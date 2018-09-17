import argparse
import gym
from gridworld import GridWorld
from itertools import count

import torch

from modules import PolicyWithValueFn
from algorithms import a2c
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch REINFORCE w/ Baseline example')
parser.add_argument('--grid_size', type=int, default=11, metavar='G',
                    help='size of the gridworld environment')
parser.add_argument('--dim', type=int, default=128, metavar='G',
                    help='model dimensionality (default: 128')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--log-dir', required=True,
                    help='log directory')
args = parser.parse_args()


writer = SummaryWriter(args.log_dir)
env = GridWorld(args.grid_size, state_config='2D')
# env = gym.make('CartPole-v1')
# env.seed(args.seed)
torch.manual_seed(args.seed)

n_actions = env.action_shape[0]
n_input = env.state_shape[0]
# n_actions = env.action_space.n
# n_input = env.observation_space.shape[0]

policy = PolicyWithValueFn(n_input, n_actions, args.dim)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3, amsgrad=True)


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()

        epi_rewards = []
        epi_log_probs = []
        epi_values = []
        epi_entropies = []
        g_t = 0

        for t in range(1000):  # Don't infinite loop while learning
            value, log_prob, action, entropy = policy.select_action(
                torch.from_numpy(state).float()
            )
            state, reward, done = env.step(action)
            g_t += reward

            if args.render:
                env.render()

            epi_rewards.append(reward)
            epi_log_probs.append(log_prob)
            epi_values.append(value)
            epi_entropies.append(entropy)

            if done:
                break

        if done:
            epi_values.append(torch.tensor([0., ]))
        else:
            epi_values.append(policy(
                torch.from_numpy(state).float()
            )[1])

        writer.add_scalar('/return/', g_t, i_episode)
        running_reward = running_reward * 0.99 + g_t * 0.01
        a2c(epi_rewards, epi_log_probs, epi_entropies, epi_values, optimizer, args.gamma)

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))

        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break


if __name__ == '__main__':
    main()
