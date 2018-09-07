import argparse
import gym
from itertools import count
import torch

from modules import Policy
from algorithms import reinforce
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--dim', type=int, default=128,
                    help='model dimensionality (default: 128')
parser.add_argument('--gamma', type=float, default=0.95,
                    help='discount factor (default: 0.95)')
parser.add_argument('--seed', type=int, default=543,
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10,
                    help='interval between training status logs (default: 10)')
parser.add_argument('--log-dir', required=True,
                    help='log directory')
args = parser.parse_args()


writer = SummaryWriter(args.log_dir)
env = gym.make('Acrobot-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

n_actions = env.action_space.n
n_input = env.observation_space.shape[0]

policy = Policy(n_input, n_actions, args.dim)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2, amsgrad=True)


def main():
    running_reward = -500
    for i_episode in count(1):
        state = env.reset()

        epi_rewards = []
        epi_log_probs = []

        g_t = 0
        for t in range(10000):  # Don't infinite loop while learning
            log_prob, action = policy.select_action(
                torch.from_numpy(state).float()
            )
            state, reward, done, _ = env.step(action)
            g_t += reward

            if args.render:
                env.render()

            epi_rewards.append(reward)
            epi_log_probs.append(log_prob)

            if done:
                break

        writer.add_scalar('/return/', g_t, i_episode)
        running_reward = running_reward * 0.99 + g_t * 0.01
        reinforce(epi_rewards, epi_log_probs, optimizer, args.gamma)

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage return: {:.2f}'.format(
                i_episode, t, running_reward))

        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break


if __name__ == '__main__':
    main()
