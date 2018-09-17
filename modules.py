from collections import deque
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ExperienceReplay(object):
    def __init__(self, size):
        self.buffer_size = size
        self.len = 0

        # Create buffers for (s_t, a_t, r_t, s_t+1, term)
        self.buffer = deque(maxlen=self.buffer_size)

    def sample(self, count):
        count = min(count, self.len)
        states = random.sample(self.buffer, count)
        return states

    def add(self, s_t):
        self.len += 1
        if self.len > self.buffer_size:
            self.len = self.buffer_size
        self.buffer.append(s_t)


class PolicyWithValueFn(nn.Module):
    def __init__(self, n_input, n_actions, dim):
        super().__init__()
        self.affine1 = nn.Linear(n_input, dim)
        self.affine2 = nn.Linear(dim, n_actions)
        self.affine3 = nn.Linear(dim, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        value = self.affine3(x)
        return F.softmax(action_scores, dim=-1), value

    def select_action(self, s_t):
        action_probs, value = self.forward(s_t)
        dist = Categorical(action_probs)
        a_t = dist.sample()
        log_prob = dist.log_prob(a_t)
        entropy = (-action_probs * action_probs.log()).sum()
        return value, log_prob, a_t.item(), entropy


class Policy(nn.Module):
    def __init__(self, n_input=4, n_actions=2, dim=128):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(n_input, dim)
        self.affine2 = nn.Linear(dim, n_actions)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=-1)

    def select_action(self, s_t):
        action_probs = self.forward(s_t)
        dist = Categorical(action_probs)
        a_t = dist.sample()
        log_prob = dist.log_prob(a_t)
        return log_prob, a_t.item()


class MLP_Generator(nn.Module):
    def __init__(self, output_dim, z_dim, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, output_dim)
        )

    def forward(self, z):
        return self.main(z)


class MLP_Discriminator(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, 1)
        )

    def forward(self, z):
        return self.main(z)


class MLP_Classifier(nn.Module):
    def __init__(self, input_dim, z_dim, dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim + z_dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        return self.main(x)
