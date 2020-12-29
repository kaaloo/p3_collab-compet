import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Network(nn.Module):

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


class Actor(Network):

    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim):
        super(Actor, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_in_dim)
        self.bn2 = nn.BatchNorm1d(hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.bn3 = nn.BatchNorm1d(hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, output_dim)
        self.nonlin = f.relu  # leaky_relu
        self.reset_parameters()

    def forward(self, x):
        can_norm = len(x.shape) > 1 and x.shape[0] > 1
        x = self.nonlin(self.fc1(self.bn1(x) if can_norm else x))
        x = self.nonlin(self.fc2(self.bn2(x) if can_norm else x))
        x = self.fc3(self.bn3(x) if can_norm else x)
        return torch.tanh(x)


class Critic(Network):

    def __init__(self, state_size, action_size, hidden_in_dim, hidden_out_dim, output_dim, num_agents):
        super(Critic, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.bn1 = nn.BatchNorm1d(num_agents)
        self.fc1 = nn.Linear(state_size, state_size)
        self.bn2 = nn.BatchNorm1d(num_agents)
        self.fc2 = nn.Linear(state_size + action_size, hidden_in_dim)
        self.bn3 = nn.BatchNorm1d(num_agents)
        self.fc3 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.bn4 = nn.BatchNorm1d(num_agents)
        self.fc4 = nn.Linear(hidden_out_dim, output_dim)
        self.nonlin = f.relu  # leaky_relu
        self.reset_parameters()

    def forward(self, state, action):
        can_norm = len(state.shape) > 1 and state.shape[0] > 1
        x = self.nonlin(self.fc1(self.bn1(state) if can_norm else state))
        x = torch.cat((self.bn2(x) if can_norm else x, action), dim=2)
        x = self.nonlin(self.fc2(self.bn2(x) if can_norm else x))
        x = self.nonlin(self.fc3(self.bn3(x) if can_norm else x))
        x = self.fc4(self.bn4(x) if can_norm else x)
        return x
