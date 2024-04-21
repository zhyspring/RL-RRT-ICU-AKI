# RL model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.)


class QNetwork_64(nn.Module):
    def __init__(self, state_dim=16, nb_actions=None):
        super(QNetwork_64, self).__init__()

        self.state_dim = state_dim
        self.nb_actions = nb_actions
        
        self.fc = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.nb_actions)
        )
        self.fc.apply(init_weights)
        
    def forward(self, x):
        x = self.fc(x)
        return x


class QNetwork_128(nn.Module):
    def __init__(self, state_dim=16, nb_actions=None):
        super(QNetwork_128, self).__init__()

        self.state_dim = state_dim
        self.nb_actions = nb_actions
        
        self.fc = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.nb_actions)
        )
        self.fc.apply(init_weights)
        
    def forward(self, x):
        x = self.fc(x)
        return x


class QNetwork_6464(nn.Module):
    def __init__(self, state_dim=16, nb_actions=None):
        super(QNetwork_6464, self).__init__()

        self.state_dim = state_dim
        self.nb_actions = nb_actions
        
        self.fc = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.nb_actions)
        )
        self.fc.apply(init_weights)
        
    def forward(self, x):
        x = self.fc(x)
        return x


# State transfer model

class AISGenerate_1(nn.Module):
    def __init__(self, state_dim, obs_dim, num_actions):
        super(AISGenerate_1, self).__init__()
        self.l1 = nn.Linear(obs_dim + num_actions, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.GRUCell(128, state_dim)
    def forward(self, x, h):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        h = self.l3(x, h)
        return h


class AISGenerate_2(nn.Module):
    def __init__(self, state_dim, obs_dim, num_actions):
        super(AISGenerate_2, self).__init__()
        self.l1 = nn.Linear(obs_dim + num_actions, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.GRUCell(64, state_dim)
    def forward(self, x, h):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        h = self.l4(x, h)
        return h


class AISPredict_1(nn.Module):
    def __init__(self, state_dim, obs_dim, num_actions):
        super(AISPredict_1, self).__init__()
        self.l1 = nn.Linear(state_dim + num_actions, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, obs_dim)
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        obs = self.l3(x)
        return obs


class AISPredict_2(nn.Module):
    def __init__(self, state_dim, obs_dim, num_actions):
        super(AISPredict_2, self).__init__()
        self.l1 = nn.Linear(state_dim + num_actions, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, obs_dim)
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        obs = self.l4(x)
        return obs
