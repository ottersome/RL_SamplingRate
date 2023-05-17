import torch
from torch import nn
from collections import deque, namedtuple
import torch.nn.functional as F
import numpy as np
import random

class RNNContinuousPolicy(nn.Module):
    def __init__(self, num_params,hidden_size,samp_rate_max=32):
        super(RNNContinuousPolicy,self).__init__()
        #self.state_limit = num_states
        #self.fc1 = nn.Linear(num_params, hidden_size)
        #self.fc2 = nn.Linear(hidden_size, 1)
        self.samp_rate_max = samp_rate_max
        self.model = nn.Sequential(
            nn.Linear(num_params, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )

    def forward(self, X):
        # Prpping
        #_,(hiddn,_) = self.rnn(X)
        #logit = self.fc2(F.relu(self.fc1(X)))
        logit = self.model(X)
        output = torch.sigmoid(logit)
        output_normalized = output*self.samp_rate_max
        return output_normalized
        # We are giving X as a 

# Critic Network
class Critic(nn.Module):
    def __init__(self, num_param, action_size, hidden_size):
        super(Critic, self).__init__()
        #self.fc1 = nn.Linear(num_param + action_size, hidden_size)
        #self.fc2 = nn.Linear(hidden_size, 1)
        self.model = nn.Sequential(
            nn.Linear(num_param+action_size, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )

    def forward(self, states, action):# States are the paremeters
        #_,(h,_)  = self.rnn(mstates)
        x = torch.cat((states, action), dim=1)
        return self.model(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)*10
# Replay buffer

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, chain_length, num_states):
        self.chain_length = chain_length
        self.num_states = num_states
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "samp_rate", "errors" ])

    def add(self, state, samp_rate, errors):
        e = self.experience(state, samp_rate, errors)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states =  torch.empty((self.batch_size,2**2))# TODO Remove hard code to the number of entries in generator matrix

        # The states will be encoded as tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.samp_rate for e in experiences if e is not None])).float()
        errors = torch.from_numpy(np.vstack([e.errors for e in experiences if e is not None])).float()

        return states, actions, errors

    def __len__(self):
        return len(self.memory)