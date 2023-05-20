import torch
from scipy.linalg import expm
from torch import nn
from sp_sims.simulators.stochasticprocesses import RaceOfExponentials
from sp_sims.detectors.pearsonneyman import multiplicity_guess
from sp_sims.statistics.statistics import quick_sample_budget
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
        return torch.sigmoid(self.model(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)*10
# Replay buffer

class ReplayBuffer:

    def __init__(self, 
                sampbudget,
                buffer_size=None):

        self.sampbudget = sampbudget
        self.memory = deque(maxlen=buffer_size)

        # Calculate Furthest Sample

        #self.experience = namedtuple("Experience", field_names=["state", "samp_rate", "errors" ])
        #self.experience = namedtuple("Experience", field_names=["state", "samp_rate", "errors" ])

    def add(self, state, samp_rate, errors):
        #e = self.experience(state, samp_rate, errors)
        self.memory.append((state, samp_rate, errors))

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states =  torch.empty((batch_size,2**2))# TODO Remove hard code to the number of entries in generator matrix

        # The states will be encoded as tensors
        #states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        #actions = torch.from_numpy(np.vstack([e.samp_rate for e in experiences if e is not None])).float()
        #errors = torch.from_numpy(np.vstack([e.errors for e in experiences if e is not None])).float()
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float()
        errors = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float()

        return states, actions, errors
    def __len__(self):
        return len(self.memory)

    def populate_replay_buffer(self,num_of_examples, guesses_per_rate = 1000):
        ## Start By Generating States
        # TODO: Maybe Try Uniform
        # Make Sure lam > mu
        #rates0[torch.chat(idcs, torch.ones((num_of_examples,1)))] = rates0[idcs:0] + 1

        # Generate some random sampling rates
        rates= gen_rates_bd('uniform', num_of_examples)
        rates0,rates1 = rates

        # Get Errors  # ACTIONS
        smp_rates = np.random.uniform(0,2**4, num_of_examples) # Actions
        errors = [0]*len(smp_rates)# Amount of errors per action

        # TODO: PArallelize this through threads 
        for i in range(num_of_examples):# For Every State-Action
            #print('Samp Rate {} thus windows {}'.format(smp_rates[i],self.sampbudget*(1/smp_rates[i])))
            true_hyps = np.random.choice(2,guesses_per_rate)# Prepare for the amount of ensuing errors 
            
            p0 = expm((1/smp_rates[i])*np.array([[-rates0[i,0],rates0[i,0]],[rates0[i,1],-rates0[i,1]]]))# 0=lam, 1=mu
            p1 = expm((1/smp_rates[i])*np.array([[-rates1[i,0],rates1[i,0]],[rates1[i,1],-rates1[i,1]]]))

            for j in range(guesses_per_rate): # Sample a bunch of paths
                roe = RaceOfExponentials(self.sampbudget*(1/smp_rates[i]), rates[true_hyps[j]][i],max_state=1)# TODO Remove that hardcoded 1
                holdTimes_tape, state_tape = roe.generate_history(0)# TODO here I am hardcoding the initial state
                # Action Performance(Sampling At Rate)
                #tape = quick_sample(smp_rates[i],state_tape,holdTimes_tape,args2.num_samples)
                tmpSampTape, replicas = quick_sample_budget(smp_rates[i], state_tape,holdTimes_tape, budget=self.sampbudget)

                # Get the corresponding Losses
                # TODO: Maybe scale down the errors /kj
                errors[i] += multiplicity_guess(tmpSampTape,replicas, p0, p1) != true_hyps[j]

        errors = np.array(errors)/guesses_per_rate
        #errors = np.array(errors)

        for i in range(num_of_examples):
            self.add(list(rates0[i])+list(rates1[i]), smp_rates[i],errors[i])


    def add_actions_to_buffer(self, states, actions, guesses_per_rate = 1000):
        errors = [0]*len(actions)# Amount of errors per action

        # TODO this could be parallelized into threads
        for i in range(actions.shape[0]):# For Every State-Action
            true_hyps = np.random.choice(2,guesses_per_rate)

            p0 = expm((1/actions[i])*np.array([[-states[i,0],states[i,0]],[states[i,1],-states[i,1]]]))# 0=lam, 1=mu
            p1 = expm((1/actions[i])*np.array([[-states[i,2],states[i,2]],[states[i,3],-states[i,3]]]))

            for j in range(guesses_per_rate): # Sample a bunch of paths
                offset = true_hyps[j]*1

                #roe = RaceOfExponentials(args2.length,states[i,offset:offset+2].detach(),state_limit=args2.state_limit)
                roe = RaceOfExponentials(guesses_per_rate*1/(actions[i]),states[i,offset:offset+2].detach())
                holdTimes_tape, state_tape = roe.generate_history(args2.init_state)
                # Action Performance(Sampling At Rate)
                tape = quick_sample_budget(actions[i],state_tape, holdTimes_tape,self.sampbudget)
                #tape = quick_sample(actions[i],state_tape,holdTimes_tape,args2.num_samples)# Could be that this was plain wrong. 

                # Get the corresponding Losses
                #errors[i] += take_a_guess(tape, p0,p1)  != true_hyps[j]
                errors[i] += take_guess_multiplicity(tape, p0,p1)  != true_hyps[j]

        errors = np.array(errors) / guesses_per_rate

        for i in range(states.shape[0]):
            self.add(list(states[i,:]), actions[i], errors[i])


def gen_rates_bd(method, num_of_examples):
    rates0,rates1 = np.zeros((num_of_examples,2)), np.zeros((num_of_examples,2))
    if method == 'gaussian':
        rates0[:,0] = np.random.normal(4/10, 0.1,num_of_examples)
        rates0[:,1] = np.random.normal(12/10,0.1,num_of_examples)
        idcs = rates0[:,0] > rates0[:,1] 
        rates0[idcs, 1] = rates0[idcs,0] + 1

        rates1[:,0] = np.random.normal(100/10,1, num_of_examples)
        rates1[:,1] = np.random.normal(122/10,1, num_of_examples)
        idcs = rates1[:,0] > rates1[:,1] 
        rates1[idcs, 1] = rates1[idcs,0] + 1

    elif method == 'uniform':
        rates0[:,0] = np.random.uniform(0, 32,num_of_examples)
        rates0[:,1] = np.random.uniform(0,32,num_of_examples)
        rates0[:, 1] += rates0[:,0] # This should provide ample enough space to cover 64 sampling rates 

        rates1[:,0] = np.random.uniform(0,32, num_of_examples)
        rates1[:,1] = np.random.uniform(0,32, num_of_examples)
        rates1[:, 1] += rates1[:,0] # This should provide ample enough space to cover 64 sampling rates 
    else:
        raise Exception('Invalid Sampling Method for Rates')

    rates0 = rates0.clip(0.1,None)
    rates1 = rates1.clip(0.1,None)
    return [rates0,rates1]