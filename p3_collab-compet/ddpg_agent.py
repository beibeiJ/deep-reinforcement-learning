import numpy as np
import random

from model import Actor, Critic
from oun_noise import OUNoise
from reply_memory import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.95            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
UPDATE_EVERY = 1        # how often to update the network
WEIGHT_DECAY = 0        # L2 weight decay
# LEAKINESS = 0.01        # leaky in Leaky ReLu


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
adjusted from https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
"""

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, n_agents, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            n_agents (int): number of agents
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(random_seed)
        random.seed(random_seed)

        # Actor Network (w/ Target Network)
        # self.actor_local = Actor(state_size, action_size, seed=random_seed, leak=LEAKINESS).to(device)
        # self.actor_target = Actor(state_size, action_size, seed=random_seed, leak=LEAKINESS).to(device)
        self.actor_local = Actor(state_size, action_size, seed=random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed=random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        # self.critic_local = Critic(state_size, action_size, seed=random_seed, leak=LEAKINESS).to(device)
        # self.critic_target = Critic(state_size, action_size, seed=random_seed, leak=LEAKINESS).to(device)
        self.critic_local = Critic(state_size, action_size, seed=random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed=random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        # self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # initialize targets same as original networks
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        # Noise process
        # self.noise = OUNoise((n_agents, action_size), random_seed)
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        # self.memory.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        self.t_step = self.t_step + 1

        if (len(self.memory) > BATCH_SIZE) and (self.t_step % UPDATE_EVERY == 0):
            # experiences = self.memory.sample()
            # self.learn(experiences, GAMMA)
            for _ in range(10):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)


    def reset(self):
        self.noise.reset()


    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences


        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)

        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # Added because I got converge problems
        # then I found in adaptationio's solution (https://github.com/adaptationio/DDPG-Continuous-Control), he added this.
        # Here to see the purpose of doing so: https://discuss.pytorch.org/t/about-torch-nn-utils-clip-grad-norm/13873
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)  # Added because I got converge problems
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, target, source):
        """
        Copy network parameters from source to target
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

