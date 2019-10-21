import numpy as np
import random

from model import Actor, Critic
from reply_memory import ReplayBuffer
from prioritized_memory import PreReplayBuffer
from oun_noise import OUNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024       # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
UPDATE_EVERY = 20       # how often to update the network
ALPHA = 0.6             # reliance of sampling on prioritization
BETA = 0.4              # reliance of importance sampling weight on prioritization
LEAKINESS = 0.01        # leaky in Leaky ReLu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, n_agents, state_size, action_size, random_seed, prioritized_reply = False):
        """Initialize an Agent object.

        Params
        ======
            n_agents (int): number of agents
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            prioritized_reply (bool): True or False for using prioritized reply buffer, default is False
        """
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(random_seed)
        random.seed(random_seed)
        self.prioritized_reply = prioritized_reply

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed=random_seed, leak=LEAKINESS).to(device)
        self.actor_target = Actor(state_size, action_size, seed=random_seed, leak=LEAKINESS).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed=random_seed, leak=LEAKINESS).to(device)
        self.critic_target = Critic(state_size, action_size, seed=random_seed, leak=LEAKINESS).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Noise process
        self.noise = OUNoise((20, action_size), random_seed)

        # Replay memory
        if self.prioritized_reply:
            self.memory = PreReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed, ALPHA)
            # Initialize learning step for updating beta
            self.learn_step = 0
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.t_step = self.t_step + 1
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        # If enough samples are available in memory, get random subset and learn
        if (len(self.memory) > BATCH_SIZE) and (self.t_step % UPDATE_EVERY == 0):
            for _ in range(10):
                if self.prioritized_reply:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA, BETA)
                else:
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

    def learn(self, experiences, gamma, beta=None):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            beta (float): reliance of importance sampling weight on prioritization
        """
        if self.prioritized_reply:
            # Beta will reach 1 after 25,000 training steps (~325 episodes)
            b = min(1.0, beta + self.learn_step * (1.0 - beta) / 25000)
            self.learn_step += 1
            states, actions, rewards, next_states, dones, probabilities, indices = experiences
        else:
            states, actions, rewards, next_states, dones = experiences


        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)

        if self.prioritized_reply:
            # Compute and update new priorities
            new_priorities = (abs(Q_expected - Q_targets) + 0.2).detach()
            self.memory.update_priority(new_priorities, indices)

            # Compute and apply importance sampling weights to TD Errors
            ISweights = (((1 / len(self.memory)) * (1 / probabilities)) ** b)
            max_ISweight = torch.max(ISweights)
            ISweights /= max_ISweight
            Q_targets *= ISweights
            Q_expected *= ISweights

        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)  # Added because I got converge problems
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
