import numpy as np
import random
from collections import namedtuple, deque

from model import DQNnetwork, duelingDQNnetwork
from prioritized_memory import PreReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
ALPHA = 0.6             # reliance of sampling on prioritization
BETA = 0.4              # reliance of importance sampling weight on prioritization


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size=None, action_size=None, dqn_type="double-DQN", prioritized_reply = False, seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            dqn_type (str): "double-DQN" or "DQN" or "dueling-DQN" or "double-dueling-DQN", default is "double-DQN"
            prioritized_reply (bool): True or False for prioritized reply buffer, default is False
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.dqn_type = dqn_type
        self.prioritized_reply = prioritized_reply
        self.seed = random.seed(seed)

        # Q-Network
        if self.dqn_type == "dueling-DQN" or self.dqn_type == "double-dueling-DQN" :
            self.qnetwork_local = duelingDQNnetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = duelingDQNnetwork(state_size, action_size, seed).to(device)
        else:
            self.qnetwork_local = DQNnetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = DQNnetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if self.prioritized_reply:
            self.memory = PreReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, ALPHA)
            # Initialize learning step for updating beta
            self.learn_step = 0
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                if self.prioritized_reply:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA, BETA)
                else:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, beta=None):
        """Update value parameters using given batch of experience tuples.

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

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.dqn_type == "double-DQN" or self.dqn_type == "double-dueling-DQN":
            # Double DQN
            q_best_action = self.qnetwork_local(next_states).max(1)[1]
            Q_targets_next = self.qnetwork_target(next_states).gather(1, q_best_action.unsqueeze(-1))
        else:
            # DQN
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)


        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

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

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)