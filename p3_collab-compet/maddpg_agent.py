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
adjusted from 
https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum 
and 
https://github.com/katnoria/unityml-tennis/
"""

def flatten(tensor):
    return torch.reshape(tensor, (tensor.shape[0], -1,))


class MADDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, n_agents, current_agent, random_seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            n_agents (int): number of agents
            current_agent (int): index of current agent
            random_seed (int): random seed
        """
        self.n_agents = n_agents
        self.current_agent = current_agent
        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(random_seed)
        random.seed(random_seed)

        # Actor Network (w/ Target Network)
        # self.actor_local = Actor(state_size, action_size, seed=random_seed, leak=LEAKINESS, use_bn=False).to(device)
        # self.actor_target = Actor(state_size, action_size, seed=random_seed, leak=LEAKINESS, use_bn=False).to(device)
        self.actor_local = Actor(state_size, action_size, seed=random_seed, use_bn=False).to(device)
        self.actor_target = Actor(state_size, action_size, seed=random_seed, use_bn=False).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        # self.critic_local = Critic(state_size * n_agents, action_size * n_agents,
        #                            seed=random_seed, leak=LEAKINESS, use_bn=False).to(device)
        # self.critic_target = Critic(self.state_size * n_agents, self.action_size * n_agents,
        #                             seed=random_seed, leak=LEAKINESS, use_bn=False).to(device)
        self.critic_local = Critic(state_size * n_agents, action_size * n_agents,
                                   seed=random_seed, use_bn=False).to(device)
        self.critic_target = Critic(self.state_size * n_agents, self.action_size * n_agents,
                                    seed=random_seed, use_bn=False).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        # self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        # self.noise = OUNoise((n_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        # Run inference in eval mode
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        # add noise if true
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)


    def reset(self):
        self.noise.reset()


    def learn(self, agents, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            agents (Agent class): all agents object
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences


        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = torch.zeros((len(states), self.n_agents, self.action_size)).to(device)
        for i, agent in enumerate(agents):
            actions_next[:, i] = agent.actor_target(states[:, i, :])

        # print('\nactions_next:', actions_next.size(), '\nnext_states:', next_states.size())
        # # Flatten state and action
        # # e.g from state (100,2,24) --> (100, 48)
        critic_states = flatten(next_states)
        actions_next = flatten(actions_next)
        # print('after flatten:\nactions_next:', actions_next.size(), '\nnext_states:', critic_states.size())

        # calculate target and expected
        Q_targets_next = self.critic_target(critic_states, actions_next)
        # Q_targets_next = self.critic_target(next_states, actions_next)
        # print('Q_targets_next:', Q_targets_next.size())

        Q_targets = rewards[:, self.current_agent, :] + (gamma * Q_targets_next * (1 - dones[:, self.current_agent, :]))
        # print('Q_targets', Q_targets.size())
        # print('\nactions:', actions.size(), '\nstates:', states.size())

        # print('after flatten:\nactions:', flatten(actions).size(), '\nstates:', flatten(states).size())
        Q_expected = self.critic_local(flatten(states), flatten(actions))
        # Q_expected = self.critic_local(states, actions)

        # # Compute critic loss, use mse loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # critic_loss_value = critic_loss.item()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # Added because I got converge problems
        # then I found in adaptationio's solution (https://github.com/adaptationio/DDPG-Continuous-Control), he added this.
        # Here to see the purpose of doing so: https://discuss.pytorch.org/t/about-torch-nn-utils-clip-grad-norm/13873

        # for param in self.critic_local.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Update the predicted action of current agent
        actions_pred = torch.zeros((len(states), self.n_agents, self.action_size)).to(device)
        actions_pred.data.copy_(actions.data)
        actions_pred[:, self.current_agent] = self.actor_local(states[:, self.current_agent])
        actor_loss = -self.critic_local(flatten(states), flatten(actions_pred)).mean()
        # actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1) # Added because I got converge problems
        # for param in self.critic_local.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        if self.t_step == 0:
            # One time only, start local and target with same parameters
            self._copy_weights(self.critic_local, self.critic_target)
            self._copy_weights(self.actor_local, self.actor_target)
        else:
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

        self.t_step += 1


    def _copy_weights(self, source_network, target_network):
        """Copy source network weights to target"""
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(source_param.data)


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


class MADDPGAgentTrainer():
    """Manages the interaction between the agents and the environment"""

    def __init__(self, state_size, action_size, n_agents, random_seed=0):
        """Initialise the trainer object
        Parameters:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            n_agents (int): number of agents
            current_agent (int): index (id) of current agent
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents

        # initialise all agents
        self.agents = [MADDPGAgent(state_size, action_size, self.n_agents,
                                   current_agent=i, random_seed=random_seed) for i in range(self.n_agents)]
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.t_step = 0


    def act(self, states, add_noise=True):
        """Executes act on all the agents
        Parameters:
            states (list): list of states, one for each agent
            add_noise (bool): whether to apply noise to the actions
        """
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.act(states[i], add_noise)
            actions.append(action)
        return actions


    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        # store a single entry for each step i.e the experience of each agent for a step
        # gets stored as single entry.
        states = np.expand_dims(states, 0)
        actions = np.expand_dims(np.array(actions).reshape(self.n_agents, self.action_size), 0)
        rewards = np.expand_dims(np.array(rewards).reshape(self.n_agents, -1), 0)
        dones = np.expand_dims(np.array(dones).reshape(self.n_agents, -1), 0)
        next_states = np.expand_dims(np.array(next_states).reshape(self.n_agents, -1), 0)

        self.memory.add(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        #     self.memory.add(state, action, reward, next_state, done)

        self.t_step = self.t_step + 1

        # Learn, if enough samples are available in memory
        if (len(self.memory) > BATCH_SIZE) and (self.t_step % UPDATE_EVERY == 0):
            experiences = self.memory.sample()
            for agent in self.agents:
                agent.learn(self.agents, experiences, GAMMA)

    def reset(self):
        """Resets the noise for each agent"""
        for agent in self.agents:
            agent.reset()
