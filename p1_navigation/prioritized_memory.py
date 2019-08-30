import random
import numpy as np
import torch
from collections import namedtuple
from SumTree import SumTree


"""
    Prioritized experience reply buffer
    Get from https://github.com/austinsilveria/Banana-Collection-DQN/blob/master/Banana_DoubleDQN_PER.py which was adjusted from original source: https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PreReplayBuffer:
    """Fixed-size buffer to store experience objects."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): reliance of sampling on prioritization
        """
        self.action_size = action_size
        self.memory = SumTree(buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.alpha = alpha
        self.max_priority = 0
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, priority=10):
        """Add a new experience to memory."""
        # Assign priority of new experiences to max priority to insure they are played at least once
        if len(self.memory) > self.batch_size + 5:
            e = self.experience(state, action, reward, next_state, done, self.max_priority)
        else:
            e = self.experience(state, action, reward, next_state, done, int(priority) ** self.alpha)
        self.memory.add(e)

    def update_priority(self, new_priorities, indices):
        """Updates priority of experience after learning."""
        for new_priority, index in zip(new_priorities, indices):
            old_e = self.memory[index]
            new_p = new_priority.item() ** self.alpha
            new_e = self.experience(old_e.state, old_e.action, old_e.reward, old_e.next_state, old_e.done, new_p)
            self.memory.update(index, new_e)
            if new_p > self.max_priority:
                self.max_priority = new_p

    def sample(self):
        """Sample a batch of experiences from memory based on TD Error priority.
           Return indices of sampled experiences in order to update their
           priorities after learning from them.
        """
        experiences = []
        indices = []
        sub_array_size = self.memory.get_sum() / self.batch_size
        for i in range(self.batch_size):
            choice = np.random.uniform(sub_array_size * i, sub_array_size * (i + 1))
            e, index = self.memory.retrieve(1, choice)
            experiences.append(e)
            indices.append(index)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        probabilities = torch.from_numpy(
            np.vstack([e.priority / self.memory.get_sum() for e in experiences])).float().to(device)
        indices = torch.from_numpy(np.vstack([i for i in indices])).int().to(device)

        return states, actions, rewards, next_states, dones, probabilities, indices

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

