import torch
import torch.nn as nn
import torch.nn.functional as F

"""
- adjusted from https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
- some ideas, e.g., use Leaky ReLu and Batch normalization, were from the report:
https://github.com/dalmia/udacity-deep-reinforcement-learning/blob/master/3%20-%20Policy-based%20methods/Project%202%20-%20Continuous%20Control/Report.md
"""

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=0,
                 fc1_units=128, fc2_units=64, leak=0.01, use_bn=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            leak: amount of leakiness in leaky relu
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.leak = leak
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

        if use_bn:
            self.use_bn = use_bn
            self.bn = nn.BatchNorm1d(state_size)
            self.bn1 = nn.BatchNorm1d(fc1_units)
            self.bn2 = nn.BatchNorm1d(fc2_units)
        else:
            self.use_bn = False


    def reset_parameters(self):
        """ Initilaize the weights using He et al (2015) weights """
        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)


    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if self.use_bn:
            state = self.bn(state)
        x = F.leaky_relu(self.fc1(state), negative_slope=self.leak)
        if self.use_bn:
            x = self.bn1(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leak)
        if self.use_bn:
            x = self.bn2(x)
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed=0,
                 fc1_units=256, fc2_units=128, fc3_units=64, leak=0.01, use_bn=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
            leak: amount of leakiness in leaky relu
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.leak = leak
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

        if use_bn:
            self.use_bn = use_bn
            self.bn = nn.BatchNorm1d(state_size)
            self.bn2 = nn.BatchNorm1d(fc2_units)
            self.bn3 = nn.BatchNorm1d(fc3_units)
        else:
            self.use_bn = False


    def reset_parameters(self):
        """ Initilaize the weights using He et al (2015) weights """
        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)


    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if self.use_bn:
            state = self.bn(state)
        x = F.leaky_relu(self.fc1(state), negative_slope=self.leak)
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leak)
        if self.use_bn:
            x = self.bn2(x)
        x = F.leaky_relu(self.fc3(x), negative_slope=self.leak)
        if self.use_bn:
            x = self.bn3(x)
        return self.fc4(x)



