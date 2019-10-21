## Continuous Control

---

In this notebook, you will learn how to use the Unity ML-Agents environment for the second project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.

### 1. Start the Environment

We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).


```python
from unityagents import UnityEnvironment
import numpy as np
```

Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.

- **Mac**: `"path/to/Reacher.app"`
- **Windows** (x86): `"path/to/Reacher_Windows_x86/Reacher.exe"`
- **Windows** (x86_64): `"path/to/Reacher_Windows_x86_64/Reacher.exe"`
- **Linux** (x86): `"path/to/Reacher_Linux/Reacher.x86"`
- **Linux** (x86_64): `"path/to/Reacher_Linux/Reacher.x86_64"`
- **Linux** (x86, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86"`
- **Linux** (x86_64, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86_64"`

For instance, if you are using a Mac, then you downloaded `Reacher.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
```
env = UnityEnvironment(file_name="Reacher.app")
```


```python
env = UnityEnvironment(file_name='Reacher-2.app',  no_graphics=True)
```

    INFO:unityagents:
    'Academy' started successfully!
    Unity Academy name: Academy
            Number of Brains: 1
            Number of External Brains : 1
            Lesson number : 0
            Reset Parameters :
    		goal_speed -> 1.0
    		goal_size -> 5.0
    Unity brain name: ReacherBrain
            Number of Visual Observations (per agent): 0
            Vector Observation space type: continuous
            Vector Observation space size (per agent): 33
            Number of stacked Vector Observation: 1
            Vector Action space type: continuous
            Vector Action space size (per agent): 4
            Vector Action descriptions: , , , 
    

Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.


```python
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
```

### 2. Examine the State and Action Spaces

In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of `33` variables corresponding to position, rotation, velocity, and angular velocities of the arm.  Each action is a vector with four numbers, corresponding to torque applicable to two joints.  Every entry in the action vector must be a number between `-1` and `1`.

Run the code cell below to print some information about the environment.


```python
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
```

    Number of agents: 20
    Size of each action: 4
    There are 20 agents. Each observes a state with length: 33
    The state for the first agent looks like: [ 0.00000000e+00 -4.00000000e+00  0.00000000e+00  1.00000000e+00
     -0.00000000e+00 -0.00000000e+00 -4.37113883e-08  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00 -1.00000000e+01  0.00000000e+00
      1.00000000e+00 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  5.75471878e+00 -1.00000000e+00
      5.55726624e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
     -1.68164849e-01]
    

### 4. It's Your Turn!

Now it's your turn to train your own agent to solve the environment!  When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
```python
env_info = env.reset(train_mode=True)[brain_name]
```

#### Algorithm to Use: DDPG

We have just learned Deep Deterministic Policy Gradients (DDPG), a actor-critic algorithm or a DQN method for continuous action space. The actor approximates the optimal policy deterministically and output the best believed action for any given state. The critic learns the optimal value function by using the actors best believed action. There are two main features of DDPG:

- experience reply buffer which refers to learn from the previous experience (memory), as in DQN. We set a Reply buffer with fixed size (BUFFER_SIZE) and stored the experiences in the buffer. When updating the parameters, we sample a few (batch_size) memory from the buffer. In this way, we can break the sequential nature of experiences and stabilize the learning algorithm.
- soft updates to the target networks
    
    Note: there are 4 neural networks:
        - two regular networks (like the local/evaluate network in DQN whose parameters update all the time): one for the actor and one for the critic
        - two target networks (like the target network in DQN whose parameters update after a certain steps): one for the actor and one for the critic
    The target networks update using a soft updates strategy, e.g, when updating the target network $\theta_{target}$, 
    $$\theta_{target} = \tau \cdot \theta_{local} + (1 - \tau)\cdot \theta_{target} $$
    Here, $\tau \leq 1$ and $\theta_{target}$ can be for either actor target network or critic target network. 

#### Define DDPG run pipeline


```python
# import
from collections import deque
import matplotlib.pyplot as plt
%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchviz import make_dot

from ddpg_agent import Agent
from model import Actor, Critic

import time
import pickle
import pandas as pd
```

`ddpg_agent` contains `Agent` class, e.g., 
```python
Agent(n_agents=num_agents, state_size=state_size, action_size=action_size, random_seed=0)
```

In DDPG `Agent` class, the architect of the neural networks look like below:
- The actor network
    
    It builds an actor (policy) network that maps states -> actions
    - The model has batch normalization and 3 fully connected layers
    - The first layer takes in the state (batch normalized) passes it through 256 nodes with Leaky ReLu activation
    - The second layer take the output from first layer and passes through 128 nodes with Leaky ReLu activation
    - The third layer takes the output from the previous layer and outputs the action size with Tanh activation 
    - Adam optimizer.


```python
model = Actor(state_size, action_size)
model.eval()
x = Variable(torch.randn(1,state_size))
y = model(x)
             
make_dot(y, params=dict(list(model.named_parameters())))
```




![svg](output_14_0.svg)



- the critic network
  
  Build a critic (value) network that maps (state, action) pairs -> Q-values.
    - The model has batch normalization and 4 fully connected layers
    - The first layer takes the state (batch normalized) and passes through 256 nodes with Leaky ReLu activation
    - Then we take the output from the first layer and concatenate it with the action size
    - We then pass this to second layer which forwards through 128 nodes with Leaky ReLu activation
    - The third layer takes the output from the previous layer passes it through 64 nodes with Leaky ReLu activation
    - The fourth layer take the previous layer outputs 1
    - Adam optimizer


```python
model = Critic(state_size, action_size, seed=0)
model.eval()
x = Variable(torch.randn(1,state_size))
z = Variable(torch.randn(1,action_size))
y = model(x, z)
             
make_dot(y, params=dict(list(model.named_parameters())))
```




![svg](output_16_0.svg)




```python
def ddpg_train(env = None,
               agent = None,
               num_agents = 1,
               n_episodes=2000,
               window_size=100, 
               score_threshold=30.0, 
               print_every=50):
    """
    Params
    ======
        env: environment
        agent: agent class
        num_agents (int): number of agents
        n_episodes (int): maximum number of training episodes
        window_size (int): window size used to get last window_size scores
        score_threshold (float): minimum value to reach (required by the project assinment)
        print_every (float): print out every print_every episodes
    """

    scores_deque = deque(maxlen=window_size) # last window_size scores
    scores = []        
    
    n_episode_reach_requirement = None # in how many episode, the problem is solved
    
    for i_episode in range(1, n_episodes+1):
        
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        
        agent.reset()
        episode_scores = np.zeros(num_agents) 

        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones)
            episode_scores += np.array(rewards)
            states = next_states
            if np.any(dones):
                break
                
        episode_score = np.mean(episode_scores)
        scores_deque.append(episode_score)
        scores.append(episode_score)
        average_score = np.mean(scores_deque)

        print('\rEpisode: {}\tAverage Score: {:.2f}\tCurrent Score: {:.2f}'.format(i_episode, average_score, episode_score), end="")
        if i_episode % print_every == 0:
            print('\rEpisode: {}\tAverage Score: {:.2f}\tCurrent Score: {:.2f}'.format(i_episode, average_score, episode_score))

        if average_score >= score_threshold:
            if n_episode_reach_requirement is None:
                n_episode_reach_requirement = i_episode - window_size
                print('\nEnvironment solved in {} episodes!\tAverage Score: {:.2f}'.format(i_episode-window_size, average_score))
            torch.save(agent.actor_local.state_dict(), ''.join('checkpoint_actor.pth'))
            torch.save(agent.critic_local.state_dict(), ''.join('checkpoint_critic.pth'))

    return scores, n_episode_reach_requirement
```

The other hyper-parameters which are not seen in notebook but in .py files are

- BUFFER_SIZE: int(1e6)  # replay buffer size
- BATCH_SIZE: 128        # minibatch size
- GAMMA: 0.99            # discount factor
- TAU: 1e-3              # for soft update of target parameters
- LR_ACTOR = 1e-4        # learning rate of the actor 
- LR_CRITIC = 3e-4       # learning rate of the critic
- UPDATE_EVERY = 20      # how often to update the network
- LEAKINESS = 0.01       # leaky in Leaky ReLu

#### Training


```python
agent = Agent(n_agents=num_agents, state_size=state_size, action_size=action_size, random_seed=42)
scores, n_episode_reach_requirement = ddpg_train(env = env, agent = agent, 
                                                 num_agents = num_agents,
                                                 n_episodes=200,
                                                 window_size=100, 
                                                 score_threshold=30.0, 
                                                 print_every=10)

# save the scores
with open('training_scores.pickle', 'wb') as handle:
    pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

    Episode: 10	Average Score: 0.89	Current Score: 1.12
    Episode: 20	Average Score: 1.25	Current Score: 2.06
    Episode: 30	Average Score: 2.04	Current Score: 4.53
    Episode: 40	Average Score: 3.28	Current Score: 8.86
    Episode: 50	Average Score: 5.16	Current Score: 15.38
    Episode: 60	Average Score: 7.43	Current Score: 22.98
    Episode: 70	Average Score: 10.20	Current Score: 32.39
    Episode: 80	Average Score: 13.35	Current Score: 35.91
    Episode: 90	Average Score: 16.06	Current Score: 37.76
    Episode: 100	Average Score: 18.31	Current Score: 38.96
    Episode: 110	Average Score: 22.11	Current Score: 39.05
    Episode: 120	Average Score: 25.82	Current Score: 38.25
    Episode: 130	Average Score: 29.28	Current Score: 38.12
    Episode: 133	Average Score: 30.25	Current Score: 37.10
    Environment solved in 33 episodes!	Average Score: 30.25
    Episode: 140	Average Score: 32.30	Current Score: 37.79
    Episode: 150	Average Score: 34.74	Current Score: 36.63
    Episode: 160	Average Score: 36.52	Current Score: 37.32
    Episode: 170	Average Score: 37.53	Current Score: 37.45
    Episode: 180	Average Score: 37.74	Current Score: 36.58
    Episode: 190	Average Score: 37.66	Current Score: 36.95
    Episode: 200	Average Score: 37.49	Current Score: 36.72
    

#### Plot


```python
# plot the scores
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Average score')
plt.xlabel('Number of episode')
plt.title("Continuous Control with DDPG")
plt.savefig("traning_plot.pdf")
plt.show()
```


![png](output_22_0.png)


When finished, close the environment.


```python
env.close()
```

### Feature work

When solving the first project: Banana, I have tried to use prioritized experience reply with different DQN based methods. The results showed that with prioritized experience reply, less episodes are needed for solving a problem. I would like also to integrate prioritized experience reply with DDPG and compare it to DDPG without prioritized experience reply. Please see the in progress DDPG with prioritized experience reply solution: [Continuous_Control_Multi_Agent_Solution_PER.ipynb] (https://github.com/beibeiJ/deep-reinforcement-learning/blob/master/p2_continuous-control/Continuous_Control_Multi_Agent_Solution_PER.ipynb). However, the initial try had converge problem. I will keep trying.


During implementation, I have suffered from the high variance of different test runs. Many times I failed to reach the required scores after training the agent 300-400 episodes. Also, it took quite long to finish one test on the CPU. Algorithm like Asynchronous Advantage Actor Critic (A3C) which consists of multiple independent agents (networks) with their own weights, who interact with a different copy of the environment in parallel. Thus, they can explore a bigger part of the state-action space in much less time (https://sergioskar.github.io/Actor_critics/). It could be an option to explore the performance of A3C on this project. 


```python

```
