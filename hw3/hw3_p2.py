import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

env = gym.make('MsPacman-v0').unwrapped
env.seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

!nvidia-smi

mspacman_color = 210 + 164 + 74
def preprocess_observation(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(88, 80, 1)

def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else just set to 1
    return np.reshape(image.astype(np.float).ravel(), [80,80])

"""## ReplayBuffer"""

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

"""## DQN"""

class DQN(nn.Module):

    def __init__(self, num_classes=4):
        super(DQN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

"""## select action"""

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

screen_height, screen_width = 80,80
n_actions = env.action_space.n

n_actions

policy_net = DQN(num_classes=n_actions).to(device)
target_net = DQN(num_classes=n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayBuffer(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []

def optimize_model(maxq):
    if len(memory) < BATCH_SIZE:
        return(maxq)
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    maxq = max(torch.max(state_action_values).data.cpu().detach(), maxq)

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return(maxq)

"""## Main"""

q_hist = []
r_hist = []

num_episodes = 100000

for i_episode in range(num_episodes):
    state = env.reset()
    state = preprocess_observation(state-state)
    state = torch.from_numpy(state).type(torch.FloatTensor).view(1,1,88,80).cuda()
    maxq = -9999
    reward_sum = 0
    for t in count():
        last_state = state
        action = select_action(last_state)
        state, reward, done, _ = env.step(action.item())
        reward_sum += reward
        state = preprocess_observation(state)
        state = torch.from_numpy(state).type(torch.FloatTensor).view(1,1,88,80).cuda()
        reward = torch.tensor([reward], device=device)
        if not done:
            next_state = state - last_state
        else:
            next_state = None
        memory.push(state, action, next_state, reward)
        state = next_state
        maxq = optimize_model(maxq)
        #print(maxq)
        #print(maxq)
        if done:
            episode_durations.append(t + 1)
            print(t,maxq, reward_sum)
            q_hist.append(maxq)
            r_hist.append(reward_sum)
            
            break
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
print('Complete')

episode_durations

plt.plot(q_hist[1:])

plt.plot(q_hist[1:])

def select_action_test(state):
    with torch.no_grad():
        return target_net(state).max(1)[1].view(1, 1)

target_net.eval()
for i_episode in range(num_episodes):
    with torch.no_grad():
        state = env.reset()
        state = preprocess_observation(state-state)
        state = torch.from_numpy(state).type(torch.FloatTensor).view(1,1,88,80).cuda()
        reward_sum = 0
        for t in count():
            last_state = state
            action = select_action_test(last_state)
            state, reward, done, _ = env.step(action.item())
            if (t%10000==0):
                print(t, reward_sum)
            reward_sum += reward
            state = preprocess_observation(state)
            state = torch.from_numpy(state).type(torch.FloatTensor).view(1,1,88,80).cuda()
            if not done:
                next_state = state - last_state
            else:
                next_state = None
            
            state = next_state
            if done:
                episode_durations.append(t + 1)
                print(t,reward_sum)            
                break

