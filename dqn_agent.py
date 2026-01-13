import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque


# Replay Buffer Class
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, next_state, reward, action):
        self.buffer.append((state, next_state, reward, action))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, next_states, rewards, actions = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
        )

    def __len__(self):
        return len(self.buffer)


# Neural Network with Dueling Architecture and Attention
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(64)
        )

        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 64)
        )

        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        features = self.feature(x)
        attn_weights = torch.sigmoid(self.attention(features))
        features = features * attn_weights

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals


# Dueling + Double DQN Agent
class DQN_agent:
    def __init__(self, device):
        self.device = device
        self.model = NeuralNetwork().to(device)
        self.target_model = NeuralNetwork().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.buffer = ReplayBuffer(capacity=10000)

        self.lr = 1e-3
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.target_update_freq = 100
        self.step_count = 0

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def act(self, state, train=True):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        if train and np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])
        else:
            with torch.no_grad():
                q_vals = self.model(state)
                return q_vals.argmax().item()

    def train(self):
        if len(self.buffer) < self.batch_size:
            return 0

        # Sample batch
        state_batch, next_state_batch, reward_batch, action_batch = self.buffer.sample(self.batch_size)

        state_batch = state_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        action_batch = action_batch.to(self.device).unsqueeze(1)

        # Current Q values
        current_q = self.model(state_batch).gather(1, action_batch)

        with torch.no_grad():
            # Double DQN: next actions from model, Q from target
            next_actions = self.model(next_state_batch).argmax(1, keepdim=True)
            next_q = self.target_model(next_state_batch).gather(1, next_actions)
            target_q = reward_batch.unsqueeze(1) + self.gamma * next_q

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())
