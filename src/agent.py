import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Dict, Any, Optional, Sequence

from src.networks import QNetwork
from src.config import Config

class Agent:
    """Deep Q-learning agent that encapsulates the online and target networks,
    optimizer, loss function, epsilon-greedy policy, and learning updates.

    Args:
        config (Config): Configuration with hyperparameters.
        num_actions (int): Number of discrete actions in the environment.
        device (str): Device string ('cpu' or 'cuda') for tensor operations.
    """
    def __init__(self, config: Config, num_actions: int, device: str) -> None:
        # Networks
        self.online_net = QNetwork(num_actions).to(device)
        if config.use_target_network:
            self.target_net = QNetwork(num_actions).to(device)
            self.target_net.load_state_dict(self.online_net.state_dict())
        else:
            self.target_net = self.online_net

        # Hyperparameters
        self.gamma: float = config.gamma
        self.eps_start: float = config.eps_start
        self.eps_end: float = config.eps_end
        self.anneal_length: float = config.anneal_length
        self.eps_curr: float = self.eps_start
        self.step_counter: int = 0

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.device = device

    def act(self, state: torch.Tensor, enough_frames_in_history: bool, evaluation: bool = False) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            state (torch.Tensor): Input state tensor of shape (C, H, W).
            evaluation (bool): If True, always choose greedy action (epsilon=0).

        Returns:
            int: Chosen action index.
        """
        self.online_net.eval()
        action = random.randrange(self.online_net.fc2.out_features)
        if enough_frames_in_history:
            if evaluation or (random.random() > self.eps_curr):
                with torch.no_grad():
                    q_values = self.online_net(state.unsqueeze(0).to(self.device))
                action = int(torch.argmax(q_values, dim=-1).item())
        return action

    def learn(self, batch: Sequence[Dict[str, Any]]) -> float:
        """Perform a learning step on a batch of transitions.

        Args:
            batch: Array of transition dicts with keys
                'state_representation', 'action', 'reward',
                'state_representation_next', 'is_terminal_state'.

        Returns:
            float: The computed loss value for this batch.
        """
        # Convert batch dictionaries to tensors for efficient computation
        states = torch.stack([sample["state_representation"] for sample in batch]).to(self.device)
        next_states = torch.stack([sample["state_representation_next"] for sample in batch]).to(self.device)
        actions = torch.tensor([sample["action"] for sample in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([sample["reward"] for sample in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([sample["is_terminal_state"] for sample in batch], dtype=torch.bool, device=self.device)

        # Compute target Q-values
        with torch.no_grad():
            self.target_net.eval()
            q_next = self.target_net(next_states)
            max_q_next = torch.max(q_next, dim=-1).values
            targets = rewards + self.gamma * max_q_next * (~dones)

        # Compute current Q-values
        self.online_net.train()
        q_values = self.online_net(states)
        action_qs = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Loss and backward
        loss = self.loss_fn(action_qs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        self.step_counter += 1
        decay = (self.eps_start - self.eps_end) / self.anneal_length
        self.eps_curr = max(self.eps_end, self.eps_curr - decay)

        return loss.item()

    def sync_target(self) -> None:
        """Copy online network parameters to the target network."""
        if self.online_net is not self.target_net:
            print(f"\tSyncing target network...")
            self.target_net.load_state_dict(self.online_net.state_dict())

