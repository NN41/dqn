
import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym

from src.config import Config
from src.networks import PolicyMLP, ValueMLP

class Agent():
    """
    An agent that learns to solve an environment using a policy gradient algorithm.
    Action space must be discrete.

    Contains the networks to approximate the policy and the value function, and the logic
    for selecting an action and updating the network parameters.
    """
    def __init__(self, env: gym.Env, config: Config):
        """
        Initializes the Agent.

        Args:
            env (gym.Env): The environment in which the agent will act.
            config (Config): The configuration object containing the hyperparameters of the Agent.
        """
        self.num_actions = env.action_space.n
        self.num_features = env.observation_space.shape[0]
        self.config = config

        # Initialize MLPs and their optimizers
        self.policy_network = PolicyMLP(self.num_features, self.config.policy_hidden_size, self.num_actions).to(self.config.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.config.policy_learning_rate)
        self.value_network = ValueMLP(self.num_features, self.config.value_hidden_size, 1).to(self.config.device)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.config.value_learning_rate, weight_decay=self.config.value_weight_decay)

    def select_action(self, observation: np.ndarray, inference_mode: bool = False) -> tuple[int, torch.Tensor]:
        """
        Randomly samples an action from the policy based on the observed state of the environment.

        Args:
            observation (np.ndarray): An observation of the state of the environment.
            inference_mode (bool): If True, uses evaluation mode for the policy network. Used for rendering.

        Returns:
            tuple[int, torch.Tensor]: A tuple containing:
                - int: The randomly chosen action.
                - torch.Tensor: The corresponding log-probability as a Tensor to preserve the computational graph for backprop
        """
        observation_tensor = torch.tensor(observation, dtype=torch.float32).to(self.config.device)
        if not inference_mode:
            self.policy_network.train()
            logits = self.policy_network(observation_tensor)
        else:
            self.policy_network.eval()
            with torch.no_grad():
                logits = self.policy_network(observation_tensor)

        # Get action probabilities from the logits using Softmax
        action_probs = nn.Softmax(dim=-1)(logits)
        
        # Randomly choose an action and compute the corresponding log-probability
        action = int(np.random.choice(self.num_actions, p=action_probs.detach().cpu().numpy()))
        log_prob = torch.log(action_probs[action])
        
        return action, log_prob

    def update_policy_network(self, batch_loss_policy_network: torch.Tensor):
        """Performs a single gradient descent step using backprop."""
        self.policy_optimizer.zero_grad()
        batch_loss_policy_network.backward()
        self.policy_optimizer.step()

    def reset_value_optimizer(self):
        """Resets the optimizer of the value function network."""
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.config.value_learning_rate, weight_decay=self.config.value_weight_decay)
 
 