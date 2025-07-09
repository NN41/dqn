import torch
import numpy as np
from typing import List, Sequence, Any
import gymnasium as gym
from torch import Tensor

class Evaluator:
    """Builds and scores a held-out frame test set to compute avg_max_q for monitoring.

    Args:
        num_test_frames (int): Number of random frames to collect for evaluation.
        transforms (Any): Torchvision transform pipeline to apply to frames.
        device (str): Device string ('cpu' or 'cuda') for tensor operations.
    """
    def __init__(self, num_test_frames: int, transforms: Any, device: str) -> None:
        self.num_test_frames: int = num_test_frames
        self.transforms = transforms
        self.device: str = device
        self.frame_test_set: List[Tensor] = []

    def build_test_set(self, env: gym.Env, seed: int) -> None:
        """Generate and cache a fixed set of transformed frames from random policy.

        Args:
            env: Gym environment instance.
            seed (int): Seed for reproducibility of frame sampling.
        """
        np.random.seed(seed)
        self.frame_test_set.clear()
        obs, info = env.reset(seed=seed)
        env.action_space.seed(seed)
        for _ in range(self.num_test_frames):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            frame = self.transforms(torch.tensor(obs).permute((2,0,1)))
            self.frame_test_set.append(frame)
            if terminated or truncated:
                obs, info = env.reset()
        env.close()

    def compute_avg_max_q(self, agent: "Agent") -> float:
        """Compute the average max Q-value over the cached test set using the agent's online net.

        Args:
            agent: DQNAgent with attributes `online_net`.

        Returns:
            float: Mean of max Q-values across the test frames.
        """
        agent.online_net.eval()
        max_qs = []
        # Number of frames expected by the convolutional layers
        stack_size = agent.online_net.conv1.in_channels
        for idx in range(stack_size, len(self.frame_test_set)):
            frames_to_stack = self.frame_test_set[idx - stack_size:idx]
            frame_stack = torch.cat(frames_to_stack, dim=0).to(self.device)
            with torch.no_grad():
                max_qs.append(torch.max(agent.online_net(frame_stack)).item())        return float(np.mean(max_qs))
