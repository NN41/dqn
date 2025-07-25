import gymnasium as gym
import ale_py
from typing import Tuple

gym.register_envs(ale_py)

class EnvironmentManager():
    """Wraps the environment to handle seeding of the environment.
    
    Args:
        env_name (str): Name of the Gym environment (e.g. "ALE/Pong-v5")
        seed (int): Random seed for reproducibility.
    """
    def __init__(self, env_name: str, seed: int) -> None:
        self.env = gym.make(env_name)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        self._obs, self._info = self.env.reset(seed=seed) # initial reset with seed
        self.num_actions: int = self.env.action_space.n

