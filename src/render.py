import gymnasium as gym
import ale_py
import os
import torch

from src.networks import QNetwork

gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5")

model_dir = "models/trained"
model_name = "trained_base_and_target_networks.pth"
model_path = os.path.join(model_dir, model_name)
state_dict = torch.load(model_path)

dqn = QNetwork(env.action_space.n)
dqn.load_state_dict(state_dict["model_state"])

obs, info = env.reset()
for _ in range(100):
    pass

