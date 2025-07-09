"""Utility script to render a trained DQN agent playing Pong."""

import os

import ale_py
import gymnasium as gym
import torch
from torchvision.transforms import v2

from src.networks import QNetwork
from src.utils import FrameBuffer


gym.register_envs(ale_py)


# Parameters controlling visualization
MODEL_NAME = "trained_base_and_target_networks.pth"
STEPS = 1000


if __name__ == "__main__":
    model_path = os.path.join("models", "trained", MODEL_NAME)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    env = gym.make("ALE/Pong-v5", render_mode="human")
    state_dict = torch.load(model_path, map_location="cpu")

    qnet = QNetwork(env.action_space.n)
    if "model_state" in state_dict:
        qnet.load_state_dict(state_dict["model_state"])
    else:
        qnet.load_state_dict(state_dict)
    qnet.eval()

    transforms = v2.Compose(
        [v2.Grayscale(), v2.ToDtype(torch.float32, scale=True), v2.Resize((84, 84))]
    )
    frame_buffer = FrameBuffer(4, "cpu", transforms)

    obs, _ = env.reset()
    frame_buffer.append(obs)
    for _ in range(STEPS):
        if len(frame_buffer) == 4:
            state = frame_buffer.preprocess().unsqueeze(0)
            with torch.no_grad():
                action = int(torch.argmax(qnet(state)))
        else:
            action = env.action_space.sample()

        obs, _, terminated, truncated, _ = env.step(action)
        env.render()
        frame_buffer.append(obs)
        if terminated or truncated:
            frame_buffer.clear()
            obs, _ = env.reset()
            frame_buffer.append(obs)

    env.close()


