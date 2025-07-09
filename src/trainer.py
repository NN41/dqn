
import numpy as np
import gymnasium as gym
import torch
import os
import random
import time
from torchvision.transforms import v2

from src.utils import ReplayBuffer, FrameBuffer
from src.config import Config
from src.agent import Agent
from src.environment import EnvironmentManager
from src.evaluator import Evaluator
from src.checkpoint_manager import CheckpointManager
from src.logger import Logger

class Trainer:
    """Simple DQN training orchestrator."""

    def __init__(self, config: Config) -> None:
        """Set up all components of the training loop."""
        self.config = config

        self._set_seed(config.seed)

        self.env_mgr = EnvironmentManager("ALE/Pong-v5", config.seed)
        self.transforms = v2.Compose([
            v2.Grayscale(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(84, 84)),
        ])

        self.replay_memory = ReplayBuffer(config.capacity)
        self.frame_history = FrameBuffer(config.num_frames_in_stack, config.device, self.transforms)
        self.agent = Agent(config, self.env_mgr.num_actions, config.device)
        self.evaluator = Evaluator(config.num_test_frames, self.transforms, config.device)
        self.checkpoint_manager = CheckpointManager(config.save_dir)
        self.logger = Logger()

        self.evaluator.build_test_set(gym.make("ALE/Pong-v5"), config.seed)

        self.t0_seconds = time.time()
        self.t0_update = time.time()
        self.ep_length = 0
        self.ep_reward = 0
        self.ep_lengths: list[int] = []
        self.ep_rewards: list[float] = []
        self.ep_durations: list[float] = []

        self.total_reward_between_updates = 0.0
        self.total_loss_between_updates = 0.0
        self.samples_between_updates = 0
        self.state_representation = None

        ckpt = self.checkpoint_manager.load()
        if ckpt is not None:
            self.start_step = ckpt["step"]
            self.agent.eps_curr = ckpt["eps_curr"]
            self.agent.online_net.load_state_dict(ckpt["model_state"])
            self.agent.target_net.load_state_dict(ckpt["target_model_state"])
            self.agent.optimizer.load_state_dict(ckpt["optim_state"])
            print(f"Resuming from step {self.start_step}, ε={self.agent.eps_curr:.3f}")
        else:
            self.start_step = 0
            self.agent.eps_curr = self.agent.eps_start
            print("No checkpoint found; starting from scratch")

    @staticmethod
    def _set_seed(seed: int) -> None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def train(self) -> None:
        """Run the training loop."""
        cfg = self.config
        for step in range(self.start_step, cfg.num_samples):

            if (step + 1) % cfg.target_network_sync_interval == 0:
                self.agent.sync_target()

            enough_frames_in_history = len(self.frame_history) == cfg.num_frames_in_stack
            if enough_frames_in_history:
                self.state_representation = self.frame_history.preprocess()

            action = self.agent.act(self.state_representation, enough_frames_in_history)

            obs, reward, terminated, truncated, _ = self.env_mgr.env.step(action)

            done = terminated or truncated
            self.frame_history.append(obs)
            if enough_frames_in_history:
                state_next = self.frame_history.preprocess()
                transition = {
                    "state_representation": self.state_representation.detach().to("cpu"),
                    "action": action,
                    "reward": reward,
                    "state_representation_next": state_next.detach().to("cpu"),
                    "is_terminal_state": done,
                }
                self.replay_memory.append(transition)

            if len(self.replay_memory) >= cfg.minibatch_size:
                batch = self.replay_memory.sample(cfg.minibatch_size)
                loss_value = self.agent.learn(batch)
                self.total_loss_between_updates += loss_value * cfg.minibatch_size
                self.samples_between_updates += cfg.minibatch_size

            self.ep_reward += reward
            self.total_reward_between_updates += reward
            self.ep_length += 1
            self.agent.eps_curr = max(
                cfg.eps_end,
                self.agent.eps_curr - (cfg.eps_start - cfg.eps_end) / cfg.anneal_length,
            )

            if (step + 1) % cfg.update_interval == 0:
                avg_loss = self.total_loss_between_updates / max(1, self.samples_between_updates)
                avg_max_q = self.evaluator.compute_avg_max_q(self.agent)
                update_duration = time.time() - self.t0_update
                print(f"Step {step+1} / {cfg.num_samples} | Avg loss since last update: {avg_loss:.8f}")
                print(f"\tAvg max_q: {avg_max_q:.8f}")
                self.logger.log_update(
                    avg_loss,
                    avg_max_q,
                    self.total_reward_between_updates,
                    update_duration / 60,
                    step,
                )
                self.total_loss_between_updates = 0.0
                self.samples_between_updates = 0
                self.total_reward_between_updates = 0.0
                self.t0_update = time.time()

            if (step + 1) % cfg.checkpoint_interval == 0:
                print("\tSaving Model...")
                self.checkpoint_manager.save(step + 1, self.agent, self.agent.optimizer)
                print(f"Saved checkpoint at step {step+1}")

            if done:
                t1_seconds = time.time()
                ep_duration = t1_seconds - self.t0_seconds
                self.ep_lengths.append(self.ep_length)
                self.ep_rewards.append(self.ep_reward)
                self.ep_durations.append(ep_duration)
                episode = len(self.ep_lengths)
                print(
                    f"\tEpisode {episode} finished! | Length {self.ep_length} | Reward {self.ep_reward} | Duration {ep_duration:.0f} seconds"
                )
                self.logger.log_episode(
                    ep_duration,
                    self.ep_length,
                    self.ep_reward,
                    episode,
                )
                self.ep_length = 0
                self.ep_reward = 0
                self.t0_seconds = time.time()
                self.frame_history.clear()
                obs, _ = self.env_mgr.env.reset()

        self.logger.close()
        self.env_mgr.env.close()

