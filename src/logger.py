import os
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Utility for logging training metrics to TensorBoard."""

    def __init__(self, log_dir: str = "runs") -> None:
        """Initialize the writer.

        Args:
            log_dir (str): Directory in which to store event files.
        """
        self.writer = SummaryWriter(log_dir)

    def log_update(self, avg_loss: float, avg_max_q: float, total_reward: float, duration_min: float, step: int) -> None:
        """Log metrics collected at update time.

        Args:
            avg_loss (float): Average loss since last update.
            avg_max_q (float): Average maximum Q-value over test frames.
            total_reward (float): Sum of rewards since last update.
            duration_min (float): Duration of updates in minutes.
            step (int): Environment step count.
        """
        self.writer.add_scalar("Update_Metrics/Avg_Loss", avg_loss, step)
        self.writer.add_scalar("Update_Metrics/Avg_Max_Q", avg_max_q, step)
        self.writer.add_scalar("Update_Metrics/Total_Reward", total_reward, step)
        self.writer.add_scalar("Update_Metrics/Duration_Min", duration_min, step)

    def log_episode(self, duration_sec: float, length: int, reward: float, episode: int) -> None:
        """Log metrics for a finished episode.

        Args:
            duration_sec (float): Episode duration in seconds.
            length (int): Episode length in steps.
            reward (float): Episode reward.
            episode (int): Episode count.
        """
        self.writer.add_scalar("Episode_Metrics/Duration_Sec", duration_sec, episode)
        self.writer.add_scalar("Episode_Metrics/Length", length, episode)
        self.writer.add_scalar("Episode_Metrics/Reward", reward, episode)

    def close(self) -> None:
        """Close the underlying SummaryWriter."""
        self.writer.close()
