import os
import torch
from typing import Optional, Dict, Any

class CheckpointManager:
    """Handles saving and loading of training checkpoints.

    Args:
        save_dir (str): Directory to store checkpoint files.
    """
    def __init__(self, save_dir: str) -> None:
        self.save_dir: str = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.ckpt_path: str = os.path.join(self.save_dir, "ckpt.pth")

    def load(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint if it exists.

        Returns:
            Optional[Dict[str, Any]]: Checkpoint dict with keys 'step', 'eps_curr',
            'model_state', 'target_model_state', and 'optim_state'; or None if none found.
        """
        if not os.path.isfile(self.ckpt_path):
            return None
        return torch.load(self.ckpt_path, map_location="cpu")

    def save(self, step: int, agent, optimizer: torch.optim.Optimizer, eps_curr: float) -> None:
        """Save a training checkpoint at the given step.

        Args:
            step (int): Current training step.
            agent: Agent instance with `online_net` and `target_net`.
            optimizer (Optimizer): Optimizer whose state to save.
            eps_curr (float): Current epsilon value for the agent.
        """
        state = {
            "step": step,
            "eps_curr": eps_curr,
            "model_state": agent.online_net.state_dict(),
            "target_model_state": agent.target_net.state_dict(),
            "optim_state": optimizer.state_dict(),
        }
        torch.save(state, self.ckpt_path)