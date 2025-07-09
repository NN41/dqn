import os
from typing import Optional, Dict, Any

import torch


class CheckpointManager:
    """Utility class for saving and loading training checkpoints."""

    def __init__(self, save_dir: str) -> None:
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.ckpt_path = os.path.join(self.save_dir, "ckpt.pth")

    def load(self) -> Optional[Dict[str, Any]]:
        """Load a checkpoint if present."""
        if not os.path.isfile(self.ckpt_path):
            return None
        return torch.load(self.ckpt_path, map_location="cpu")

    def save(self, step: int, agent, optimizer: torch.optim.Optimizer) -> None:
        """Save a checkpoint for the current training state."""
        state = {
            "step": step,
            "eps_curr": agent.eps_curr,
            "model_state": agent.online_net.state_dict(),
            "target_model_state": agent.target_net.state_dict(),
            "optim_state": optimizer.state_dict(),
        }        torch.save(state, self.ckpt_path)

