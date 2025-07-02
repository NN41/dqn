import torch
from dataclasses import dataclass, field

@dataclass
class Config:
    """
    Configuration class containing hyperparameters and settings for the agent and training process.
    An instance of this class acts as the single source of truth for a given run and can be
    serialized for logging and reproducibility purposes of experiments.
    """
    # Training loop
    num_samples_per_epoch: int = 10_000 
    num_epochs: int = 64
    seed: int = 42
    
    @property
    def num_samples(self) -> int:
        return self.num_samples_per_epoch * self.num_epochs

    # Network and training and testing
    target_network_sync_interval: int = 500
    use_target_network: bool = True
    gamma: float = 0.99
    num_test_frames: int = 2_000
    learning_rate: float = 1e-4

    # Epsilon annealing
    eps_start: float = 1.0
    eps_end: float = 0.1

    @property
    def anneal_length(self) -> float:
        return self.num_samples * 0.5

    # Replay memory and preprocessing
    capacity: int = 25_000
    num_frames_in_stack: int = 4
    minibatch_size: int = 32

    # Saving, info and logging
    update_interval: int = 5_000 # number of environment steps between updates about training process
    checkpoint_interval: int = 10_000 # number of environment steps between creating a checkpoint of the trained network
    save_dir: str  = "models/checkpoints"

    # Other
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
 