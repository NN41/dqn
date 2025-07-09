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
    num_samples_per_epoch: int = 10_000  # transitions collected in each epoch
    num_epochs: int = 64                 # number of epochs to run
    seed: int = 42                       # random seed for reproducibility
    
    @property
    def num_samples(self) -> int:
        return self.num_samples_per_epoch * self.num_epochs

    # Network, training and testing
    target_network_sync_interval: int = 500  # steps between syncing target net
    use_target_network: bool = True          # if False, only an online network
    gamma: float = 0.99                      # discount factor
    num_test_frames: int = 2_000             # frames to evaluate per epoch
    learning_rate: float = 1e-4              # optimizer learning rate

    # Epsilon annealing
    eps_start: float = 1.0  # initial exploration rate
    eps_end: float = 0.1    # final exploration rate

    @property
    def anneal_length(self) -> float:
        return self.num_samples * 0.5

    # Replay memory and preprocessing
    capacity: int = 25_000          # replay buffer capacity
    num_frames_in_stack: int = 4    # frames stacked per state
    minibatch_size: int = 32        # transitions sampled per update

    # Saving, info and logging
    update_interval: int = 5_000       # env steps between logging updates
    checkpoint_interval: int = 10_000  # env steps between creating checkpoints
    save_dir: str = "models/checkpoints"

    # Other    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")  # computation device
 
