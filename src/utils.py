import torch
from torchvision.transforms import v2
import numpy as np
from collections import deque
from typing import Any, Dict, Sequence

class ReplayBuffer(deque[Dict[str, Any]]):
    """
    Fixed-capacity replay buffer to store transitions for experience replay mechanism.
    This buffer holds arbitrary transitions dictionaries and allows random sampling
    of minibatches.

    Args:
        capacity (int): Maximum number of transitions to store.
    """
    def __init__(self, capacity: int) -> None:
        super().__init__(maxlen=capacity)

    def sample(self, batch_size: int) -> np.array:
        """Randomly sample a batch of transitions of a given size without replacement.
        
        Args:
            batch_size (int): Number of transitions to sample.
            
        Returns:
            np.array: A numpy array containing a randomly sampled batch of transitions.
        """
        return np.random.choice(self, size=batch_size, replace=False)

class FrameBuffer(deque):
    """A fixed-capacity buffer to store the most recent frames.
    
    Provides a method to preprocess frames and stack them into a tensor
    representing the state of the system, to be passed as input into the Q-network.
    
    Args:
        capacity (int): Number of frames to hold (e.g. 4 for ALE/Pong-v5)
        device (str): Device to perform tensor operations on ('cpu' or 'cuda').
        transforms (v2.Compose): Torchvision frame preprocessing pipeline to apply per frame.
    """
    def __init__(self, capacity: int, device: str, transforms: v2.Compose) -> None:
        super().__init__(maxlen=capacity)
        self.device = device
        self.transforms = transforms

    def preprocess(self) -> torch.Tensor:
        """Preprocess individual frames stored in buffer and stack them.
        
        Returns:
            torch.Tensor: Preprocessed frame stack representing the state. Of shape (C, H, W),
                where C is the buffer capacity, that is, the number of frames to stack.
        """
        img_t_list = []
        for frame in self:
            img = torch.permute(torch.tensor(frame), (2,0,1))
            img_t = self.transforms(img)
            img_t_list.append(img_t)
        img_stack = torch.cat(img_t_list).to(self.device)
        return img_stack

    def clear(self) -> None:
        """Remove all stored frames from buffer."""
        super().clear()

