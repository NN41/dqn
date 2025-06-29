import torch
from torchvision.transforms import v2
import numpy as np
from collections import deque

class ReplayBuffer(deque):
    def __init__(self, capacity: int):
        super().__init__(maxlen=capacity)

    def sample(self, batch_size: int) -> np.array:
        return np.random.choice(self, size=batch_size, replace=False)

class FrameBuffer(deque):
    def __init__(self, capacity: int, device: torch.device):
        super().__init__(maxlen=capacity)
        self.device = device
        self.transforms = v2.Compose([
            v2.Grayscale(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(84,84)),
        ])

    def preprocess(self) -> torch.Tensor:
        img_t_list = []
        for frame in self:
            img = torch.permute(torch.tensor(frame), (2,0,1))
            img_t = self.transforms(img)
            img_t_list.append(img_t)
        img_stack = torch.cat(img_t_list).to(self.device)
        return img_stack

