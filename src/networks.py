import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4) # for input (H, W) = (84, 84), gives output (20, 20)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2) # for input (H, W) = (20, 20), gives output (9, 9)
        self.fc1 = nn.Linear(in_features=32*9*9, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_actions)

    def forward(self, x): # input must be of shape (batch_size, 4, 84, 84) or (4, 84, 84)
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h2_flat = torch.flatten(h2, start_dim=-3)
        h3 = F.relu(self.fc1(h2_flat))
        output = self.fc2(h3) # these are not logits (unnormalized probabilities), they are Q-values for each action
        return output
 
 