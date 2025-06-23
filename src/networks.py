import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyMLP(nn.Module):
    """
    A densely-connected network representing the policy.
    Has one hidden layer with ReLU activation and outputs logits.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits
    
class ValueMLP(nn.Module):
    """
    A densely-conncted network representing the value function.
    Has one hidden layer with ReLU activation and outputs a scalar.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        output = self.fc2(h)
        return output
 
 