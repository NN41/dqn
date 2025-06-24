import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from collections import deque

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque()
        self.capacity = capacity # paper uses 1e6
        self.memory_size = 0

    def add_experience(self, sample: list):
        self.memory.append(sample)
        self.memory_size += 1
        if self.memory_size > self.capacity:
            self.memory.popleft()

def compute_discounted_reverse_cumsums(vals, disc_factor):
    """Helper function for computing cumulative sums in reverse order with a discount factor."""
    disc_sums = []
    disc_sum = 0
    for val in vals[::-1]:
        disc_sum = val + disc_factor * disc_sum
        disc_sums.append(disc_sum)
    return disc_sums[::-1]

def compute_discounted_future_returns(ep_rewards, disc_factor):
    """Computes the discounted future returns (discounted rewards-to-go) of an episode."""
    return compute_discounted_reverse_cumsums(ep_rewards, disc_factor)

# def compute_td_errors(ep_observations, ep_rewards, disc_factor, value_network):
#     """Computes the TD(1) (one-step Time Difference) errors of an episode."""
#     ep_rews = np.array(ep_rewards) 
#     ep_obs = torch.tensor(ep_observations, device=device)
#     value_network.eval()
#     with torch.no_grad():
#         ep_vals = value_network(ep_obs).squeeze().cpu().numpy()
#     ep_vals[-1] = 0 # value of being in the last (terminated) state is 0.
#     ep_vals_next = np.append(ep_vals[1:], 0) # similar reasoning as for the line above.
#     ep_td_errors = ep_rews + disc_factor * ep_vals_next - ep_vals # = Rt + gamma * V(St+1) - V(St)
#     return ep_td_errors.tolist()

# def compute_gaes(ep_observations: list, ep_rewards: list, gamma_gae: float, lambda_gae: float, value_network: nn.Module, set_to_zero: bool = False) -> list[float]:
#     """
#     Computes the GAEs (Generalized Advantage Estimators) of an episode. The GAE is defined as the
#     discounted (in)finit-horizon sum of TD errors, discounted using lambda * gamma. Note that
#     when lambda_gae = 0, evaluates to TD errors. When lambda_gae = 1, evaluates to discounted future returns minus a value function baseline.
#     Both of this is confirmed numerically (of course, only when set_to_zero = False).

#     Args:
#         ep_observations (list): The episode's observations/states.
#         ep_rewards (list): The episodes step-by-step rewards.
#         gamma_gae (float): The discount factor for future returns.
#         lambda_gae (float): Scalar value to manage bias-variance tradeoff.
#         value_network (nn.Module): Network representing the value function.
#         set_to_zero (bool): If set to True, overrides the penultimate TD error to an intuitively sensible value

#     Returns:
#         list[float]: A list containing the GAEs for each time step t of the episode. 
#     """
#     ep_td_errors = compute_td_errors(ep_observations, ep_rewards, gamma_gae, value_network)
#     if set_to_zero:
#         ep_td_errors[-2] = 0 # We might need this for stability
#     gae_discount_factor = gamma_gae * lambda_gae 
#     ep_gaes = compute_discounted_reverse_cumsums(ep_td_errors, gae_discount_factor)
#     return ep_gaes
