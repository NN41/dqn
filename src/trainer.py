
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from typing import TypeAlias
import os
import random
import time
from torchvision.transforms import v2

from src.utils import ReplayBuffer, FrameBuffer, compute_discounted_future_returns, compute_gaes
from src.config import Config
from src.agent import Agent
from src.environment import EnvironmentManager
from src.evaluator import Evaluator
from src.checkpoint_manager import CheckpointManager
from src.logger import Logger

UpdateInfo = tuple[list[float], list[int]] # Alias used for type hinting of output in some functions

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PGTrainer:
    """
    Manages the training of the Agent class, while recording metrics regarding the training process.
    Can render episodes for visualization.    
    """

    def __init__(self, env: gym.Env, agent: Agent, config: Config):
        """
        Initializes the Trainer using a configuration.

        Args:
            env (gym.Env): The environment the Agent will act in and receive rewards from.
            agent (Agent): The agent that will try to maximize expected returns through RL.
            config (Config): A config dataclass containing all hyperparameters for the training process,
                             including for logging, rendering and experimentation.
        """
        self.agent = agent
        self.config = config
        self.env = env
        self.writer = SummaryWriter(config.log_dir) # Instantiate a writer for visualization in Tensorboard

    def train(self):
        """
        Main training loop of the policy network, performing update steps
        for a certain number of epochs. Manages logging, rendering, the env and the writer.
        """
        for epoch in range(self.config.num_epochs_policy_network):
            print(f"(Policy Net) Epoch {epoch+1} / {self.config.num_epochs_policy_network}:")

            # Train one epoch and log metrics related to training and network parameters
            metrics_to_log = self._train_one_epoch()
            self._log_metrics(metrics_to_log, epoch)

            # Visualize an episode
            if (epoch+1) % self.config.render_every_n_epochs == 0:
                print(f"\tVisualizing episode...")
                self._render_episode()

            avg_batch_return = metrics_to_log['Metrics/Avg_Episode_Return']
            avg_batch_length = metrics_to_log['Metrics/Avg_Episode_Length']
            print(f"\tAvg return: {avg_batch_return:.1f} | Avg length: {avg_batch_length:.1f}")

        # Properly close the writer and the environment
        self.writer.flush()
        self.writer.close()
        self.env.close()    
    
    def _train_one_epoch(self) -> dict:
        """
        Performs a single update step for the policy network
        after collecting a fresh batch of training data.

        Returns:
            dict: A dictionary containing training metrics to be visualized in TensorBoard.
        """
        # simulate multiple episodes to collect batch training data
        batch, metrics_to_log = self._collect_batch()
        
        # compute loss and update network policy
        print(f"\tUpdating policy...")
        batch_loss = self._compute_policy_loss(batch)
        self.agent.update_policy_network(batch_loss)

        if self.config.weight_kind in ['gae', 'td', 'dfrb']:
            print(f"\tTraining value function...")
            train_dataloader, test_dataloader = create_dataloaders_for_value_network(batch['obs'], batch['disc_fut_returns'])
            self.agent.reset_value_optimizer()
            test_loss_info, train_loss_info = train_value_network(self.agent.value_network, nn.MSELoss(), self.agent.value_optimizer, train_dataloader, test_dataloader, self.config.num_epochs_value_network, n_updates=-1)

        return metrics_to_log

    def _collect_batch(self) -> tuple[dict[str, list], dict]:
        """
        Collect a batch of training data by simulating a certain number of episodes.
        Also computes the weights corresponding to the policy's log-probabilities
        based on various methods.

        Returns:
            tuple[dict[str, list], dict]: A tuple containing:
                - dict[str, list]: A dictionary containing the necessary data to estimate the policy gradient over that batch.
                - dict: A dictionary containing training metrics to be visualized in TensorBoard.
        """

        # Set up variables for the GAEs (Generalized Advantage Estimators)
        gamma_gae = self.config.gamma_gae # Standard discounting factor
        lambda_gae = self.config.lambda_gae
        value_network = self.agent.value_network

        # Reset batch data
        batch_returns = [] # Full returns of trajectories
        batch_lengths = [] # Trajectory lengths
        batch_full_returns = [] # Full returns of trajectories, used for the weight calculation
        batch_fut_returns = [] # Future returns (rewards-to-go) at each step in the trajectory
        batch_disc_fut_returns = [] # Discounted future returns (rewards-to-go) at each step in the trajectory
        batch_gaes = [] # GAEs (Generalized Advantage Estimators) at each step in the trajectory
        batch_td_errors = [] # TD(1) Errors (one-step Time Difference Errors) at each step in the trajectory
        batch_dfr_baseline = [] # Discounted Future Returns minus Value Function Baseline at each step in the trajectory
        batch_obs = [] # States/observations
        batch_lprobs = [] # Log-probabilities of policy network 

        print(f"\tSimulating {self.config.num_episodes} episodes...")
        for ep in range(self.config.num_episodes):

            # Simulated episode and collect episode data
            episode_data = self._run_episode()
            ep_rewards = episode_data['rewards']
            ep_obs = episode_data['observations']
            ep_lprobs = episode_data['log_probs']

            # Compute episode statistics
            ep_return = sum(ep_rewards)
            ep_length = len(ep_rewards)
            ep_full_returns = [ep_return] * ep_length
            ep_fut_returns = compute_discounted_future_returns(ep_rewards, 1)
            ep_disc_fut_returns = compute_discounted_future_returns(ep_rewards, gamma_gae)
            ep_gaes = compute_gaes(ep_obs, ep_rewards, gamma_gae, lambda_gae, value_network, set_to_zero=False)
            ep_td_errors = compute_gaes(ep_obs, ep_rewards, gamma_gae, 0, value_network, set_to_zero=False)
            ep_dfr_baseline = compute_gaes(ep_obs, ep_rewards, gamma_gae, 1, value_network, set_to_zero=False)
            # value_network.eval()
            # with torch.no_grad():
            #     ep_vals = value_network(torch.tensor(ep_obs, dtype=torch.float32).to(device)).squeeze().cpu().numpy().tolist()
            #     ep_vals[-1] = 0 # Value of being in the last (terminated) state is 0. This is ensures that GAEs with lambda = 1 indeed match disc future returns minus value baseline

            batch_returns.append(ep_return)
            batch_lengths.append(ep_length)
            batch_full_returns += ep_full_returns
            batch_fut_returns += ep_fut_returns
            batch_disc_fut_returns += ep_disc_fut_returns
            batch_gaes += ep_gaes
            batch_td_errors += ep_td_errors
            batch_dfr_baseline += ep_dfr_baseline
            batch_obs += ep_obs
            batch_lprobs += ep_lprobs

        # Compute the weights corresponding to the log-probs of each observation-action pair
        weight_kind = self.config.weight_kind
        if weight_kind in ['r']:
            batch_weights = batch_full_returns
        elif weight_kind in ['fr']:
            batch_weights = batch_fut_returns
        elif weight_kind in ['dfr']:
            batch_weights = batch_disc_fut_returns
        elif weight_kind in ['gae']:
            batch_weights = batch_gaes
        elif weight_kind in ['td']:
            batch_weights = batch_td_errors
        elif weight_kind in ['dfrb']:
            batch_weights = batch_dfr_baseline

        # Store metrics for TensorBoard
        metrics_to_log = {
            'Distributions/Episode_Returns': np.array(batch_returns),
            'Metrics/Avg_Episode_Return': np.mean(batch_returns),
            'Metrics/Avg_Episode_Length': np.mean(batch_lengths),
            'Weight_Variances/Full_Returns': float(np.var(batch_full_returns)),
            'Weight_Variances/Future_Returns': float(np.var(batch_fut_returns)),
            'Weight_Variances/Disc_Future_Returns': float(np.var(batch_disc_fut_returns)),
            'Weight_Variances/GAEs': float(np.var(batch_gaes)),
            'Weight_Variances/TD_Errors': float(np.var(batch_td_errors)),
            'Weight_Variances/DFR_Baseline': float(np.var(batch_dfr_baseline)),
        }

        # Store batch data for computing the batch loss
        batch_data = {
            'obs': batch_obs,
            'lprobs': batch_lprobs,
            'disc_fut_returns': batch_disc_fut_returns,
            'weights': batch_weights
        }

        return batch_data, metrics_to_log
    
    def _run_episode(self) -> dict[str, list]:
        """
        Simulate a single episode by sampling from a stochastic policy.
        Returns a dict containing episode data necessary for estimating the policy gradient.
        """
        # Set up variables
        episode_data = {
            'rewards': [],
            'observations': [],
            'log_probs': []
        }
        episode_done = False
        observation, info = self.env.reset(seed=self.config.seed)

        while not episode_done:

            # Sample action from agent, act in environment and determine if epsisode is over
            action, log_prob = self.agent.select_action(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)
            episode_done = terminated or truncated

            # Log data
            episode_data['rewards'].append(reward)
            episode_data['observations'].append(observation.tolist())
            episode_data['log_probs'].append(log_prob)

        return episode_data

    def _compute_policy_loss(self, batch: dict[str, list]) -> torch.Tensor:
        """Compute batch loss for updating policy network. Returns a tensor representing the loss."""
        policy_gradient_terms = [lp * w for lp, w in zip(batch['lprobs'], batch['weights'])]
        if self.config.avg_kind == 'a': # take sample mean over all state-action pairs
            loss = -sum(policy_gradient_terms) / len(policy_gradient_terms)
        elif self.config.avg_kind == 't': # take sample mean only wrt the number of trajectories
            loss = -sum(policy_gradient_terms) / self.config.num_episodes
        return loss

    def _log_metrics(self, metrics: dict, epoch: int):
        """Log all metrics from the training process to TensorBoard."""
        # Log metrics from the batch data collection
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                self.writer.add_histogram(key, value, epoch)
            else:
                self.writer.add_scalar(key, value, epoch)
 
        # Log network parameters periodically
        if epoch % self.config.log_params_every_n_epochs == 0:
            print(f"\tLogging network params info...")
            for name, param in self.agent.policy_network.named_parameters():
                self.writer.add_histogram(f'Policy_Param_Values/{name}', param.data, epoch)
                self.writer.add_scalar(f'Policy_Param_Values_Norm/{name}', param.data.norm().item(), epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Policy_Param_Grads/{name}', param.grad, epoch)
                    self.writer.add_scalar(f'Policy_Param_Grads_Norm/{name}', param.grad.norm().item(), epoch)
            for name, param in self.agent.value_network.named_parameters():
                self.writer.add_histogram(f'Value_Param_Values/{name}', param.data, epoch)
                self.writer.add_scalar(f'Value_Param_Values_Norm/{name}', param.data.norm().item(), epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Value_Param_Grads/{name}', param.grad, epoch)
                    self.writer.add_scalar(f'Value_Param_Grads_Norm/{name}', param.grad.norm().item(), epoch)

    def _render_episode(self):
        """Visualize a single episode."""
        render_env = gym.make(self.config.env_name, render_mode='human')
        obs, info = render_env.reset()
        done = False
        while not done:
            act, lprob = self.agent.select_action(obs, inference_mode=True)
            obs, rew, term, trunc, info = render_env.step(act)
            done = term or trunc
        render_env.close()

def create_dataloaders_for_value_network(batch_observations: list[list[float]], batch_future_returns: list[float]) -> tuple[DataLoader, DataLoader]:
    """Function for creating train and test dataloaders for training value function network."""
    X = torch.tensor(batch_observations, dtype=torch.float32)
    y = torch.tensor(batch_future_returns, dtype=torch.float32)
    full_dataset = TensorDataset(X, y)
    train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2]) # 80/20 train/test split
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_dataloader, test_dataloader

def test(model: nn.Module, loss: nn.Module, test_dataloader: DataLoader) -> float:
    """Function for computing loss of the value function network on the test set."""
    loss_total = 0
    n_samples = 0
    model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X).squeeze()
            n_samples += len(y)
            loss_total += loss(y_pred, y).item() * len(y)
    avg_loss = loss_total / n_samples
    return avg_loss

def train(model: nn.Module, loss: nn.Module, optimizer: torch.optim.Optimizer, train_dataloader: DataLoader, n_updates: int = 0) -> tuple[float, UpdateInfo]:
    """
    Trains the value function network for a single epoch by passing over the train set once in batches.
    
    Args:
        model (nn.Module): The network to be trained.
        loss (nn.Module): The objective function to be minimized through gradient descent.
        optimizer (torch.optim.Optimizer): Optimizer to take the gradient descent step.
        train_dataloader (DataLoader): The train data, passed to the optimization algorithm in batches.
        n_updates (int): The number of progress updates about the training process during a single training epoch.
                         N means N updates (evenly spaced out), -1 means update at every batch. Default is 0 updates.

    Returns:
        tuple[float, UpdateInfo]: A tuple containing:
            - float: The average train loss of the network over the train data.
            - UpdateInfo: A tuple with a list of average test losses between updates and a list of the batch indices for each update.
    """

    # Set up variables
    n_batches = len(train_dataloader)
    total_loss_update = 0
    total_loss = 0
    n_samples_update = 0
    n_samples = 0
    avg_loss_updates = []

    # Determine batch indices for evenly-spaced progress updates
    if n_updates == -1:
        n_updates = n_batches
    update_batches = np.linspace(-1, n_batches-1, min(n_batches,n_updates)+1, dtype=int)[1:] # roughly evenly spaced out updates 

    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X).squeeze()
        batch_loss = loss(y_pred, y)

        # backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 3) # uncomment if you need gradient clipping
        optimizer.step()

        # update the total train loss
        n_samples += len(y)
        total_loss += batch_loss.item() * len(y)

        # update the total train loss in between two updates and potentially print information
        n_samples_update += len(y)
        total_loss_update += batch_loss.item() * len(y)
        if batch in update_batches:
            avg_loss_update = total_loss_update / n_samples_update
            avg_loss_updates.append(avg_loss_update)
            print(f"\t{batch+1} / {n_batches} | avg train loss {avg_loss_update:.5f}")
            n_samples_update = 0
            total_loss_update = 0
    
    avg_loss = total_loss / n_samples
    return avg_loss, (avg_loss_updates, update_batches.tolist())

def train_value_network(model: nn.Module, loss: nn.Module, optimizer: torch.optim.Optimizer, train_dataloader: DataLoader, test_dataloader: DataLoader, n_epochs: int, n_updates: int = 0) -> tuple[UpdateInfo, UpdateInfo]:
    """
    Performs a complete training loop for the value function network, training it for a given number of epochs
    on the train data and testing it on the test data, while outputting information about the training process. 

    Args:
        model (nn.Module): The network to be trained.
        loss (nn.Module): The objective function to be minimized through gradient descent.
        optimizer (torch.optim.Optimizer): Optimizer to take the gradient descent step.
        train_dataloader (DataLoader): The train data, passed to the optimization algorithm in batches.
        test_dataloader (DataLoader): The test data, to be used for computing the test error.
        n_epochs (int): The total number of times we pass over the training data to train the network.
        n_updates (int): The number of progress updates about the training process during a complete training run.
                         N means N updates (evenly spaced out across epochs), -1 means update at every epoch. Default is 0 updates.

    Returns:
        tuple[UpdateInfo, UpdateInfo]: A tuple containing:
            - UpdateInfo: A tuple with a list of average test losses between updates and a list of the epoch indices for each update.
            - UpdateInfo: A tuple with a list of average train losses between updates and a list of the epoch indices for each update.
    """

    # Set up variables
    test_losses = []
    train_losses = []

    # Compute test error before training starts.
    if n_updates != 0:
        test_loss = test(model, loss, test_dataloader)
        print(f"\t\t(Value Net) Epoch {0} / {n_epochs} | train loss = NaN    | test loss = {test_loss:.5f}")

    # Determine epoch indices for evenly-spaced progress updates
    if n_updates == -1:
        n_updates = n_epochs
    update_epochs = np.linspace(0, n_epochs-1, min(n_epochs,n_updates), dtype=int)

    # Perform training steps, while sometimes computing test error and outputting information
    for epoch in range(n_epochs):
        train_loss, _ = train(model, loss, optimizer, train_dataloader, n_updates=0)
        train_losses.append(train_loss)
        if epoch in update_epochs:
            test_loss = test(model, loss, test_dataloader)
            test_losses.append(test_loss)
            print(f"\t\t(Value Net) Epoch {epoch+1} / {n_epochs} | train loss = {train_loss:.5f} | test loss = {test_loss:.5f}")

    test_loss_info = (test_losses, update_epochs.tolist())
    train_loss_info = (train_losses, list(range(n_epochs)))
    return test_loss_info, train_loss_info


class Trainer:
    """Simple DQN training orchestrator."""

    def __init__(self, config: Config) -> None:
        """Set up all components of the training loop."""
        self.config = config

        self._set_seed(config.seed)

        self.env_mgr = EnvironmentManager("ALE/Pong-v5", config.seed)
        self.transforms = v2.Compose([
            v2.Grayscale(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(84, 84)),
        ])

        self.replay_memory = ReplayBuffer(config.capacity)
        self.frame_history = FrameBuffer(config.num_frames_in_stack, config.device, self.transforms)
        self.agent = Agent(config, self.env_mgr.num_actions, config.device)
        self.evaluator = Evaluator(config.num_test_frames, self.transforms, config.device)
        self.checkpoint_manager = CheckpointManager(config.save_dir)
        self.logger = Logger()

        self.evaluator.build_test_set(gym.make("ALE/Pong-v5"), config.seed)

        self.t0_seconds = time.time()
        self.t0_update = time.time()
        self.ep_length = 0
        self.ep_reward = 0
        self.ep_lengths: list[int] = []
        self.ep_rewards: list[float] = []
        self.ep_durations: list[float] = []

        self.total_reward_between_updates = 0.0
        self.total_loss_between_updates = 0.0
        self.samples_between_updates = 0
        self.state_representation = None

        ckpt = self.checkpoint_manager.load()
        if ckpt is not None:
            self.start_step = ckpt["step"]
            self.agent.eps_curr = ckpt["eps_curr"]
            self.agent.online_net.load_state_dict(ckpt["model_state"])
            self.agent.target_net.load_state_dict(ckpt["target_model_state"])
            self.agent.optimizer.load_state_dict(ckpt["optim_state"])
            print(f"Resuming from step {self.start_step}, ε={self.agent.eps_curr:.3f}")
        else:
            self.start_step = 0
            self.agent.eps_curr = self.agent.eps_start
            print("No checkpoint found; starting from scratch")

    @staticmethod
    def _set_seed(seed: int) -> None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def train(self) -> None:
        """Run the training loop."""
        cfg = self.config
        for step in range(self.start_step, cfg.num_samples):

            if (step + 1) % cfg.target_network_sync_interval == 0:
                self.agent.sync_target()

            enough_frames_in_history = len(self.frame_history) == cfg.num_frames_in_stack
            if enough_frames_in_history:
                self.state_representation = self.frame_history.preprocess()

            action = self.agent.act(self.state_representation, enough_frames_in_history)

            obs, reward, terminated, truncated, _ = self.env_mgr.env.step(action)

            done = terminated or truncated
            self.frame_history.append(obs)
            if enough_frames_in_history:
                state_next = self.frame_history.preprocess()
                transition = {
                    "state_representation": self.state_representation.detach().to("cpu"),
                    "action": action,
                    "reward": reward,
                    "state_representation_next": state_next.detach().to("cpu"),
                    "is_terminal_state": done,
                }
                self.replay_memory.append(transition)

            if len(self.replay_memory) >= cfg.minibatch_size:
                batch = self.replay_memory.sample(cfg.minibatch_size)
                loss_value = self.agent.learn(batch)
                self.total_loss_between_updates += loss_value * cfg.minibatch_size
                self.samples_between_updates += cfg.minibatch_size

            self.ep_reward += reward
            self.total_reward_between_updates += reward
            self.ep_length += 1
            self.agent.eps_curr = max(
                cfg.eps_end,
                self.agent.eps_curr - (cfg.eps_start - cfg.eps_end) / cfg.anneal_length,
            )

            if (step + 1) % cfg.update_interval == 0:
                avg_loss = self.total_loss_between_updates / max(1, self.samples_between_updates)
                avg_max_q = self.evaluator.compute_avg_max_q(self.agent)
                update_duration = time.time() - self.t0_update
                print(f"Step {step+1} / {cfg.num_samples} | Avg loss since last update: {avg_loss:.8f}")
                print(f"\tAvg max_q: {avg_max_q:.8f}")
                self.logger.log_update(
                    avg_loss,
                    avg_max_q,
                    self.total_reward_between_updates,
                    update_duration / 60,
                    step,
                )
                self.total_loss_between_updates = 0.0
                self.samples_between_updates = 0
                self.total_reward_between_updates = 0.0
                self.t0_update = time.time()

            if (step + 1) % cfg.checkpoint_interval == 0:
                print("\tSaving Model...")
                self.checkpoint_manager.save(step + 1, self.agent, self.agent.optimizer)
                print(f"Saved checkpoint at step {step+1}")

            if done:
                t1_seconds = time.time()
                ep_duration = t1_seconds - self.t0_seconds
                self.ep_lengths.append(self.ep_length)
                self.ep_rewards.append(self.ep_reward)
                self.ep_durations.append(ep_duration)
                episode = len(self.ep_lengths)
                print(
                    f"\tEpisode {episode} finished! | Length {self.ep_length} | Reward {self.ep_reward} | Duration {ep_duration:.0f} seconds"
                )
                self.logger.log_episode(
                    ep_duration,
                    self.ep_length,
                    self.ep_reward,
                    episode,
                )
                self.ep_length = 0
                self.ep_reward = 0
                self.t0_seconds = time.time()
                self.frame_history.clear()
                obs, _ = self.env_mgr.env.reset()

        self.logger.close()
        self.env_mgr.env.close()

