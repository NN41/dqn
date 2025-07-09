# import ctypes

# # Flags to tell Windows “we’re in a long-running, system-required activity”
# ES_CONTINUOUS       = 0x80000000
# ES_SYSTEM_REQUIRED  = 0x00000001

# # Prevent sleep
# ctypes.windll.kernel32.SetThreadExecutionState(
#     ES_CONTINUOUS | ES_SYSTEM_REQUIRED
# )

# import traceback, sys
# log_file = open("train.log", "a", buffering=1)

# import faulthandler, sys
# faulthandler.enable(file=sys.stderr, all_threads=True)

import gymnasium as gym
import ale_py
import numpy as np
import os
import random
import time

import torch
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter

from src.utils import ReplayBuffer, FrameBuffer
from src.networks import QNetwork
from src.config import Config
from src.environment import EnvironmentManager
from src.agent import Agent
from src.checkpoint_manager import CheckpointManager

# ale_py.register_v5_envs()
gym.register_envs(ale_py)

config = Config(
    num_test_frames=5,
    num_samples_per_epoch=100,
    num_epochs=10
)

device = config.device
print(f"Using {device} device")

def set_seed(seed: int):
    """Sets the random seed for reproducility of experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

set_seed(config.seed)

env_mgr = EnvironmentManager("ALE/Pong-v5", config.seed)
obs, info = env_mgr._obs, env_mgr._info

transforms = v2.Compose([
            v2.Grayscale(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(84,84)),
        ])

def generate_frame_test_set(num_test_frames: int, transforms: v2.Compose, env: gym.Env):

    env = gym.make("ALE/Pong-v5")
    obs, info = env.reset(seed=config.seed)
    env.action_space.seed(config.seed)

    frame_test_set = []
    for _ in range(num_test_frames):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frame = transforms(torch.tensor(obs).permute((2,0,1))) # 
        frame_test_set.append(frame)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

    return frame_test_set

def compute_avg_max_q(frame_test_set, q_network):

    q_network.eval()

    max_qs = []
    for idx in range(num_frames_in_stack, len(frame_test_set)):
        frames_to_stack = frame_test_set[idx-num_frames_in_stack:idx]
        frame_stack = torch.cat(frames_to_stack).to(device)

        with torch.no_grad():
            max_q = torch.max(q_network(frame_stack)).item()
        max_qs.append(max_q)

    return float(np.mean(max_qs))

save_dir = config.save_dir
checkpoint_manager = CheckpointManager(save_dir)

capacity = config.capacity
num_samples_per_epoch = config.num_samples_per_epoch
num_epochs = config.num_epochs
num_samples = config.num_samples
update_interval = config.update_interval
checkpoint_interval = config.checkpoint_interval
target_network_sync_interval = config.target_network_sync_interval
use_target_network = config.use_target_network

eps_start = config.eps_start
eps_end = config.eps_end
anneal_length = config.anneal_length


gamma = config.gamma
num_frames_in_stack = config.num_frames_in_stack
minibatch_size = config.minibatch_size
num_test_frames = config.num_test_frames

replay_memory = ReplayBuffer(capacity) 
frame_history = FrameBuffer(num_frames_in_stack, device, transforms)
agent = Agent(config, env_mgr.num_actions, device)
# q_network = QNetwork(env_mgr.num_actions).to(device)
# q_network_target = QNetwork(env_mgr.num_actions).to(device) if use_target_network else q_network
# loss = nn.MSELoss()
# optimizer = torch.optim.Adam(q_network.parameters(), lr=config.learning_rate)
writer = SummaryWriter()

ep_lengths = []
ep_rewards = []
ep_durations = []

print(f"Generating frame test set...")
frame_test_set = generate_frame_test_set(num_test_frames, transforms, env_mgr.env)
# avg_max_qs = []

# check_list = []
# check_updates = []

ep_length = 0
ep_reward = 0
t0_seconds = time.time()
total_reward_between_updates = 0
total_loss_between_updates = 0
min_loss_between_updates = +1e9
max_loss_between_updates = -1e9
samples_between_updates = 0
t0_update = time.time()
# all_batch_losses = []
state_representation = None

ckpt = checkpoint_manager.load()
if ckpt is not None:
    start_step = ckpt["step"]
    agent.eps_curr = ckpt["eps_curr"]
    agent.online_net.load_state_dict(ckpt["model_state"])
    agent.target_net.load_state_dict(ckpt["target_model_state"])
    agent.optimizer.load_state_dict(ckpt["optim_state"])
    print(f"Resuming from step {start_step}, ε={agent.eps_curr:.3f}")
else:
    start_step = 0
    agent.eps_curr = agent.eps_start
    print("No checkpoint found; starting from scratch")

# try:
for step in range(start_step, num_samples):

    if (step + 1) % target_network_sync_interval == 0:
        agent.sync_target()

    # check if we have enough frames, and if so, preprocess
    enough_frames_in_history = len(frame_history) == num_frames_in_stack
    if enough_frames_in_history:
        state_representation = frame_history.preprocess()

    # # select epsilon-greedy action
    # choose_greedy_action = np.random.rand() >= eps_curr
    # if enough_frames_in_history and choose_greedy_action:
    #     with torch.no_grad():
    #         q_network.eval()
    #         q_values = q_network(state_representation)
    #     action = torch.argmax(q_values).item()
    # else:
    #     action = int(env_mgr.env.action_space.sample())
    # # check_list.append((enough_frames_in_history, choose_greedy_action, action))
    action = agent.act(state_representation, enough_frames_in_history)

    # act in environment
    obs, reward, terminated, truncated, info = env_mgr.env.step(action)
    
    # process results of action
    done = terminated or truncated
    frame_history.append(obs) # this gives s_t+1
    if enough_frames_in_history:
        state_representation_next = frame_history.preprocess()
        transition = {
            "state_representation": state_representation.detach().to("cpu"), # storing on the cpu to avoid OOM issues on the gpu
            "action": action,
            "reward": reward,
            "state_representation_next": state_representation_next.detach().to("cpu"), # storing on the cpu to avoid OOM issues on the gpu
            "is_terminal_state": done
        }
        replay_memory.append(transition)

    if len(replay_memory) >= config.minibatch_size:

        batch = replay_memory.sample(config.minibatch_size)
        loss_value = agent.learn(batch)
        total_loss_between_updates += loss_value * config.minibatch_size
        samples_between_updates += config.minibatch_size

        # batch_phi = []
        # batch_phi_next = []
        # batch_action_indices = []
        # batch_reward = []
        # batch_done = []

        # for sample in batch:
        #     batch_phi.append(sample["state_representation"])
        #     batch_phi_next.append(sample["state_representation_next"])
        #     batch_action_indices.append(sample["action"])
        #     batch_reward.append(sample["reward"])
        #     batch_done.append(sample["is_terminal_state"])

        # batch_phi = torch.stack(batch_phi).to(device)
        # batch_phi_next = torch.stack(batch_phi_next).to(device)
        # batch_action_indices = torch.tensor(batch_action_indices).to(device)
        # batch_reward = torch.tensor(batch_reward).to(device)
        # batch_done = torch.tensor(batch_done).to(device)

        # q_network_target.eval()
        # with torch.no_grad():
        #     q_values_next = q_network_target(batch_phi_next)
        # max_q_values_next = torch.max(q_values_next, dim=-1).values

        # batch_targets = batch_reward + gamma * max_q_values_next * (~batch_done)

        # q_network.train()
        # rows = torch.arange(minibatch_size).to(device)
        # cols = batch_action_indices
        # batch_q_values = q_network(batch_phi)[rows, cols]

        # batch_loss = loss(batch_q_values, batch_targets)

        # optimizer.zero_grad()
        # batch_loss.backward()
        # optimizer.step()
        
        # total_loss_between_updates += batch_loss.item() * minibatch_size
        # samples_between_updates += minibatch_size
    
    ep_reward += reward
    total_reward_between_updates += reward
    ep_length += 1
    agent.eps_curr = max(
        eps_end,
        agent.eps_curr - (eps_start - eps_end) / anneal_length,
    )

    if (step+1) % update_interval == 0:
        avg_loss_between_updates = total_loss_between_updates / samples_between_updates
        avg_max_q = compute_avg_max_q(frame_test_set, agent.online_net)
        update_duration = time.time() - t0_update

        print(f"Step {step+1} / {num_samples} | Avg loss since last update: {avg_loss_between_updates:.8f}")
        print(f"\tAvg max_q: {avg_max_q:.8f}")

        writer.add_scalar("Update_Metrics/Avg_Loss", avg_loss_between_updates, step)
        writer.add_scalar("Update_Metrics/Avg_Max_Q", avg_max_q, step)
        writer.add_scalar("Update_Metrics/Total_Reward", total_reward_between_updates, step)
        writer.add_scalar("Update_Metrics/Duration_Min", update_duration / 60, step)
        
        total_loss_between_updates = 0
        min_loss_between_updates = +1e9
        max_loss_between_updates = -1e9
        samples_between_updates = 0
        total_reward_between_updates = 0
        t0_update = time.time()

    if (step + 1) % checkpoint_interval == 0:
        print(f"\tSaving Model...")
        checkpoint_manager.save(step + 1, agent, agent.optimizer)
        print(f"Saved checkpoint at step {step+1}")

    if done:
        
        t1_seconds = time.time()
        ep_duration = t1_seconds - t0_seconds
        
        ep_lengths.append(ep_length)
        ep_rewards.append(ep_reward)
        ep_durations.append(ep_duration)

        episode = len(ep_lengths)
        print(f"\tEpisode {episode} finished! | Length {ep_length} | Reward {ep_reward} | Duration {ep_duration:.0f} seconds")

        writer.add_scalar("Episode_Metrics/Duration_Sec", ep_duration, episode)
        writer.add_scalar("Episode_Metrics/Length", ep_length, episode)
        writer.add_scalar("Episode_Metrics/Reward", ep_reward, episode)

        ep_length = 0
        ep_reward = 0
        t0_seconds = time.time()

        frame_history.clear()
        obs, info = env_mgr.env.reset()env_mgr.env.close()
