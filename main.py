import ctypes

# Flags to tell Windows “we’re in a long-running, system-required activity”
ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001

# Prevent sleep
ctypes.windll.kernel32.SetThreadExecutionState(
    ES_CONTINUOUS | ES_SYSTEM_REQUIRED
)

import traceback, sys
log_file = open("train.log", "a", buffering=1)

import faulthandler, sys
faulthandler.enable(file=sys.stderr, all_threads=True)

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

from src.utils import FixedSizeMemory
from src.networks import QNetwork

# ale_py.register_v5_envs()
gym.register_envs(ale_py)

device = "cuda" if torch.cuda.is_available() else "cpu"
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

SEED = 42
set_seed(SEED)
env = gym.make('ALE/Pong-v5')
obs, info = env.reset(seed=SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)
num_actions = env.action_space.n

transforms = v2.Compose([
    v2.Grayscale(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(size=(84,84)),
])

def preprocess(frame_history: FixedSizeMemory) -> torch.Tensor:
    img_t_list = []
    for frame in frame_history.memory:
        img = torch.permute(torch.tensor(frame), (2,0,1))
        img_t = transforms(img)
        img_t_list.append(img_t)
    img_stack = torch.cat(img_t_list).to(device)
    return img_stack

def generate_frame_test_set(num_test_frames: int = 1000):

    env = gym.make("ALE/Pong-v5")
    obs, info = env.reset(seed=SEED)
    env.action_space.seed(SEED)

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

def load_latest_ckpt(save_dir):
    ckpts = sorted(
        [f for f in os.listdir(save_dir) if f.startswith("ckpt_")],
        key=lambda fn: int(fn.split("_")[1].split(".")[0]),
    )
    if not ckpts:
        return None
    path = os.path.join(save_dir, ckpts[-1])
    return torch.load(path)

save_dir  = "models/baseline"
ckpt_path = os.path.join(save_dir, "ckpt.pth")
os.makedirs(save_dir, exist_ok=True)

capacity = 25_000
num_samples_per_epoch = 10_000 # takes roughly 15 minutes, or ~14 episodes with random sampling
num_epochs = 4 * 16
num_samples = num_samples_per_epoch * num_epochs
update_interval = 5_000
checkpoint_interval = 10_000
target_network_sync_interval = 500
use_target_network = True

eps_start = 1
eps_end = 0.1
anneal_length = num_samples * 0.5
eps_curr = eps_start

gamma = 0.99
num_frames_in_stack = 4
minibatch_size = 32
num_test_frames = 2_000

replay_memory = FixedSizeMemory(capacity) 
frame_history = FixedSizeMemory(num_frames_in_stack)
q_network = QNetwork(num_actions).to(device)
q_network_target = QNetwork(num_actions).to(device) if use_target_network else q_network
loss = nn.MSELoss()
optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-4)
writer = SummaryWriter()

ep_lengths = []
ep_rewards = []
ep_durations = []

print(f"Generating frame test set...")
frame_test_set = generate_frame_test_set(num_test_frames)
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

if os.path.isfile(ckpt_path):
    ckpt       = torch.load(ckpt_path, map_location=device)
    start_step = ckpt["step"]
    eps_curr   = ckpt["eps_curr"]
    q_network.load_state_dict(ckpt["model_state"])
    q_network_target.load_state_dict(ckpt["target_model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    print(f"[✔] Resuming from step {start_step}, ε={eps_curr:.3f}")
else:
    start_step = 0
    eps_curr   = eps_start
    print("[–] No checkpoint found; starting from scratch")

try:
    for step in range(start_step, num_samples):

        if ((step + 1) % target_network_sync_interval == 0) and use_target_network:
            print(f"\tCopying weights to target network at step {step}")
            weights = q_network.state_dict()
            q_network_target.load_state_dict(weights)

        # check if we have enough frames, and if so, preprocess
        enough_frames_in_history = frame_history.memory_size == num_frames_in_stack
        if enough_frames_in_history:
            state_representation = preprocess(frame_history)

        # select epsilon-greedy action
        choose_greedy_action = np.random.rand() >= eps_curr
        if enough_frames_in_history and choose_greedy_action:
            with torch.no_grad():
                q_network.eval()
                q_values = q_network(state_representation)
            action = torch.argmax(q_values).item()
        else:
            action = int(env.action_space.sample())
        # check_list.append((enough_frames_in_history, choose_greedy_action, action))

        # act in environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # process results of action
        done = terminated or truncated
        frame_history.add(obs) # this gives s_t+1
        if enough_frames_in_history:
            state_representation_next = preprocess(frame_history)
            transition = {
                "state_representation": state_representation.detach().to("cpu"),
                "action": action,
                "reward": reward,
                "state_representation_next": state_representation_next.detach().to("cpu"),
                "is_terminal_state": done
            }
            replay_memory.add(transition)

        if replay_memory.memory_size >= minibatch_size:

            batch = np.random.choice(replay_memory.memory, size=minibatch_size, replace=False)

            batch_phi = []
            batch_phi_next = []
            batch_action_indices = []
            batch_reward = []
            batch_done = []

            for sample in batch:
                batch_phi.append(sample["state_representation"])
                batch_phi_next.append(sample["state_representation_next"])
                batch_action_indices.append(sample["action"])
                batch_reward.append(sample["reward"])
                batch_done.append(sample["is_terminal_state"])

            batch_phi = torch.stack(batch_phi).to(device)
            batch_phi_next = torch.stack(batch_phi_next).to(device)
            batch_action_indices = torch.tensor(batch_action_indices).to(device)
            batch_reward = torch.tensor(batch_reward).to(device)
            batch_done = torch.tensor(batch_done).to(device)

            q_network_target.eval()
            with torch.no_grad():
                q_values_next = q_network_target(batch_phi_next)
            max_q_values_next = torch.max(q_values_next, dim=-1).values

            batch_targets = batch_reward + gamma * max_q_values_next * (~batch_done)

            q_network.train()
            rows = torch.arange(minibatch_size).to(device)
            cols = batch_action_indices
            batch_q_values = q_network(batch_phi)[rows, cols]

            batch_loss = loss(batch_q_values, batch_targets)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # check_updates.append([step, batch_q_values, batch_targets, batch_loss.item()])
            # all_batch_losses.append([step, batch_loss.item()])
            # min_loss_between_updates = min(min_loss_between_updates, batch_loss.item())
            # max_loss_between_updates = max(max_loss_between_updates, batch_loss.item())
            total_loss_between_updates += batch_loss.item() * minibatch_size
            samples_between_updates += minibatch_size
        
        ep_reward += reward
        total_reward_between_updates += reward
        ep_length += 1
        eps_curr = max([eps_end, eps_curr - (eps_start - eps_end) / anneal_length])

        if (step+1) % update_interval == 0:
            avg_loss_between_updates = total_loss_between_updates / samples_between_updates
            avg_max_q = compute_avg_max_q(frame_test_set, q_network)
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
            state = {
                "step":      step + 1,
                "eps_curr":  eps_curr,
                "model_state": q_network.state_dict(),
                "target_model_state": q_network_target.state_dict(),
                "optim_state": optimizer.state_dict(),
            }
            torch.save(state, ckpt_path)
            print(f"[💾] Saved checkpoint at step {step+1}")

        if (step+1) % 1000 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] step {step+1}", file=log_file)

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

            frame_history.empty()
            obs, info = env.reset()
    env.close()
except Exception:
    traceback.print_exc(file=log_file)
    print(f"[!] Crashed at step {step}", file=log_file)
    raise
finally:
    log_file.close()