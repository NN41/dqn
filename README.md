# Q-Learning from Scratch: Playing Pong with DQN

This project implements a Deep Q-Network (DQN) from scratch using PyTorch and Gymnasium, with the goal to learn to play [Pong](https://ale.farama.org/environments/pong/), a classic Atari 2600 game implemented in the Arcade Learning Environment ([ALE](https://ale.farama.org/)).

The goal of this project is to gain experience  with Q-learning and explore some more advanced variations to improve training. This implementation directly follows that of the original 2013 paper by Mnih et al, "Playing Atari with Deep Reinforcement Learning".

Even though time and hardware constraints limited the performance the agent could achieve, we managed to achieve significant improvement.

## Overview of Implementations
* **ALE/Pong-v5**: The latest ALE environment implementing Pong with frameskip and non-zero action repeat probability.
* **DQN**: Implementation of a deep Q-network (DQN) to learn the action-value function Q. The network consists of two convolutional and two fully-connected layers and is trained using stochastic gradient descent and a Q-learning algorithm.
* **Replay Experience**: We use *replay experience* to handle non-stationarity and correlations in the data, and ...
* **Target DQN**: Implementation of a target DQN, helping against Q-value overestimation issue and significantly improving training performance.

## Setup & Usage
### Installation

This project uses dependencies that are best managed with Conda to ensure a smooth setup. To clone the repository, create a Conda environment and activate it, run the following:
```bash
git clone https://github.com/NN41/dqn.git
cd dqn
conda env create -f environment.yml
conda activate dqn
```
Note: you might need to manually put the pong.bin file in the correct directory.

### 2. Running a Single Training Loop
To run a single training loop with the default hyperparameters:

```bash
python main.py
```
Training metrics are stored directly in the `runs/` directory and can be visualized in TensorBoard.

### 3. Monitoring with TensorBoard
Relevant training metrics are logged to the `runs/` directory. To view them in TensorBoard, run the following command:
```bash
tensorboard --logdir runs
```

## Background & Implementation
The implementation in this project is directly based on the 2013 paper by Mnih et al. "Playing Atari with Deep Reinforcement Learning", who were the first to successfully use a deep learning model to learn to play Atari 2600 games directly from visual input. The theory is furthermore based on OpenAI's [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html).

During the training process, we collect sequences $s_t$ of frames $x_t$ and actions $a_t$, that is, $s_t = \{x_1, a_1, \ldots, a_{t-1}, x_t\}$. A frame is a vector of raw pixel values representing the current screen, which in the case of ``ALE/Pong-v5`` is an RGB image of dimensions 210 x 160. We preprocess each sequence into a *state representation* $\phi_t$ by taking the last four frames of $s_t$, transforming them to grayscale, downsizing them to $84$x$84$ and stacking them. This state representation is then fed into our Deep Q-Network (DQN).

As in the paper, our DQN is comprised of two convolutional layers (first 16 8x8 kernels with stride 4, then 32 4x4 kernels with stride 2), followed by two fully-connected layers (first with 256 hidden output units, then 6 output units for each possible action). We use ReLU activation in all hidden layers. Each element in the output vector contains the predicted Q-value of the corresponding action for the input state.

We also implement a basic *experience replay* mechanism, which stores the agent's experiences in the form of transitions $e_t = (\phi_t, a_t, r_t, \phi_{t+1})$ in a *replay memory*. At each step (frame) during training, we uniformly sample a minibatch of size 32 from the replay memory, compute the MSE loss with respect to the one-step temporal difference target and update the weights $\theta$ using Stochastic Gradient Descent (SGD). This one-step temporal difference target $y_j = r_j + \gamma \max_{a}Q(\phi_{j+1}, a; \theta)$ is based on the Bellman Equations. The experience replay mechanism has several benefits. It reuses data, stabilizes the data distribution and breaks correlations.

This implementation deviates from the original paper in a few ways. First, we use an Adam optimizer (instead of RMSprop) with a learning rate of 1e-4. Even though RMSprop theoretically better handles non-stationarity of input distributions, in this project Adam is giving smoother loss curves. Besides, implementations I found online also use Adam with 1e-4 learning rate. Next, we train the agent for ~600k steps with a replay memory of 25k transitions, instead of 10 million and 1 million as in the paper, respectively. The reason is hardware and time constraints. Last, we anneal $\epsilon$ of the $\epsilon$-greedy policy from 1 to 0.1 over only 320k frames, instead of 1 million. 

## Experiments
### Online DQN and Target DQN

In the figure below, you see two training runs. The y-axis shows the final score per completed game and the x-axis shows the number of games played.

The blue line is vanilla DQN as described in the Background section: it uses only a single DQN for both the online and the target network. Even though at takes a long time to see some noticeable improvement (after ~200k environment/update steps), it does clearly improve, with its best run achieving -8 points.

The pink line shows a training run where, in addition to the online network, we also use a target network that we synchronize with the online network every 500 steps. It is well-known that this (simple) modification to the training algorithm helps combat the problem of Q-value overestimation and stabilizes and improves training. In our case, we start seeing noticeable learning after way fewer steps (~100k), and it learns quicker and better. Its best game achieved a score of +3.

![Results](assets/Screenshot%202025-07-02%20133726.png)

Note that the figure displays the number of completed games on the x-axis, while the training process end was determined by the number of environment steps. Because the pink line belongs to the more successful algorithm, the agent is able to survive longer and hence play fewer complete games in the same number of environment steps. This explains why the pink line "stops" before the blue line.

## Future Work
- [ ] Improve storage of transitions in replay memory. Instead of storing two 4-frame stacks for each transition, store the (preprocessed) frames individually, then create the two frame stacks on the fly, reducing memory footprint by 8x. This allows us to keep a larger replay memory in the GPU memory, avoiding having to move frame stacks to and from the GPU, speeding up training as well. 
- [ ] Implement a circular buffer for replay memory. I implemented the replay memory as a deque for O(1) popleft (instead of O(N) for a list), but later realized that this results in O(32 * N) random sampling (instead of O(1) for a list), so both implementations are O(N) but the deque has a 32x different constant. A circular buffer is a list of size N with a pointer moving through the list, overwriting old values. This gives both O(1) "popleft" and O(32) sampling
- [ ] Proper seeding for full reproducibility.
- [ ] Implement vectorized environments for training speedup.
- [ ] Implement prioritized experience replay, where you sample transitions from the replay memory non-uniformly, based on the cases where the network attains the worst errors. See e.g. [this article](https://medium.com/@joachimiak.krzysztof/learning-to-play-pong-with-pytorch-tianshou-a9b8d2f1b8bd)