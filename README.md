# Flappy Bird DQN (Deep Q-Network)

This repository implements a Deep Q-Network (DQN) reinforcement learning agent to learn and play the classic Flappy Bird game. The agent is built using **PyTorch**, **Gymnasium**, and the **flappy-bird-gymnasium** environment.

---
 ## Gameplay GIF
 <img width="1280" height="720" alt="compressed-1783586674874 (2) (online-video-cutter (1)" src="https://github.com/user-attachments/assets/66495fdc-e35f-4432-b0e4-e3dea782a9b5" />



---

## 📂 Project Structure

- **[agent.py](file:///c:/Users/ABHINAV%20GAUR/RL_DQN/agent.py)**: The main orchestrator for training and evaluating the DQN agent.
- **[dqn.py](file:///c:/Users/ABHINAV%20GAUR/RL_DQN/dqn.py)**: Defines the Deep Q-Network architecture (a 3-layer feedforward neural network).
- **[experience_replay.py](file:///c:/Users/ABHINAV%20GAUR/RL_DQN/experience_replay.py)**: Implements the `ReplayMemory` buffer to store and sample state transitions.
- **[game_flappy_bird.py](file:///c:/Users/ABHINAV%20GAUR/RL_DQN/game_flappy_bird.py)**: A manual gameplay script allowing you to control the bird using PyGame keyboard events.
- **[parameters.yaml](file:///c:/Users/ABHINAV%20GAUR/RL_DQN/parameters.yaml)**: Configuration file for RL hyperparameters (learning rate, discount factor, epsilon decay, etc.).

---

## ⚙️ Requirements & Installation

Make sure you have Python 3.8+ installed. You can install the required dependencies using `pip`:

```bash
pip install torch gymnasium flappy-bird-gymnasium pygame pyyaml
```

---

## 🚀 How to Use

### 1. Play Manually
Test the game environment and your own reaction times by playing manually:
```bash
python game_flappy_bird.py
```
* **Control**: Press the `SPACE` key to flap.

### 2. Train the RL Agent
To train the DQN agent using the configuration set `flappybirdv0` defined in `parameters.yaml`:
```bash
python agent.py flappybirdv0 --train
```
- During training, the agent dynamically decays its exploration rate ($\varepsilon$) and saves the best model parameters to `runs/flappybirdv0.pt` and training progress to `runs/flappybirdv0.log`.

### 3. Evaluate / Test the Trained Agent
Once trained, run the agent in evaluation mode to watch it play:
```bash
python agent.py flappybirdv0
```

---

## 🎛️ Hyperparameters (`parameters.yaml`)

You can tune training parameters inside `parameters.yaml`:

| Parameter | Description |
|---|---|
| `env_id` | Gymnasium environment ID (`FlappyBird-v0`) |
| `epsilon_init` | Starting value for epsilon ($\varepsilon$-greedy exploration) |
| `epsilon_min` | Minimum exploration rate |
| `epsilon_decay` | Decay factor multiplied per episode |
| `replay_memory_size` | Capacity of the experience replay buffer |
| `mini_batch_size` | Size of the batch sampled from replay memory for optimization |
| `network_sync_rate` | Number of steps between target network syncs |
| `alpha` | Learning rate (Adam optimizer) |
| `gamma` | Discount factor for future rewards |
| `reward_threshold` | Target reward threshold to terminate an episode early if reached |
