# Deep Reinforcement Learning for Joint Ramp Metering & VSL — Codebase Mental Map

This document serves as a comprehensive technical overview of the project, highlighting its architecture, components, and the currently active VSL variant.

## 1. Project Overview & Goals
**Goal:** Use Deep Reinforcement Learning (DRL) to control a highway on-ramp merge bottleneck to maintain free-flow speeds, maximize throughput, and prevent ramp queue spillback.
**Current Scope:** A `1ramp_1x3` network simulated in SUMO. The most recent active variant simultaneously controls the ramp meter (green-time duration) and applies a Variable Speed Limit (VSL) to the mainline lane closest to the ramp (`vsl_zone_0`).

## 2. Currently Active Variant: `rm_vsl_macro_with_setMaxSpeed`
**Location:** `env/custom_env/rm_vsl_macro_with_setMaxSpeed/`
This variant replaces the old pure-ramp-metering approach with a joint **Ramp Metering + VSL** approach.

### Action Space (Discrete: 42 Actions)
The agent chooses one of 42 actions every 40-second control cycle.
* **Green Time Levels (7):** 10, 15, 20, 25, 30, 35, 40 seconds.
* **VSL Speed Limits (6):** 13.89, 16.67, 19.44, 22.22, 25.0, 27.78 m/s (50 km/h to 100 km/h).
* *Execution Logic:* The VSL speed limit is applied to the target lane (`vsl_zone_0`) **only during the green sub-phase** to facilitate safe merging. During the red sub-phase, the lane reverts to free-flow default speed.

### Observation Space (15-d Flat Vector)
Macro-only traffic state aggregated over the last 40s cycle:
1-7. **Mainline Macro:** Upstream flow, merging flow, upstream occupancy, upstream speed, bottleneck occupancy, bottleneck speed, ramp queue.
8-13. **Lane 0 (VSL Lane) Macro:** Flow, occupancy, speed for both bottleneck and upstream regions on the specific rightmost lane.
14-15. **Last Action:** Last chosen green time and last chosen VSL speed (normalized).

### Reward Function
A weighted linear combination of normalized KPIs (Theoretical range approx. `[-24, +3]`):
* **(+) Rewards:** Merge Speed (x1.5), Upstream Speed (x1.0), Downstream Speed (x0.5).
* **(-) Penalties:** Bottleneck Occ (x2.0), Upstream Occ (x1.0), Ramp Queue (x1.0), Heavy Spillback (x20.0).

### Neural Network (`dqn_config.py`)
Standard MLP Body: `Input(15) → FC(256) → ReLU → FC(128) → ReLU → Output(128)`.

---

## 3. Architecture & Tech Stack
* **Language & Package Manager:** Python 3.10+ / `uv`
* **ML Framework:** PyTorch (Dueling Double DQN with PER options). Models saved as `msgpack` arrays.
* **Traffic Simulation:** SUMO v1.21.0 interfaced via TraCI.
* **RL Environment API:** Gymnasium (`env_wrap.py` bridges TraCI logic to standard RL loops).
* **Visualization:** Matplotlib, Seaborn, Tensorboard.

### Agent Hierarchy (`dqn/agent.py`)
`Agent` → `DoubleAgent` → `DuelingDoubleDQNAgent` (Currently configured algorithm).

---

## 4. Codebase Structure

```
├── train.py               # Main training script (buffer warm-up + training loop)
├── evaluate.py            # Batch evaluation against baselines
├── play.py / observe.py   # Run visualization or human-play modes
├── env/
│   ├── dqn_config.py      # Main config router
│   ├── dqn_env.py         # Gymnasium adapter
│   └── custom_env/        # Environment variants
│       ├── rm_vsl_macro_with_setMaxSpeed/  # CURRENT ACTIVE VARIANT (Joint RM + VSL)
│       ├── baselines.py   # Classical controllers (ALINEA, Fixed Cycle)
│       ├── rl_controller.py # Active environment logic
│       └── sumo_env.py    # TraCI base class wrapper
├── dqn/                   # Deep RL implementation
│   ├── agent.py           # Agent hierarchy
│   ├── network.py         # PyTorch architectures
│   └── replay_memory.py   # Naive & Prioritized (SumTree) buffers
└── evaluation/            # Scripts/Notebooks for metric parsing and plotting
```

---

## 5. Typical Workflow Commands

**Train the Agent:**
```bash
uv run python train.py
```
*(Check `dqn_config.py` in the active variant to modify hyperparameters like `max_total_steps` or `VARIANT_TAG`).*

**Evaluate the Agent:**
```bash
uv run python evaluate.py
```

**Monitor Training (Tensorboard):**
```bash
uv run tensorboard --logdir logs/train/1ramp_1x3
```
*(Make sure to run without an active external `.venv` to avoid path conflicts).*

---

## 6. Known Codebase Gotchas & Weaknesses (from README / Analysis)
1. **Penetration Rate Bug:** In training route generation, vehicles are spawned as 100% connected regardless of the drawn penetration rate.
2. **Double State Computation:** `_get_current_observation()` and `_calculate_reward()` are called redundantly by both the RL Controller and the Gym Wrapper every step.
3. **Queue Measurement:** Uses total vehicles on the edge rather than strictly halted/queued vehicles.
4. **Vectorization Overhead:** Uses `DummyVecEnv` for a single environment, adding slight overhead.
5. **No Gradient Clipping:** Might cause instability early in training given the large negative reward limits.

---

## 7. Abstraction Philosophy

### 7.1 Agent Module Hierarchy
The RL agents are designed using an object-oriented inheritance model to maximize code reuse:
* `Agent` (Base Class): Defines the interface for all agents (choose action, store transition, learn, save/load).
* `SimpleAgent` / `DoubleAgent` / `PerDoubleAgent`: Implement specific training logics (e.g., standard Q-learning vs. Double Q-learning vs. Prioritized Experience Replay).
* Concrete Classes (`DQNAgent`, `DuelingDoubleDQNAgent`, etc.): Combine a specific logic class with a specific Neural Network architecture (e.g., `DeepQNetwork` or `DuelingDeepQNetwork`).

### 7.2 Environment Module Hierarchy
The environment strictly separates generic simulator logic from RL-specific logic:
* `SumoEnv` (`env/custom_env/sumo_env.py`): The base simulator wrapper. Handles pure TraCI communication, starting/stopping SUMO, routing, and low-level loop queries. Knows nothing about state/actions/rewards.
* `RLController` (`env/custom_env/rl_controller.py`): Inherits from `SumoEnv`. Implements the MDP (Markov Decision Process). Translates abstract RL actions (like action index `0-41`) into specific TraCI commands (like setting phase and VSL limit), computes the state vector, and calculates the reward.
* `DqnEnv` (`env/dqn_env.py`): A structural adapter that wraps the `RLController` (or baselines) to expose standard `reset()`, `step()`, `obs()`, and `rew()` methods.
* `CustomEnvWrapper` (`dqn/env_wrap.py`): The outermost Gymnasium-compatible wrapper that translates `DqnEnv` outputs into the standard `(obs, reward, terminated, truncated, info)` tuple expected by most modern RL frameworks.

### 7.3 The Training, Observation, and Evaluation Workflow (Layer-by-Layer)
The system executes a control step by cascading through several layers of abstraction:
1. **Agent Layer (`train.py`):** The training loop queries the agent for an action (e.g., `action = 15`) and calls `env.step(action)`.
2. **Wrapper Layer (`CustomEnvWrapper` / `DqnEnv`):** Receives the action and delegates it down to the active environment variant.
3. **Logic Layer (`RLController`):** Decodes `action = 15` into `green_time = 20s`, `vsl_speed = 19.44 m/s`. It sends these commands to the simulator layer and calculates the necessary red-time duration.
4. **Simulator Layer (`SumoEnv` / TraCI):** Executes the literal traffic light phase change and max-speed override in SUMO. It then ticks the microscopic simulation step-by-step for the duration of the green/red phases.
5. **Observation & Reward Computation:** Once the cycle completes, `RLController` queries `SumoEnv` for aggregated sensor data (flow, occupancy, speed), normalizes this data into a 15-d state vector, and computes the weighted scalar reward.
6. **Return to Agent:** The state, reward, and done flag bubble back up to the training loop, where the agent stores the transition in its replay buffer and triggers a learning step (`loss` computation and backpropagation).
