# Copilot.md

This file provides guidance to Codex (anthropic/Codex) when working with code in this repository.

---

## Commands

```bash
# Install dependencies
uv sync

# Install with analysis extras (Jupyter, Matplotlib, etc.)
uv sync --extra analysis

# Lint (auto-fix safe issues)
uv run ruff check --fix .

# Format
uv run ruff format .

# Lint + format in one go
uv run ruff check --fix . && uv run ruff format .

# Lint with unsafe fixes
uv run ruff check --fix --unsafe-fixes .

# Training
uv run python train.py
uv run python train.py --load True     # resume from checkpoint
uv run python train.py --gui True      # with SUMO GUI

# Baselines / inference / evaluation
uv run python play.py --strategy AlwaysGreen
uv run python observe.py --load <path>
uv run python evaluate.py

# TensorBoard
uv run tensorboard --logdir logs/train/1ramp_1x3/
```

> No automated tests exist in this project (see Known Issues).

---

## Project Overview

**Deep Reinforcement Learning for Automatic Ramp Metering Control** on a highway corridor.

A DRL agent controls a traffic signal (ramp meter) at a highway on-ramp to:

1. Maintain free-flow speed on the mainline at and upstream of the merge zone.
2. Maximize downstream throughput.
3. Limit on-ramp queue length to prevent spillback onto the arterial network.

The agent is trained and evaluated in **SUMO** (Simulation of Urban MObility) via the **TraCI** Python interface.

---

## Technology Stack

| Component         | Technology                      |
| ----------------- | ------------------------------- |
| Language          | Python ≥ 3.12                   |
| Package Manager   | `uv` (pyproject.toml + uv.lock) |
| DL Framework      | PyTorch                         |
| Traffic Simulator | SUMO (via TraCI / sumolib)      |
| RL API            | Gymnasium                       |
| Logging           | TensorBoard                     |
| Serialization     | msgpack (model save/load)       |
| Visualization     | Matplotlib, Seaborn, Jupyter    |

---

## Directory Structure

```text
.
├── train.py                 # Training entry point
├── play.py                  # Run baselines or human-play mode
├── observe.py               # Run trained agent for inference/observation
├── evaluate.py              # Batch evaluation pipeline (N episodes, seeded)
├── pyproject.toml           # uv-managed project config
│
├── env/                     # Environment package
│   ├── __init__.py          # Exports DqnEnv, DQN_CONFIG
│   ├── dqn_config.py        # Legacy top-level config (current work uses variant config below)
│   ├── dqn_env.py           # DqnEnv — adapter between framework and SUMO env
│   ├── view.py              # Pyglet visualization (disabled by default)
│   └── custom_env/
│       ├── __init__.py      # Exports RLController, baselines, SUMO_PARAMS
│       ├── utils.py         # SUMO_PARAMS — global simulation config dict
│       ├── sumo_env.py      # SumoEnv — base TraCI interface (lifecycle, detectors, routes)
│       ├── rl_controller.py # Legacy top-level RLController
│       ├── baselines.py     # Classical baselines: AlwaysGreen, FixedCycle, ALINEA, PI-ALINEA
│       ├── macro + lane/        # Variant 2 configs (14-d state, MLP)
│       └── rm_lcc_macro_with_setMaxSpeed/
│           ├── rl_controller.py # Main Variant: Joint RM + setMaxSpeed VSL controller (current workstream)
│           └── dqn_config.py    # Macro-only MLP config for the joint controller
│
├── dqn/                     # Agent package
│   ├── __init__.py          # Re-exports all agent classes and env utilities
│   ├── agent.py             # Agent hierarchy: DQN → Double → Dueling → PER variants
│   ├── network.py           # DeepQNetwork, DuelingDeepQNetwork, TwoStreamHybridNetwork
│   ├── replay_memory.py     # ReplayMemoryNaive (uniform), ReplayMemoryPrioritized (SumTree)
│   ├── env_wrap.py          # CustomEnvWrapper — Gymnasium-compatible reset/step
│   ├── env_make.py          # Environment factory with wrappers
│   └── utils/
│       ├── sum_tree.py      # SumTree data structure for PER
│       ├── better_abc.py    # Enhanced ABC metaclass
│       ├── msgpack_numpy.py # Model serialization helpers
│       └── baselines_wrappers/
│           ├── monitor.py         # Episode monitoring
│           ├── dummy_vec_env.py   # Single-process vectorized env wrapper
│           └── subproc_vec_env.py # Multi-process vectorized env wrapper
│
├── evaluation/
│   ├── parsers.py           # Parse tripinfo.xml, SUMO logs, framework CSVs
│   ├── results/             # Per-strategy result CSVs + Jupyter notebooks + plots/
│   └── reward/              # Training reward visualization notebook
│
├── save/                    # Saved model checkpoints (msgpack format)
│   └── 1ramp_1x3/
├── logs/                    # TensorBoard event files
│   └── train/1ramp_1x3/
├── tripinfo_runs/           # SUMO trip info XML outputs
└── bin/
    └── environment.yml      # Legacy Conda environment definition
```

---

## Architecture

### Environment Stack (bottom → top)

```text
SumoEnv          ← TraCI lifecycle, detector queries, route generation, vehicle subscriptions
    ↑
RLController     ← 15-d macro state, joint RM + setMaxSpeed VSL action execution, reward
    ↑
DqnEnv           ← Thin adapter matching framework's generic interface
    ↑
CustomEnvWrapper ← Gymnasium-compatible reset()/step() API
    ↑
DummyVecEnv      ← Vectorized env wrapper (always n_env=1)
```

### Agent Hierarchy

```text
Agent (abstract)
├── SimpleAgent          → DQNAgent (vanilla DQN)
├── DoubleAgent          → DoubleDQNAgent, DuelingDoubleDQNAgent  ← ACTIVE
└── PerDoubleAgent       → PerDuelingDoubleDQNAgent
```

### Network Architectures

- **DeepQNetwork**: Configurable MLP body → Q-values
- **DuelingDeepQNetwork**: MLP body → value stream + advantage stream → Q = V + (A − mean(A))
- **Macro MLP (active workstream)**: `rm_lcc_macro_with_setMaxSpeed/dqn_config.py` uses a 2-layer MLP body (256 → 128) for a 15-d macro state.

---

## SUMO Network

- **File**: `1ramp_1x3.net.xml` — single on-ramp, 3-lane mainline highway
- **Key edges**: `entry`, `off_ramp_up_stream`, `main_road`, `on_ramp`, `passage_area`, `acceleration_area`, `end_main_road`
- **Ramp meter**: Traffic light `ramp_meter` with two phases (G/r)
- **Detectors**: Upstream (3 lanes), bottleneck (4 lanes), downstream (3 lanes), ramp queue — all with 40s collection period

---

## Active Configuration

| Parameter               | Value                                                       |
| ----------------------- | ----------------------------------------------------------- |
| Algorithm               | `DuelingDoubleDQNAgent`                                     |
| State dimension         | 15 (macro-only + last RM action + last VSL speed)           |
| Action space            | 42 discrete joint actions (7 RM green times × 6 VSL speeds) |
| Control cycle           | 40 seconds                                                  |
| Learning rate           | 1e-4 (Adam)                                                 |
| Discount (γ)            | 0.99                                                        |
| Epsilon                 | 1.0 → 0.01 (exponential, over 2M steps)                     |
| Batch size              | 32                                                          |
| Replay buffer           | 1M transitions (uniform sampling)                           |
| Min buffer before learn | 100K transitions                                            |
| Target update freq      | 30000                                                       |
| Target update           | Soft Polyak (τ = 1e-3) every step                           |
| Loss                    | SmoothL1Loss (Huber)                                        |
| Max training steps      | 2.1M agent steps                                            |
| Episode length          | 3600s (1 hour sim time, ~90 steps)                          |
| Variant tag             | `joint_rm_vsl_42`                                           |

---

## Reward Function

Weighted multi-objective: speed rewards (positive) + occupancy/queue penalties (negative).

```text
reward = 1.5 × r_speed_merge      (+)
       + 1.0 × r_speed_upstream   (+)
       + 0.5 × r_speed_downstream (+)
       - 2.0 × p_occ_bottleneck   (−)
       - 1.0 × p_occ_upstream     (−)
       - 1.0 × p_queue            (−)
       - 20.0 × p_spillback       (−, triggers at 90% max queue)
```

Theoretical range: **[−24, +3]**.

---

## Entry Points

| Command                         | Description                                 |
| ------------------------------- | ------------------------------------------- |
| `python train.py`               | Train from scratch                          |
| `python train.py --load True`   | Resume training from checkpoint             |
| `python train.py --gui True`    | Train with SUMO GUI visible                 |
| `python play.py --strategy X`   | Run a baseline strategy (AlwaysGreen, etc.) |
| `python observe.py --load path` | Run trained model for inference             |
| `python evaluate.py`            | Batch evaluate over N episodes              |

---

## Baselines

| Baseline    | Strategy                                                   |
| ----------- | ---------------------------------------------------------- |
| AlwaysGreen | No metering — ramp signal permanently green                |
| FixedCycle  | Fixed 20s green / 20s red alternation                      |
| ALINEA      | Proportional feedback on downstream occupancy (O_crit=17%) |
| PI-ALINEA   | Proportional-integral feedback (K_P=60, K_I=10)            |

---

## Model Persistence

Models are saved as **msgpack** files (not PyTorch `.pt`), containing:

- Network parameters as NumPy arrays
- Training step count, episode count, mean reward, mean episode length

Location: `save/1ramp_1x3/<AgentClass>_lr<lr>/`

---

## Known Issues (from original project)

These are documented weaknesses in the base project:

1. **Penetration rate bug**: In `SumoEnv._generate_route_file()`, nearly all vehicles spawn as connected regardless of the drawn penetration rate. The correct split code is commented out.
2. **Double observation computation**: `RLController.step()` computes observation/reward, then `CustomEnvWrapper` computes them again via separate accessor methods.
3. **Queue measurement**: `get_edge_ls_queue_length_vehicles()` uses `getLastStepVehicleNumber()` (total vehicles) instead of `getLastStepHaltingNumber()` (halting vehicles).
4. **No yellow phase**: Direct green↔red transitions (yellow time defined but unused).
5. **Hard-coded IDs**: Detector and edge IDs are string literals scattered across the code.
6. **No gradient clipping**: None of the `learn()` methods apply gradient clipping.
7. **No validation during training**: No periodic evaluation on held-out scenarios.
8. **No automated tests**: No unit or integration tests exist.
9. **Model variants via file swapping**: Changing architectures requires manually copying files between subdirectories.

---

## Conventions

- **Config management**: Shared simulation config in `env/custom_env/utils.py` (`SUMO_PARAMS`), with current controller config in `env/custom_env/rm_lcc_macro_with_setMaxSpeed/dqn_config.py`.
- **Current workstream**: Edit `env/custom_env/rm_lcc_macro_with_setMaxSpeed/rl_controller.py` and `.../dqn_config.py` for joint RM+VSL behavior.
- **Logging**: TensorBoard for training metrics → `logs/train/1ramp_1x3/<agent>_lr<lr>/`
- **Evaluation results**: CSVs per strategy → `evaluation/results/results_<StrategyName>.csv`
- **Console styling**: Uses `colorama` for colored terminal output.
- **Episode seeding**: Evaluation uses deterministic seeds (`master_seed + episode_id`) for reproducibility across strategies.

---

## Demand Scenarios

Flows are drawn from weighted discrete distributions each episode:

| Flow Type | Values (veh/h)                           | Bias           |
| --------- | ---------------------------------------- | -------------- |
| Mainline  | 4000, 4500, 5000, 5500, 6000, 6500       | Towards higher |
| On-ramp   | 1400, 1500, 1600, 1700, 1800, 1900, 2000 | Towards higher |
| Off-ramp  | 100, 300, 500                            | Uniform        |

---

## Current State Representation (Joint RM+VSL Variant)

The active controller in `rm_lcc_macro_with_setMaxSpeed` uses a macro-only state:

- **Dimension**: 15 features
- **Core content**: mainline/ramp macro traffic features + lane-0 local features + previous RM and lane actions
- **No micro grid**: this variant does not use the 27×5×2 connected-vehicle grid

---

_This file is intended for AI assistant context. See `README.md` for the full technical analysis._

## How the Two Packages Work Together

### The Full Data Flow (One Training Step)

```text
train.py
  └─ DummyVecEnv.step(action)
       └─ Monitor → CustomEnvWrapper.step(action)   [Gymnasium API]
           └─ DqnEnv.step(action)                   [thin adapter — just delegates]
               └─ RLController.step(action)         [REAL work happens here]
                   └─ SumoEnv.*                     [TraCI calls]

```

---

### The Environment Package (`env/`)

- **`env/custom_env/utils.py`:** The root config dictionary `SUMO_PARAMS`. Everything else reads from here: network shape, demand distributions, cycle timing, grid dimensions. Change one value here and it ripples everywhere.
- **`env/custom_env/sumo_env.py` (`SumoEnv`):** Owns the SUMO process. Responsibilities include:
- Launches/kills the `traci` subprocess on `__init__` and on every `simulation_reset()`.
- Builds the `internal_to_destination_map` from the network file (maps internal junction lanes → destination lanes, used by the micro grid).
- Provides all detector query helpers: `get_loops_flow_interval`, `get_loops_occupancy_interval`, `get_loops_flow_weigthed_mean_speed`, etc.
- Generates the `.rou.xml` route file each episode (stochastic demand).
- Abstract methods `reset()`, `step()`, `obs()`, `rew()`, `done()` — must be implemented by subclasses.

- **`env/custom_env/rm_lcc_macro_with_setMaxSpeed/rl_controller.py` (`RLController(SumoEnv)`):** The current RL brain. Responsibilities include:
- `step(action_index)`: Decodes a joint action (0–41) into `(green_idx, vsl_idx)` where green is in {10,15,...,40}s and VSL is a direct speed choice.
- `_apply_lane_vsl(speed_mps)`: Applies lane control via `traci.lane.setMaxSpeed("vsl_zone_0", speed_mps)`.
- `_collect_data_at_cycle_end()`: Reads all detectors once per cycle into `self.processed_*` attributes.
- `_get_current_observation()`: Builds the 15-d normalized macro state (including last RM and last VSL speed).
- `_calculate_reward()`: Weighs 7 components with hardcoded weights.
- `reset()`: Calls `simulation_reset()`, runs 5 warm-up steps, collects initial data.
- **Note:** This is the file you touch to change state features, action space, and reward weights.

- **`env/dqn_env.py` (`DqnEnv`):** A thin adapter. Its only real job is to instantiate the right `RLController` or `Baselines.*` depending on mode (train/observe/play) and to expose `action_space_n` and `observation_space_n` to the layer above. The `obs()`, `rew()`, `step()`, `reset()` methods just delegate 1:1 to `sumo_env`. There is no logic here.
- **`env/custom_env/rm_lcc_macro_with_setMaxSpeed/dqn_config.py`:**
- `HYPER_PARAMS` dict: Current variant hyperparameters.
- `network_config()` factory: Active 2-layer MLP body (256→128), returned as `(net, fc_out_dim, optimizer_class, loss_class)`.

---

### The Agent Package (`dqn/`)

- **`dqn/env_wrap.py` (`CustomEnvWrapper(gym.Env)`):** Wraps `DqnEnv` to comply with Gymnasium's `reset() → (obs, info)` and `step() → (obs, rew, terminated, truncated, info)` API. Also has `log_info_writer()` for CSV logging during evaluation.
- **`dqn/env_make.py` (`make_env`):** Applies optional wrappers: `RepeatActionWrapper`, `MaxEpisodeStepsWrapper`, `DummyVecEnv`. Always returns a vectorized env (`n_env=1` uses `DummyVecEnv`).
- **`dqn/network.py` (`DeepQNetwork`, `DuelingDeepQNetwork`):** PyTorch modules. They receive `nn_conf_func` (the factory from `dqn_config.py`) and call it to get `self.net` (the body). Then they bolt on their own output heads:
- `DeepQNetwork`: Adds `fc_out` → Q(s,a) linear.
- `DuelingDeepQNetwork`: Adds `fc_val` → V(s) and `fc_adv` → A(s,a), aggregates to Q.

- **`dqn/agent.py`:** The learning logic. Class hierarchy:
- `Agent` (abstract): Replay buffer management, epsilon-greedy, soft/hard target update, tensorboard logging, save/load.
- `SimpleAgent.learn()`: Vanilla DQN Bellman target.
- `DoubleAgent.learn()`: Double DQN — online network selects action, target network evaluates it.
- `PerDoubleAgent.learn()`: Adds importance-sampling weights from `ReplayMemoryPrioritized`.
- Concrete classes (`DuelingDoubleDQNAgent`, etc.) only add `__init__` that instantiates the right network + replay memory.

---

### What `train.py` Does

```text
train.py  →  CustomEnvWrapper(DqnEnv("train"))
                                     ↓
                 make_env(env, max_episode_steps=1000)  →  DummyVecEnv
                                     ↓
         DuelingDoubleDQNAgent(nn_conf_func=network_config, ...)
                                     ↓
         init_replay_memory_buffer()   ← 100K random steps first
         train_loop()                  ← ε-greedy → step → store → learn → soft update

```

---

### Where to Make Changes

| Goal                            | File(s) to Edit                                                                                                |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Change reward weights**       | `env/custom_env/rm_lcc_macro_with_setMaxSpeed/rl_controller.py` (`_calculate_reward`)                          |
| **Add/remove state features**   | `_get_current_observation()` and `observation_space_n` in `.../rm_lcc_macro_with_setMaxSpeed/rl_controller.py` |
| **Change joint action space**   | `green_time_actions_sec`, `vsl_speed_actions_mps`, and `action_space_n` in `.../rl_controller.py`              |
| **Change lane VSL behavior**    | `TARGET_VSL_LANE_ID`, `VSL_SPEEDS_MPS`, `_apply_lane_vsl()` in `.../rl_controller.py`                          |
| **Change network architecture** | `env/custom_env/rm_lcc_macro_with_setMaxSpeed/dqn_config.py` (`network_config`)                                |
| **Change hyperparameters**      | `env/custom_env/rm_lcc_macro_with_setMaxSpeed/dqn_config.py` (`HYPER_PARAMS`)                                  |
| **Change demand distributions** | `env/custom_env/utils.py:33` (`SUMO_PARAMS`)                                                                   |
| **Switch algorithm**            | `HYPER_PARAMS["algo"]` in `env/custom_env/rm_lcc_macro_with_setMaxSpeed/dqn_config.py`                         |
| **Switch to another variant**   | Point `DqnEnv` to a different controller/config pair under `env/custom_env/`                                   |

---

### One Non-Obvious Coupling

> **Important:** `DqnEnv.step()` calls `sumo_env.step(action)` but discards the return value `(obs, rew, done, info)`. The framework then calls `sumo_env.obs()`, `sumo_env.rew()`, `sumo_env.done()` separately via `CustomEnvWrapper._obs()/_rew()/_done()`. This means the observation and reward are computed twice per step — this is the "Double observation computation" bug in `AGENTS.md`. If you add expensive computation to `step()`, cache it; don't recompute in `obs()`/`rew()`.

[Introduction to Gym and Stable Baselines for Reinforcement Learning](https://www.youtube.com/watch?v=lZ-F9C6cGIA)
This video provides an excellent visual introduction to using the Gym interface alongside Stable Baselines, which covers many of the core concepts in your codebase.
