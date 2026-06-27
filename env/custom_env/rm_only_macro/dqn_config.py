import torch.nn as nn
import torch.optim as optim

from ..utils import SUMO_PARAMS

CONFIG = SUMO_PARAMS.get("config", "1ramp_1x3")

# One agent step = 40 simulation seconds.
MAX_SIMULATION_SECONDS_PER_EPISODE = SUMO_PARAMS.get("steps", 3600)
AGENT_CONTROL_CYCLE_SEC = 40.0
MAX_AGENT_STEPS_PER_EPISODE = int(
    MAX_SIMULATION_SECONDS_PER_EPISODE / AGENT_CONTROL_CYCLE_SEC
)

# Ablation control for `rm_lcc_macro_with_setMaxSpeed`: ramp metering only,
# same macro state (minus last_lane_action) and same reward function/weights.
VARIANT_TAG = "rm_only_macro"

HYPER_PARAMS = {
    "gpu": "0",
    "n_env": 1,
    "lr": 1e-4,
    "gamma": 0.99,
    "eps_start": 1.0,
    "eps_min": 0.01,
    "eps_dec": 2e6,
    "eps_dec_exp": True,
    "bs": 32,
    "min_mem": 100000,
    "max_mem": 1000000,
    "target_update_freq": 30000,
    "target_soft_update": True,
    "target_soft_update_tau": 1e-3,
    "save_freq": 10000,
    "log_freq": 4500,
    "save_dir": "./save/" + CONFIG + "/" + VARIANT_TAG + "/",
    "log_dir": "./logs/train/" + CONFIG + "/" + VARIANT_TAG + "/",
    "load": True,
    "repeat": 0,
    "max_episode_steps": 1000,
    "max_total_steps": 21e5,
    "algo": "DuelingDoubleDQNAgent",
}


def network_config(input_dim_space):
    """MLP body for the 8-d macro state.

    Action-head width (8) is set by the framework from env.action_space.n.
    Same architecture and hyperparameters as `rm_lcc_macro_with_setMaxSpeed`
    so the two variants are comparable.
    """
    num_input_features = input_dim_space.shape[0]

    fc_dims = (256, 128)
    activation = nn.ReLU()

    net = nn.Sequential(
        nn.Linear(num_input_features, fc_dims[0]),
        activation,
        nn.Linear(fc_dims[0], fc_dims[1]),
        activation,
    )

    fc_out_dim = fc_dims[-1]
    optim_func = optim.Adam
    loss_func = nn.SmoothL1Loss

    return net, fc_out_dim, optim_func, loss_func
