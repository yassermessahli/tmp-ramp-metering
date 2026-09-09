import os

import msgpack
import torch as T
import torch.nn as nn

from .utils import msgpack_numpy_patch

msgpack_numpy_patch()


class Network(nn.Module):
    """Base neural network class handling serialization."""

    def __init__(self, device, nn_conf_func, input_dim):
        """Initializes network architecture and optimizer."""
        super().__init__()

        self.net, self.fc_out_dim, optim_func, loss_func = nn_conf_func(input_dim)
        self.optim_func = lambda params, lr: optim_func(params, lr=lr)
        self.loss_func = lambda reduction: loss_func(reduction=reduction)

        self.device = device

    def forward(self, s):
        pass

    def actions(self, obses):
        pass

    def save(self, save_path, step, episode_count, rew_mean, len_mean):
        """Serializes network parameters and optimizer state to disk."""
        import io

        params_dict = {
            "parameters": {
                k: v.detach().cpu().numpy() for k, v in self.state_dict().items()
            },
            "step": step,
            "episode_count": episode_count,
            "rew_mean": rew_mean,
            "len_mean": len_mean,
        }

        if hasattr(self, "optimizer"):
            buffer = io.BytesIO()
            T.save(self.optimizer.state_dict(), buffer)
            params_dict["optimizer_state"] = buffer.getvalue()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(msgpack.dumps(params_dict))

    def load(self, load_path):
        """Loads network parameters and optimizer state from disk."""
        import io

        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path, "rb") as f:
            params_dict = msgpack.loads(f.read())

        parameters = {
            k: T.as_tensor(v, device=self.device)
            for k, v in params_dict["parameters"].items()
        }
        self.load_state_dict(parameters)

        if "optimizer_state" in params_dict and hasattr(self, "optimizer"):
            buffer = io.BytesIO(params_dict["optimizer_state"])
            opt_state = T.load(buffer, map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(opt_state)

        return (
            params_dict["step"],
            params_dict["episode_count"],
            params_dict["rew_mean"],
            params_dict["len_mean"],
        )


class DeepQNetwork(Network):
    """Standard Deep Q-Network implementation."""

    def __init__(
        self, device, lr, nn_conf_func, input_dim, output_dim, reduction="mean"
    ):
        """Initializes DQN with linear output layer."""
        super().__init__(device, nn_conf_func, input_dim)

        self.fc_out = nn.Linear(self.fc_out_dim, output_dim)

        self.optimizer = self.optim_func(self.parameters(), lr=lr)
        self.loss = self.loss_func(reduction=reduction)

        self.to(self.device)

    def forward(self, s):
        """Forward pass to compute Q-values."""
        net = self.net(s)
        a = self.fc_out(net)

        return a

    def actions(self, obses):
        """Selects greedy actions based on Q-values."""
        obses_t = T.as_tensor(obses, dtype=T.float32).to(self.device)
        q_values = self(obses_t)

        max_q_indices = T.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()

        return actions


class DuelingDeepQNetwork(Network):
    """Dueling Deep Q-Network implementation."""

    def __init__(
        self, device, lr, nn_conf_func, input_dim, output_dim, reduction="mean"
    ):
        """Initializes Dueling DQN with value and advantage streams."""
        super().__init__(device, nn_conf_func, input_dim)

        self.fc_val = nn.Linear(self.fc_out_dim, 1)
        self.fc_adv = nn.Linear(self.fc_out_dim, output_dim)
        # Aggregation layer: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        self.aggregate_layer = lambda val, adv: T.add(
            val, (adv - adv.mean(dim=1, keepdim=True))
        )

        self.optimizer = self.optim_func(self.parameters(), lr=lr)
        self.loss = self.loss_func(reduction=reduction)

        self.to(self.device)

    def forward(self, s):
        """Forward pass computing aggregated Q-values."""
        net = self.net(s)
        val = self.fc_val(net)
        adv = self.fc_adv(net)
        agg = self.aggregate_layer(val, adv)

        return agg

    def value(self, s):
        """Computes state value V(s)."""
        net = self.net(s)
        val = self.fc_val(net)

        return val

    def advantages(self, s):
        """Computes advantage values A(s,a)."""
        net = self.net(s)
        adv = self.fc_adv(net)

        return adv

    def actions(self, obses):
        """Selects greedy actions based on advantages."""
        obses_t = T.as_tensor(obses, dtype=T.float32).to(self.device)
        adv_q_values = self.advantages(obses_t)

        max_adv_q_indices = T.argmax(adv_q_values, dim=1)
        actions = max_adv_q_indices.detach().tolist()

        return actions
