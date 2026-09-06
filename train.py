import argparse
import itertools
import os

import numpy as np
from colorama import Fore
from tqdm import tqdm

from dqn import Agents, CustomEnvWrapper, make_env
from env import HYPER_PARAMS, CustomEnv, network_config

BUFFER_SAVE_DIR = "./save/replay_buffer/"


class Train:
    """Manages the training loop for DQN agents."""

    def __init__(self, args):
        """Initializes environment and agent configuration."""
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        self.env = make_env(
            env=CustomEnvWrapper(CustomEnv("train")),
            repeat=args.repeat,
            max_episode_steps=args.max_episode_steps,
            n_env=args.n_env,
        )

        self.agent = getattr(Agents, args.algo)(
            n_env=args.n_env,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.eps_start,
            epsilon_min=args.eps_min,
            epsilon_decay=args.eps_dec,
            epsilon_exp_decay=args.eps_dec_exp,
            nn_conf_func=network_config,
            input_dim=self.env.observation_space,
            output_dim=self.env.action_space.n,
            batch_size=args.bs,
            min_buffer_size=args.min_mem,
            buffer_size=args.max_mem,
            update_target_frequency=args.target_update_freq,
            target_soft_update=args.target_soft_update,
            target_soft_update_tau=args.target_soft_update_tau,
            save_frequency=args.save_freq,
            log_frequency=args.log_freq,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            load=args.load,
            algo=args.algo,
            gpu=args.gpu,
        )
        print(Fore.LIGHTYELLOW_EX, self.agent.device, Fore.RESET)
        self.agent.load_model()

        print()
        print("TRAIN")
        print()
        print(args.algo)
        print()
        print(self.agent.online_network)
        print()
        [print(arg, "=", getattr(args, arg)) for arg in vars(args)]

        self.max_total_steps = args.max_total_steps
        self.load_buffer_path = args.load_buffer

        obs_dim = self.env.observation_space.shape[0]
        variant = os.path.basename(os.path.normpath(args.save_dir))
        self.buffer_path = os.path.join(
            BUFFER_SAVE_DIR,
            f"{variant}_obs{obs_dim}_mem{self.agent.min_buffer_size}.npz",
        )

    def _save_buffer(self):
        """Serialises the replay buffer to disk as a compressed npz archive."""
        os.makedirs(BUFFER_SAVE_DIR, exist_ok=True)
        buf = list(self.agent.replay_memory_buffer.replay_buffer)
        obs      = np.array([t[0] for t in buf], dtype=np.float32)
        actions  = np.array([t[1] for t in buf], dtype=np.int32)
        rews     = np.array([t[2] for t in buf], dtype=np.float32)
        dones    = np.array([t[3] for t in buf], dtype=bool)
        new_obs  = np.array([t[4] for t in buf], dtype=np.float32)
        np.savez_compressed(self.buffer_path, obs=obs, actions=actions, rews=rews, dones=dones, new_obs=new_obs)
        size_mb = os.path.getsize(self.buffer_path) / 1e6
        print(Fore.LIGHTCYAN_EX + f"Buffer saved → {self.buffer_path}  ({size_mb:.1f} MB)" + Fore.RESET)

    def _load_buffer(self, path):
        """Restores a previously saved buffer. Returns True on success, False otherwise."""
        if not os.path.exists(path):
            print(Fore.LIGHTRED_EX + f"Buffer file not found: {path} — filling from scratch." + Fore.RESET)
            return False

        data = np.load(path)
        saved_obs_dim = data["obs"].shape[1]
        expected_obs_dim = self.env.observation_space.shape[0]
        if saved_obs_dim != expected_obs_dim:
            print(
                Fore.LIGHTRED_EX
                + f"Buffer obs dim mismatch: file has {saved_obs_dim}-d, env expects {expected_obs_dim}-d. "
                + "Filling from scratch."
                + Fore.RESET
            )
            return False

        deque = self.agent.replay_memory_buffer.replay_buffer
        for obs, action, rew, done, new_obs in zip(
            data["obs"], data["actions"], data["rews"], data["dones"], data["new_obs"], strict=False
        ):
            deque.append((obs, int(action), float(rew), bool(done), new_obs))

        print(Fore.LIGHTCYAN_EX + f"Buffer loaded ← {path}  ({len(deque)} transitions)" + Fore.RESET)
        return True

    def init_replay_memory_buffer(self):
        """Fills replay buffer with initial experiences."""
        print()
        print("Initialize Replay Memory Buffer")

        if self.load_buffer_path and self._load_buffer(self.load_buffer_path):
            return

        total_init = self.agent.min_buffer_size // self.agent.n_env
        obses = self.env.reset()
        with tqdm(
            total=self.agent.min_buffer_size,
            initial=self.agent.resume_step * self.agent.n_env,
            desc="Fill Buffer",
            unit="step",
            dynamic_ncols=True,
            colour="yellow",
        ) as pbar:
            for t in range(total_init):
                if t >= total_init - self.agent.resume_step:
                    actions = self.agent.choose_actions(obses)
                else:
                    actions = [
                        self.env.action_space.sample() for _ in range(self.agent.n_env)
                    ]

                new_obses, rews, dones, _ = self.env.step(actions)
                self.agent.store_transitions(obses, actions, rews, dones, new_obses, None)
                obses = new_obses
                pbar.update(self.agent.n_env)

        self._save_buffer()

    def train_loop(self):
        """Executes main training loop."""
        print()
        print("Start Training")

        total_train = int(self.max_total_steps) if bool(self.max_total_steps) else None
        obses = self.env.reset()
        with tqdm(
            total=total_train,
            initial=self.agent.resume_step * self.agent.n_env,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            colour="green",
        ) as pbar:
            for step in itertools.count(start=self.agent.resume_step):
                self.agent.step = step

                actions = self.agent.choose_actions(obses)
                new_obses, rews, dones, infos = self.env.step(actions)
                self.agent.store_transitions(obses, actions, rews, dones, new_obses, infos)
                obses = new_obses

                self.agent.learn()
                self.agent.update_target_network()
                self.agent.log()
                self.agent.save_model()

                pbar.update(self.agent.n_env)
                if step % 100 == 0:
                    pbar.set_postfix(
                        eps=f"{self.agent.epsilon():.3f}",
                        ep=self.agent.episode_count,
                        rew=f"{self.agent.info_mean('r'):.1f}" if self.agent.ep_info_buffer else "n/a",
                    )

                if (
                    bool(self.max_total_steps)
                    and (step * self.agent.n_env) >= self.max_total_steps
                ):
                    self.agent.save_model(force=True)
                    return

    def run(self):
        """Starts the training process."""
        self.init_replay_memory_buffer()
        self.train_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRAIN")

    def str2bool(v):
        return v.lower() in ("yes", "y", "true", "t", "1")

    parser.add_argument("-gpu", type=str, default=HYPER_PARAMS["gpu"], help="GPU #")
    parser.add_argument(
        "-n_env",
        type=int,
        default=HYPER_PARAMS["n_env"],
        help="Multi-processing environments",
    )
    parser.add_argument(
        "-lr", type=float, default=HYPER_PARAMS["lr"], help="Learning rate"
    )
    parser.add_argument(
        "-gamma", type=float, default=HYPER_PARAMS["gamma"], help="Discount factor"
    )
    parser.add_argument(
        "-eps_start",
        type=float,
        default=HYPER_PARAMS["eps_start"],
        help="Epsilon start",
    )
    parser.add_argument(
        "-eps_min", type=float, default=HYPER_PARAMS["eps_min"], help="Epsilon min"
    )
    parser.add_argument(
        "-eps_dec", type=float, default=HYPER_PARAMS["eps_dec"], help="Epsilon decay"
    )
    parser.add_argument(
        "-eps_dec_exp",
        type=str2bool,
        default=HYPER_PARAMS["eps_dec_exp"],
        help="Epsilon exponential decay",
    )
    parser.add_argument("-bs", type=int, default=HYPER_PARAMS["bs"], help="Batch size")
    parser.add_argument(
        "-min_mem",
        type=int,
        default=HYPER_PARAMS["min_mem"],
        help="Replay memory buffer min size",
    )
    parser.add_argument(
        "-max_mem",
        type=int,
        default=HYPER_PARAMS["max_mem"],
        help="Replay memory buffer max size",
    )
    parser.add_argument(
        "-target_update_freq",
        type=int,
        default=HYPER_PARAMS["target_update_freq"],
        help="Target network update frequency",
    )
    parser.add_argument(
        "-target_soft_update",
        type=str2bool,
        default=HYPER_PARAMS["target_soft_update"],
        help="Target network soft update",
    )
    parser.add_argument(
        "-target_soft_update_tau",
        type=float,
        default=HYPER_PARAMS["target_soft_update_tau"],
        help="Target network soft update tau rate",
    )
    parser.add_argument(
        "-save_freq", type=int, default=HYPER_PARAMS["save_freq"], help="Save frequency"
    )
    parser.add_argument(
        "-log_freq", type=int, default=HYPER_PARAMS["log_freq"], help="Log frequency"
    )
    parser.add_argument(
        "-save_dir", type=str, default=HYPER_PARAMS["save_dir"], help="Save directory"
    )
    parser.add_argument(
        "-log_dir", type=str, default=HYPER_PARAMS["log_dir"], help="Log directory"
    )
    parser.add_argument(
        "-load", type=str2bool, default=HYPER_PARAMS["load"], help="Load model"
    )
    parser.add_argument(
        "-load_buffer",
        type=str,
        default=None,
        help="Path to a saved buffer .npz file to load instead of filling from scratch",
    )
    parser.add_argument(
        "-repeat", type=int, default=HYPER_PARAMS["repeat"], help="Steps repeat action"
    )
    parser.add_argument(
        "-max_episode_steps",
        type=int,
        default=HYPER_PARAMS["max_episode_steps"],
        help="Episode step limit",
    )
    parser.add_argument(
        "-max_total_steps",
        type=int,
        default=HYPER_PARAMS["max_total_steps"],
        help="Max total training steps",
    )
    parser.add_argument(
        "-algo",
        type=str,
        default=HYPER_PARAMS["algo"],
        help="DQNAgent "
        + "DoubleDQNAgent "
        + "DuelingDoubleDQNAgent "
        + "PerDuelingDoubleDQNAgent",
    )

    Train(parser.parse_args()).run()
