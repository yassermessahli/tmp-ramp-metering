import argparse
import os

import numpy as np
from torch import cuda, device

from dqn import CustomEnvWrapper, Networks, make_env
from env import HYPER_PARAMS, CustomEnv, View, network_config


class Observe(View):
    """Visualizes agent performance using trained models."""

    def __init__(self, args):
        """Initializes observation environment and loads model."""
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        # Set SUMO seed before SUMO starts (read by sumo_env.set_params())
        if args.seed >= 0:
            os.environ["SUMO_EVAL_SEED"] = str(args.seed)
        elif "SUMO_EVAL_SEED" in os.environ:
            del os.environ["SUMO_EVAL_SEED"]

        # Determine GUI override from args if present
        gui_override = None
        if hasattr(args, "headless") and args.headless:
            gui_override = False

        super().__init__(
            type(self).__name__.upper(),
            make_env(
                env=CustomEnvWrapper(CustomEnv("observe", gui_override=gui_override)),
                max_episode_steps=args.max_s,
            ),
        )

        model_pack = args.d.split("/")[-1].split("_model.pack")[0]

        self.network = getattr(
            Networks,
            {
                "DQNAgent": "DeepQNetwork",
                "DoubleDQNAgent": "DeepQNetwork",
                "DuelingDoubleDQNAgent": "DuelingDeepQNetwork",
                "PerDuelingDoubleDQNAgent": "DuelingDeepQNetwork",
            }[model_pack.split("_lr")[0]],
        )(
            device(("cuda:" + args.gpu) if cuda.is_available() else "cpu"),
            float(model_pack.split("_lr")[1].split("_")[0]),
            network_config,
            self.env.observation_space,
            self.env.action_space.n,
        )

        self.network.load(args.d)

        self.obs = np.zeros(self.env.observation_space.shape, dtype=np.float32)

        self.repeat = 0
        self.action = 0
        self.ep = 0

        print()
        print("OBSERVE")
        print()
        [print(arg, "=", getattr(args, arg)) for arg in vars(args)]

        self.max_episodes = args.max_e
        self.log = (args.log, args.log_s, args.log_dir + model_pack)

    def setup(self):
        """Resets environment for new observation episode."""
        self.obs = self.env.reset()
        self.ep_green_times = []
        self.ep_lane_closed_count = 0
        self.ep_rewards = []

    def close(self):
        """Closes the environment."""
        self.env.close()

    def loop(self):
        """Executes inference step."""
        if self.repeat % (HYPER_PARAMS["repeat"] or 1) == 0:
            # Select action using learned policy
            self.action = self.network.actions([self.obs.tolist()])[0]

        self.repeat += 1

        # Execute action in environment
        self.obs, reward, terminated, truncated, info = self.env.step(self.action)
        # Check for episode termination
        done = terminated or truncated

        # Accumulate stats
        if "chosen_green_time_sec" in info:
            self.ep_green_times.append(info["chosen_green_time_sec"])
        if info.get("lane_closed", 0) == 1:
            self.ep_lane_closed_count += 1

        # 'reward' here might be None from step, but it's typically a float.
        # Alternatively we can use info.get("reward", 0.0) if it's there.
        # But 'reward' is returned from step.
        self.ep_rewards.append(reward if reward is not None else 0.0)

        # Log episode statistics
        self.env.log_info_writer(info, done, *self.log)

        # to fix additional episode at the end
        if done:
            self.repeat = 0
            self.ep += 1

            print()
            print("Episode :", self.ep)
            # You can still print the info from the episode that just finished
            [print(k, ":", info[k]) for k in info]

            print("\n--- Episode Statistics ---")
            avg_green = np.mean(self.ep_green_times) if self.ep_green_times else 0.0
            print(f"Average Green Time: {avg_green:.2f} sec")
            print(f"Lane Closed Count: {self.ep_lane_closed_count} times")
            total_reward = info.get(
                "r", np.sum(self.ep_rewards)
            )  # Use accumulated episode reward from Monitor if available
            print(f"Total Episode Reward: {total_reward:.2f}")
            print(f"Average Step Reward: {np.mean(self.ep_rewards):.2f}")
            print("--------------------------\n")

            # Check if we should exit BEFORE resetting the environment
            if bool(self.max_episodes) and self.ep >= self.max_episodes:
                # The environment (and SUMO) from the completed run is still open.
                # We can now safely close it and exit the script.
                self.env.close()  # Gracefully close the TraCI connection
                exit()

            # If we are NOT exiting, only then do we reset for the next episode.
            self.setup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OBSERVE")

    def str2bool(v):
        return v.lower() in ("yes", "y", "true", "t", "1")

    parser.add_argument("-d", type=str, default="", help="Directory", required=True)
    parser.add_argument("-gpu", type=str, default="0", help="GPU #")
    parser.add_argument(
        "-seed", type=int, default=-1, help="SUMO random seed (-1 = random)"
    )
    parser.add_argument(
        "-max_s", type=int, default=0, help="Max steps per episode if > 0, else inf"
    )
    parser.add_argument(
        "-max_e", type=int, default=0, help="Max episodes if > 0, else inf"
    )
    parser.add_argument(
        "-log", type=str2bool, default=False, help="Log csv to ./logs/test/"
    )
    parser.add_argument(
        "-log_s", type=int, default=0, help="Log step if > 0, else episode"
    )
    parser.add_argument(
        "-log_dir", type=str, default="./logs/test/", help="Log directory"
    )

    Observe(parser.parse_args()).run()
