# evaluate.py
import argparse
import os
import random
import sys
import time

import pandas as pd
from colorama import Fore, Style
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- The only parser we need for XML/logs ---
from evaluation.parsers import (
    parse_framework_log,
    parse_sumo_log,
    parse_tripinfo_for_episode_stats,
)
from observe import Observe
from play import Play

STRATEGIES = {
    "DQNAgent": Observe,
    "AlwaysGreenBaseline": Play,
    "FixedCycleBaseline": Play,
    "AlineaDsBaseline": Play,
    "PiAlineaDsBaseline": Play,
}


def run_single_episode(env_instance):
    """Executes a single evaluation episode."""
    # Reset environment for new episode
    obs, info = env_instance.env.reset()
    done = truncated = False
    while not (done or truncated):
        # Select action based on strategy type
        action = (
            env_instance.get_play_action()
            if isinstance(env_instance, Play)
            else env_instance.network.actions([obs.tolist()])[0]
        )
        # Execute action in environment
        obs, _, terminated, truncated, info = env_instance.env.step(action)
        done = terminated
        # Log step info
        env_instance.env.log_info_writer(info, done or truncated, *env_instance.log)


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Run evaluation benchmark for ramp metering strategies."
    )
    # ... (all arguments are the same as before) ...
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        required=True,
        choices=list(STRATEGIES.keys()),
        help="The control strategy to evaluate.",
    )
    parser.add_argument(
        "-n",
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to run for the evaluation.",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=42,
        help="The master seed for reproducibility.",
    )
    parser.add_argument(
        "-d",
        "--model-path",
        type=str,
        default=None,
        help="Path to the trained DRL agent model (.pack file), required for DQNAgent.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./logs/evaluation/results/",
        help="Directory to save the final results CSV.",
    )
    parser.add_argument(
        "-g", "--gpu", type=str, default="0", help="GPU to use for the agent."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run SUMO in headless mode (no GUI) for faster evaluation.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    strategy_class = STRATEGIES[args.strategy]

    # --- Output paths ---
    temp_sumo_log_path = os.path.join(
        args.output_dir, f"temp_sumo_log_{args.strategy}.log"
    )
    tripinfo_root = os.path.join(".", "tripinfo_runs", args.strategy)
    os.makedirs(tripinfo_root, exist_ok=True)

    all_episode_metrics = []
    print(
        f"{Fore.CYAN}--- Starting Evaluation for: {Style.BRIGHT}{args.strategy}{Style.RESET_ALL} ---"
    )

    for episode in tqdm(
        range(args.num_episodes), desc=f"Evaluating {args.strategy}", unit="episode"
    ):
        current_seed = args.master_seed + episode
        os.environ["SUMO_EVAL_SEED"] = str(current_seed)
        os.environ["SUMO_EVAL_LOG_FILE"] = temp_sumo_log_path
        tripinfo_xml_path = os.path.join(
            tripinfo_root, f"tripinfo_{args.strategy}_ep{episode:04d}.xml"
        )
        os.environ["SUMO_TRIPINFO_FILE"] = tripinfo_xml_path
        random.seed(current_seed)

        mock_args_dict = {
            "max_s": 0,
            "max_e": 1,
            "log": True,
            "log_s": 1,
            "log_dir": args.output_dir,
            "headless": args.headless,
            "seed": current_seed,
        }

        if strategy_class == Play:
            mock_args_dict["player"] = args.strategy
            temp_framework_log_path = os.path.join(args.output_dir, args.strategy)
        else:
            if not args.model_path:
                print(
                    f"{Fore.RED}\nError: --model-path is required for DQNAgent.{Style.RESET_ALL}"
                )
                return
            mock_args_dict.update({"d": args.model_path, "gpu": args.gpu})
            model_pack_name = args.model_path.split("/")[-1].split("_model.pack")[0]
            temp_framework_log_path = os.path.join(args.output_dir, model_pack_name)

        mock_args = argparse.Namespace(**mock_args_dict)
        env_instance = strategy_class(mock_args)

        run_single_episode(env_instance)

        scenario_info = env_instance.env.get_env().get_scenario_info()
        env_instance.close()

        # --- Parsing is now simpler ---
        # Allow SUMO to flush tripinfo output before parsing
        time.sleep(3)
        # Extract trip stats from XML
        trip_and_emission_stats = parse_tripinfo_for_episode_stats(tripinfo_xml_path)
        # Parse SUMO logs
        sumo_stats = parse_sumo_log(temp_sumo_log_path)
        # Parse framework-specific logs
        framework_stats = parse_framework_log(
            temp_framework_log_path, spillback_threshold=20
        )

        combined_stats = {
            "episode_id": episode,
            "seed": current_seed,
            **scenario_info,
            **trip_and_emission_stats,
            **sumo_stats,
            **framework_stats,
        }
        all_episode_metrics.append(combined_stats)

        # --- Cleanup is simpler ---
        if os.path.exists(temp_sumo_log_path):
            os.remove(temp_sumo_log_path)
        if os.path.exists(temp_framework_log_path):
            os.remove(temp_framework_log_path)

    if all_episode_metrics:
        results_df = pd.DataFrame(all_episode_metrics)
        # For DRL strategies, include the model identifier so different
        # checkpoints (e.g. lane_control vs the prior ramp-only variant) don't
        # silently overwrite each other.
        if strategy_class == Play:
            results_basename = f"results_{args.strategy}.csv"
        else:
            model_tag = os.path.basename(os.path.dirname(args.model_path)) or "model"
            results_basename = f"results_{args.strategy}_{model_tag}.csv"
        final_csv_path = os.path.join(args.output_dir, results_basename)
        results_df.to_csv(final_csv_path, index=False, float_format="%.4f")
        print(
            f"\n{Fore.GREEN}--- Evaluation Complete: {args.strategy} ---{Style.RESET_ALL}"
        )
        print(f"Results for {args.num_episodes} episodes saved to: {final_csv_path}")
    else:
        print(
            f"\n{Fore.YELLOW}Warning: No metrics were collected. Evaluation may have failed.{Style.RESET_ALL}"
        )


if __name__ == "__main__":
    main()
