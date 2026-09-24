"""Training pipeline for ChaosRL agents using MaskablePPO.
Trains policies across randomized microservice architectures to learn generalizable fault-selection strategies.
"""

import os
import time
import argparse
import json
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback

from chaosrl.env import ChaosEnv


class MetricLoggingCallback(BaseCallback):
    """Tracks episode rewards, weakness discovery, and damage during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_discoveries: List[int] = []
        self.episode_damages: List[float] = []
        self.timesteps: List[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "total_found" in info and self.locals.get("dones", [False])[0]:
                self.episode_discoveries.append(info["total_found"])
                self.episode_damages.append(info.get("total_damage", 0.0))
                self.timesteps.append(self.num_timesteps)
        return True


def train_single_seed(
    seed: int,
    num_services: int = 20,
    num_nodes: int = 6,
    budget: int = 40,
    total_timesteps: int = 6000,
    reward_variant: str = "R2",
    include_history: bool = True,
    compound_mode: bool = False,
    output_dir: str = "results/checkpoints",
) -> Dict[str, Any]:
    """Train MaskablePPO on a distribution of synthetic systems using a distinct seed."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    # Note: By omitting system_seed, reset() generates new random architectures every episode
    env = ChaosEnv(
        num_services=num_services,
        num_nodes=num_nodes,
        budget=budget,
        reward_variant=reward_variant,
        include_history=include_history,
        compound_mode=compound_mode,
        mask_already_tried=True,
    )

    metric_cb = MetricLoggingCallback()

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=min(budget * 5, 256),
        batch_size=32,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=seed,
        verbose=0,
    )

    start_time = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=metric_cb)
    elapsed = time.perf_counter() - start_time

    model_path = os.path.join(output_dir, f"ppo_s{num_services}_seed{seed}_{reward_variant}.zip")
    model.save(model_path)

    training_stats = {
        "seed": seed,
        "num_services": num_services,
        "timesteps": total_timesteps,
        "elapsed_seconds": elapsed,
        "model_path": model_path,
        "final_mean_discovery": float(np.mean(metric_cb.episode_discoveries[-10:])) if metric_cb.episode_discoveries else 0.0,
        "final_mean_damage": float(np.mean(metric_cb.episode_damages[-10:])) if metric_cb.episode_damages else 0.0,
    }

    # Save log CSV
    log_df = pd.DataFrame({
        "timestep": metric_cb.timesteps,
        "discoveries": metric_cb.episode_discoveries,
        "damage": metric_cb.episode_damages,
    })
    log_csv = os.path.join("results/logs", f"train_log_seed{seed}_{reward_variant}.csv")
    log_df.to_csv(log_csv, index=False)

    return training_stats


def run_training_suite(
    seeds: List[int] = [42, 43, 44, 45, 46],
    num_services: int = 20,
    timesteps: int = 5000,
    reward_variant: str = "R2",
    compound_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Runs multi-seed training suite and records variance across training seeds."""
    all_stats = []
    for s in seeds:
        print(f"[Training] Training MaskablePPO with seed {s} ({timesteps} steps, {reward_variant})...")
        stat = train_single_seed(
            seed=s,
            num_services=num_services,
            total_timesteps=timesteps,
            reward_variant=reward_variant,
            compound_mode=compound_mode,
        )
        all_stats.append(stat)
        print(f"  Finished in {stat['elapsed_seconds']:.1f}s - Final Mean Weaknesses Found: {stat['final_mean_discovery']:.2f}")

    with open("results/training_summary.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--services", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=5000)
    parser.add_argument("--reward", type=str, default="R2")
    parser.add_argument("--compound", action="store_true")
    args = parser.parse_args()

    run_training_suite(
        seeds=args.seeds,
        num_services=args.services,
        timesteps=args.timesteps,
        reward_variant=args.reward,
        compound_mode=args.compound,
    )
