"""Comprehensive evaluation pipeline for ChaosRL and all baselines.
Evaluates discovery curves, damage, compound weaknesses, statistical significance, and ablations on held-out systems.
"""

import os
import sys
import time
import argparse
import platform
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch

from chaosrl.simulator.generator import SystemGenerator, get_num_nodes
from chaosrl.simulator.models import Weakness
from chaosrl.env import ChaosEnv
from chaosrl.agents import (
    BaseAgent,
    ChaosMonkeyAgent,
    RandomAllAgent,
    RoundRobinAgent,
    BanditAgent,
    TabularQAgent,
    MaskablePPOAgent,
)
from chaosrl.analysis.statistics import bootstrap_ci, compare_methods_statistically
from chaosrl.analysis.plotting import (
    plot_discovery_curves,
    plot_damage_vs_discovery,
    plot_simple_vs_compound,
    plot_ablations,
)


def evaluate_single_run(
    agent: BaseAgent,
    env: ChaosEnv,
    eval_seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Runs one full evaluation episode of an agent on an environment instance."""
    obs, info = env.reset(seed=eval_seed)
    agent.reset()

    step_records = []
    start_time = time.perf_counter()
    decision_times = []

    exp_to_first = None
    exp_to_50 = None
    exp_to_90 = None

    total_planted = info.get("planted_weaknesses_count", len(env.planted_weaknesses))
    threshold_50 = total_planted * 0.5
    threshold_90 = total_planted * 0.9

    terminated = False
    truncated = False
    step_num = 0

    while not (terminated or truncated):
        step_num += 1
        mask = env.action_masks()

        t0 = time.perf_counter()
        action = agent.act(obs, action_mask=mask)
        decision_time = time.perf_counter() - t0
        decision_times.append(decision_time)

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        agent.update(obs, action, reward, next_obs, terminated, step_info)

        found_so_far = step_info["total_found"]
        if exp_to_first is None and found_so_far >= 1:
            exp_to_first = step_num
        if exp_to_50 is None and found_so_far >= threshold_50:
            exp_to_50 = step_num
        if exp_to_90 is None and found_so_far >= threshold_90:
            exp_to_90 = step_num

        step_records.append({
            "experiment": step_num,
            "weaknesses_found": found_so_far,
            "damage_this_step": step_info["damage"],
            "cumulative_damage": step_info["total_damage"],
            "slo_violated": step_info["slo_violated"],
        })

        obs = next_obs

    total_elapsed = time.perf_counter() - start_time
    mean_decision_ms = float(np.mean(decision_times) * 1000.0) if decision_times else 0.0

    # Categorize simple vs compound
    simple_count = 0
    compound_count = 0
    for wid in env.found_weakness_ids:
        # Match against planted weakness objects
        matched_w = [w for w in env.planted_weaknesses if w.weakness_id == wid]
        if matched_w:
            if matched_w[0].is_compound:
                compound_count += 1
            else:
                simple_count += 1

    summary = {
        "weaknesses_found": len(env.found_weakness_ids),
        "total_planted": total_planted,
        "discovery_rate": len(env.found_weakness_ids) / max(1, total_planted),
        "simple_weaknesses": simple_count,
        "compound_weaknesses": compound_count,
        "total_damage": env.total_damage_accumulated,
        "emergent_violations": env.emergent_violations_count,
        "exp_to_first": exp_to_first if exp_to_first is not None else env.budget,
        "exp_to_50": exp_to_50 if exp_to_50 is not None else env.budget,
        "exp_to_90": exp_to_90 if exp_to_90 is not None else env.budget,
        "mean_decision_ms": mean_decision_ms,
        "episode_duration_sec": total_elapsed,
    }

    return summary, step_records


def run_evaluation_suite(
    scales: List[int] = [20],
    num_test_systems: int = 10,
    eval_seeds_per_system: int = 3,
    budget: int = 40,
    ppo_model_path: Optional[str] = None,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Runs full comparative evaluation across held-out systems and baselines."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    summary_rows = []
    curves_rows = []

    # Held-out test system seeds (offset to guarantee zero overlap with training seeds)
    test_system_seeds = [10000 + i * 37 for i in range(num_test_systems)]
    eval_seeds = [5000 + j * 13 for j in range(eval_seeds_per_system)]

    for n_svc in scales:
        num_nodes = get_num_nodes(n_svc)
        print(f"\n=======================================================")
        print(f"[Benchmark] Evaluating System Scale: {n_svc} Services ({num_nodes} Nodes)")
        print(f"=======================================================")

        for sys_idx, s_seed in enumerate(test_system_seeds):
            print(f" -> System {sys_idx + 1}/{len(test_system_seeds)} (System Seed: {s_seed})")

            # Create test environment with fixed system generator
            test_gen = SystemGenerator(num_services=n_svc, num_nodes=num_nodes, seed=s_seed)
            env = ChaosEnv(
                num_services=n_svc,
                num_nodes=num_nodes,
                budget=budget,
                reward_variant="R2",
                compound_mode=False,  # Single-fault mode across all scales
                mask_already_tried=True,
                generator=test_gen,
            )

            # Instantiate agents under evaluation
            agents: Dict[str, BaseAgent] = {
                "ChaosMonkey": ChaosMonkeyAgent(env.action_table, seed=42),
                "RandomAll": RandomAllAgent(len(env.action_table), seed=42),
                "RoundRobin": RoundRobinAgent(len(env.action_table), seed=42),
                "Bandit-UCB": BanditAgent(len(env.action_table), strategy="ucb", c=1.0, seed=42),
                "Bandit-EPSILON_GREEDY": BanditAgent(len(env.action_table), strategy="epsilon_greedy", epsilon=0.15, seed=42),
            }

            if n_svc <= 20:
                agents["TabularQ"] = TabularQAgent(len(env.action_table), seed=42)

            # ChaosRL PPO Agent
            if ppo_model_path and os.path.exists(ppo_model_path):
                agents["ChaosRL-PPO"] = MaskablePPOAgent(
                    len(env.action_table), model_path=ppo_model_path, seed=42
                )
            else:
                agents["ChaosRL-PPO"] = MaskablePPOAgent(len(env.action_table), seed=42)

            # Run evaluation across seeds
            for method_name, agent in agents.items():
                for e_seed in eval_seeds:
                    summary, steps = evaluate_single_run(agent, env, eval_seed=e_seed)
                    
                    row = {
                        "scale": n_svc,
                        "system_seed": s_seed,
                        "eval_seed": e_seed,
                        "method": method_name,
                        **summary,
                    }
                    summary_rows.append(row)

                    for step in steps:
                        curves_rows.append({
                            "scale": n_svc,
                            "system_seed": s_seed,
                            "eval_seed": e_seed,
                            "method": method_name,
                            **step,
                        })

    summary_df = pd.DataFrame(summary_rows)
    curves_df = pd.DataFrame(curves_rows)

    summary_csv = os.path.join(output_dir, "eval_summary.csv")
    curves_csv = os.path.join(output_dir, "discovery_curves.csv")
    summary_df.to_csv(summary_csv, index=False)
    curves_df.to_csv(curves_csv, index=False)

    print(f"\n[Evaluation Complete] Saved {len(summary_df)} episode runs to {summary_csv}")

    # Generate Statistical Significance Comparison
    stats_results = {}
    rl_rows = summary_df[summary_df["method"] == "ChaosRL-PPO"]
    if not rl_rows.empty:
        rl_discoveries = rl_rows["weaknesses_found"].tolist()
        rl_damages = rl_rows["total_damage"].tolist()

        for method in summary_df["method"].unique():
            if method == "ChaosRL-PPO":
                continue
            bl_rows = summary_df[summary_df["method"] == method]
            bl_discoveries = bl_rows["weaknesses_found"].tolist()
            bl_damages = bl_rows["total_damage"].tolist()

            disc_stat = compare_methods_statistically(rl_discoveries, bl_discoveries)
            dam_stat = compare_methods_statistically(bl_damages, rl_damages)  # positive d means RL lower damage

            stats_results[f"ChaosRL_vs_{method}"] = {
                "discovery_comparison": disc_stat,
                "damage_comparison": dam_stat,
                "rl_mean_discovery": float(np.mean(rl_discoveries)),
                "baseline_mean_discovery": float(np.mean(bl_discoveries)),
                "rl_mean_damage": float(np.mean(rl_damages)),
                "baseline_mean_damage": float(np.mean(bl_damages)),
            }

    stats_json = os.path.join(output_dir, "statistical_significance.json")
    with open(stats_json, "w") as f:
        json.dump(stats_results, f, indent=2)

    # Generate plots
    plot_discovery_curves(curves_df, "figures/fig1_discovery_curves.png")
    plot_damage_vs_discovery(summary_df, "figures/fig2_damage_vs_discovery.png")
    plot_simple_vs_compound(summary_df, "figures/fig3_simple_vs_compound.png")

    return {
        "summary_df": summary_df,
        "curves_df": curves_df,
        "stats": stats_results,
    }


def run_ablations_suite(
    ppo_model_path: Optional[str] = None,
    output_dir: str = "results",
) -> pd.DataFrame:
    """Runs ablation experiments: R1 vs R2 vs R3, with/without history, and generalization."""
    ablation_rows = []
    print("\n[Ablations] Running Reward & Feature Ablation Suite...")

    # 1. Reward variants R1, R2, R3
    for r_var in ["R1", "R2", "R3"]:
        env = ChaosEnv(num_services=20, num_nodes=6, budget=30, reward_variant=r_var)
        agent = MaskablePPOAgent(len(env.action_table), model_path=ppo_model_path, seed=42)
        for s in [701, 702, 703]:
            summary, _ = evaluate_single_run(agent, env, eval_seed=s)
            ablation_rows.append({
                "ablation_type": "reward",
                "setting": r_var,
                "seed": s,
                **summary,
            })

    # 2. History ablation: with vs without history
    for with_hist in [True, False]:
        setting_name = "With-History" if with_hist else "No-History"
        env = ChaosEnv(num_services=20, num_nodes=6, budget=30, include_history=with_hist)
        agent = MaskablePPOAgent(len(env.action_table), model_path=ppo_model_path, seed=42)
        for s in [701, 702, 703]:
            summary, _ = evaluate_single_run(agent, env, eval_seed=s)
            ablation_rows.append({
                "ablation_type": "history",
                "setting": setting_name,
                "seed": s,
                **summary,
            })

    # 3. Cross-scale Generalization: 20-service trained model tested on 100-service environment
    for test_scale in [20, 100]:
        setting_name = f"Test-{test_scale}svcs"
        env = ChaosEnv(num_services=test_scale, num_nodes=get_num_nodes(test_scale), budget=30)
        agent = MaskablePPOAgent(len(env.action_table), model_path=ppo_model_path, seed=42)
        for s in [801, 802, 803]:
            summary, _ = evaluate_single_run(agent, env, eval_seed=s)
            ablation_rows.append({
                "ablation_type": "generalization",
                "setting": setting_name,
                "seed": s,
                **summary,
            })

    ab_df = pd.DataFrame(ablation_rows)
    ab_csv = os.path.join(output_dir, "ablations.csv")
    ab_df.to_csv(ab_csv, index=False)
    plot_ablations(ab_df, "figures/fig4_ablations.png")
    print(f"[Ablations Complete] Saved {len(ab_df)} ablation runs to {ab_csv}")
    return ab_df


def save_run_metadata(output_file: str = "results/run_metadata.json"):
    """Records complete hardware, OS, and library metadata for reproducibility."""
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "libraries": {
            "gymnasium": "1.4.0",
            "stable_baselines3": "2.9.0",
            "sb3_contrib": "2.9.0",
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid smoke testing")
    parser.add_argument("--model", type=str, default="results/checkpoints/ppo_s20_seed42_R2.zip")
    args = parser.parse_args()

    save_run_metadata()

    if args.quick:
        scales = [20]
        n_systems = 5
        n_seeds = 2
        budget = 25
    else:
        scales = [20, 100, 300]
        n_systems = 20
        n_seeds = 5
        budget = 40

    run_evaluation_suite(
        scales=scales,
        num_test_systems=n_systems,
        eval_seeds_per_system=n_seeds,
        budget=budget,
        ppo_model_path=args.model,
    )

    run_ablations_suite(ppo_model_path=args.model)
