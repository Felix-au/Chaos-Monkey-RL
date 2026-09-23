"""Plotting and visualization routines producing publication-quality figures."""

import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_plot_style():
    """Sets a clean, modern aesthetic style for scientific figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
        "axes.edgecolor": "#cccccc",
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })


def plot_discovery_curves(
    curves_df: pd.DataFrame,
    output_path: str = "figures/fig1_discovery_curves.png",
):
    """Plot cumulative weaknesses found vs experiment number with confidence bands."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Color palette
    colors = {
        "ChaosRL-PPO": "#1f77b4",
        "Bandit-UCB": "#2ca02c",
        "Bandit-EPSILON_GREEDY": "#8c564b",
        "TabularQ": "#9467bd",
        "RoundRobin": "#ff7f0e",
        "RandomAll": "#e377c2",
        "ChaosMonkey": "#d62728",
    }

    methods = curves_df["method"].unique()
    for method in methods:
        sub = curves_df[curves_df["method"] == method]
        grouped = sub.groupby("experiment")["weaknesses_found"]
        mean_vals = grouped.mean()
        sem_vals = grouped.sem().fillna(0.0)

        x = mean_vals.index.values
        y = mean_vals.values
        err = 1.96 * sem_vals.values

        color = colors.get(method, "#333333")
        label = "ChaosRL (MaskablePPO)" if "PPO" in method else method
        ax.plot(x, y, label=label, color=color)
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.18)

    ax.set_title("Weakness Discovery Trajectory Across Experiment Budget")
    ax.set_xlabel("Experiment Number (Budget Used)")
    ax.set_ylabel("Cumulative Weaknesses Found")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_damage_vs_discovery(
    summary_df: pd.DataFrame,
    output_path: str = "figures/fig2_damage_vs_discovery.png",
):
    """Plot safety tradeoff: Total Damage vs Weaknesses Discovered."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    methods = summary_df["method"].unique()
    colors = {
        "ChaosRL-PPO": "#1f77b4",
        "Bandit-UCB": "#2ca02c",
        "Bandit-EPSILON_GREEDY": "#8c564b",
        "TabularQ": "#9467bd",
        "RoundRobin": "#ff7f0e",
        "RandomAll": "#e377c2",
        "ChaosMonkey": "#d62728",
    }

    for method in methods:
        sub = summary_df[summary_df["method"] == method]
        x_mean = sub["total_damage"].mean()
        x_err = 1.96 * sub["total_damage"].sem() if len(sub) > 1 else 0.0
        y_mean = sub["weaknesses_found"].mean()
        y_err = 1.96 * sub["weaknesses_found"].sem() if len(sub) > 1 else 0.0

        color = colors.get(method, "#333333")
        label = "ChaosRL (Ours)" if "PPO" in method else method
        ax.errorbar(
            x_mean, y_mean, xerr=x_err, yerr=y_err,
            fmt="o", markersize=9, color=color, capsize=4, label=label
        )
        ax.annotate(
            label, (x_mean, y_mean),
            textcoords="offset points", xytext=(8, -4), fontsize=9
        )

    ax.set_title("Efficiency & Safety Tradeoff: Discovery vs Incurred Damage")
    ax.set_xlabel("Total Collateral Damage (SLO Overrun / Error Cost)")
    ax.set_ylabel("Weaknesses Discovered")
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_simple_vs_compound(
    summary_df: pd.DataFrame,
    output_path: str = "figures/fig3_simple_vs_compound.png",
):
    """Bar chart comparing simple vs compound weaknesses discovered."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    grouped = summary_df.groupby("method")[["simple_weaknesses", "compound_weaknesses"]].mean()
    methods = grouped.index.tolist()
    x = np.arange(len(methods))
    width = 0.35

    ax.bar(x - width/2, grouped["simple_weaknesses"], width, label="Simple Weaknesses", color="#4a90e2")
    ax.bar(x + width/2, grouped["compound_weaknesses"], width, label="Compound Weaknesses", color="#e94e77")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_title("Simple vs Compound Weakness Discovery Breakdown")
    ax.set_ylabel("Mean Weaknesses Found")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_ablations(
    ablation_df: pd.DataFrame,
    output_path: str = "figures/fig4_ablations.png",
):
    """Plot comparisons for reward variants, history features, and budget."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    set_plot_style()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # 1. Reward variants R1 vs R2 vs R3
    reward_data = ablation_df[ablation_df["ablation_type"] == "reward"]
    if not reward_data.empty:
        r_grouped = reward_data.groupby("setting")[["weaknesses_found", "total_damage"]].mean()
        r_settings = r_grouped.index.tolist()
        x = np.arange(len(r_settings))
        axs[0].bar(x - 0.18, r_grouped["weaknesses_found"], 0.36, label="Weaknesses Found", color="#2ca02c")
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(r_settings)
        axs[0].set_title("Reward Formulations (R1 vs R2 vs R3)")
        axs[0].set_ylabel("Weaknesses Found")
        
        # Second y-axis for damage
        ax_dam = axs[0].twinx()
        ax_dam.plot(x, r_grouped["total_damage"], "ro--", label="Total Damage")
        ax_dam.set_ylabel("Damage", color="red")
        ax_dam.grid(False)

    # 2. Generalization 20 to 100 services
    gen_data = ablation_df[ablation_df["ablation_type"] == "generalization"]
    if not gen_data.empty:
        g_grouped = gen_data.groupby("setting")["weaknesses_found"].mean()
        g_settings = g_grouped.index.tolist()
        axs[1].bar(g_settings, g_grouped.values, color=["#1f77b4", "#ff7f0e"])
        axs[1].set_title("Generalization: 20-Svc Model on 100-Svc Test")
        axs[1].set_ylabel("Weaknesses Found")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
