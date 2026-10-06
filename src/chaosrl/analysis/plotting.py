"""Plotting and visualization routines producing publication-quality static (PNG/SVG) and interactive (HTML) figures."""

import os
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    """Plot cumulative weaknesses found vs experiment number with confidence bands (PNG & SVG)."""
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
    svg_path = output_path.replace(".png", ".svg")
    fig.savefig(svg_path)
    plt.close(fig)


def plot_discovery_curves_interactive(
    curves_df: pd.DataFrame,
    output_path: str = "figures/fig1_discovery_curves.html",
):
    """Generates an interactive Plotly chart for weakness discovery trajectories with confidence intervals."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig = go.Figure()

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

        # Convert hex color to rgba for shading
        h = color.lstrip("#")
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

        # Upper bound
        fig.add_trace(go.Scatter(
            x=x, y=y + err,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Lower bound with fill
        fig.add_trace(go.Scatter(
            x=x, y=np.maximum(0, y - err),
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.18)",
            showlegend=False,
            hoverinfo='skip'
        ))
        # Main trace
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines+markers',
            name=label,
            line=dict(color=color, width=3),
            marker=dict(size=4),
            customdata=err,
            hovertemplate=f"<b>{label}</b><br>Experiment: %{{x}}<br>Weaknesses Found: %{{y:.2f}} ± %{{customdata:.2f}}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text="Figure 1: Weakness Discovery Trajectory Across Experiment Budget", font=dict(size=16, color="#f8fafc")),
        xaxis=dict(title="Experiment Number (Budget Used)", gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(title="Cumulative Weaknesses Found", gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.2)"),
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        margin=dict(l=60, r=40, t=60, b=60),
    )

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return fig


def plot_damage_vs_discovery(
    summary_df: pd.DataFrame,
    output_path: str = "figures/fig2_damage_vs_discovery.png",
):
    """Plot safety tradeoff: Total Damage vs Weaknesses Discovered (PNG & SVG)."""
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
    svg_path = output_path.replace(".png", ".svg")
    fig.savefig(svg_path)
    plt.close(fig)


def plot_damage_vs_discovery_interactive(
    summary_df: pd.DataFrame,
    output_path: str = "figures/fig2_damage_vs_discovery.html",
):
    """Plot safety tradeoff: Total Damage vs Weaknesses Discovered (Interactive Plotly)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig = go.Figure()

    colors = {
        "ChaosRL-PPO": "#1f77b4",
        "Bandit-UCB": "#2ca02c",
        "Bandit-EPSILON_GREEDY": "#8c564b",
        "TabularQ": "#9467bd",
        "RoundRobin": "#ff7f0e",
        "RandomAll": "#e377c2",
        "ChaosMonkey": "#d62728",
    }

    methods = summary_df["method"].unique()
    for method in methods:
        sub = summary_df[summary_df["method"] == method]
        x_mean = sub["total_damage"].mean()
        x_err = 1.96 * sub["total_damage"].sem() if len(sub) > 1 else 0.0
        y_mean = sub["weaknesses_found"].mean()
        y_err = 1.96 * sub["weaknesses_found"].sem() if len(sub) > 1 else 0.0

        color = colors.get(method, "#333333")
        label = "ChaosRL (Ours)" if "PPO" in method else method

        fig.add_trace(go.Scatter(
            x=[x_mean],
            y=[y_mean],
            mode='markers+text',
            name=label,
            text=[label],
            textposition="top right",
            marker=dict(size=14, color=color, symbol='circle'),
            error_x=dict(type='data', array=[x_err], visible=True, color=color),
            error_y=dict(type='data', array=[y_err], visible=True, color=color),
            hovertemplate=f"<b>{label}</b><br>Total Damage: %{{x:.2f}} ± {x_err:.2f}<br>Weaknesses Found: %{{y:.2f}} ± {y_err:.2f}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text="Figure 2: Efficiency & Safety Tradeoff: Discovery vs Incurred Damage", font=dict(size=16, color="#f8fafc")),
        xaxis=dict(title="Total Collateral Damage (SLO Overrun / Error Cost)", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Weaknesses Discovered", gridcolor="rgba(255,255,255,0.1)"),
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        margin=dict(l=60, r=40, t=60, b=60),
    )

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return fig


def plot_simple_vs_compound(
    summary_df: pd.DataFrame,
    output_path: str = "figures/fig3_simple_vs_compound.png",
):
    """Bar chart comparing simple vs compound weaknesses discovered (PNG & SVG)."""
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
    svg_path = output_path.replace(".png", ".svg")
    fig.savefig(svg_path)
    plt.close(fig)


def plot_simple_vs_compound_interactive(
    summary_df: pd.DataFrame,
    output_path: str = "figures/fig3_simple_vs_compound.html",
):
    """Bar chart comparing simple vs compound weaknesses discovered (Interactive Plotly)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grouped = summary_df.groupby("method")[["simple_weaknesses", "compound_weaknesses"]].mean().reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=grouped["method"],
        y=grouped["simple_weaknesses"],
        name="Simple Weaknesses",
        marker_color="#4a90e2",
        hovertemplate="<b>%{x}</b><br>Simple Found: %{y:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        x=grouped["method"],
        y=grouped["compound_weaknesses"],
        name="Compound Weaknesses",
        marker_color="#e94e77",
        hovertemplate="<b>%{x}</b><br>Compound Found: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        barmode='group',
        title=dict(text="Figure 3: Simple vs Compound Weakness Discovery Breakdown", font=dict(size=16, color="#f8fafc")),
        xaxis=dict(title="Method", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Mean Weaknesses Found", gridcolor="rgba(255,255,255,0.1)"),
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        margin=dict(l=60, r=40, t=60, b=60),
    )

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return fig


def plot_ablations(
    ablation_df: pd.DataFrame,
    output_path: str = "figures/fig4_ablations.png",
):
    """Plot comparisons for reward variants, history features, and budget (PNG & SVG)."""
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
    svg_path = output_path.replace(".png", ".svg")
    fig.savefig(svg_path)
    plt.close(fig)


def plot_ablations_interactive(
    ablation_df: pd.DataFrame,
    output_path: str = "figures/fig4_ablations.html",
):
    """Plot comparisons for reward variants and generalization (Interactive Subplots)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Reward Formulations (R1 vs R2 vs R3)", "Generalization: 20-Svc Model on 100-Svc Test"),
        specs=[[{"secondary_y": True}, {"secondary_y": False}]]
    )

    reward_data = ablation_df[ablation_df["ablation_type"] == "reward"]
    if not reward_data.empty:
        r_grouped = reward_data.groupby("setting").agg({"weaknesses_found": "mean", "total_damage": "mean"}).reset_index()
        fig.add_trace(
            go.Bar(
                x=r_grouped["setting"],
                y=r_grouped["weaknesses_found"],
                name="Weaknesses Found",
                marker_color="#2ca02c",
                hovertemplate="Reward %{x}<br>Weaknesses: %{y:.2f}<extra></extra>"
            ),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=r_grouped["setting"],
                y=r_grouped["total_damage"],
                name="Total Damage",
                mode="lines+markers",
                line=dict(color="#ef4444", width=3),
                marker=dict(size=8),
                hovertemplate="Reward %{x}<br>Damage: %{y:.2f}<extra></extra>"
            ),
            row=1, col=1, secondary_y=True
        )

    gen_data = ablation_df[ablation_df["ablation_type"] == "generalization"]
    if not gen_data.empty:
        g_grouped = gen_data.groupby("setting")["weaknesses_found"].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=g_grouped["setting"],
                y=g_grouped["weaknesses_found"],
                name="Scale Generalization",
                marker_color=["#1f77b4", "#ff7f0e"],
                hovertemplate="Setting %{x}<br>Weaknesses: %{y:.2f}<extra></extra>"
            ),
            row=1, col=2
        )

    fig.update_layout(
        title=dict(text="Figure 4: Ablation Studies & Generalization", font=dict(size=16, color="#f8fafc")),
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        showlegend=True,
        legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        margin=dict(l=60, r=40, t=60, b=60),
    )
    fig.update_yaxes(title_text="Weaknesses Found", row=1, col=1, secondary_y=False, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(title_text="Total Damage", row=1, col=1, secondary_y=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(title_text="Weaknesses Found", row=1, col=2, gridcolor="rgba(255,255,255,0.1)")

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return fig


def generate_all_plots(
    curves_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ablation_df: pd.DataFrame = None,
    fig_dir: str = "figures",
):
    """Master generator producing PNG, SVG, and interactive HTML files for all figures."""
    os.makedirs(fig_dir, exist_ok=True)
    
    # Static PNG + SVG
    plot_discovery_curves(curves_df, os.path.join(fig_dir, "fig1_discovery_curves.png"))
    plot_damage_vs_discovery(summary_df, os.path.join(fig_dir, "fig2_damage_vs_discovery.png"))
    plot_simple_vs_compound(summary_df, os.path.join(fig_dir, "fig3_simple_vs_compound.png"))

    # Interactive HTML
    plot_discovery_curves_interactive(curves_df, os.path.join(fig_dir, "fig1_discovery_curves.html"))
    plot_damage_vs_discovery_interactive(summary_df, os.path.join(fig_dir, "fig2_damage_vs_discovery.html"))
    plot_simple_vs_compound_interactive(summary_df, os.path.join(fig_dir, "fig3_simple_vs_compound.html"))

    if ablation_df is not None and not ablation_df.empty:
        plot_ablations(ablation_df, os.path.join(fig_dir, "fig4_ablations.png"))
        plot_ablations_interactive(ablation_df, os.path.join(fig_dir, "fig4_ablations.html"))

