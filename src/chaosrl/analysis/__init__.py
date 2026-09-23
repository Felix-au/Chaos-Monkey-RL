"""Analysis and reporting package."""
from chaosrl.analysis.statistics import bootstrap_ci, compare_methods_statistically
from chaosrl.analysis.plotting import (
    plot_discovery_curves,
    plot_damage_vs_discovery,
    plot_simple_vs_compound,
    plot_ablations,
)
from chaosrl.analysis.report_generator import generate_report

__all__ = [
    "bootstrap_ci",
    "compare_methods_statistically",
    "plot_discovery_curves",
    "plot_damage_vs_discovery",
    "plot_simple_vs_compound",
    "plot_ablations",
    "generate_report",
]
