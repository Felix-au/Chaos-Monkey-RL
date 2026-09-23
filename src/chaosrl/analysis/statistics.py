"""Statistical evaluation routines: 95% bootstrap confidence intervals, Mann-Whitney U tests, and effect sizes."""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import stats


def bootstrap_ci(
    data: List[float],
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
) -> Tuple[float, float, float]:
    """Computes mean and 95% bootstrap confidence intervals [ci_lower, ci_upper]."""
    arr = np.array(data, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    mean_val = float(np.mean(arr))
    boot_means = []
    rng = np.random.default_rng(42)

    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))

    return mean_val, lower, upper


def compare_methods_statistically(
    rl_data: List[float], baseline_data: List[float]
) -> Dict[str, Any]:
    """Computes Mann-Whitney U test, p-value, Cohen's d, and rank-biserial correlation."""
    rl_arr = np.array(rl_data, dtype=np.float64)
    bl_arr = np.array(baseline_data, dtype=np.float64)

    if len(rl_arr) < 2 or len(bl_arr) < 2:
        return {
            "u_stat": 0.0,
            "p_value": 1.0,
            "significant_at_05": False,
            "cohens_d": 0.0,
            "rank_biserial": 0.0,
        }

    # Mann-Whitney U test (two-sided)
    try:
        res = stats.mannwhitneyu(rl_arr, bl_arr, alternative="two-sided")
        u_stat = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        u_stat = 0.0
        p_val = 1.0

    # Cohen's d effect size
    s1 = np.std(rl_arr, ddof=1)
    s2 = np.std(bl_arr, ddof=1)
    pooled_sd = np.sqrt(((len(rl_arr) - 1) * s1**2 + (len(bl_arr) - 1) * s2**2) / max(1, len(rl_arr) + len(bl_arr) - 2))
    cohens_d = float((np.mean(rl_arr) - np.mean(bl_arr)) / max(1e-6, pooled_sd))

    # Rank-biserial correlation = 1 - (2*U) / (n1 * n2)
    n1, n2 = len(rl_arr), len(bl_arr)
    rank_biserial = float(1.0 - (2.0 * u_stat) / (n1 * n2))

    return {
        "u_stat": u_stat,
        "p_value": p_val,
        "significant_at_05": bool(p_val < 0.05),
        "cohens_d": cohens_d,
        "rank_biserial": rank_biserial,
    }
