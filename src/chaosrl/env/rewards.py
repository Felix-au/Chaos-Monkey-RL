"""Reward function formulations for the ChaosRL environment:
R1: Pure discovery novelty (+1 per newly found weakness)
R2: Novelty minus damage penalty (safety-constrained chaos)
R3: Novelty minus damage + exploration bonus & repeat penalty
"""

from typing import List
from chaosrl.simulator.models import Weakness


def compute_reward(
    reward_variant: str,
    newly_found: List[Weakness],
    damage: float,
    slo_violated: bool,
    is_repeat_action: bool,
    partial_degradation: bool = False,
) -> float:
    """Compute step reward according to configured variant.
    
    Args:
        reward_variant: 'R1', 'R2', or 'R3'
        newly_found: Planted weaknesses newly discovered this step
        damage: Accumulated damage incurred during the experiment
        slo_violated: Whether the experiment breached system SLO
        is_repeat_action: Whether this exact action was already tried
        partial_degradation: Sub-SLO latency/error increase (information gain)
    """
    novelty_score = float(len(newly_found))

    if reward_variant == "R1":
        # Pure weakness discovery count
        return novelty_score

    elif reward_variant == "R2":
        # Discovery penalized by collateral damage (encourages minimal-damage discovery)
        damage_penalty = min(2.0, damage * 0.15)
        return novelty_score - damage_penalty

    elif reward_variant == "R3":
        # R2 + penalty for repeating redundant faults + small bonus for informative sub-SLO probing
        damage_penalty = min(2.0, damage * 0.15)
        repeat_penalty = 0.25 if is_repeat_action else 0.0
        info_bonus = 0.05 if (partial_degradation and not slo_violated and not is_repeat_action) else 0.0
        return novelty_score - damage_penalty - repeat_penalty + info_bonus

    else:
        raise ValueError(f"Unknown reward variant: {reward_variant}")
