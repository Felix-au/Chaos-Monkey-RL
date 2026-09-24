"""Evaluation metrics for chaos experiments: weakness discovery matching, damage tracking, and emergent failure identification."""

from typing import List, Tuple, Set
from chaosrl.simulator.models import Fault, Weakness


def evaluate_discovery(
    faults: List[Fault],
    slo_violated: bool,
    planted_weaknesses: List[Weakness],
    already_found_ids: Set[str],
) -> Tuple[List[Weakness], bool]:
    """Determines which planted weaknesses (if any) were newly discovered by this experiment.
    
    A weakness counts as found IF AND ONLY IF:
    1. The experiment's fault set satisfies the weakness trigger condition, AND
    2. The SLO was actually violated in simulation.
    
    Also identifies whether an emergent violation occurred (SLO violated without matching any planted weakness).
    """
    newly_found: List[Weakness] = []
    matched_any = False

    if slo_violated:
        for w in planted_weaknesses:
            if w.matches_fault_set(faults):
                matched_any = True
                if w.weakness_id not in already_found_ids:
                    newly_found.append(w)

    emergent_violation = (slo_violated and not matched_any)
    return newly_found, emergent_violation
