"""Implementation of baseline agents:
1. ChaosMonkeyAgent: Faithful reproduction of Netflix Chaos Monkey (pure random pod kill)
2. RandomAllAgent: Uniform random over all fault types and targets
3. RoundRobinAgent: Exhaustive sweep in deterministic order
4. BanditAgent: Multi-Armed Bandit with UCB and Epsilon-Greedy strategies
"""

import math
from typing import List, Optional, Dict, Any
import numpy as np

from chaosrl.agents.base import BaseAgent
from chaosrl.simulator.models import FaultType, Fault


class ChaosMonkeyAgent(BaseAgent):
    """Chaos Monkey: Picks pod kills uniformly at random across services."""

    def __init__(self, action_table: List[List[Fault]], seed: int = 42):
        super().__init__("ChaosMonkey", seed=seed)
        self.action_table = action_table
        # Identify indices that are pure pod kills
        self.pod_kill_indices = [
            idx for idx, f_list in enumerate(action_table)
            if len(f_list) == 1 and f_list[0].fault_type == FaultType.POD_KILL
        ]
        if not self.pod_kill_indices:
            raise ValueError("No POD_KILL actions available in action table for Chaos Monkey!")

    def act(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        candidates = self.pod_kill_indices
        if action_mask is not None:
            valid_candidates = [idx for idx in candidates if action_mask[idx]]
            if valid_candidates:
                candidates = valid_candidates
        return int(self.rng.choice(candidates))


class RandomAllAgent(BaseAgent):
    """Uniform random injection across all fault types and targets."""

    def __init__(self, num_actions: int, seed: int = 42):
        super().__init__("RandomAll", seed=seed)
        self.num_actions = num_actions

    def act(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        if action_mask is not None:
            valid = np.where(action_mask)[0]
            if len(valid) > 0:
                return int(self.rng.choice(valid))
        return int(self.rng.integers(0, self.num_actions))


class RoundRobinAgent(BaseAgent):
    """Deterministic exhaustive round-robin sweep over actions."""

    def __init__(self, num_actions: int, seed: int = 42):
        super().__init__("RoundRobin", seed=seed)
        self.num_actions = num_actions
        self.current_idx = 0

    def act(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        attempts = 0
        while attempts < self.num_actions:
            action = self.current_idx
            self.current_idx = (self.current_idx + 1) % self.num_actions
            if action_mask is None or action_mask[action]:
                return action
            attempts += 1
        return self.current_idx

    def reset(self):
        self.current_idx = 0


class BanditAgent(BaseAgent):
    """Multi-Armed Bandit agent with UCB1 and Epsilon-Greedy strategies."""

    def __init__(
        self,
        num_actions: int,
        strategy: str = "ucb",  # 'ucb' or 'epsilon_greedy'
        c: float = 1.0,          # UCB exploration constant
        epsilon: float = 0.15,   # Epsilon for epsilon-greedy
        seed: int = 42,
    ):
        super().__init__(f"Bandit-{strategy.upper()}", seed=seed)
        self.num_actions = num_actions
        self.strategy = strategy.lower()
        self.c = c
        self.epsilon = epsilon
        
        self.counts = np.zeros(num_actions, dtype=np.int32)
        self.values = np.zeros(num_actions, dtype=np.float32)
        self.total_pulls = 0

    def act(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        self.total_pulls += 1
        valid_indices = (
            np.where(action_mask)[0]
            if action_mask is not None and np.any(action_mask)
            else np.arange(self.num_actions)
        )

        # Unpulled arms first
        unpulled = [idx for idx in valid_indices if self.counts[idx] == 0]
        if unpulled:
            return int(self.rng.choice(unpulled))

        if self.strategy == "epsilon_greedy":
            if self.rng.random() < self.epsilon:
                return int(self.rng.choice(valid_indices))
            sub_values = self.values[valid_indices]
            best_local = np.argmax(sub_values)
            return int(valid_indices[best_local])

        elif self.strategy == "ucb":
            # UCB1 score
            log_t = math.log(max(1, self.total_pulls))
            ucb_scores = []
            for idx in valid_indices:
                score = self.values[idx] + self.c * math.sqrt(log_t / self.counts[idx])
                ucb_scores.append(score)
            best_local = np.argmax(ucb_scores)
            return int(valid_indices[best_local])

        else:
            return int(self.rng.choice(valid_indices))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        info: Dict[str, Any],
    ):
        self.counts[action] += 1
        n = self.counts[action]
        # Incremental average update
        self.values[action] += (reward - self.values[action]) / n

    def reset(self):
        # We can either maintain cross-system memory or reset per episode
        pass
