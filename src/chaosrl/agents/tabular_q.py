"""Tabular Q-learning agent with compact state discretization for small-scale systems."""

from typing import Optional, Dict, Any
import numpy as np
from chaosrl.agents.base import BaseAgent


class TabularQAgent(BaseAgent):
    """Tabular Q-learning agent using feature discretization."""

    def __init__(
        self,
        num_actions: int,
        alpha: float = 0.15,
        gamma: float = 0.95,
        epsilon: float = 0.20,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        seed: int = 42,
    ):
        super().__init__("TabularQ", seed=seed)
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # State space:
        # Latency bin (3): <0.2, 0.2-0.5, >=0.5
        # Error bin (2): <0.05, >=0.05
        # Budget bin (3): >0.66, 0.33-0.66, <=0.33
        # Most degraded layer (3): 0, 1, 2
        # Total = 3 * 2 * 3 * 3 = 54 states
        self.num_states = 54
        self.q_table = np.zeros((self.num_states, num_actions), dtype=np.float32)

    def _discretize_state(self, obs: np.ndarray) -> int:
        norm_lat = obs[0]
        norm_err = obs[1]
        budget_ratio = obs[2]

        lat_bin = 0 if norm_lat < 0.2 else (1 if norm_lat < 0.5 else 2)
        err_bin = 0 if norm_err < 0.05 else 1
        budget_bin = 0 if budget_ratio > 0.66 else (1 if budget_ratio > 0.33 else 2)

        # Check layer max errors from per-layer aggregates (indices 6, 12, 18)
        layer_errs = [obs[6], obs[12], obs[18]]
        deg_layer = int(np.argmax(layer_errs))

        state_idx = lat_bin * 18 + err_bin * 9 + budget_bin * 3 + deg_layer
        return min(self.num_states - 1, max(0, state_idx))

    def act(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        state = self._discretize_state(obs)
        valid_indices = (
            np.where(action_mask)[0]
            if action_mask is not None and np.any(action_mask)
            else np.arange(self.num_actions)
        )

        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_indices))

        q_vals = self.q_table[state, valid_indices]
        best_sub = np.argmax(q_vals)
        return int(valid_indices[best_sub])

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        info: Dict[str, Any],
    ):
        s = self._discretize_state(obs)
        s_next = self._discretize_state(next_obs)
        
        target = reward
        if not terminated:
            target += self.gamma * np.max(self.q_table[s_next])

        td_error = target - self.q_table[s, action]
        self.q_table[s, action] += self.alpha * td_error

        # Decay exploration
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
