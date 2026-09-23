"""Base agent interface for all ChaosRL and baseline policies."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for chaos injection agents."""

    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def act(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        """Select an action index given the current observation and optional boolean mask."""
        pass

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        info: Dict[str, Any],
    ):
        """Update internal policy or value estimates based on transition (for learning agents)."""
        pass

    def reset(self):
        """Reset internal agent episode state if needed."""
        pass
