"""MaskablePPO agent wrapper leveraging sb3-contrib for invalid action masking and continuous policy optimization."""

import os
from typing import Optional, Dict, Any
import numpy as np
from sb3_contrib import MaskablePPO
from chaosrl.agents.base import BaseAgent


class MaskablePPOAgent(BaseAgent):
    """PPO agent with invalid action masking via sb3-contrib."""

    def __init__(
        self,
        num_actions: int,
        model: Optional[MaskablePPO] = None,
        model_path: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__("ChaosRL-PPO", seed=seed)
        self.num_actions = num_actions
        self.model = model

        if model_path is not None and os.path.exists(model_path):
            self.model = MaskablePPO.load(model_path)

    def act(self, obs: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        if self.model is not None:
            model_act_dim = self.model.action_space.n
            curr_act_dim = len(action_mask) if action_mask is not None else self.num_actions

            if model_act_dim == curr_act_dim:
                action, _ = self.model.predict(
                    obs,
                    action_masks=action_mask,
                    deterministic=True,
                )
                return int(action)
            else:
                # Cross-scale evaluation (e.g., 20-svc trained model evaluated on 100-svc system)
                sub_mask = action_mask[:model_act_dim] if action_mask is not None else np.ones(model_act_dim, dtype=bool)
                if np.any(sub_mask):
                    action, _ = self.model.predict(
                        obs,
                        action_masks=sub_mask,
                        deterministic=True,
                    )
                    return int(action)
                else:
                    valid_indices = np.where(action_mask)[0] if action_mask is not None else np.arange(curr_act_dim)
                    return int(self.rng.choice(valid_indices)) if len(valid_indices) > 0 else 0

        # Fallback if no model loaded
        valid_indices = (
            np.where(action_mask)[0]
            if action_mask is not None and np.any(action_mask)
            else np.arange(self.num_actions)
        )
        return int(self.rng.choice(valid_indices))
