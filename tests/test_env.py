"""Unit tests for ChaosEnv Gymnasium API compliance, observation bounds, action masking, and episode dynamics."""

import pytest
import numpy as np
from gymnasium.utils.env_checker import check_env
from chaosrl.env import ChaosEnv


def test_gymnasium_api_compliance():
    """Verify standard Gymnasium API compliance."""
    env = ChaosEnv(num_services=20, num_nodes=6, budget=20, mask_already_tried=False)
    # env_checker checks reset, step, observation space, action space
    check_env(env.unwrapped)


def test_action_masking():
    """Verify that action_masks correctly masks tested actions when enabled."""
    env = ChaosEnv(num_services=20, num_nodes=6, budget=10, mask_already_tried=True)
    obs, info = env.reset(seed=42)
    
    mask0 = env.action_masks()
    assert np.all(mask0), "Initial mask should allow all actions"

    # Step action 5
    action_to_try = 5
    obs, reward, terminated, truncated, info = env.step(action_to_try)
    
    mask1 = env.action_masks()
    assert not mask1[action_to_try], f"Action {action_to_try} was not masked after execution!"
    assert np.sum(mask1) == len(mask1) - 1


def test_episode_budget_termination():
    """Verify that episode terminates exactly when budget is exhausted."""
    budget = 15
    env = ChaosEnv(num_services=20, num_nodes=6, budget=budget)
    obs, info = env.reset(seed=123)

    steps = 0
    terminated = False
    while not terminated:
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        action = int(valid_actions[0])
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    assert steps == budget
    assert terminated is True
    assert env.experiments_conducted == budget
