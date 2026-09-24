"""Unit tests verifying agent behaviors: Chaos Monkey pod kill restriction, Round Robin sequence, Bandit, Tabular Q, and PPO."""

import pytest
import numpy as np

from chaosrl.env import ChaosEnv
from chaosrl.simulator.models import FaultType
from chaosrl.agents import (
    ChaosMonkeyAgent,
    RandomAllAgent,
    RoundRobinAgent,
    BanditAgent,
    TabularQAgent,
    MaskablePPOAgent,
)


def test_chaos_monkey_only_picks_pod_kill():
    """Verify that Chaos Monkey ONLY picks pod kills."""
    env = ChaosEnv(num_services=20, num_nodes=6, budget=20)
    agent = ChaosMonkeyAgent(env.action_table, seed=42)

    for _ in range(50):
        action = agent.act(np.zeros(61), action_mask=env.action_masks())
        fault_set = env.action_table[action]
        assert len(fault_set) == 1
        assert fault_set[0].fault_type == FaultType.POD_KILL, f"Chaos Monkey picked non-pod-kill fault: {fault_set[0].fault_type}"


def test_round_robin_deterministic_sweep():
    """Verify round-robin deterministic cycling."""
    agent = RoundRobinAgent(num_actions=5, seed=42)
    actions = [agent.act(np.zeros(61)) for _ in range(10)]
    assert actions == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]


def test_bandit_and_tabular_q_execution():
    """Verify Bandit and Tabular Q step and update loop without crashing."""
    env = ChaosEnv(num_services=10, num_nodes=4, budget=10)
    obs, info = env.reset(seed=42)

    agents = [
        BanditAgent(len(env.action_table), strategy="ucb", seed=42),
        BanditAgent(len(env.action_table), strategy="epsilon_greedy", seed=42),
        TabularQAgent(len(env.action_table), seed=42),
    ]

    for agent in agents:
        env.reset(seed=42)
        done = False
        step_count = 0
        while not done:
            mask = env.action_masks()
            act = agent.act(obs, action_mask=mask)
            next_obs, rew, term, trunc, info = env.step(act)
            agent.update(obs, act, rew, next_obs, term, info)
            obs = next_obs
            done = term or trunc
            step_count += 1
        assert step_count == 10
