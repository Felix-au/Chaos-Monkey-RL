"""ChaosEnv: Gymnasium environment for microservice chaos engineering and fault injection policies."""

from typing import Dict, List, Optional, Tuple, Any, Set
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from chaosrl.simulator.models import (
    Service,
    Node,
    Fault,
    FaultType,
    IntensityLevel,
    Weakness,
    TelemetrySnapshot,
)
from chaosrl.simulator.generator import SystemGenerator
from chaosrl.simulator.dynamics import SystemSimulator
from chaosrl.simulator.metrics import evaluate_discovery
from chaosrl.env.observations import ObservationBuilder
from chaosrl.env.rewards import compute_reward


class ChaosEnv(gym.Env):
    """Gymnasium Environment modeling microservice fault injection as a Markov Decision Process."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_services: int = 20,
        num_nodes: int = 6,
        budget: int = 50,
        reward_variant: str = "R2",
        include_history: bool = True,
        compound_mode: bool = False,
        mask_already_tried: bool = True,
        system_seed: Optional[int] = None,
        generator: Optional[SystemGenerator] = None,
    ):
        super().__init__()
        self.num_services = num_services
        self.num_nodes = num_nodes
        self.budget = budget
        self.reward_variant = reward_variant
        self.include_history = include_history
        self.compound_mode = compound_mode
        self.mask_already_tried = mask_already_tried
        self.system_seed = system_seed
        self.fixed_generator = generator

        self.obs_builder = ObservationBuilder(include_history=include_history)
        
        # Build action table
        self.action_table = self._build_action_table()
        self.action_space = spaces.Discrete(len(self.action_table))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(ObservationBuilder.OBS_DIM,),
            dtype=np.float32,
        )

        # Episode state variables
        self.services: Dict[str, Service] = {}
        self.nodes: Dict[str, Node] = {}
        self.planted_weaknesses: List[Weakness] = []
        self.simulator: Optional[SystemSimulator] = None
        self.experiments_conducted = 0
        self.found_weakness_ids: Set[str] = set()
        self.tested_action_indices: Set[int] = set()
        self.history_records: List[Dict[str, Any]] = []
        self.last_damage = 0.0
        self.last_discovered = False
        self.last_snapshot: Optional[TelemetrySnapshot] = None
        self.total_damage_accumulated = 0.0
        self.emergent_violations_count = 0

    def _build_action_table(self) -> List[List[Fault]]:
        """Precomputes mapping from discrete action index to list of Fault objects."""
        actions: List[List[Fault]] = []
        fault_types = list(FaultType)
        intensity = IntensityLevel.HIGH  # Use high intensity for decisive weakness discovery

        # 1. Single faults on services
        for s_idx in range(self.num_services):
            sid = f"svc_{s_idx}"
            for ft in fault_types:
                if ft == FaultType.NODE_FAILURE:
                    continue  # Node failure targets nodes, not services
                actions.append([Fault(fault_type=ft, target=sid, intensity=intensity)])

        # 2. Node failures
        for n_idx in range(self.num_nodes):
            nid = f"node_{n_idx}"
            actions.append([Fault(fault_type=FaultType.NODE_FAILURE, target=nid, intensity=intensity)])

        # 3. If compound mode is enabled, add compound pairs (simultaneous 2 faults)
        if self.compound_mode:
            # Latency on downstream + Packet loss upstream
            for s1_idx in range(min(5, self.num_services)):
                for s2_idx in range(min(5, self.num_services)):
                    if s1_idx != s2_idx:
                        actions.append([
                            Fault(FaultType.LATENCY_INJECTION, f"svc_{s1_idx}", IntensityLevel.HIGH),
                            Fault(FaultType.PACKET_LOSS, f"svc_{s2_idx}", IntensityLevel.MEDIUM),
                        ])
            # Pod kill + CPU stress
            for s_idx in range(min(10, self.num_services)):
                actions.append([
                    Fault(FaultType.POD_KILL, f"svc_{s_idx}", IntensityLevel.HIGH),
                    Fault(FaultType.CPU_STRESS, f"svc_{s_idx}", IntensityLevel.HIGH),
                ])

        return actions

    def action_masks(self) -> np.ndarray:
        """Returns boolean mask of valid actions for MaskablePPO."""
        masks = np.ones(len(self.action_table), dtype=bool)

        if self.mask_already_tried:
            for idx in self.tested_action_indices:
                if idx < len(masks):
                    masks[idx] = False

        # If all actions were masked out, unmask all so the agent can still step
        if not np.any(masks):
            masks[:] = True

        return masks

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Decide generation seed
        gen_seed = seed if seed is not None else (self.system_seed if self.system_seed is not None else int(self.np_random.integers(0, 1000000)))
        
        if self.fixed_generator is not None:
            gen = self.fixed_generator
        else:
            gen = SystemGenerator(
                num_services=self.num_services,
                num_nodes=self.num_nodes,
                seed=gen_seed,
            )

        self.services, self.nodes, self.planted_weaknesses = gen.generate()
        self.simulator = SystemSimulator(self.services, self.nodes, seed=gen_seed)

        self.experiments_conducted = 0
        self.found_weakness_ids.clear()
        self.tested_action_indices.clear()
        self.history_records.clear()
        self.last_damage = 0.0
        self.last_discovered = False
        self.total_damage_accumulated = 0.0
        self.emergent_violations_count = 0

        # Run 2 initial ticks to stabilize baseline snapshot
        self.last_snapshot = self.simulator.step_tick()
        self.last_snapshot = self.simulator.step_tick()

        obs = self.obs_builder.build(
            services=self.services,
            nodes=self.nodes,
            last_snapshot=self.last_snapshot,
            budget_remaining=self.budget,
            total_budget=self.budget,
            found_weakness_count=0,
            history_records=self.history_records,
            last_damage=0.0,
            last_discovered=False,
        )

        info = {
            "planted_weaknesses_count": len(self.planted_weaknesses),
            "found_weakness_ids": list(self.found_weakness_ids),
            "action_mask": self.action_masks(),
        }

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.experiments_conducted += 1
        is_repeat = (action in self.tested_action_indices)
        self.tested_action_indices.add(action)

        fault_set = self.action_table[action]

        # Execute experiment in simulator
        slo_violated, damage, snapshots = self.simulator.run_experiment(
            fault_set, experiment_ticks=6, recovery_ticks=3
        )
        self.last_snapshot = snapshots[-1]
        self.last_damage = damage
        self.total_damage_accumulated += damage

        # Evaluate weakness discovery
        newly_found, emergent = evaluate_discovery(
            fault_set, slo_violated, self.planted_weaknesses, self.found_weakness_ids
        )
        for w in newly_found:
            self.found_weakness_ids.add(w.weakness_id)

        self.last_discovered = len(newly_found) > 0
        if emergent:
            self.emergent_violations_count += 1

        # Check partial degradation (sub-SLO latency increase)
        partial_degradation = (
            not slo_violated
            and any(s.system_latency_p99 > self.simulator.slo_latency_threshold * 0.7 for s in snapshots)
        )

        # Compute reward
        reward = compute_reward(
            reward_variant=self.reward_variant,
            newly_found=newly_found,
            damage=damage,
            slo_violated=slo_violated,
            is_repeat_action=is_repeat,
            partial_degradation=partial_degradation,
        )

        # Update history
        record = {
            "action": action,
            "fault_types": [f.fault_type for f in fault_set],
            "slo_violated": slo_violated,
            "damage": damage,
            "discovered": len(newly_found) > 0,
        }
        self.history_records.append(record)

        budget_remaining = max(0, self.budget - self.experiments_conducted)
        terminated = budget_remaining <= 0
        truncated = False

        obs = self.obs_builder.build(
            services=self.services,
            nodes=self.nodes,
            last_snapshot=self.last_snapshot,
            budget_remaining=budget_remaining,
            total_budget=self.budget,
            found_weakness_count=len(self.found_weakness_ids),
            history_records=self.history_records,
            last_damage=damage,
            last_discovered=self.last_discovered,
        )

        info = {
            "slo_violated": slo_violated,
            "damage": damage,
            "total_damage": self.total_damage_accumulated,
            "newly_found": [w.weakness_id for w in newly_found],
            "total_found": len(self.found_weakness_ids),
            "total_planted": len(self.planted_weaknesses),
            "discovery_rate": len(self.found_weakness_ids) / max(1, len(self.planted_weaknesses)),
            "emergent_violations": self.emergent_violations_count,
            "is_repeat": is_repeat,
            "action_mask": self.action_masks(),
        }

        return obs, reward, terminated, truncated, info
