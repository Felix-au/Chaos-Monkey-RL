"""Scalable observation builders for ChaosEnv.
Provides fixed-size feature vectors invariant to the number of microservices (20, 100, 300+).
"""

from typing import Dict, List, Set, Any
import numpy as np
from chaosrl.simulator.models import Service, Node, TelemetrySnapshot, FaultType


class ObservationBuilder:
    """Constructs fixed-size, scale-invariant observation representations."""

    OBS_DIM = 61  # 3 system + 18 layer-wise + 30 top-k + 10 history

    def __init__(self, include_history: bool = True, top_k: int = 5):
        self.include_history = include_history
        self.top_k = top_k

    def build(
        self,
        services: Dict[str, Service],
        nodes: Dict[str, Node],
        last_snapshot: TelemetrySnapshot,
        budget_remaining: int,
        total_budget: int,
        found_weakness_count: int,
        history_records: List[Dict[str, Any]],
        last_damage: float,
        last_discovered: bool,
    ) -> np.ndarray:
        features: List[float] = []

        # 1. System-level metrics (3 features)
        norm_lat = min(1.0, last_snapshot.system_latency_p99 / 1000.0)
        norm_err = min(1.0, last_snapshot.system_error_rate)
        budget_ratio = budget_remaining / max(1.0, float(total_budget))
        features.extend([norm_lat, norm_err, budget_ratio])

        # 2. Per-layer aggregates for layers 0, 1, 2 (3 * 6 = 18 features)
        for layer_idx in [0, 1, 2]:
            layer_svcs = [s for s in services.values() if s.layer == layer_idx]
            if not layer_svcs:
                features.extend([0.0] * 6)
                continue

            latencies = [s.p99_latency for s in layer_svcs]
            errors = [s.error_rate for s in layer_svcs]
            utils = [min(2.0, s.utilization) for s in layer_svcs]
            rep_ratios = [
                len([r for r in s.replicas if r.is_alive and nodes[r.node_id].is_alive])
                / max(1, len(s.replicas))
                for s in layer_svcs
            ]

            features.append(min(1.0, float(np.mean(latencies)) / 1000.0))
            features.append(min(1.0, float(np.max(latencies)) / 1000.0))
            features.append(min(1.0, float(np.mean(errors))))
            features.append(min(1.0, float(np.max(errors))))
            features.append(float(np.mean(utils)) / 2.0)
            features.append(float(np.mean(rep_ratios)))

        # 3. Top-k most degraded or critical services (k * 6 = 30 features)
        def risk_score(s: Service) -> float:
            deg = len(s.callers) + len(s.dependencies)
            return s.error_rate * 3.0 + min(2.0, s.utilization) + (s.p99_latency / 500.0) + (deg * 0.05)

        sorted_svcs = sorted(services.values(), key=risk_score, reverse=True)
        for i in range(self.top_k):
            if i < len(sorted_svcs):
                s = sorted_svcs[i]
                alive_reps = len([r for r in s.replicas if r.is_alive and nodes[r.node_id].is_alive])
                deg = len(s.callers) + len(s.dependencies)
                features.extend([
                    s.layer / 2.0,
                    min(1.0, s.p99_latency / 1000.0),
                    min(1.0, s.error_rate),
                    min(1.0, s.utilization / 2.0),
                    alive_reps / max(1, len(s.replicas)),
                    min(1.0, deg / 20.0),
                ])
            else:
                features.extend([0.0] * 6)

        # 4. History feature vector (10 features)
        if self.include_history:
            total_exp = len(history_records)
            exp_progress = min(1.0, total_exp / max(1.0, float(total_budget)))
            weakness_ratio = min(1.0, found_weakness_count / 10.0)
            last_disc_val = 1.0 if last_discovered else 0.0
            norm_last_damage = min(1.0, last_damage / 5.0)

            # Fault type frequencies (6 fault types)
            fault_counts = {ft: 0 for ft in FaultType}
            for rec in history_records:
                for ft in rec.get("fault_types", []):
                    if ft in fault_counts:
                        fault_counts[ft] += 1

            total_faults_tried = max(1, sum(fault_counts.values()))
            fault_dist = [fault_counts[ft] / total_faults_tried for ft in FaultType]

            features.extend([exp_progress, weakness_ratio, last_disc_val, norm_last_damage])
            features.extend(fault_dist)
        else:
            features.extend([0.0] * 10)

        obs = np.array(features, dtype=np.float32)
        assert len(obs) == self.OBS_DIM, f"Observation dimension mismatch: {len(obs)} != {self.OBS_DIM}"
        return obs
