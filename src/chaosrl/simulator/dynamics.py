"""Discrete-time aggregate dynamics for microservice systems.
Computes load propagation, queueing latency, error rates, cascading timeouts, retry storms, circuit breakers, and telemetry.
"""

from typing import Dict, List, Tuple
import numpy as np

from chaosrl.simulator.models import (
    Node,
    Replica,
    Service,
    Fault,
    FaultType,
    IntensityLevel,
    TelemetrySnapshot,
)


class SystemSimulator:
    """Simulates tick-by-tick system dynamics, fault injections, and metric emissions."""

    def __init__(
        self,
        services: Dict[str, Service],
        nodes: Dict[str, Node],
        baseline_load: float = 50.0,
        slo_latency_threshold: float = 300.0,  # ms
        slo_error_threshold: float = 0.05,     # 5%
        noise_level: float = 0.05,
        seed: int = 42,
    ):
        self.services = services
        self.nodes = nodes
        self.baseline_load = baseline_load
        self.slo_latency_threshold = slo_latency_threshold
        self.slo_error_threshold = slo_error_threshold
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)
        
        self.current_tick = 0
        self.active_faults: List[Fault] = []
        self.backup_state = self._snapshot_initial_state()

    def _snapshot_initial_state(self) -> Dict:
        """Saves baseline configuration for deterministic rollback."""
        return {
            "services": {
                sid: {
                    "base_latency": s.base_latency,
                    "capacity_per_replica": s.capacity_per_replica,
                    "timeout": s.timeout,
                    "max_retries": s.max_retries,
                    "circuit_breaker_enabled": s.circuit_breaker_enabled,
                    "cb_state": "CLOSED",
                    "cb_timer": 0,
                    "cb_fail_count": 0,
                    "fallback_enabled": s.fallback_enabled,
                    "autoscaling_enabled": s.autoscaling_enabled,
                    "min_replicas": s.min_replicas,
                    "replicas": [
                        {"id": r.replica_id, "node": r.node_id, "alive": True, "stress": 0.0}
                        for r in s.replicas
                    ],
                }
                for sid, s in self.services.items()
            },
            "nodes": {
                nid: {
                    "is_alive": n.is_alive,
                    "running_replicas": list(n.running_replicas),
                }
                for nid, n in self.nodes.items()
            },
        }

    def rollback(self):
        """Roll back system state to pre-fault baseline."""
        self.active_faults.clear()
        
        # Restore nodes
        for nid, n_data in self.backup_state["nodes"].items():
            node = self.nodes[nid]
            node.is_alive = n_data["is_alive"]
            node.running_replicas = list(n_data["running_replicas"])

        # Restore services and replicas
        for sid, s_data in self.backup_state["services"].items():
            svc = self.services[sid]
            svc.base_latency = s_data["base_latency"]
            svc.capacity_per_replica = s_data["capacity_per_replica"]
            svc.timeout = s_data["timeout"]
            svc.max_retries = s_data["max_retries"]
            svc.circuit_breaker_enabled = s_data["circuit_breaker_enabled"]
            svc.cb_state = "CLOSED"
            svc.cb_timer = 0
            svc.cb_fail_count = 0
            svc.fallback_enabled = s_data["fallback_enabled"]
            svc.autoscaling_enabled = s_data["autoscaling_enabled"]
            svc.current_load = 0.0
            svc.effective_capacity = 0.0
            svc.utilization = 0.0
            svc.p50_latency = 0.0
            svc.p99_latency = 0.0
            svc.error_rate = 0.0
            svc.overload_ticks = 0

            # Restore replicas
            svc.replicas = []
            for r_data in s_data["replicas"]:
                svc.replicas.append(
                    Replica(
                        replica_id=r_data["id"],
                        service_id=sid,
                        node_id=r_data["node"],
                        is_alive=True,
                        cpu_stress=0.0,
                    )
                )

    def inject_fault(self, fault: Fault):
        """Register a fault for active application."""
        self.active_faults.append(fault)

    def apply_active_faults(self):
        """Applies physical disruptions according to active faults."""
        intensity_map = {
            IntensityLevel.LOW: 0.25,
            IntensityLevel.MEDIUM: 0.50,
            IntensityLevel.HIGH: 0.85,
        }

        for fault in self.active_faults:
            factor = intensity_map[fault.intensity]

            if fault.fault_type == FaultType.NODE_FAILURE:
                if fault.target in self.nodes:
                    node = self.nodes[fault.target]
                    node.is_alive = False
                    # Kill all replicas on this node
                    for s in self.services.values():
                        for r in s.replicas:
                            if r.node_id == node.node_id:
                                r.is_alive = False

            elif fault.fault_type == FaultType.POD_KILL:
                if fault.target in self.services:
                    svc = self.services[fault.target]
                    alive_reps = [r for r in svc.replicas if r.is_alive]
                    if alive_reps:
                        # Kill the first alive replica
                        alive_reps[0].is_alive = False

            elif fault.fault_type == FaultType.CPU_STRESS:
                # Can target service or node
                if fault.target in self.nodes:
                    for s in self.services.values():
                        for r in s.replicas:
                            if r.node_id == fault.target:
                                r.cpu_stress = min(1.0, r.cpu_stress + factor)
                elif fault.target in self.services:
                    svc = self.services[fault.target]
                    for r in svc.replicas:
                        r.cpu_stress = min(1.0, r.cpu_stress + factor)

    def step_tick(self) -> TelemetrySnapshot:
        """Simulate one discrete time tick across the entire service mesh."""
        self.current_tick += 1
        self.apply_active_faults()

        # 1. Gateway external arrival load
        diurnal_factor = 1.0 + 0.15 * np.sin(self.current_tick / 15.0)
        noise = float(self.rng.normal(0.0, self.noise_level))
        ext_load = max(10.0, self.baseline_load * diurnal_factor * (1.0 + noise))

        # Reset tick telemetry on services
        for s in self.services.values():
            s.current_load = 0.0

        gateways = [s for s in self.services.values() if s.layer == 0]
        for gw in gateways:
            gw.current_load += ext_load / len(gateways)

        # 2. Downward load propagation (Gateway -> Business -> Data)
        # Topological ordering: by layer (0 -> 1 -> 2)
        services_by_layer = sorted(self.services.values(), key=lambda s: s.layer)

        for s in services_by_layer:
            if not s.dependencies or s.current_load <= 0.0:
                continue

            num_deps = len(s.dependencies)
            load_per_dep = s.current_load / num_deps

            for dep_id in s.dependencies:
                dep_svc = self.services[dep_id]
                
                # Check circuit breaker status
                if s.circuit_breaker_enabled and s.cb_state == "OPEN":
                    # Breaker open: calls short-circuited!
                    continue

                # Retry amplification (retry storm dynamics)
                amplification = 1.0
                if s.max_retries > 0 and dep_svc.error_rate > 0.05:
                    amplification += s.max_retries * min(1.0, dep_svc.error_rate * 1.5)

                dep_svc.current_load += load_per_dep * amplification

        # 3. Compute capacity, queueing utilization, and local latency per service
        for s in self.services.values():
            alive_replicas = [r for r in s.replicas if r.is_alive and self.nodes[r.node_id].is_alive]
            
            # Autoscaling check
            if s.autoscaling_enabled and s.utilization > 0.8:
                s.overload_ticks += 1
                if s.overload_ticks >= s.autoscaling_delay and len(s.replicas) < s.max_replicas:
                    # Spawn new replica
                    new_rep_id = f"autoscale_{len(s.replicas)}_{s.service_id}"
                    available_nodes = [nid for nid, n in self.nodes.items() if n.is_alive]
                    if available_nodes:
                        chosen_n = available_nodes[0]
                        new_rep = Replica(new_rep_id, s.service_id, chosen_n, is_alive=True)
                        s.replicas.append(new_rep)
                        self.nodes[chosen_n].running_replicas.append(new_rep_id)
            else:
                s.overload_ticks = max(0, s.overload_ticks - 1)

            # Re-evaluate alive replicas after potential autoscale
            alive_replicas = [r for r in s.replicas if r.is_alive and self.nodes[r.node_id].is_alive]
            
            if not alive_replicas:
                s.effective_capacity = 0.001
                s.utilization = 10.0
                s.error_rate = 1.0
                s.p50_latency = 1000.0
                s.p99_latency = 2000.0
                continue

            # Effective capacity taking CPU stress into account
            cap = sum(s.capacity_per_replica * (1.0 - r.cpu_stress) for r in alive_replicas)
            s.effective_capacity = max(1.0, cap)
            s.utilization = s.current_load / s.effective_capacity

            # Local queueing latency: saturating non-linear model
            # Latency = Base * (1 + U^2 / (1.01 - min(U, 0.99)))
            u_clamped = min(s.utilization, 0.99)
            queue_mult = 1.0 + (s.utilization ** 2) / (1.01 - u_clamped)
            local_latency = s.base_latency * queue_mult

            # Drop rate if overloaded
            drop_error = 0.0
            if s.utilization > 1.0:
                drop_error = min(1.0, (s.utilization - 1.0) / s.utilization)

            s.p50_latency = local_latency
            s.p99_latency = local_latency * 1.5
            s.error_rate = drop_error

        # 4. Apply Faults directly altering metrics (Latency injection, Packet loss, Dependency error)
        for fault in self.active_faults:
            if fault.target in self.services:
                svc = self.services[fault.target]
                if fault.fault_type == FaultType.LATENCY_INJECTION:
                    added_ms = {
                        IntensityLevel.LOW: 200.0,
                        IntensityLevel.MEDIUM: 600.0,
                        IntensityLevel.HIGH: 1800.0,
                    }[fault.intensity]
                    svc.p50_latency += added_ms
                    svc.p99_latency += added_ms * 1.6

                elif fault.fault_type == FaultType.PACKET_LOSS:
                    loss_rate = {
                        IntensityLevel.LOW: 0.20,
                        IntensityLevel.MEDIUM: 0.50,
                        IntensityLevel.HIGH: 0.85,
                    }[fault.intensity]
                    svc.error_rate = 1.0 - (1.0 - svc.error_rate) * (1.0 - loss_rate)

                elif fault.fault_type == FaultType.DEPENDENCY_ERROR:
                    forced_err = {
                        IntensityLevel.LOW: 0.35,
                        IntensityLevel.MEDIUM: 0.70,
                        IntensityLevel.HIGH: 0.95,
                    }[fault.intensity]
                    svc.error_rate = max(svc.error_rate, forced_err)

        # 5. Upward propagation (Data -> Business -> Gateway)
        # Latency and errors bubble up; circuit breakers and timeouts trigger
        reverse_services = sorted(self.services.values(), key=lambda s: s.layer, reverse=True)

        for s in reverse_services:
            if not s.dependencies:
                continue

            max_dep_latency = 0.0
            combined_dep_error = 0.0

            for dep_id in s.dependencies:
                dep_svc = self.services[dep_id]

                # Check if caller circuit breaker is OPEN
                if s.circuit_breaker_enabled and s.cb_state == "OPEN":
                    if s.fallback_enabled:
                        max_dep_latency = max(max_dep_latency, s.fallback_latency)
                        combined_dep_error = max(combined_dep_error, s.fallback_error_rate)
                    else:
                        combined_dep_error = max(combined_dep_error, 0.3)
                    continue

                dep_lat = dep_svc.p99_latency
                dep_err = dep_svc.error_rate

                # Timeout check: did dependency call exceed configured timeout?
                if dep_lat > s.timeout:
                    dep_err = max(dep_err, 0.8)  # Timeout triggers high failure rate
                    dep_lat = s.timeout          # Caller stops waiting at timeout

                    # If caller has NO circuit breaker and high timeout, caller gets clogged
                    if not s.circuit_breaker_enabled:
                        s.utilization = min(5.0, s.utilization * 1.8)
                        s.error_rate = min(1.0, s.error_rate + 0.4)

                max_dep_latency = max(max_dep_latency, dep_lat)
                combined_dep_error = 1.0 - (1.0 - combined_dep_error) * (1.0 - dep_err)

                # Circuit breaker trip tracking
                if s.circuit_breaker_enabled:
                    if dep_err > s.cb_failure_threshold:
                        s.cb_fail_count += 1
                        if s.cb_fail_count >= 2:
                            s.cb_state = "OPEN"
                            s.cb_timer = s.cb_recovery_ticks
                    else:
                        s.cb_fail_count = max(0, s.cb_fail_count - 1)

            # Circuit breaker timer update
            if s.cb_state == "OPEN":
                s.cb_timer -= 1
                if s.cb_timer <= 0:
                    s.cb_state = "CLOSED"
                    s.cb_fail_count = 0

            # Compound service metrics with dependency latency and errors
            s.p50_latency += max_dep_latency * 0.7
            s.p99_latency += max_dep_latency
            s.error_rate = 1.0 - (1.0 - s.error_rate) * (1.0 - combined_dep_error)

        # 6. Gateway aggregation & SLO verification
        gw_latencies = [gw.p99_latency for gw in gateways]
        gw_errors = [gw.error_rate for gw in gateways]

        # Add measurement noise to observability
        meas_noise = float(self.rng.normal(1.0, 0.03))
        system_p99 = float(np.mean(gw_latencies) * meas_noise)
        system_error = float(np.mean(gw_errors))

        slo_violated = (
            system_p99 > self.slo_latency_threshold
            or system_error > self.slo_error_threshold
        )

        # Collect snapshot
        snapshot = TelemetrySnapshot(
            tick=self.current_tick,
            system_latency_p99=system_p99,
            system_error_rate=system_error,
            slo_violated=slo_violated,
            service_latencies={s.service_id: s.p99_latency for s in self.services.values()},
            service_error_rates={s.service_id: s.error_rate for s in self.services.values()},
            service_utilizations={s.service_id: s.utilization for s in self.services.values()},
            service_healthy_replicas={
                s.service_id: len([r for r in s.replicas if r.is_alive and self.nodes[r.node_id].is_alive])
                for s in self.services.values()
            },
            active_faults_count=len(self.active_faults),
        )

        return snapshot

    def run_experiment(
        self, faults: List[Fault], experiment_ticks: int = 10, recovery_ticks: int = 5
    ) -> Tuple[bool, float, List[TelemetrySnapshot]]:
        """Executes a single fault injection experiment: inject -> simulate T ticks -> rollback -> recovery."""
        for f in faults:
            self.inject_fault(f)

        snapshots: List[TelemetrySnapshot] = []
        slo_violated_in_experiment = False
        accumulated_damage = 0.0

        # Simulate fault duration
        for _ in range(experiment_ticks):
            snap = self.step_tick()
            snapshots.append(snap)
            if snap.slo_violated:
                slo_violated_in_experiment = True
                lat_excess = max(0.0, snap.system_latency_p99 - self.slo_latency_threshold)
                # Damage metric: error rate penalty + latency overrun penalty
                accumulated_damage += snap.system_error_rate + (lat_excess / 1000.0)

        # Rollback faults
        self.rollback()

        # Simulate recovery period to ensure system returns to steady state
        for _ in range(recovery_ticks):
            snap = self.step_tick()
            snapshots.append(snap)

        return slo_violated_in_experiment, accumulated_damage, snapshots
