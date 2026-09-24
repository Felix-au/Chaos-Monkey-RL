"""Microservice system generator: synthesizes realistic layered DAG topologies, infrastructure node mappings, service parameters, and planted weaknesses."""

import math
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np

from chaosrl.simulator.models import (
    Node,
    Replica,
    Service,
    Weakness,
    FaultType,
    IntensityLevel,
)


def get_num_nodes(num_services: int) -> int:
    """Standard node count helper for system generation."""
    if num_services <= 20:
        return 6
    elif num_services <= 100:
        return 20
    elif num_services <= 300:
        return 40
    else:
        return max(4, num_services // 8)


class SystemGenerator:
    """Generates deterministic synthetic microservice architectures given a seed and configuration."""

    def __init__(
        self,
        num_services: int = 20,
        num_nodes: int = 8,
        max_replicas: int = 3,
        seed: int = 42,
    ):
        self.num_services = max(5, num_services)
        self.num_nodes = max(2, num_nodes)
        self.max_replicas = max(1, max_replicas)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self) -> Tuple[Dict[str, Service], Dict[str, Node], List[Weakness]]:
        """Generate the services, infrastructure nodes, and planted ground-truth weaknesses."""
        nodes = self._create_nodes()
        services = self._create_services()
        self._build_topology(services)
        self._place_replicas(services, nodes)
        weaknesses = self._plant_weaknesses(services, nodes)
        return services, nodes, weaknesses

    def _create_nodes(self) -> Dict[str, Node]:
        nodes = {}
        for i in range(self.num_nodes):
            nid = f"node_{i}"
            nodes[nid] = Node(
                node_id=nid,
                cpu_capacity=100.0,
                memory_capacity=100.0,
                is_alive=True,
                running_replicas=[],
            )
        return nodes

    def _create_services(self) -> Dict[str, Service]:
        services: Dict[str, Service] = {}
        
        # Determine layers:
        # Layer 0: Gateway (1 for small, up to 2 for large)
        # Layer 1: Business services (~55%)
        # Layer 2: Data/Storage backend services (~40%)
        num_gateways = 1 if self.num_services < 50 else 2
        num_business = max(2, int((self.num_services - num_gateways) * 0.55))
        num_data = self.num_services - num_gateways - num_business

        all_ids = [f"svc_{i}" for i in range(self.num_services)]
        
        gateway_ids = all_ids[:num_gateways]
        business_ids = all_ids[num_gateways : num_gateways + num_business]
        data_ids = all_ids[num_gateways + num_business :]

        for sid in gateway_ids:
            services[sid] = Service(
                service_id=sid,
                name=f"Gateway-{sid}",
                layer=0,
                base_latency=float(self.rng.uniform(5.0, 15.0)),
                capacity_per_replica=float(self.rng.uniform(200.0, 400.0)),
                timeout=500.0,
                max_retries=1,
                circuit_breaker_enabled=True,
                fallback_enabled=False,
                autoscaling_enabled=True,
                min_replicas=2,
                max_replicas=self.max_replicas + 2,
            )

        for sid in business_ids:
            services[sid] = Service(
                service_id=sid,
                name=f"Business-{sid}",
                layer=1,
                base_latency=float(self.rng.uniform(10.0, 30.0)),
                capacity_per_replica=float(self.rng.uniform(80.0, 150.0)),
                timeout=float(self.rng.choice([150.0, 300.0, 500.0])),
                max_retries=int(self.rng.choice([0, 1, 2, 3])),
                circuit_breaker_enabled=bool(self.rng.random() > 0.3),
                fallback_enabled=bool(self.rng.random() > 0.5),
                autoscaling_enabled=bool(self.rng.random() > 0.4),
                min_replicas=int(self.rng.choice([1, 2, self.max_replicas])),
                max_replicas=self.max_replicas + 1,
            )

        for sid in data_ids:
            services[sid] = Service(
                service_id=sid,
                name=f"Data-{sid}",
                layer=2,
                base_latency=float(self.rng.uniform(15.0, 45.0)),
                capacity_per_replica=float(self.rng.uniform(60.0, 120.0)),
                timeout=float(self.rng.choice([200.0, 400.0, 800.0])),
                max_retries=0,  # Data tier usually doesn't call downstream
                circuit_breaker_enabled=True,
                fallback_enabled=bool(self.rng.random() > 0.6),
                autoscaling_enabled=bool(self.rng.random() > 0.5),
                min_replicas=int(self.rng.choice([1, 2, self.max_replicas])),
                max_replicas=self.max_replicas + 1,
            )

        return services

    def _build_topology(self, services: Dict[str, Service]):
        """Create a directed acyclic graph (DAG) respecting layer hierarchy."""
        gateways = [s for s in services.values() if s.layer == 0]
        business = [s for s in services.values() if s.layer == 1]
        data = [s for s in services.values() if s.layer == 2]

        # 1. Gateways call a subset of business services (fan-out)
        for gw in gateways:
            fanout = min(len(business), max(2, int(len(business) * 0.4)))
            chosen_business = self.rng.choice(business, size=fanout, replace=False)
            for b in chosen_business:
                gw.dependencies.append(b.service_id)
                b.callers.append(gw.service_id)

        # 2. Business services can call other business services (DAG order) and data services
        # Heavy-tailed preferential attachment for data services (hub services)
        data_weights = np.array([1.0 / (i + 1) ** 0.8 for i in range(len(data))])
        data_weights /= data_weights.sum()

        for idx, b in enumerate(business):
            # Internal business service dependencies (strict forward DAG)
            if idx > 0 and self.rng.random() < 0.35:
                upstream_candidates = business[:idx]
                target = self.rng.choice(upstream_candidates)
                target.dependencies.append(b.service_id)
                b.callers.append(target.service_id)

            # Connect to data tier
            num_data_deps = min(len(data), int(self.rng.integers(1, max(2, min(4, len(data) + 1)))))
            chosen_data = self.rng.choice(data, size=num_data_deps, replace=False, p=data_weights)
            for d in chosen_data:
                if d.service_id not in b.dependencies:
                    b.dependencies.append(d.service_id)
                if b.service_id not in d.callers:
                    d.callers.append(b.service_id)

        # Ensure all business services have at least one caller
        for b in business:
            if not b.callers and gateways:
                chosen_gw = self.rng.choice(gateways)
                chosen_gw.dependencies.append(b.service_id)
                b.callers.append(chosen_gw.service_id)

        # Ensure all data services have at least one caller so none are disconnected
        for d in data:
            if not d.callers and business:
                chosen_b = self.rng.choice(business)
                chosen_b.dependencies.append(d.service_id)
                d.callers.append(chosen_b.service_id)

    def _place_replicas(self, services: Dict[str, Service], nodes: Dict[str, Node]):
        """Place service replicas onto infrastructure nodes."""
        node_ids = list(nodes.keys())
        replica_counter = 0

        for s in services.values():
            s.replicas = []
            num_reps = s.min_replicas
            # Choose nodes for this service's replicas (could co-locate with other services)
            chosen_nodes = self.rng.choice(node_ids, size=num_reps, replace=(num_reps > len(node_ids)))
            for nid in chosen_nodes:
                rep_id = f"rep_{replica_counter}_{s.service_id}"
                replica_counter += 1
                rep = Replica(replica_id=rep_id, service_id=s.service_id, node_id=nid, is_alive=True)
                s.replicas.append(rep)
                nodes[nid].running_replicas.append(rep_id)

    def _plant_weaknesses(
        self, services: Dict[str, Service], nodes: Dict[str, Node]
    ) -> List[Weakness]:
        """Plant deterministic ground-truth weaknesses into the system."""
        weaknesses: List[Weakness] = []
        w_counter = 0

        # Weakness 1: Single-replica critical hub service
        # Pick a data or business service with high caller degree, set replica=1
        hub_candidates = sorted(
            [s for s in services.values() if s.layer in (1, 2) and len(s.callers) >= 1],
            key=lambda s: len(s.callers),
            reverse=True,
        )
        if hub_candidates:
            hub = hub_candidates[0]
            # Enforce single replica
            hub.min_replicas = 1
            hub.max_replicas = 1
            # Prune replicas to 1
            kept_rep = hub.replicas[0]
            # Remove others from nodes
            for r in hub.replicas[1:]:
                nodes[r.node_id].running_replicas = [
                    x for x in nodes[r.node_id].running_replicas if x != r.replica_id
                ]
            hub.replicas = [kept_rep]

            w_counter += 1
            weaknesses.append(
                Weakness(
                    weakness_id=f"W{w_counter}_SINGLE_REPLICA_HUB",
                    weakness_type="SINGLE_REPLICA_CRITICAL",
                    description=f"Critical hub {hub.service_id} with {len(hub.callers)} callers has only 1 replica.",
                    is_compound=False,
                    trigger_fault_types=[FaultType.POD_KILL],
                    target_services=[hub.service_id],
                    required_min_intensity=IntensityLevel.LOW,
                )
            )

        # Weakness 2: Missing timeout on hot path
        # A caller has infinite / high timeout (e.g. 5000ms) to a dependency
        for s in services.values():
            if s.layer == 1 and s.dependencies:
                dep_id = s.dependencies[0]
                s.timeout = 5000.0  # Excessive timeout
                w_counter += 1
                weaknesses.append(
                    Weakness(
                        weakness_id=f"W{w_counter}_MISSING_TIMEOUT",
                        weakness_type="MISSING_TIMEOUT",
                        description=f"Service {s.service_id} has unbounded timeout to {dep_id}.",
                        is_compound=False,
                        trigger_fault_types=[FaultType.LATENCY_INJECTION],
                        target_services=[dep_id],
                        required_min_intensity=IntensityLevel.HIGH,
                    )
                )
                break

        # Weakness 3: No circuit breaker on flaky dependency
        for s in services.values():
            if s.layer == 1 and s.dependencies:
                dep_id = s.dependencies[-1]
                s.circuit_breaker_enabled = False
                s.fallback_enabled = False
                w_counter += 1
                weaknesses.append(
                    Weakness(
                        weakness_id=f"W{w_counter}_NO_CIRCUIT_BREAKER",
                        weakness_type="NO_CIRCUIT_BREAKER",
                        description=f"Service {s.service_id} lacks circuit breaker or fallback for {dep_id}.",
                        is_compound=False,
                        trigger_fault_types=[FaultType.DEPENDENCY_ERROR],
                        target_services=[dep_id],
                        required_min_intensity=IntensityLevel.MEDIUM,
                    )
                )
                break

        # Weakness 4: Correlated node co-location of critical replicas
        # Place single replicas of two key services on the same node
        if len(hub_candidates) >= 2 and len(nodes) >= 2:
            s1 = hub_candidates[0]
            s2 = hub_candidates[1]
            target_node_id = list(nodes.keys())[0]
            # Relocate s1 rep 0 and s2 rep 0 to target_node
            r1 = s1.replicas[0]
            r2 = s2.replicas[0]
            nodes[r1.node_id].running_replicas = [
                x for x in nodes[r1.node_id].running_replicas if x != r1.replica_id
            ]
            nodes[r2.node_id].running_replicas = [
                x for x in nodes[r2.node_id].running_replicas if x != r2.replica_id
            ]
            r1.node_id = target_node_id
            r2.node_id = target_node_id
            nodes[target_node_id].running_replicas.extend([r1.replica_id, r2.replica_id])

            w_counter += 1
            weaknesses.append(
                Weakness(
                    weakness_id=f"W{w_counter}_CORRELATED_COLOCATION",
                    weakness_type="CORRELATED_COLOCATION",
                    description=f"Critical services {s1.service_id} and {s2.service_id} share node {target_node_id}.",
                    is_compound=False,
                    trigger_fault_types=[FaultType.NODE_FAILURE],
                    target_services=[s1.service_id, s2.service_id],
                    target_nodes=[target_node_id],
                    required_min_intensity=IntensityLevel.LOW,
                )
            )

        # Weakness 5: Low capacity margin with no autoscaling
        for s in services.values():
            if s.layer == 1 and not s.autoscaling_enabled:
                s.capacity_per_replica = 45.0  # Marginally small capacity
                w_counter += 1
                weaknesses.append(
                    Weakness(
                        weakness_id=f"W{w_counter}_LOW_CAPACITY_MARGIN",
                        weakness_type="LOW_CAPACITY_NO_AUTOSCALE",
                        description=f"Service {s.service_id} has thin capacity margin and no autoscaling.",
                        is_compound=False,
                        trigger_fault_types=[FaultType.CPU_STRESS],
                        target_services=[s.service_id],
                        required_min_intensity=IntensityLevel.MEDIUM,
                    )
                )
                break

        # Weakness 6: COMPOUND: Downstream Latency + Aggressive Upstream Retries -> Retry Storm
        for s in services.values():
            if s.layer == 1 and len(s.dependencies) >= 1:
                dep_id = s.dependencies[0]
                dep_svc = services[dep_id]
                s.max_retries = 3
                s.circuit_breaker_enabled = False
                dep_svc.circuit_breaker_enabled = False
                w_counter += 1
                weaknesses.append(
                    Weakness(
                        weakness_id=f"W{w_counter}_COMPOUND_RETRY_STORM",
                        weakness_type="COMPOUND_RETRY_STORM",
                        description=f"Latency on {dep_id} combined with packet loss or retries on {s.service_id} triggers retry storm.",
                        is_compound=True,
                        trigger_fault_types=[FaultType.LATENCY_INJECTION, FaultType.PACKET_LOSS],
                        target_services=[dep_id, s.service_id],
                        required_min_intensity=IntensityLevel.LOW,
                    )
                )
                break

        # Weakness 7: COMPOUND: Pod kill on one replica + CPU stress on another
        multi_rep_services = [s for s in services.values() if len(s.replicas) >= 2]
        if multi_rep_services:
            target_svc = multi_rep_services[0]
            w_counter += 1
            weaknesses.append(
                Weakness(
                    weakness_id=f"W{w_counter}_COMPOUND_POD_AND_CPU",
                    weakness_type="COMPOUND_POD_AND_CPU",
                    description=f"Pod kill on {target_svc.service_id} coupled with CPU stress exhausts remaining replica capacity.",
                    is_compound=True,
                    trigger_fault_types=[FaultType.POD_KILL, FaultType.CPU_STRESS],
                    target_services=[target_svc.service_id],
                    required_min_intensity=IntensityLevel.MEDIUM,
                )
            )

        return weaknesses
