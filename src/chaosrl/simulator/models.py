"""Data models representing microservice nodes, replicas, services, faults, and planted weaknesses."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set


class FaultType(str, Enum):
    POD_KILL = "POD_KILL"
    NODE_FAILURE = "NODE_FAILURE"
    LATENCY_INJECTION = "LATENCY_INJECTION"
    PACKET_LOSS = "PACKET_LOSS"
    CPU_STRESS = "CPU_STRESS"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"


class IntensityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Replica:
    replica_id: str
    service_id: str
    node_id: str
    is_alive: bool = True
    cpu_stress: float = 0.0  # Fraction of capacity consumed by stress (0.0 to 1.0)


@dataclass
class Node:
    node_id: str
    cpu_capacity: float = 100.0  # Percentage or units
    memory_capacity: float = 100.0
    is_alive: bool = True
    running_replicas: List[str] = field(default_factory=list)  # replica_ids


@dataclass
class Service:
    service_id: str
    name: str
    layer: int  # 0: Gateway, 1: Business, 2: Data/Storage
    base_latency: float  # ms
    capacity_per_replica: float  # requests/tick
    timeout: float  # ms timeout when calling downstream
    max_retries: int  # retries on failure/timeout
    backoff_factor: float = 1.5
    circuit_breaker_enabled: bool = True
    cb_failure_threshold: float = 0.4  # error rate threshold to trip
    cb_recovery_ticks: int = 5
    cb_state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    cb_fail_count: int = 0
    cb_timer: int = 0
    fallback_enabled: bool = False
    fallback_latency: float = 20.0
    fallback_error_rate: float = 0.0
    autoscaling_enabled: bool = False
    autoscaling_delay: int = 4
    min_replicas: int = 1
    max_replicas: int = 5
    health_check_delay: int = 2
    
    # Graph connections
    dependencies: List[str] = field(default_factory=list)  # downstream
    callers: List[str] = field(default_factory=list)        # upstream
    
    # Replicas
    replicas: List[Replica] = field(default_factory=list)
    
    # Dynamic runtime state per tick
    current_load: float = 0.0
    effective_capacity: float = 0.0
    utilization: float = 0.0
    p50_latency: float = 0.0
    p99_latency: float = 0.0
    error_rate: float = 0.0
    overload_ticks: int = 0


@dataclass
class Fault:
    fault_type: FaultType
    target: str  # service_id or node_id
    intensity: IntensityLevel = IntensityLevel.MEDIUM
    duration: int = 10  # ticks
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_type": self.fault_type.value,
            "target": self.target,
            "intensity": self.intensity.value,
            "duration": self.duration,
            "extra_params": self.extra_params,
        }


@dataclass
class Weakness:
    weakness_id: str
    weakness_type: str
    description: str
    is_compound: bool
    trigger_fault_types: List[FaultType]
    target_services: List[str]
    target_nodes: List[str] = field(default_factory=list)
    required_min_intensity: IntensityLevel = IntensityLevel.LOW
    
    def matches_fault_set(self, faults: List[Fault]) -> bool:
        """Check if injected fault set satisfies the deterministic trigger conditions."""
        if not faults:
            return False
            
        intensity_rank = {IntensityLevel.LOW: 1, IntensityLevel.MEDIUM: 2, IntensityLevel.HIGH: 3}
        min_rank = intensity_rank[self.required_min_intensity]

        if not self.is_compound:
            # Single weakness: at least one fault must match type, target, and intensity
            for f in faults:
                if f.fault_type in self.trigger_fault_types:
                    if (f.target in self.target_services) or (f.target in self.target_nodes):
                        if intensity_rank[f.intensity] >= min_rank:
                            return True
            return False
        else:
            # Compound weakness: requires simultaneous presence of all trigger conditions
            matched_conditions: Set[int] = set()
            for idx, required_type in enumerate(self.trigger_fault_types):
                for f in faults:
                    if f.fault_type == required_type:
                        if (f.target in self.target_services) or (f.target in self.target_nodes):
                            if intensity_rank[f.intensity] >= min_rank:
                                matched_conditions.add(idx)
                                break
            return len(matched_conditions) == len(self.trigger_fault_types)


@dataclass
class TelemetrySnapshot:
    tick: int
    system_latency_p99: float
    system_error_rate: float
    slo_violated: bool
    service_latencies: Dict[str, float] = field(default_factory=dict)
    service_error_rates: Dict[str, float] = field(default_factory=dict)
    service_utilizations: Dict[str, float] = field(default_factory=dict)
    service_healthy_replicas: Dict[str, int] = field(default_factory=dict)
    active_faults_count: int = 0
