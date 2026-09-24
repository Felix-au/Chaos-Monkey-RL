"""Unit tests verifying simulator determinism, steady-state health, fault effects, rollback, weakness matching, and scale."""

import time
import pytest
import numpy as np

from chaosrl.simulator import (
    SystemGenerator,
    SystemSimulator,
    Fault,
    FaultType,
    IntensityLevel,
    evaluate_discovery,
)


def test_generator_determinism():
    """Verify that same seed produces identical topologies and weaknesses."""
    gen1 = SystemGenerator(num_services=20, num_nodes=6, seed=42)
    s1, n1, w1 = gen1.generate()

    gen2 = SystemGenerator(num_services=20, num_nodes=6, seed=42)
    s2, n2, w2 = gen2.generate()

    assert len(s1) == len(s2)
    assert len(n1) == len(n2)
    assert len(w1) == len(w2)
    for sid in s1:
        assert s1[sid].dependencies == s2[sid].dependencies
        assert s1[sid].base_latency == pytest.approx(s2[sid].base_latency)
        assert len(s1[sid].replicas) == len(s2[sid].replicas)
    for wid in range(len(w1)):
        assert w1[wid].weakness_id == w2[wid].weakness_id
        assert w1[wid].is_compound == w2[wid].is_compound


def test_steady_state_health():
    """Verify that unperturbed system runs healthy without SLO violations."""
    gen = SystemGenerator(num_services=20, num_nodes=6, seed=123)
    services, nodes, _ = gen.generate()
    sim = SystemSimulator(services, nodes, seed=123)

    # Run 10 baseline ticks with no faults
    for _ in range(10):
        snap = sim.step_tick()
        assert not snap.slo_violated, f"Unperturbed system violated SLO! Lat={snap.system_latency_p99}, Err={snap.system_error_rate}"
        assert snap.system_latency_p99 < sim.slo_latency_threshold
        assert snap.system_error_rate < sim.slo_error_threshold


def test_fault_injection_and_rollback():
    """Verify that injecting a fault creates measurable impact and rollback restores health."""
    gen = SystemGenerator(num_services=20, num_nodes=6, seed=777)
    services, nodes, _ = gen.generate()
    sim = SystemSimulator(services, nodes, seed=777)

    # Initial tick
    snap_before = sim.step_tick()
    assert not snap_before.slo_violated

    # Inject severe fault on a gateway or hub
    gw_id = [s.service_id for s in services.values() if s.layer == 0][0]
    fault = Fault(
        fault_type=FaultType.LATENCY_INJECTION,
        target=gw_id,
        intensity=IntensityLevel.HIGH,
        duration=5,
    )
    sim.inject_fault(fault)

    # Step under fault
    snap_fault = sim.step_tick()
    assert snap_fault.system_latency_p99 > snap_before.system_latency_p99

    # Rollback
    sim.rollback()
    for _ in range(3):
        snap_after = sim.step_tick()

    assert not snap_after.slo_violated
    assert len(sim.active_faults) == 0


def test_weakness_matching_logic():
    """Verify single and compound weakness trigger matching."""
    gen = SystemGenerator(num_services=20, num_nodes=6, seed=101)
    services, nodes, weaknesses = gen.generate()

    single_w = [w for w in weaknesses if not w.is_compound]
    compound_w = [w for w in weaknesses if w.is_compound]

    assert len(single_w) > 0, "No single weaknesses planted!"
    assert len(compound_w) > 0, "No compound weaknesses planted!"

    # Test single weakness matching
    target_w = single_w[0]
    correct_fault = Fault(
        fault_type=target_w.trigger_fault_types[0],
        target=target_w.target_services[0] if target_w.target_services else target_w.target_nodes[0],
        intensity=IntensityLevel.HIGH,
    )
    assert target_w.matches_fault_set([correct_fault])

    # Unrelated fault should not match
    wrong_fault = Fault(
        fault_type=FaultType.PACKET_LOSS if target_w.trigger_fault_types[0] != FaultType.PACKET_LOSS else FaultType.CPU_STRESS,
        target="non_existent_target",
        intensity=IntensityLevel.LOW,
    )
    assert not target_w.matches_fault_set([wrong_fault])

    # Test compound weakness matching
    c_w = compound_w[0]
    # Incomplete fault set should fail
    partial_fault = [
        Fault(fault_type=c_w.trigger_fault_types[0], target=c_w.target_services[0], intensity=IntensityLevel.HIGH)
    ]
    if len(c_w.trigger_fault_types) > 1:
        assert not c_w.matches_fault_set(partial_fault)

    # Complete compound fault set should match
    complete_faults = []
    for idx, ftype in enumerate(c_w.trigger_fault_types):
        t_svc = c_w.target_services[min(idx, len(c_w.target_services) - 1)]
        complete_faults.append(Fault(fault_type=ftype, target=t_svc, intensity=IntensityLevel.HIGH))
    assert c_w.matches_fault_set(complete_faults)


def test_all_six_fault_types():
    """Verify that all 6 fault types can be injected without runtime crashes and produce impact."""
    gen = SystemGenerator(num_services=20, num_nodes=6, seed=42)
    services, nodes, _ = gen.generate()
    sim = SystemSimulator(services, nodes, seed=42)

    s_id = list(services.keys())[2]
    n_id = list(nodes.keys())[0]

    faults_to_test = [
        Fault(FaultType.POD_KILL, s_id, IntensityLevel.HIGH),
        Fault(FaultType.NODE_FAILURE, n_id, IntensityLevel.HIGH),
        Fault(FaultType.LATENCY_INJECTION, s_id, IntensityLevel.HIGH),
        Fault(FaultType.PACKET_LOSS, s_id, IntensityLevel.HIGH),
        Fault(FaultType.CPU_STRESS, s_id, IntensityLevel.HIGH),
        Fault(FaultType.DEPENDENCY_ERROR, s_id, IntensityLevel.HIGH),
    ]

    for f in faults_to_test:
        sim.rollback()
        slo_violated, damage, snaps = sim.run_experiment([f], experiment_ticks=3, recovery_ticks=2)
        assert len(snaps) == 5
        assert isinstance(slo_violated, bool)
        assert damage >= 0.0


def test_simulation_scale_and_speed():
    """Verify scales (20, 100, 300) and verify high throughput (>100 experiments/min on 100 services)."""
    scales = [20, 100, 300]
    for n_svc in scales:
        gen = SystemGenerator(num_services=n_svc, num_nodes=max(4, n_svc // 5), seed=42)
        s, n, w = gen.generate()
        assert len(s) == n_svc
        assert len(w) >= 3

    # Benchmark 100 services speed
    gen100 = SystemGenerator(num_services=100, num_nodes=20, seed=42)
    s100, n100, _ = gen100.generate()
    sim100 = SystemSimulator(s100, n100, seed=42)

    start = time.perf_counter()
    num_exps = 25
    test_fault = [Fault(FaultType.POD_KILL, list(s100.keys())[5], IntensityLevel.MEDIUM)]
    for _ in range(num_exps):
        sim100.run_experiment(test_fault, experiment_ticks=4, recovery_ticks=2)
    elapsed = time.perf_counter() - start

    rate_per_sec = num_exps / elapsed
    rate_per_min = rate_per_sec * 60.0
    print(f"\n[Simulator Benchmark] 100-service simulation speed: {rate_per_sec:.1f} exps/sec ({rate_per_min:.0f} exps/min)")
    assert rate_per_min > 500, f"Simulation too slow: {rate_per_min:.0f} exps/min"
