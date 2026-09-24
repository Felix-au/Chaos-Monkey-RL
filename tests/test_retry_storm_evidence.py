"""Sanity verification demonstrating:
1. Healthy steady state with zero faults
2. Plausible degradation under individual faults
3. Concrete evidence of a retry storm under planted compound weakness
"""

from chaosrl.simulator import SystemGenerator, SystemSimulator, Fault, FaultType, IntensityLevel


def run_sanity_checks():
    print("=== ChaosRL Simulator Sanity Demonstration ===")
    gen = SystemGenerator(num_services=20, num_nodes=6, seed=42)
    services, nodes, weaknesses = gen.generate()
    sim = SystemSimulator(services, nodes, seed=42)

    # 1. Healthy steady state
    print("\n--- 1. Baseline Steady State (No Faults) ---")
    for _ in range(5):
        snap = sim.step_tick()
    print(f"System p99 Latency: {snap.system_latency_p99:.2f} ms (SLO Threshold: {sim.slo_latency_threshold} ms)")
    print(f"System Error Rate:  {snap.system_error_rate:.4f} (SLO Threshold: {sim.slo_error_threshold})")
    print(f"SLO Violated:       {snap.slo_violated}")
    assert not snap.slo_violated

    # 2. Plausible degradation under single fault
    print("\n--- 2. Single Fault Injection (Latency Injection on Gateway Dependency) ---")
    gw = [s for s in services.values() if s.layer == 0][0]
    dep_id = gw.dependencies[0]
    fault_lat = Fault(FaultType.LATENCY_INJECTION, dep_id, IntensityLevel.HIGH)
    sim.inject_fault(fault_lat)
    snap_fault = sim.step_tick()
    print(f"Under Latency Fault on {dep_id}:")
    print(f"System p99 Latency: {snap_fault.system_latency_p99:.2f} ms")
    print(f"System Error Rate:  {snap_fault.system_error_rate:.4f}")
    print(f"SLO Violated:       {snap_fault.slo_violated}")
    sim.rollback()

    # 3. Retry Storm demonstration
    print("\n--- 3. Compound Retry Storm Demonstration ---")
    retry_weaknesses = [w for w in weaknesses if w.weakness_type == "COMPOUND_RETRY_STORM"]
    assert len(retry_weaknesses) > 0, "No retry storm weakness found in system!"
    rw = retry_weaknesses[0]
    print(f"Weakness: {rw.weakness_id}")
    print(f"Description: {rw.description}")

    # Measure downstream load before fault
    dep_svc = services[rw.target_services[0]]
    load_before = dep_svc.current_load
    print(f"Target downstream service {dep_svc.service_id} load before: {load_before:.2f} req/tick")

    # Inject compound retry storm trigger
    faults = [
        Fault(FaultType.LATENCY_INJECTION, rw.target_services[0], IntensityLevel.HIGH),
        Fault(FaultType.PACKET_LOSS, rw.target_services[1], IntensityLevel.MEDIUM),
    ]
    for f in faults:
        sim.inject_fault(f)

    # Step simulation under retry storm
    snap_storm = sim.step_tick()
    load_during = dep_svc.current_load
    print(f"Target downstream service {dep_svc.service_id} load during storm: {load_during:.2f} req/tick")
    print(f"Load amplification factor: {load_during / max(1.0, load_before):.2f}x")
    print(f"Gateway Error Rate during storm: {snap_storm.system_error_rate:.4f}")
    print(f"Gateway p99 Latency during storm: {snap_storm.system_latency_p99:.2f} ms")
    print(f"SLO Violated: {snap_storm.slo_violated}")
    assert snap_storm.slo_violated
    assert load_during > load_before

    sim.rollback()
    snap_recovered = sim.step_tick()
    print(f"Post-rollback recovery latency: {snap_recovered.system_latency_p99:.2f} ms, SLO Violated: {snap_recovered.slo_violated}")
    assert not snap_recovered.slo_violated
    print("\n=== All Sanity Demonstrations Successfully Passed! ===")


if __name__ == "__main__":
    run_sanity_checks()
