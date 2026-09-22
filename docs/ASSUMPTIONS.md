# ChaosRL Simulator Assumptions & Limitations

This document explicitly details all assumptions, approximations, and simplifications made in the ChaosRL microservice simulator, detailing what is realistic, what is stylized, and the rationale behind each choice.

## 1. Discrete-Time Aggregate Dynamics vs. Continuous-Time Packet Simulation

- **Assumption**: The simulator evaluates system state in discrete time ticks (e.g., representing 1-second or 5-second aggregate intervals) rather than individual packet/RPC event queues (such as OMNeT++ or ns-3).
- **Rationale**: Simulating 100 to 500 microservices with thousands of requests per second down to individual RPC threads would require hours per single experiment. Discrete-time aggregate equations allow evaluating hundreds of experiments per minute, enabling RL agents to complete training within CPU-feasible time budgets.
- **Unrealistic Aspect**: Microsecond-level packet jitter, queue buffer overrun bursts below the tick resolution, and transient TCP SYN timeouts are smoothed into fractional loss and average queueing delays.

## 2. Queueing & Latency Model

- **Assumption**: Service response latency follows an augmented M/M/c or saturating utilization curve:
  $$\text{Latency} = \text{Base Latency} + \frac{\text{Queue Delay}}{1 - \min(U, 0.99)}$$
  where $U$ is CPU/replica utilization. When load exceeds capacity, queueing latency spikes non-linearly.
- **Rationale**: Captures real-world tail-latency behavior observed in cloud services without running real kernel scheduling.
- **Unrealistic Aspect**: Real microservices may exhibit GC pauses, thread pool starvation, or IO deadlocks with step-function latency jumps rather than continuous smooth curves.

## 3. Retries, Timeouts, and Retry Amplification (Storms)

- **Assumption**: If a dependency call exceeds the configured `timeout` or returns an error, the caller attempts up to `max_retries` with exponential backoff.
- **Cascading Effect**: When an overloaded downstream service slows down, upstream services timeout and retry, magnifying total requests ($R_{\text{total}} = R_0 \times (1 + \text{retries})$). This naturally precipitates retry storms.
- **Realistic Aspect**: Retry storms are one of the most common real-world failure modes in microservices (e.g., AWS, Netflix outage post-mortems).
- **Unrealistic Aspect**: Jitter in backoff algorithms in real systems slightly desynchronizes retries; our discrete ticks simulate aggregate retry volume within the tick interval.

## 4. Circuit Breakers & Fallbacks

- **Assumption**: Services can be configured with circuit breakers that track rolling error/timeout rates. If the error threshold (e.g. 50%) is breached, the breaker opens, rejecting or routing requests to local fallbacks/caches without querying downstream.
- **Realistic Aspect**: Mimics Envoy/Resilience4j circuit breaking semantics accurately.

## 5. Replica Placement and Multi-Tenancy

- **Assumption**: Microservices consist of $R$ replicas distributed across $M$ nodes/VMs. Unrelated services can share nodes.
- **Failure Impact**: When a node fails, all pod replicas running on that node are simultaneously killed. If two critical services happen to have replicas co-located on the same node, node failure causes correlated multi-service degradation.
- **Realistic Aspect**: Multi-tenant Kubernetes clusters regularly exhibit correlated pod disruptions during node drain, OOM killer strikes, or hypervisor reboots.

## 6. Fault Injections & Rollback

- **Assumption**: Faults (pod kill, node failure, latency injection, packet loss, CPU stress, dependency error) persist for the experiment duration $T_{\text{ticks}}$, after which the system undergoes full rollback/remediation and a recovery period.
- **Realistic Aspect**: Modern chaos tooling (Chaos Mesh, Gremlin) injects time-bounded sidecar/iptables/cgroup disruptions and cleans them up after the duration.

## 7. Planted Ground-Truth Weaknesses

- **Assumption**: Ground-truth weaknesses represent specific latent architectural defects:
  1. *Single-replica critical path*: Service on critical path with replica count = 1.
  2. *Missing timeout / unbounded wait*: Service calling slow dependency with infinite/huge timeout.
  3. *Missing circuit breaker on flaky dependency*: Cascading failure vulnerability.
  4. *Correlated co-location*: Multiple single-points-of-failure co-located on the same node.
  5. *Compound retry storm*: Latency on downstream + aggressive retries upstream.
- **Unrealistic Aspect**: In real systems, weaknesses are not binary labels; they depend on unpredictable production traffic spikes and obscure state transitions. In our benchmark, ground-truth labels allow rigorous, quantitative precision/recall evaluation.
