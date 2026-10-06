# ChaosRL: Plain-Language Project Explainer

This document is written in clear, plain language so that any team member, student, or examiner can understand, explain, and defend every aspect of the ChaosRL project.

---

## 1. What Problem Are We Solving?

Modern web systems (like Netflix, Uber, or Amazon) are built out of dozens or hundreds of small, interconnected programs called **microservices**. For example, one service handles user logins, another searches products, and another processes credit cards.

Because these services depend on one another, a tiny glitch in one service can snowball into a total system blackout. To find these weaknesses before real users are hurt, engineers practice **Chaos Engineering**: intentionally injecting faults (like killing a server or slowing down a network link) to see if the system recovers gracefully.

### The Problem With Chaos Monkey
Netflix popularized **Chaos Monkey**, a tool that randomly kills server instances. While useful, Chaos Monkey has critical flaws:
1. **Blindness**: It has no memory of what it tested in the past.
2. **Limited Scope**: It only kills servers (pod kills); it cannot test slow networks, high CPU load, or retry storms.
3. **Expensive Damage**: Every failure in staging or canary production degrades user experience and consumes error budgets.

### The ChaosRL Solution
**ChaosRL** replaces random guessing with a **Reinforcement Learning (RL) agent**. The agent looks at a high-level summary of the system and selects intelligent, targeted experiments (single or compound faults) designed to find hidden vulnerabilities quickly while causing as little collateral damage as possible.

---

## 2. How Does the Simulator Work?

Because injecting thousands of faults into real cloud servers would cost thousands of dollars and take weeks, we built a fast, deterministic **discrete-time microservice simulator**:

- **Topology (The Map):** A layered Directed Acyclic Graph (DAG):
  - **Gateway Layer (Layer 0):** The front door where user requests arrive.
  - **Business Layer (Layer 1):** Services handling business logic (orders, cart, authentication).
  - **Data Layer (Layer 2):** Backend databases and storage services.
- **Infrastructure (The Hardware):** Services run multiple copies (**replicas**) placed across virtual machines (**nodes**). If two unrelated services share the same node and that node crashes, both services suffer simultaneously.
- **The Dynamics (The Physics of Traffic):**
  - Traffic enters at the gateway and flows down to dependencies.
  - As traffic approaches a service's capacity limit, queueing latency spikes non-linearly (just like traffic jams on a highway).
  - If a service fails or times out, upstream callers may retry. If too many callers retry at once, it triggers a **retry storm**, which can take down healthy services.
  - Services equipped with **circuit breakers** and **fallbacks** stop calling failing dependencies, protecting the rest of the system.
- **Rollback:** After each experiment, the system automatically rolls back all faults, allowing services to return to their healthy baseline state.

---

## 3. Ground-Truth Weaknesses: How We Measure Success

To evaluate methods scientifically, we plant specific, realistic architectural weaknesses when a system is generated:
1. **Single-Replica Hub**: A crucial service that many others depend on, but with only one replica. If it dies, the whole system fails.
2. **Missing Timeout**: A service waiting forever for a slow dependency, causing its own threads to pile up and crash.
3. **No Circuit Breaker**: A service that keeps hammering an erroring backend instead of fast-failing.
4. **Correlated Co-location**: Two critical services accidentally sharing the same physical node.
5. **Low Capacity Margin**: A service barely handling normal traffic with no autoscaling.
6. **Compound Retry Storm**: A downstream latency glitch combined with aggressive retries upstream, overwhelming the gateway.
7. **Compound Pod + CPU Stress**: One replica is killed while the remaining replica is throttled with CPU load.

**The Discovery Rule:** A weakness is officially "discovered" if and only if:
1. The agent's injected fault matches the weakness's trigger condition, **AND**
2. The system-level Service Level Objective (SLO) is actually violated in simulation.

---

## 4. The Reinforcement Learning Formulation

- **State / Observation (What the Agent Sees):**
  A fixed 61-number summary:
  - System-level latency and error rate.
  - Per-layer averages and maximums (latency, errors, utilization, alive replicas).
  - Telemetry of the top 5 most degraded services.
  - History vector (progress through budget, fraction of weaknesses found, fault types previously tried).
- **Action Space (What the Agent Can Do):**
  A list of discrete experiments: targeting a specific service or node with a specific fault (kill, latency, packet loss, CPU stress, dependency error).
- **Action Masking:**
  Using `MaskablePPO`, invalid or already-tested actions are masked out so the agent doesn't waste its budget re-testing things it already checked.
- **Reward Function (What Drives the Agent):**
  We compare three reward functions:
  - **R1 (Pure Novelty):** +1 for every newly discovered weakness; 0 otherwise.
  - **R2 (Safety-Constrained):** +1 for every new discovery, minus a penalty proportional to the damage caused.
  - **R3 (Safety + Curiosity):** R2 minus a penalty for repeating actions, plus a small bonus for discovering sub-SLO degradation.

---

## 5. The Baseline Competitors & Evaluation Metrics

To make the comparison fair, every method gets the **exact same experiment budget** and tests the **exact same systems**:
1. **Chaos Monkey (Faithful):** Randomly picks a service and kills a replica. Cannot inject latency or packet loss.
2. **Random-All:** Randomly picks any service or node and any fault type.
3. **Round-Robin:** Systematically tests actions one by one in fixed order.
4. **Multi-Armed Bandit (UCB & Epsilon-Greedy):** Treats each fault injection as an "arm" on a slot machine, balancing exploration of untried faults with exploitation of high-yield ones.
5. **Tabular Q-Learning:** A simplified reinforcement learning agent using discrete bins, useful for educational comparison on small systems.

### Understanding the Evaluation Metrics

- **Weaknesses Found (Mean ± SD):** The total number of planted architectural vulnerabilities uncovered during the episode. Higher is better.
- **Discovery Rate (%):** The percentage of planted weaknesses found (Found / Planted).
- **Total Damage:** Accumulated SLO-violation time and error-rate cost incurred during experiments: `D = Σ [error_rate + (p99 - 300)/1000]`. Measures collateral system damage. **Lower is safer.**
- **Exp to 1st Weakness:** The number of experiments required to discover the very first weakness. Measures initial sample efficiency. **Lower is better.**
- **Exp to 50% / 90%:** Number of experiments required to uncover 50% / 90% of all planted weaknesses.
- **Decision Latency (ms):** Wall-clock decision time taken by the policy per experiment step (e.g. 0.65ms for ChaosRL vs 0.05ms for random). Demonstrates CPU decision speed.

### Benchmark Results & Policy Comparisons

- **ChaosRL (MaskablePPO) vs. Chaos Monkey:**
  - On 20 services, ChaosRL reaches the first weakness in **1.0 experiment** on average (instant targeted hit), whereas Chaos Monkey takes **7.8 experiments**.
  - Chaos Monkey is structurally blind to 5 out of 6 fault categories (network delay, packet drops, CPU stress, retry storms) because it only kills pods.
- **ChaosRL vs. Round-Robin & Tabular Q (Damage Mitigation):**
  - Round-Robin causes **50x higher damage** (9,436.69 vs 186.02, p = 0.00018, Cohen's d = 3.54).
  - Tabular Q causes **26x higher damage** (4,944.22 vs 186.02, p = 0.00018, Cohen's d = 4.22).
  - Round-Robin and Tabular Q blindly trigger prolonged, severe outages without safety constraints. ChaosRL's R2 reward penalty forces it to discover weaknesses cleanly without crashing the gateway.
- **Multi-Armed Bandits vs. ChaosRL:**
  - Bandits find simple weaknesses quickly in small action spaces (2.76 vs 1.99 on 20 services), but incur **6.8x higher collateral damage** (1,261.57 vs 186.02) because bandits have no safety-constrained penalty terms.

---

## 6. How to Read Each Generated Figure

- **Figure 1 (`figures/fig1_discovery_curves.png`):**
  - **X-axis:** Number of experiments used (the budget).
  - **Y-axis:** Cumulative weaknesses discovered.
  - **Shaded bands:** 95% bootstrap confidence intervals across seeds.
  - **Key takeaway:** A steeper curve that plateaus higher means the agent finds weaknesses faster and finds more of them.
- **Figure 2 (`figures/fig2_damage_vs_discovery.png`):**
  - **X-axis:** Total damage caused (lower is safer).
  - **Y-axis:** Weaknesses found (higher is better).
  - **Ideal position:** The top-left corner (maximum discovery, minimum damage).
- **Figure 3 (`figures/fig3_simple_vs_compound.png`):**
  - Compares how many single-fault weaknesses vs. tricky multi-fault compound weaknesses each method uncovered.
- **Figure 4 (`figures/fig4_ablations.png`):**
  - Compares the R1, R2, and R3 reward designs, and shows how well a policy trained on 20 services performs when tested on a 100-service system.

---

## 7. Real-Data Calibration (RCAEval)

We downloaded real telemetry traces from the **RCAEval** benchmark (Online Boutique application). The real data proved that during CPU fault injection, CPU usage spikes by over **35x** compared to normal traffic. We used this empirical ratio to calibrate our simulator's CPU stress parameters, ensuring our simulated mechanics reflect real cloud behavior.
