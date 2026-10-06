# ChaosRL Design Document

## 1. Executive Summary & Research Question

**Research Question:**
> Can a reinforcement learning agent discover more weaknesses in a microservice system, using fewer fault-injection experiments and causing less damage, than:
> (a) Chaos Monkey-style random injection (pod kills only),
> (b) Random-All (uniform random across all fault types and services),
> (c) A scripted round-robin / exhaustive sweep, and
> (d) A multi-armed bandit (epsilon-greedy and UCB)?

In microservice architectures, fault injection testing (chaos engineering) is vital for resilience. However, testing in staging or canary production environments incurs real costs (degraded user experience, error-budget consumption, compute overhead). Chaos Monkey picks instances at random with no memory. ChaosRL formulates fault selection as a Markov Decision Process (MDP) or contextual sequential decision problem, where an RL agent observes system topology and telemetry summaries, selects targeted perturbations (single or compound faults), and seeks to uncover latent resilience flaws with minimal collateral damage.

## 2. Architecture & System Overview

```
+-----------------------------------------------------------------------------------+
|                                  ChaosRL Pipeline                                 |
+-----------------------------------------------------------------------------------+
|  1. System Generator & Simulator (src/chaosrl/simulator/)                         |
|     - Layered DAG topology (Gateway -> Business Services -> Data Services)       |
|     - Replica placement on physical/virtual nodes (co-location risks)             |
|     - M/M/c queueing dynamics, retry storms, cascading timeouts, circuit breakers |
|     - Planted Ground-Truth Weaknesses (Single & Compound)                         |
+-----------------------------------------------------------------------------------+
|  2. Gymnasium Environment (src/chaosrl/env/)                                      |
|     - Observation Space: Fixed-size top-k degraded metrics, layer stats, history  |
|     - Action Space: Factored action (target service, fault type, intensity)       |
|     - Action Masking: Invalidate exhausted/invalid actions for MaskablePPO       |
|     - Rewards: R1 (Novelty), R2 (Novelty - Damage), R3 (Safety + Info Gain)       |
+-----------------------------------------------------------------------------------+
|  3. Agents & Baselines (src/chaosrl/agents/)                                      |
|     - Chaos Monkey (Pod kill only, random target)                                 |
|     - Random-All (Uniform random target, fault, intensity)                        |
|     - Round-Robin Sweep (Deterministic exhaustive sweep)                          |
|     - Multi-Armed Bandit (Epsilon-Greedy, UCB with parameter tuning)              |
|     - Tabular Q-Learning / DQN (Small scale benchmark)                            |
|     - MaskablePPO (Primary ChaosRL agent)                                         |
+-----------------------------------------------------------------------------------+
|  4. Experimentation & Evaluation Pipeline (src/chaosrl/evaluate.py, train.py)    |
|     - Train on distribution of systems; Test on strictly held-out test systems    |
|     - Scales: 20, 100, 300 services                                               |
|     - Metrics: Weakness discovery curve, time to 50%/90%, damage, emergent flaws  |
|     - Statistical significance: 95% Bootstrap CI & Mann-Whitney U tests          |
|     - Ablations: R1/R2/R3, history features, compound mode, budget curves         |
|     - Real-data Calibration: RCAEval dataset analysis & calibration               |
+-----------------------------------------------------------------------------------+
```


