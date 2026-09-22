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

## 3. Milestones & Implementation Roadmap

- **Milestone 1: Simulator & Ground Truth Generator**
  - Implement topology generator (DAG, node placement, replica counts, service attributes).
  - Implement discrete-time aggregate dynamics (load propagation, queueing latency, error rate, timeouts, retries, circuit breakers, fallbacks).
  - Implement 6 fault injection types + rollback mechanism.
  - Implement planted ground-truth weaknesses (single and compound) + deterministic matching logic.
  - Write unit tests (`tests/test_simulator.py`).

- **Milestone 2: Gymnasium Environment & Baseline Agents**
  - Implement `ChaosEnv` with Gymnasium API.
  - Build scalable observation representations (fixed-size summary + history vector).
  - Action masking integration (`sb3-contrib.common.maskable`).
  - Implement reward variants (R1, R2, R3).
  - Implement baseline agents (Chaos Monkey, Random-All, Round-Robin, UCB/Epsilon-Greedy Bandit, Tabular Q).
  - Write unit tests (`tests/test_env.py`, `tests/test_agents.py`).

- **Milestone 3: Training & Experiment Pipeline**
  - Implement `train.py` with multi-seed training, checkpointing, and evaluation tracking.
  - Implement `evaluate.py` supporting `--quick` (fast testing) and `--full` (comprehensive runs).
  - Ensure zero evaluation on training systems; strictly held-out test systems.

- **Milestone 4: RCAEval Real-Data Calibration Attempt**
  - Download / process RCAEval traces (Zenodo/GitHub) or document offline calibration.
  - Calibrate simulator parameters (latency degradation curves, error propagation).

- **Milestone 5: Execution, Analysis & Reporting**
  - Execute quick and full benchmark suites.
  - Generate plots and statistics in `figures/` and `results/`.
  - Compile comprehensive `REPORT.md`, `docs/EXPLAINER.md`, `docs/ASSUMPTIONS.md`, and `README.md`.
