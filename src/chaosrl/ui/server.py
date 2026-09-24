"""FastAPI backend server for the ChaosRL interactive portal and real-time visualizer."""

import os
import sys
import json
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np

from chaosrl.simulator.generator import SystemGenerator, get_num_nodes
from chaosrl.simulator.models import Fault, FaultType, IntensityLevel
from chaosrl.env import ChaosEnv
from chaosrl.agents import (
    BaseAgent,
    ChaosMonkeyAgent,
    RandomAllAgent,
    RoundRobinAgent,
    BanditAgent,
    TabularQAgent,
    MaskablePPOAgent,
)

app = FastAPI(title="ChaosRL Interactive Portal", version="0.1.0")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
FIGURES_DIR = os.path.abspath("figures")
RESULTS_DIR = os.path.abspath("results")
DOCS_DIR = os.path.abspath("docs")
ROOT_DIR = os.path.abspath(".")

# Mount static assets and generated figures
if os.path.exists(FIGURES_DIR):
    app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")


class ActiveSession:
    """Holds active interactive simulation and agent state."""
    def __init__(self):
        self.env: Optional[ChaosEnv] = None
        self.agent: Optional[BaseAgent] = None
        self.agent_type: str = "ChaosRL-PPO"
        self.obs: Optional[np.ndarray] = None
        self.step_count: int = 0
        self.history: List[Dict[str, Any]] = []

session = ActiveSession()


class InitRequest(BaseModel):
    num_services: int = 20
    seed: int = 42
    budget: int = 35
    agent_type: str = "ChaosRL-PPO"
    reward_variant: str = "R2"


class ManualFaultRequest(BaseModel):
    fault_type: str
    target: str
    intensity: str = "high"


@app.get("/api/status")
def get_status():
    """Returns runtime health and system information."""
    checkpoints = []
    ckpt_dir = os.path.join(RESULTS_DIR, "checkpoints")
    if os.path.exists(ckpt_dir):
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith(".zip")]

    return {
        "status": "online",
        "python_version": sys.version.split()[0],
        "available_checkpoints": checkpoints,
        "results_ready": os.path.exists(os.path.join(RESULTS_DIR, "eval_summary.csv")),
    }


@app.get("/api/docs/{doc_name}")
def get_document(doc_name: str):
    """Fetches raw markdown content for project documentation."""
    safe_name = os.path.basename(doc_name)
    paths_to_check = [
        os.path.join(ROOT_DIR, safe_name),
        os.path.join(DOCS_DIR, safe_name),
    ]

    for p in paths_to_check:
        if os.path.exists(p) and p.endswith(".md"):
            with open(p, "r", encoding="utf-8") as f:
                return {"doc_name": safe_name, "content": f.read()}

    raise HTTPException(status_code=404, detail=f"Document {safe_name} not found.")


@app.get("/api/benchmarks/summary")
def get_benchmarks_summary():
    """Returns precomputed evaluation metrics across scales and baselines."""
    summary_csv = os.path.join(RESULTS_DIR, "eval_summary.csv")
    if not os.path.exists(summary_csv):
        return {"records": [], "grouped": []}

    df = pd.read_csv(summary_csv)
    grouped = df.groupby(["scale", "method"]).agg({
        "weaknesses_found": ["mean", "std"],
        "discovery_rate": "mean",
        "simple_weaknesses": "mean",
        "compound_weaknesses": "mean",
        "total_damage": ["mean", "std"],
        "exp_to_first": "mean",
        "exp_to_50": "mean",
        "emergent_violations": "mean",
        "mean_decision_ms": "mean",
    }).reset_index()

    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    # Convert to JSON serializable list of dicts
    grouped_records = grouped.to_dict(orient="records")
    return {"grouped": grouped_records, "total_runs": len(df)}


@app.get("/api/benchmarks/stats")
def get_benchmarks_stats():
    """Returns Mann-Whitney U test results and effect sizes."""
    stats_file = os.path.join(RESULTS_DIR, "statistical_significance.json")
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            return json.load(f)
    return {}


@app.get("/api/benchmarks/calibration")
def get_calibration_data():
    """Returns empirical RCAEval calibration metrics."""
    calib_file = os.path.join(RESULTS_DIR, "rca_calibration_summary.json")
    if os.path.exists(calib_file):
        with open(calib_file, "r") as f:
            return json.load(f)
    return {}


@app.post("/api/simulation/init")
def init_simulation(req: InitRequest):
    """Initializes the live interactive simulation environment and agent."""
    num_nodes = get_num_nodes(req.num_services)
    gen = SystemGenerator(num_services=req.num_services, num_nodes=num_nodes, seed=req.seed)
    
    session.env = ChaosEnv(
        num_services=req.num_services,
        num_nodes=num_nodes,
        budget=req.budget,
        reward_variant=req.reward_variant,
        compound_mode=False,
        mask_already_tried=True,
        generator=gen,
    )

    session.obs, info = session.env.reset(seed=req.seed)
    session.step_count = 0
    session.history.clear()
    session.agent_type = req.agent_type

    # Instantiate chosen agent
    num_actions = len(session.env.action_table)
    if req.agent_type == "ChaosMonkey":
        session.agent = ChaosMonkeyAgent(session.env.action_table, seed=req.seed)
    elif req.agent_type == "RandomAll":
        session.agent = RandomAllAgent(num_actions, seed=req.seed)
    elif req.agent_type == "RoundRobin":
        session.agent = RoundRobinAgent(num_actions, seed=req.seed)
    elif req.agent_type == "Bandit-UCB":
        session.agent = BanditAgent(num_actions, strategy="ucb", c=1.0, seed=req.seed)
    elif req.agent_type == "Bandit-EPSILON_GREEDY":
        session.agent = BanditAgent(num_actions, strategy="epsilon_greedy", epsilon=0.15, seed=req.seed)
    elif req.agent_type == "TabularQ":
        session.agent = TabularQAgent(num_actions, seed=req.seed)
    else:  # ChaosRL-PPO
        model_path = os.path.join(RESULTS_DIR, "checkpoints", f"ppo_s20_seed42_{req.reward_variant}.zip")
        if not os.path.exists(model_path):
            # Fallback to any existing model
            ckpt_dir = os.path.join(RESULTS_DIR, "checkpoints")
            if os.path.exists(ckpt_dir):
                zips = [f for f in os.listdir(ckpt_dir) if f.endswith(".zip")]
                if zips:
                    model_path = os.path.join(ckpt_dir, zips[0])
        session.agent = MaskablePPOAgent(num_actions, model_path=model_path, seed=req.seed)

    # Serialize system topology for graphical canvas rendering
    topology_services = {}
    for sid, s in session.env.services.items():
        topology_services[sid] = {
            "id": s.service_id,
            "name": s.name,
            "layer": s.layer,
            "base_latency": s.base_latency,
            "capacity": s.capacity_per_replica,
            "timeout": s.timeout,
            "circuit_breaker": s.circuit_breaker_enabled,
            "fallback": s.fallback_enabled,
            "callers": s.callers,
            "dependencies": s.dependencies,
            "replicas": [
                {"id": r.replica_id, "node": r.node_id, "alive": r.is_alive}
                for r in s.replicas
            ],
            "p99_latency": s.p99_latency,
            "error_rate": s.error_rate,
            "utilization": s.utilization,
        }

    topology_nodes = {}
    for nid, n in session.env.nodes.items():
        topology_nodes[nid] = {
            "id": n.node_id,
            "is_alive": n.is_alive,
            "running_replicas": n.running_replicas,
        }

    planted = [
        {
            "id": w.weakness_id,
            "type": w.weakness_type,
            "description": w.description,
            "is_compound": w.is_compound,
            "triggers": [ft.value for ft in w.trigger_fault_types],
            "targets": w.target_services + w.target_nodes,
        }
        for w in session.env.planted_weaknesses
    ]

    return {
        "services": topology_services,
        "nodes": topology_nodes,
        "planted_weaknesses": planted,
        "budget": req.budget,
        "agent": req.agent_type,
        "slo_latency_threshold": session.env.simulator.slo_latency_threshold,
        "slo_error_threshold": session.env.simulator.slo_error_threshold,
    }


@app.post("/api/simulation/step")
def step_simulation():
    """Executes a single fault injection step by the active agent."""
    if session.env is None or session.agent is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized.")

    mask = session.env.action_masks()
    action = session.agent.act(session.obs, action_mask=mask)
    fault_set = session.env.action_table[action]

    next_obs, reward, terminated, truncated, info = session.env.step(action)
    session.agent.update(session.obs, action, reward, next_obs, terminated, info)
    session.obs = next_obs
    session.step_count += 1

    # Format fault action details
    fault_descs = [
        f"{f.fault_type.value} on {f.target} ({f.intensity.value})"
        for f in fault_set
    ]

    step_data = {
        "step": session.step_count,
        "action_index": action,
        "fault_descriptions": fault_descs,
        "faults": [f.to_dict() for f in fault_set],
        "reward": float(reward),
        "slo_violated": info["slo_violated"],
        "damage": float(info["damage"]),
        "total_damage": float(info["total_damage"]),
        "newly_found": info["newly_found"],
        "total_found": info["total_found"],
        "total_planted": info["total_planted"],
        "discovery_rate": float(info["discovery_rate"]),
        "emergent_violations": info["emergent_violations"],
        "terminated": terminated,
        "system_p99": float(session.env.last_snapshot.system_latency_p99),
        "system_error": float(session.env.last_snapshot.system_error_rate),
        "service_latencies": session.env.last_snapshot.service_latencies,
        "service_error_rates": session.env.last_snapshot.service_error_rates,
        "service_utilizations": session.env.last_snapshot.service_utilizations,
        "service_healthy_replicas": session.env.last_snapshot.service_healthy_replicas,
    }

    session.history.append(step_data)
    return step_data


@app.post("/api/simulation/manual")
def manual_inject(req: ManualFaultRequest):
    """Allows manual fault injection from the UI to test user hypotheses."""
    if session.env is None:
        raise HTTPException(status_code=400, detail="Simulation not initialized.")

    ftype = FaultType(req.fault_type)
    intensity = IntensityLevel(req.intensity.lower())
    fault = Fault(fault_type=ftype, target=req.target, intensity=intensity)

    slo_violated, damage, snapshots = session.env.simulator.run_experiment([fault], experiment_ticks=6, recovery_ticks=3)

    return {
        "manual_fault": fault.to_dict(),
        "slo_violated": slo_violated,
        "damage": float(damage),
        "last_p99": float(snapshots[-1].system_latency_p99),
        "last_error": float(snapshots[-1].system_error_rate),
    }


# Static frontend files mount
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def run_server(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    uvicorn.run("chaosrl.ui.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
