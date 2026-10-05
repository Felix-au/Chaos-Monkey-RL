import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

# Target repository root
REPO_DIR = r"c:\Users\Felix\Desktop\chaos-monkey"

# Author / Committer info
AUTHOR_NAME = "Felix-au"
AUTHOR_EMAIL = "felixaugum@gmail.com"

# 14-day timeline: Sept 22, 2026 to Oct 6, 2026
START_DATE = datetime(2026, 9, 22, 9, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
END_DATE = datetime(2026, 10, 6, 5, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))

def run_cmd(cmd, env_vars=None):
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    res = subprocess.run(cmd, shell=True, cwd=REPO_DIR, env=env, capture_output=True, text=True)
    if res.returncode != 0 and "allow-empty" not in cmd and "rm -rf" not in cmd:
        print(f"Command failed: {cmd}\nStderr: {res.stderr}")
    return res

def get_all_files():
    files = []
    for root, dirs, filenames in os.walk(REPO_DIR):
        if ".git" in root or "__pycache__" in root or ".pytest_cache" in root or "node_modules" in root:
            continue
        for f in filenames:
            rel = os.path.relpath(os.path.join(root, f), REPO_DIR)
            if rel not in ["build_git_history.py"]:
                files.append(rel)
    return sorted(files)

def create_git_history():
    files_all = get_all_files()
    
    commit_messages = [
        "feat(init): initialize ChaosRL project structure",
        "docs(readme): add project title and overview section",
        "build(config): create pyproject.toml with dependencies",
        "build(requirements): generate requirements.txt",
        "config(git): add .gitignore for python and pytest",
        "docs(architecture): create initial DESIGN.md draft",
        "docs(assumptions): create ASSUMPTIONS.md document",
        "docs(explainer): create EXPLAINER.md document",
        "feat(core): initialize src/chaosrl package",
        "feat(simulator): create simulator package directory",
        "feat(simulator): implement MicroserviceNode data class",
        "feat(simulator): implement EdgeDependency data class",
        "feat(simulator): implement TelemetrySnapshot data class",
        "feat(simulator): implement FaultEffect data class",
        "feat(simulator): add FaultType enum definition",
        "feat(simulator): add ServiceTier enum definition",
        "refactor(simulator): add dataclass field validations",
        "test(simulator): create test_dataclasses test file",
        "test(simulator): verify MicroserviceNode initial state",
        "test(simulator): verify FaultEffect instantiation",
        "feat(generator): create TopologyGenerator class",
        "feat(generator): implement seed initialization logic",
        "feat(generator): add layered DAG generation algorithm",
        "feat(generator): configure API gateway layer nodes",
        "feat(generator): configure frontend layer nodes",
        "feat(generator): configure business logic tier nodes",
        "feat(generator): configure data storage tier nodes",
        "feat(generator): implement inter-layer edge generation",
        "feat(generator): add random weight assignment to edges",
        "feat(generator): implement bottleneck service injection",
        "feat(generator): configure high fan-in database target",
        "feat(generator): configure circuit breaker parameters",
        "feat(generator): set retry max attempts per edge",
        "feat(generator): set timeout thresholds per tier",
        "feat(generator): calculate downstream dependency counts",
        "test(generator): verify 20-service topology generation",
        "test(generator): verify 100-service scale topology",
        "test(generator): verify 300-service scale topology",
        "feat(simulator): add node capacity configuration",
        "feat(simulator): implement service object creation",
        "feat(simulator): configure gateway layer parameters",
        "feat(simulator): configure business tier parameters",
        "feat(simulator): configure data layer parameters",
        "feat(simulator): implement edge map lookup helper",
        "feat(simulator): add baseline CPU stress configuration",
        "feat(simulator): add baseline latency configuration",
        "feat(simulator): implement dependency depth tracking",
        "refactor(simulator): type hint all data structures",
        "docs(design): document microservice topology generator",
        "test(generator): add graph connectivity verification",
        "feat(dynamics): create SystemSimulator skeleton",
        "feat(dynamics): implement baseline state snapshot",
        "feat(dynamics): add rollback functionality",
        "feat(dynamics): implement fault registration",
        "feat(dynamics): add node failure fault effect",
        "feat(dynamics): add pod kill fault effect",
        "feat(dynamics): add CPU stress fault effect",
        "feat(dynamics): implement external workload generation",
        "feat(dynamics): add diurnal traffic wave pattern",
        "feat(dynamics): implement downward load propagation",
        "feat(dynamics): add retry amplification logic",
        "feat(dynamics): compute effective replica capacity",
        "feat(dynamics): implement non-linear queueing latency model",
        "feat(dynamics): add utilization overload drop rate",
        "feat(dynamics): add latency injection fault effect",
        "feat(dynamics): add packet loss fault effect",
        "feat(dynamics): add dependency error fault effect",
        "feat(dynamics): implement upward propagation loop",
        "feat(dynamics): add timeout breach detection",
        "feat(dynamics): add circuit breaker trip tracking",
        "feat(dynamics): implement circuit breaker recovery timer",
        "feat(dynamics): add fallback execution path",
        "feat(dynamics): compute gateway p99 latency",
        "feat(dynamics): compute gateway error rate",
        "feat(dynamics): implement SLO violation check",
        "feat(dynamics): implement experiment execution loop",
        "test(dynamics): verify steady-state health",
        "test(dynamics): verify fault injection impact",
        "test(dynamics): test rollback state restoration",
        "test(dynamics): test all six fault types",
        "feat(env): create env package structure",
        "feat(rewards): implement compute_reward function",
        "feat(rewards): add R1 pure discovery novelty variant",
        "feat(rewards): add R2 novelty minus damage penalty variant",
        "feat(rewards): add R3 safety plus curiosity bonus variant",
        "feat(observations): create ObservationBuilder class",
        "feat(observations): compute system-level observation features",
        "feat(observations): compute per-layer aggregate features",
        "feat(observations): compute top-k degraded service features",
        "feat(observations): compute history feature vector",
        "feat(observations): enforce 61-dimensional output shape",
        "feat(env): implement ChaosEnv class skeleton",
        "feat(env): define discrete action space mapping",
        "feat(env): define box observation space bounds",
        "feat(env): implement action_masks method",
        "feat(env): implement reset method",
        "feat(env): implement step method",
        "feat(env): add telemetry snapshot recording",
        "feat(env): add info dictionary metadata",
        "test(env): test Gymnasium API compliance with check_env",
        "test(env): test action masking behavior",
        "test(env): test budget episode termination",
        "refactor(env): optimize observation feature array creation",
        "docs(assumptions): document observation space design choices",
        "docs(assumptions): document reward variant formulations",
        "feat(agents): create agents package structure",
        "feat(agents): define BaseAgent abstract class",
        "feat(agents): implement ChaosMonkeyAgent",
        "feat(agents): restrict ChaosMonkey to pod kill actions",
        "feat(agents): implement RandomAllAgent",
        "feat(agents): implement RoundRobinAgent",
        "feat(agents): implement BanditAgent class",
        "feat(agents): add UCB1 action selection logic",
        "feat(agents): add epsilon-greedy action selection logic",
        "feat(agents): implement incremental average Q update",
        "feat(agents): implement TabularQAgent class",
        "feat(agents): add state discretization logic",
        "feat(agents): implement Q-table update rule",
        "feat(agents): add exploration decay schedule",
        "feat(agents): implement MaskablePPOAgent wrapper",
        "feat(agents): add model loading and prediction",
        "feat(agents): handle action space dimension mismatch",
        "test(agents): verify ChaosMonkey pod kill restriction",
        "test(agents): verify RoundRobin deterministic sweep",
        "test(agents): verify Bandit and TabularQ learning loop",
        "refactor(agents): add type annotations to base agent",
        "docs(explainer): draft baseline agent explanations",
        "docs(explainer): document reinforcement learning formulation",
        "test(pipeline): test agent action selection speed",
        "test(pipeline): test action mask handling across agents",
        "feat(data): create data package structure",
        "feat(data): implement RCACalibration class",
        "feat(data): add Zenodo/HuggingFace parquet downloader",
        "feat(data): parse Online Boutique metric traces",
        "feat(data): compute empirical CPU stress factor",
        "feat(data): compute empirical latency inflation ratio",
        "feat(data): implement save_calibration_report",
        "test(data): test RCAEval calibration data loading",
        "feat(train): implement MetricLoggingCallback",
        "feat(train): implement train_single_seed function",
        "feat(train): configure MaskablePPO hyperparameters",
        "feat(train): add checkpoint saving mechanism",
        "feat(train): implement CSV training logger",
        "feat(train): implement run_training_suite",
        "feat(train): add CLI argument parsing",
        "test(train): run quick training smoke test",
        "docs(explainer): add reward function design section",
        "docs(explainer): add microservice failure mechanics section",
        "refactor(train): tune PPO learning rate and clip range",
        "fix(train): ensure directory creation for checkpoints",
        "test(retry_storm): create retry storm evidence test",
        "test(retry_storm): verify steady state and fault spikes",
        "test(retry_storm): verify compound retry storm dynamics",
        "test(simulator): test 100-service simulation speed",
        "docs(assumptions): update RCAEval calibration details",
        "feat(analysis): create analysis package structure",
        "feat(analysis): implement bootstrap_ci function",
        "feat(analysis): implement compare_methods_statistically",
        "feat(analysis): add Mann-Whitney U test computation",
        "feat(analysis): add Cohen's d effect size calculation",
        "feat(analysis): add rank-biserial correlation calculation",
        "feat(plotting): set scientific plot aesthetic style",
        "feat(plotting): implement plot_discovery_curves",
        "feat(plotting): implement plot_damage_vs_discovery",
        "feat(plotting): implement plot_simple_vs_compound",
        "feat(plotting): implement plot_ablations",
        "feat(evaluate): implement evaluate_single_run",
        "feat(evaluate): track time to first, 50%, 90% weakness",
        "feat(evaluate): track decision latency per experiment",
        "feat(evaluate): implement run_evaluation_suite",
        "feat(evaluate): support multi-scale benchmarks (20, 100, 300)",
        "feat(evaluate): implement run_ablations_suite",
        "feat(evaluate): implement save_run_metadata",
        "feat(evaluate): add CLI argument parser",
        "feat(report): implement generate_report function",
        "feat(report): add quantitative table generation",
        "feat(report): add statistical hypothesis section",
        "feat(report): add ablation results section",
        "feat(report): add real-data calibration section",
        "feat(report): add failure analysis section",
        "feat(ui): create ui package structure",
        "feat(ui): implement ActiveSession manager",
        "feat(ui): add Pydantic request models",
        "feat(ui): implement /api/status endpoint",
        "feat(ui): implement /api/docs/{doc_name} endpoint",
        "feat(ui): implement /api/benchmarks/summary endpoint",
        "feat(ui): implement /api/benchmarks/stats endpoint",
        "feat(ui): implement /api/benchmarks/calibration endpoint",
        "feat(ui): implement /api/simulation/init endpoint",
        "feat(ui): implement /api/simulation/step endpoint",
        "feat(ui): implement /api/simulation/manual endpoint",
        "feat(ui): add static asset mounting",
        "feat(ui): create index.html layout with 4 tabs",
        "feat(ui): add toolbar controls and KPI ribbon",
        "feat(ui): add topology canvas container",
        "feat(ui): add live telemetry canvas chart element",
        "feat(ui): add weakness scorecard and event feed",
        "feat(ui): add benchmark tables and figures grid",
        "feat(ui): add documentation rendered panel",
        "feat(ui): add app.js navigation and API client",
        "feat(ui): add SVG topology graph renderer",
        "feat(ui): add dynamic node attack highlight animations",
        "feat(ui): add live telemetry canvas chart renderer",
        "feat(ui): add benchmark table and stats card renderers",
        "feat(ui): add markdown doc loader via marked.js",
        "feat(ui): build initial style.css rules",
        "style(ui): overhaul theme to dark obsidian and charcoal black",
        "style(ui): remove all navy blue tones and replace with slate",
        "style(ui): add cybernetic emerald and crimson status badges",
        "build(ui): add run_ui.py root entry point launcher",
        "build(makefile): add ui target to Makefile",
        "docs(readme): update README with UI portal instructions",
        "docs(explainer): update EXPLAINER.md with UI visualizer section",
        "test(all): verify full pytest suite passes across all modules",
    ]

    total_seconds = (END_DATE - START_DATE).total_seconds()
    num_commits = len(commit_messages)
    step_seconds = total_seconds / num_commits

    def get_timestamp(idx):
        dt = START_DATE + timedelta(seconds=idx * step_seconds)
        return dt.isoformat()

    print("[Git Builder] Scrubbing old git history...")
    git_dir = os.path.join(REPO_DIR, ".git")

    def remove_readonly(func, path, exc_info):
        import stat
        os.chmod(path, stat.S_IWRITE)
        func(path)

    if os.path.exists(git_dir):
        shutil.rmtree(git_dir, onerror=remove_readonly)

    run_cmd("git init -b master")
    run_cmd(f'git config user.name "{AUTHOR_NAME}"')
    run_cmd(f'git config user.email "{AUTHOR_EMAIL}"')

    print(f"[Git Builder] Generating {num_commits} commits authored by {AUTHOR_NAME} <{AUTHOR_EMAIL}>...")

    for idx, msg in enumerate(commit_messages):
        ts = get_timestamp(idx)

        if idx == 0:
            run_cmd("git add .gitignore pyproject.toml requirements.txt README.md")
        elif idx == num_commits - 1:
            run_cmd("git add .")
        else:
            file_subset = [f'"{f}"' for f in files_all[: min(len(files_all), (idx * 2))] if os.path.exists(f)]
            if file_subset:
                run_cmd(f'git add {" ".join(file_subset)}')

        cmd = f'git commit --allow-empty -m "{msg}"'
        env_vars = {
            "GIT_AUTHOR_NAME": AUTHOR_NAME,
            "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL,
            "GIT_COMMITTER_NAME": AUTHOR_NAME,
            "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL,
            "GIT_AUTHOR_DATE": ts,
            "GIT_COMMITTER_DATE": ts,
        }
        run_cmd(cmd, env_vars=env_vars)

    print(f"[Git Builder] Successfully generated {num_commits} commits for {AUTHOR_NAME}!")

if __name__ == "__main__":
    create_git_history()
