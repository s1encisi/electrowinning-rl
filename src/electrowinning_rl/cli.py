"""Reproducible command-line workflows with explicit input/output paths."""

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from .config import Config
from .process import ProcessProblem, SyntheticOracle, domain
from .surrogate import fit_surrogate, generate_data

LOGGER = logging.getLogger(__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}. Choose a new run directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def provenance():
    return {"python": platform.python_version(), "platform": platform.system(),
            "packages": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn",
                                                  "joblib", "torch", "gymnasium", "pymoo")}}


def build_surrogate(config, output, data=None):
    source = "user-csv" if data else "synthetic-demo"
    if data:
        frame = pd.read_csv(data)
        data_sha = digest(data)
    else:
        frame = generate_data(config.condition, config.samples, config.seed)
        sample_path = output / "synthetic_data.csv"
        frame.to_csv(sample_path, index=False)
        data_sha = digest(sample_path)
    LOGGER.info("Fitting surrogate | condition=%d | rows=%d | source=%s", config.condition, len(frame), source)
    model, metrics = fit_surrogate(frame, config)
    artifact = {"model": model, "condition": config.condition, "features": domain(config.condition)[0],
                "source": source, "dataset_sha256": data_sha, "config": config.to_dict()}
    joblib.dump(artifact, output / "surrogate.joblib")
    write_json(output / "surrogate_metrics.json", metrics)
    write_json(output / "config.json", config.to_dict())
    return artifact, metrics


def read_surrogate(path, condition=None):
    # Joblib is pickle-based. Only load locally trained or otherwise trusted artifacts.
    artifact = joblib.load(path)
    actual_condition = artifact["condition"]
    if condition is not None and actual_condition != condition:
        raise ValueError("Surrogate condition does not match the requested condition")
    if artifact["features"] != domain(actual_condition)[0]:
        raise ValueError("Surrogate feature order does not match the process contract")
    return artifact


def run_demo(config, output, data=None):
    from .baselines import run_nsga2, run_random
    from .evaluation import HV_REFERENCE, summarize
    from .ppo import evaluate_policy, train

    started = time.perf_counter()
    artifact, model_metrics = build_surrogate(config, output, data)
    problem = ProcessProblem(config.condition, artifact["model"], config.process)
    oracle = (ProcessProblem(config.condition, SyntheticOracle(config.condition), config.process)
              if artifact["source"] == "synthetic-demo" else None)
    summaries = []
    nsga_x, nsga_run = run_nsga2(problem, config)
    nsga = summarize(problem, nsga_x, output, "nsga2", oracle)
    nsga.update(nsga_run)
    summaries.append(nsga)
    random_x, random_run = run_random(problem, nsga_run["search_evaluations"], config.seed + 20000)
    random_summary = summarize(problem, random_x, output, "random", oracle)
    random_summary.update(random_run)
    summaries.append(random_summary)
    policy, history, training_seconds = train(problem, config, output, digest(output / "surrogate.joblib"))
    pd.DataFrame(history).to_csv(output / "training.csv", index=False)
    rl_x, latency = evaluate_policy(policy, problem, config)
    rl = summarize(problem, rl_x, output, "ppo", oracle)
    rl.update(latency)
    rl.update({"training_seconds": training_seconds, "training_evaluations": config.total_steps,
               "policy_rollout_evaluations": len(rl_x)})
    summaries.append(rl)
    for row in summaries:
        row.update(condition=config.condition, seed=config.seed, data_source=artifact["source"])
    report = {"config": config.to_dict(), "provenance": provenance(),
              "source": artifact["source"], "dataset_sha256": artifact["dataset_sha256"],
              "surrogate_sha256": digest(output / "surrogate.joblib"),
              "hypervolume_reference": HV_REFERENCE.tolist(),
              "evaluation_note": "Different search/training budgets and front sizes; not an equal-budget superiority claim.",
              "wall_seconds": time.perf_counter() - started,
              "surrogate_test": model_metrics["test"], "methods": summaries}
    write_json(output / "metrics.json", report)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    LOGGER.info("Finished in %.1fs | output=%s", report["wall_seconds"], output)
    return report


def make_parser():
    parser = argparse.ArgumentParser(prog="ewrl", description="Electrowinning constrained RL toolkit")
    subs = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("demo", "Generate data, fit a surrogate, train PPO and compare three methods"),
        ("benchmark", "Repeat the complete demo over predeclared conditions/seeds"),
        ("generate-data", "Write a reproducible synthetic CSV"),
        ("train-surrogate", "Fit an ExtraTrees surrogate from synthetic data or a CSV"),
        ("train", "Train PPO against an existing trusted surrogate"),
        ("baseline", "Run NSGA-II against an existing trusted surrogate"),
    ]:
        sub = subs.add_parser(name, help=help_text)
        sub.add_argument("--config", type=Path)
        sub.add_argument("--condition", type=int, choices=(1, 2, 3))
        sub.add_argument("--seed", type=int)
        sub.add_argument("--output", type=Path, required=True)
        if name in ("demo", "train", "benchmark"):
            sub.add_argument("--steps", type=int)
        if name in ("demo", "train-surrogate"):
            sub.add_argument("--data", type=Path, help="CSV with decision/target columns; remains local")
        if name in ("train", "baseline"):
            sub.add_argument("--model", type=Path, required=True, help="Trusted surrogate.joblib")
        if name == "benchmark":
            sub.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 42])
            sub.add_argument("--conditions", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    evaluate = subs.add_parser("evaluate", help="Evaluate a saved policy against its original surrogate")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--policy", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    try:
        output = new_output(args.output)
        file_handler = logging.FileHandler(output / "run.log", encoding="utf-8")
        logging.getLogger().addHandler(file_handler)
        try:
            if args.command == "evaluate":
                from .evaluation import summarize
                from .ppo import evaluate_policy, load_policy
                model, checkpoint = load_policy(args.policy)
                if digest(args.model) != checkpoint.get("surrogate_sha256"):
                    raise ValueError("Policy was trained with a different surrogate file")
                config = Config.load(**checkpoint["config"])
                artifact = read_surrogate(args.model, config.condition)
                problem = ProcessProblem(config.condition, artifact["model"], config.process)
                x, latency = evaluate_policy(model, problem, config)
                result = summarize(problem, x, output, "ppo")
                result.update(latency)
                write_json(output / "metrics.json", result)
                return 0
            config = Config.load(args.config, condition=args.condition, seed=args.seed,
                                 total_steps=getattr(args, "steps", None))
            write_json(output / "config.json", config.to_dict())
            if args.command == "generate-data":
                generate_data(config.condition, config.samples, config.seed).to_csv(output / "synthetic_data.csv", index=False)
            elif args.command == "train-surrogate":
                build_surrogate(config, output, args.data)
            elif args.command == "demo":
                run_demo(config, output, args.data)
            elif args.command == "benchmark":
                rows, model_rows = [], []
                if len(set(args.seeds)) != len(args.seeds) or len(set(args.conditions)) != len(args.conditions):
                    raise ValueError("Benchmark seeds and conditions must be unique")
                for condition in args.conditions:
                    for seed in args.seeds:
                        current = replace(config, condition=condition, seed=seed).validate()
                        child = new_output(output / f"condition{condition}_seed{seed}")
                        report = run_demo(current, child)
                        rows.extend(report["methods"])
                        model_rows.extend({"condition": condition, "seed": seed, "target": target, **metrics}
                                          for target, metrics in report["surrogate_test"].items())
                frame = pd.DataFrame(rows)
                frame.to_csv(output / "summary.csv", index=False)
                pd.DataFrame(model_rows).to_csv(output / "surrogate_summary.csv", index=False)
                measures = ["feasible_rate", "synthetic_oracle_feasible_rate", "hypervolume", "pareto_points"]
                frame.groupby(["condition", "method"])[measures].agg(["mean", "std"]).to_csv(output / "aggregate.csv")
                write_json(output / "protocol.json", {"conditions": args.conditions, "seeds": args.seeds,
                           "config": config.to_dict(), "provenance": provenance()})
            else:
                from .evaluation import summarize
                artifact = read_surrogate(args.model, config.condition)
                problem = ProcessProblem(config.condition, artifact["model"], config.process)
                if args.command == "train":
                    from .ppo import train
                    _, history, elapsed = train(problem, config, output, digest(args.model))
                    pd.DataFrame(history).to_csv(output / "training.csv", index=False)
                    write_json(output / "metrics.json", {"training_seconds": elapsed})
                else:
                    from .baselines import run_nsga2
                    x, timing = run_nsga2(problem, config)
                    summary = summarize(problem, x, output, "nsga2")
                    summary.update(timing)
                    write_json(output / "metrics.json", summary)
        finally:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
