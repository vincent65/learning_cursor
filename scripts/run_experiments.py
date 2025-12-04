"""
Utilities for running hyperparameter sweeps programmatically.

This module exposes:
    - run_experiment(config): trains a model with overrides and logs metrics.
    - sweep_celeba(): grid search over predefined hyperparameters.
    - summarize_best_configs(): report top-performing runs per dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from celeba_embeddings import (
    get_attribute_indices,
    load_embeddings_splits,
    select_attributes,
)
from fclf_model import FCLFConfig
from scripts.eval_fclf_metrics import compute_flowed_embeddings, load_trained_model
from scripts.train_fclf import load_yaml_config, train_fclf
from scripts.viz_utils import plot_flow_losses


HP_DIR = Path("eval/hp_search")
RUNS_DIR = Path("eval/hp_runs")

TARGET_ATTRS = ["Smiling", "Young", "Male", "Eyeglasses", "Mustache"]
DATASET_TO_ATTR = {
    "celeba_smiling": "Smiling",
    "celeba_young": "Young",
    "celeba_male": "Male",
    "celeba_eyeglasses": "Eyeglasses",
    "celeba_mustache": "Mustache",
}
ATTR_TO_DATASET = {attr: dataset for dataset, attr in DATASET_TO_ATTR.items()}
ALL_DATASETS = list(DATASET_TO_ATTR.keys())


@dataclass
class ExperimentResult:
    dataset_name: str
    config: Dict
    checkpoint_path: str
    metrics_path: str
    plots_dir: str
    train_acc: float
    val_acc: float
    val_auc: float
    within_dist: float
    between_dist: float
    loss_log_path: Optional[str] = None
    loss_plot_path: Optional[str] = None
    classifier_log_path: Optional[str] = None
    baseline_path: Optional[str] = None
    baseline_val_acc: Optional[float] = None
    baseline_val_auc: Optional[float] = None
    aggregate_metric: Optional[float] = None

    def to_log_row(self) -> Dict:
        row = {
            **self.config,
            "dataset_name": self.dataset_name,
            "checkpoint_path": self.checkpoint_path,
            "metrics_path": self.metrics_path,
            "plots_dir": self.plots_dir,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "val_auc": self.val_auc,
            "within_dist": self.within_dist,
            "between_dist": self.between_dist,
            "loss_log_path": self.loss_log_path,
            "loss_plot_path": self.loss_plot_path,
            "classifier_log_path": self.classifier_log_path,
            "baseline_path": self.baseline_path,
            "baseline_val_acc": self.baseline_val_acc,
            "baseline_val_auc": self.baseline_val_auc,
            "aggregate_metric": self.aggregate_metric,
        }
        return row


def _ensure_dirs() -> None:
    HP_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_attr(dataset_name: str) -> str:
    if dataset_name not in DATASET_TO_ATTR:
        raise ValueError(
            f"Unknown dataset_name={dataset_name}. "
            f"Available: {list(DATASET_TO_ATTR.keys())}"
        )
    return DATASET_TO_ATTR[dataset_name]


def _apply_overrides(base_cfg: Dict, overrides: Dict) -> Dict:
    cfg = deepcopy(base_cfg)
    model_cfg = cfg.setdefault("model", {})
    training_cfg = cfg.setdefault("training", {})
    loss_cfg = cfg.setdefault("loss", {})
    inference_cfg = cfg.setdefault("inference", {})

    if "hidden_dim" in overrides:
        model_cfg["hidden_dim"] = overrides["hidden_dim"]
    if "K" in overrides:
        inference_cfg["num_flow_steps"] = overrides["K"]
    if "eps" in overrides:
        training_cfg["alpha"] = overrides["eps"]
        inference_cfg["step_size"] = overrides["eps"]
    if "tau" in overrides:
        loss_cfg["temperature"] = overrides["tau"]
    if "lambda_curl" in overrides:
        loss_cfg["lambda_curl"] = overrides["lambda_curl"]
    if "lambda_div" in overrides:
        loss_cfg["lambda_div"] = overrides["lambda_div"]
    if "lr" in overrides:
        training_cfg["learning_rate"] = overrides["lr"]

    return cfg


def _run_id(dataset_name: str, overrides: Dict, seed: Optional[int]) -> str:
    parts = [
        dataset_name,
        f"K{overrides.get('K', 'NA')}",
        f"eps{overrides.get('eps', 'NA')}",
        f"curl{overrides.get('lambda_curl', 'NA')}",
        f"div{overrides.get('lambda_div', 'NA')}",
        f"tau{overrides.get('tau', 'NA')}",
    ]
    if seed is not None:
        parts.append(f"seed{seed}")
    return "_".join(str(p) for p in parts)


def _train_probe(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    return clf


def _distance_metrics(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y = y.astype(bool)
    cls0 = X[~y]
    cls1 = X[y]
    within0 = np.linalg.norm(cls0 - cls0.mean(axis=0), axis=1).mean() if len(cls0) else 0.0
    within1 = np.linalg.norm(cls1 - cls1.mean(axis=0), axis=1).mean() if len(cls1) else 0.0
    within = np.mean([d for d in [within0, within1] if d > 0]) if (len(cls0) and len(cls1)) else (within0 or within1)
    if len(cls0) and len(cls1):
        between = float(np.linalg.norm(cls0.mean(axis=0) - cls1.mean(axis=0)))
    else:
        between = 0.0
    return {"within_dist": float(within), "between_dist": between}


def run_experiment(
    config: Dict,
    base_config_path: str = "configs/fclf_config_RECOMMENDED.yaml",
    embedding_dir: str = "data/embeddings",
) -> ExperimentResult:
    """
    High-level API used by sweeps and notebooks.
    """
    _ensure_dirs()
    dataset_name = config["dataset_name"]
    target_attr = _resolve_attr(dataset_name)
    seed = config.get("seed")

    base_cfg = load_yaml_config(base_config_path)
    train_cfg = _apply_overrides(base_cfg, config)
    baseline_path, baseline_metrics = _load_or_compute_baseline(dataset_name, embedding_dir)

    train_split, val_split, _ = load_embeddings_splits(embedding_dir)

    run_id = _run_id(dataset_name, config, seed)
    run_dir = RUNS_DIR / run_id
    checkpoints_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_metadata = train_fclf(
        cfg_yaml=train_cfg,
        embedding_dir=embedding_dir,
        output_dir=str(checkpoints_dir),
        checkpoint_path=None,
        resume=False,
        save_every=5,
        seed=seed,
    )
    checkpoint_path = train_metadata["checkpoint_path"]
    loss_plot_path = None
    if train_metadata.get("loss_log_path"):
        loss_plot = plots_dir / "flow_losses.png"
        plotted = plot_flow_losses(train_metadata["loss_log_path"], loss_plot)
        if plotted:
            loss_plot_path = plotted

    attr_indices = get_attribute_indices(train_split.attr_names, TARGET_ATTRS)
    train_attrs_full = select_attributes(train_split, attr_indices)
    val_attrs_full = select_attributes(val_split, attr_indices)
    attr_col = TARGET_ATTRS.index(target_attr)
    train_labels = train_attrs_full[:, attr_col]
    val_labels = val_attrs_full[:, attr_col]

    fallback_cfg = FCLFConfig()
    model = load_trained_model(checkpoint_path, cfg_fallback=fallback_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train_flow = compute_flowed_embeddings(
        model=model,
        embeddings=train_split.embeddings,
        attrs=train_attrs_full,
        num_steps=config.get("K", base_cfg.get("inference", {}).get("num_flow_steps", 10)),
        device=torch.device(device),
    )
    X_val_flow = compute_flowed_embeddings(
        model=model,
        embeddings=val_split.embeddings,
        attrs=val_attrs_full,
        num_steps=config.get("K", base_cfg.get("inference", {}).get("num_flow_steps", 10)),
        device=torch.device(device),
    )

    X_train_np = X_train_flow.numpy()
    X_val_np = X_val_flow.numpy()
    y_train_np = train_labels.numpy()
    y_val_np = val_labels.numpy()

    clf = _train_probe(X_train_np, y_train_np)
    train_acc = accuracy_score(y_train_np, clf.predict(X_train_np))
    val_preds = clf.predict(X_val_np)
    val_acc = accuracy_score(y_val_np, val_preds)

    if len(np.unique(y_val_np)) < 2:
        val_auc = float("nan")
    else:
        val_probs = clf.predict_proba(X_val_np)[:, 1]
        val_auc = roc_auc_score(y_val_np, val_probs)

    dist_metrics = _distance_metrics(X_val_np, y_val_np)

    metrics = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "val_auc": val_auc,
        **dist_metrics,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    result = ExperimentResult(
        dataset_name=dataset_name,
        config={k: v for k, v in config.items() if k != "dataset_name"},
        checkpoint_path=checkpoint_path,
        metrics_path=str(metrics_path),
        plots_dir=str(plots_dir),
        train_acc=float(train_acc),
        val_acc=float(val_acc),
        val_auc=float(val_auc),
        within_dist=dist_metrics["within_dist"],
        between_dist=dist_metrics["between_dist"],
        loss_log_path=train_metadata.get("loss_log_path"),
        loss_plot_path=loss_plot_path,
        classifier_log_path=train_metadata.get("classifier_log_path"),
        baseline_path=str(baseline_path),
        baseline_val_acc=baseline_metrics.get("val_acc"),
        baseline_val_auc=baseline_metrics.get("val_auc"),
    )
    _log_result(dataset_name, result.to_log_row())
    return result


def _log_result(dataset_name: str, row: Dict) -> None:
    csv_path = HP_DIR / f"{dataset_name}.csv"
    json_path = HP_DIR / f"{dataset_name}.json"
    fieldnames = [
        "dataset_name",
        "K",
        "eps",
        "tau",
        "hidden_dim",
        "lambda_curl",
        "lambda_div",
        "lr",
        "seed",
        "checkpoint_path",
        "metrics_path",
        "plots_dir",
        "loss_log_path",
        "loss_plot_path",
        "classifier_log_path",
        "baseline_path",
        "baseline_val_acc",
        "baseline_val_auc",
        "train_acc",
        "val_acc",
        "val_auc",
        "within_dist",
        "between_dist",
        "aggregate_metric",
    ]
    row = {k: row.get(k) for k in fieldnames}

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    if json_path.exists():
        entries = json.loads(json_path.read_text())
    else:
        entries = []
    entries.append(row)
    json_path.write_text(json.dumps(entries, indent=2))


def _baseline_cache_path(dataset_name: str) -> Path:
    return HP_DIR / f"{dataset_name}_baseline.json"


def _load_or_compute_baseline(
    dataset_name: str,
    embedding_dir: str,
) -> Tuple[Path, Dict[str, float]]:
    cache_path = _baseline_cache_path(dataset_name)
    if cache_path.exists():
        return cache_path, json.loads(cache_path.read_text())

    attr_name = _resolve_attr(dataset_name)
    train_split, val_split, _ = load_embeddings_splits(embedding_dir)
    attr_indices = get_attribute_indices(train_split.attr_names, [attr_name])
    train_labels = select_attributes(train_split, attr_indices)[:, 0]
    val_labels = select_attributes(val_split, attr_indices)[:, 0]

    X_train = train_split.embeddings.cpu().numpy()
    X_val = val_split.embeddings.cpu().numpy()
    y_train = train_labels.cpu().numpy()
    y_val = val_labels.cpu().numpy()

    clf = _train_probe(X_train, y_train)
    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    if len(np.unique(y_val)) >= 2:
        val_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    else:
        val_auc = float("nan")

    baseline = {
        "attr_name": attr_name,
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "val_auc": float(val_auc),
    }
    cache_path.write_text(json.dumps(baseline, indent=2))
    return cache_path, baseline


def _aggregate_results(
    results: List[ExperimentResult],
    metric: str = "val_acc",
) -> List[Dict]:
    grouped: Dict[Tuple, Dict] = {}
    for res in results:
        cfg_key = tuple(sorted(res.config.items()))
        entry = grouped.setdefault(
            cfg_key,
            {
                "config": res.config,
                "per_dataset": [],
            },
        )
        entry["per_dataset"].append(
            {
                "dataset_name": res.dataset_name,
                "train_acc": res.train_acc,
                "val_acc": res.val_acc,
                "val_auc": res.val_auc,
                "within_dist": res.within_dist,
                "between_dist": res.between_dist,
            }
        )

    summary: List[Dict] = []
    for entry in grouped.values():
        metric_vals = [
            ds.get(metric)
            for ds in entry["per_dataset"]
            if ds.get(metric) is not None
        ]
        metric_vals = [float(v) for v in metric_vals if not np.isnan(v)]  # type: ignore[arg-type]
        aggregate = float(np.mean(metric_vals)) if metric_vals else float("nan")
        summary.append(
            {
                "config": entry["config"],
                "aggregate_metric": aggregate,
                "per_dataset": entry["per_dataset"],
            }
        )
    summary.sort(key=lambda x: x["aggregate_metric"], reverse=True)
    return summary


def sweep_celeba(
    dataset_name: str,
    base_config_path: str,
    embedding_dir: str,
    seeds: Optional[List[int]] = None,
) -> List[ExperimentResult]:
    seeds = seeds or [0]
    grid_K = [0, 2, 4, 8]
    grid_eps = [0.05, 0.1]
    grid_curl = [0.0, 1e-3, 1e-2]

    results: List[ExperimentResult] = []
    for seed in seeds:
        for K in grid_K:
            for eps in grid_eps:
                for lam in grid_curl:
                    cfg = {
                        "dataset_name": dataset_name,
                        "K": K,
                        "eps": eps,
                        "tau": 0.1,
                        "hidden_dim": 256,
                        "lambda_curl": lam,
                        "lambda_div": lam,
                        "lr": 1e-4,
                        "seed": seed,
                    }
                    print(f"[SWEEP] Running config: {cfg}")
                    result = run_experiment(
                        config=cfg,
                        base_config_path=base_config_path,
                        embedding_dir=embedding_dir,
                    )
                    results.append(result)
    return results


def summarize_best_configs(
    dataset_name: str,
    metric: str = "val_acc",
    top_k: int = 3,
) -> List[Dict]:
    csv_path = HP_DIR / f"{dataset_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No CSV log found at {csv_path}")
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    sorted_rows = sorted(
        rows,
        key=lambda r: float(r.get(metric, float("-inf"))),
        reverse=True,
    )
    top_rows = sorted_rows[:top_k]
    print(f"=== Top {top_k} configs for {dataset_name} (sorted by {metric}) ===")
    for row in top_rows:
        print(row)
    return top_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FCLF hyperparameter sweeps.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="celeba_eyeglasses",
        choices=ALL_DATASETS + ["all"],
        help="Dataset identifier (determines target attribute).",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="data/embeddings",
        help="Directory containing train/val/test embeddings.",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/fclf_config_RECOMMENDED.yaml",
        help="Base YAML config path.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="If set, run the full predefined sweep; otherwise run a single experiment with overrides.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="When sweeping multiple datasets, compute and store aggregate rankings.",
    )
    parser.add_argument(
        "--overrides",
        type=json.loads,
        default=None,
        help="JSON string of overrides when not running the sweep.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Print the top configs from the logged CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_name == "all":
        dataset_list = ALL_DATASETS
    else:
        dataset_list = [args.dataset_name]

    if args.summarize:
        for ds in dataset_list:
            summarize_best_configs(dataset_name=ds)
        return

    if args.sweep:
        sweep_results: List[ExperimentResult] = []
        for ds in dataset_list:
            sweep_results.extend(
                sweep_celeba(
                    dataset_name=ds,
                    base_config_path=args.base_config,
                    embedding_dir=args.embedding_dir,
                )
            )
        if args.aggregate and len(dataset_list) > 1:
            summary = _aggregate_results(sweep_results)
            agg_path = HP_DIR / "celeba_all_aggregate.json"
            agg_path.write_text(json.dumps(summary, indent=2))
            print(f"[SWEEP] Saved aggregate summary to {agg_path}")
    else:
        if args.overrides is None:
            raise ValueError(
                "When --sweep is not set, you must pass --overrides as a JSON dict."
            )
        overrides = args.overrides
        for ds in dataset_list:
            overrides_ds = dict(overrides)
            overrides_ds["dataset_name"] = ds
            run_experiment(
                config=overrides_ds,
                base_config_path=args.base_config,
                embedding_dir=args.embedding_dir,
            )


if __name__ == "__main__":
    main()

