"""
Fast hyperparameter sweep helper that executes a curated set of ~20 experiments.

This script reuses `run_experiment` from `scripts/run_experiments.py`, but limits
the search to a mix of no-flow baselines, different step sizes, regularization
weights, and learning-rate variations so we can sanity-check ablations quickly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import sys
import yaml
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from celeba_embeddings import load_embeddings_splits, get_attribute_indices, select_attributes
from fclf_model import FCLFConfig
from scripts.eval_fclf_metrics import compute_flowed_embeddings, load_trained_model
from scripts.run_experiments import (
    ALL_DATASETS,
    ATTR_TO_DATASET,
    TARGET_ATTRS,
    _distance_metrics,
    _load_or_compute_baseline,
    _train_probe,
    run_experiment,
)

# Each entry describes a small configuration tweak. We pair it with the seeds
# specified on the CLI to reach ~20 total runs.
FAST_SETUPS: List[Dict] = [
    {
        "name": "no_flow_baseline_small_step",
        "base_config": "configs/fclf_config.yaml",
        "params": {"K": 0, "eps": 0.05, "lambda_curl": 0.0, "lambda_div": 0.0},
    },
    {
        "name": "no_flow_baseline_large_step",
        "base_config": "configs/fclf_config.yaml",
        "params": {"K": 0, "eps": 0.1, "lambda_curl": 1e-3, "lambda_div": 1e-3},
    },
    {
        "name": "flow_light_reg",
        "params": {"K": 2, "eps": 0.05, "lambda_curl": 1e-3, "lambda_div": 1e-3},
    },
    {
        "name": "flow_high_reg",
        "params": {"K": 4, "eps": 0.1, "lambda_curl": 1e-2, "lambda_div": 1e-2},
    },
    {
        "name": "flow_small_step",
        "params": {"K": 8, "eps": 0.03, "lambda_curl": 1e-2, "lambda_div": 1e-2},
    },
    {
        "name": "hidden_dim_512",
        "params": {"K": 4, "eps": 0.05, "hidden_dim": 512, "lambda_curl": 1e-3, "lambda_div": 1e-3},
    },
    {
        "name": "hidden_dim_128",
        "params": {"K": 4, "eps": 0.05, "hidden_dim": 128, "lambda_curl": 1e-3, "lambda_div": 1e-3},
    },
    {
        "name": "lr_high",
        "params": {"K": 4, "eps": 0.08, "lambda_curl": 1e-2, "lambda_div": 1e-2, "lr": 3e-4},
    },
    {
        "name": "lr_low",
        "params": {"K": 4, "eps": 0.08, "lambda_curl": 1e-2, "lambda_div": 1e-2, "lr": 5e-5},
    },
    {
        "name": "tau_soft",
        "params": {"K": 6, "eps": 0.06, "tau": 0.2, "lambda_curl": 1e-3, "lambda_div": 1e-3},
    },
]


def evaluate_all_attributes(
    checkpoint_path: str,
    embedding_dir: str,
    num_steps: int,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_split, val_split, _ = load_embeddings_splits(embedding_dir)
    attr_indices = get_attribute_indices(train_split.attr_names, TARGET_ATTRS)
    train_attrs = select_attributes(train_split, attr_indices)
    val_attrs = select_attributes(val_split, attr_indices)

    fallback_cfg = FCLFConfig()
    model = load_trained_model(checkpoint_path, cfg_fallback=fallback_cfg).to(device)
    X_train_flow = compute_flowed_embeddings(
        model=model,
        embeddings=train_split.embeddings,
        attrs=train_attrs,
        num_steps=num_steps,
        device=device,
    ).numpy()
    X_val_flow = compute_flowed_embeddings(
        model=model,
        embeddings=val_split.embeddings,
        attrs=val_attrs,
        num_steps=num_steps,
        device=device,
    ).numpy()

    per_attr = []
    val_accs: List[float] = []
    val_aucs: List[float] = []
    for idx, attr_name in enumerate(TARGET_ATTRS):
        y_train = train_attrs[:, idx].numpy()
        y_val = val_attrs[:, idx].numpy()
        clf = _train_probe(X_train_flow, y_train)
        train_acc = float(np.mean(clf.predict(X_train_flow) == y_train))
        y_pred = clf.predict(X_val_flow)
        val_acc = float(np.mean(y_pred == y_val))
        if len(np.unique(y_val)) >= 2:
            val_auc = float(roc_auc_score(y_val, clf.predict_proba(X_val_flow)[:, 1]))
        else:
            val_auc = float("nan")
        baseline_dataset = ATTR_TO_DATASET[attr_name]
        _, baseline_metrics = _load_or_compute_baseline(baseline_dataset, embedding_dir)
        dist = _distance_metrics(X_val_flow, y_val)
        per_attr.append(
            {
                "attr": attr_name,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "baseline_val_acc": baseline_metrics.get("val_acc"),
                "baseline_val_auc": baseline_metrics.get("val_auc"),
                "within_dist": dist["within_dist"],
                "between_dist": dist["between_dist"],
            }
        )
        val_accs.append(val_acc)
        if not np.isnan(val_auc):
            val_aucs.append(val_auc)

    aggregate_val_acc = float(np.mean(val_accs)) if val_accs else float("nan")
    aggregate_val_auc = float(np.mean(val_aucs)) if val_aucs else float("nan")
    return {
        "per_attr": per_attr,
        "aggregate_val_acc": aggregate_val_acc,
        "aggregate_val_auc": aggregate_val_auc,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast ablation sweep for FCLF.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="celeba_eyeglasses",
        choices=ALL_DATASETS + ["all"],
        help="Target dataset/attribute (or 'all' to run each attribute once).",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="data/embeddings",
        help="Directory with train/val/test embedding .pt files.",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/fclf_config.yaml",
        help="Fallback YAML config when a setup does not specify one.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1",
        help="Comma-separated seeds to pair with each setup (e.g., '0,1').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/hp_search/fast_runs_summary.json",
        help="Path to save a JSON summary of all runs.",
    )
    parser.add_argument(
        "--list_setups",
        action="store_true",
        help="List the curated setups and exit without running anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_setups:
        for idx, setup in enumerate(FAST_SETUPS):
            print(f"[{idx:02d}] {setup['name']} -> base_config={setup.get('base_config', args.base_config)} "
                  f"params={setup['params']}")
        return

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    datasets = ALL_DATASETS if args.dataset_name == "all" else [args.dataset_name]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jobs: List[tuple[str, Dict, int]] = []
    for setup in FAST_SETUPS:
        for seed in seeds:
            jobs.append((setup["name"], setup, seed))

    summary: List[Dict] = []
    for idx, (setup_name, setup, seed) in enumerate(jobs):
        dataset_name = datasets[0]
        base_config_path = setup.get("base_config", args.base_config)
        config = dict(setup["params"])
        config["dataset_name"] = dataset_name
        config["seed"] = seed
        with open(base_config_path, "r") as f:
            base_cfg = yaml.safe_load(f)
        num_steps = config.get("K", base_cfg.get("inference", {}).get("num_flow_steps", 10))
        print(f"[FAST] dataset={dataset_name} setup={setup_name} seed={seed} config={config}")
        result = run_experiment(
            config=config,
            base_config_path=base_config_path,
            embedding_dir=args.embedding_dir,
        )
        holistic = evaluate_all_attributes(
            checkpoint_path=result.checkpoint_path,
            embedding_dir=args.embedding_dir,
            num_steps=num_steps,
        )
        summary.append(
            {
                "dataset": dataset_name,
                "setup": setup_name,
                "seed": seed,
                "metrics": result.to_log_row(),
                "holistic": holistic,
            }
        )

    output_path.write_text(json.dumps(summary, indent=2))
    print(f"[FAST] Saved summary for {len(summary)} runs to {output_path}")


if __name__ == "__main__":
    main()

