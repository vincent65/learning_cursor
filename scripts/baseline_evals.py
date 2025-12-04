import argparse
import os
from typing import List

import numpy as np
import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from celeba_embeddings import (
    load_embeddings_splits,
    get_attribute_indices,
    select_attributes,
)
from scripts.viz_utils import plot_pca_umap


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def run_linear_probes(
    embedding_dir: str,
    target_attrs: List[str],
    output_path: Path,
) -> None:
    """
    Train simple linear classifiers on raw CLIP embeddings for selected attributes.
    Stores the summary in output_path.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    train, val, _ = load_embeddings_splits(embedding_dir)
    attr_indices = get_attribute_indices(train.attr_names, target_attrs)

    X_train = _to_numpy(train.embeddings)
    X_val = _to_numpy(val.embeddings)
    Y_train = select_attributes(train, attr_indices)
    Y_val = select_attributes(val, attr_indices)

    output_lines = ["=== Linear Probes on Raw CLIP Embeddings ==="]
    print("=== Linear Probes on Raw CLIP Embeddings ===")
    for j, attr_name in enumerate(attr_indices.keys()):
        y_tr = _to_numpy(Y_train[:, j])
        y_va = _to_numpy(Y_val[:, j])

        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
        clf.fit(X_train, y_tr)
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_va, y_pred)
        f1 = f1_score(y_va, y_pred)
        pos_rate = y_va.mean()

        line = (
            f"{attr_name:10s} | val_acc = {acc: .4f} | val_f1 = {f1: .4f} | "
            f"pos_rate = {pos_rate: .3f}"
        )
        print(line)
        output_lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines))
    print(f"[INFO] Saved baseline metrics to {output_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline CLIP embedding evaluations.")
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="data/embeddings",
        help="Directory with precomputed embeddings.",
    )
    parser.add_argument(
        "--attr_names",
        type=str,
        nargs="+",
        default=["Smiling", "Young", "Male", "Eyeglasses", "Mustache"],
        help="Attributes to evaluate.",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default="eval/baseline/linear_probe_metrics.txt",
        help="Path to save linear probe summary.",
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        default="eval/baseline",
        help="Directory to store PCA/UMAP plots.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=10000,
        help="Max number of embeddings to use for PCA/UMAP plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics_path = Path(args.metrics_output)
    figure_dir = Path(args.figure_dir)

    run_linear_probes(args.embedding_dir, args.attr_names, metrics_path)
    train, _, _ = load_embeddings_splits(args.embedding_dir)
    for attr in args.attr_names:
        attr_indices = get_attribute_indices(train.attr_names, [attr])
        labels = select_attributes(train, attr_indices)[:, 0]
        embeddings = train.embeddings
        N = embeddings.shape[0]
        if N > args.max_points:
            idx = torch.randperm(N)[: args.max_points]
            embeddings = embeddings[idx]
            labels = labels[idx]
        plot_pca_umap(
            z=_to_numpy(embeddings),
            y=_to_numpy(labels),
            attr_name=attr,
            title=f"Raw CLIP embeddings – {attr}",
            output_prefix=figure_dir / f"raw_clip_{attr.lower()}",
        )


