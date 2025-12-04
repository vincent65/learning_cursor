"""
Common visualization helpers shared across analysis scripts.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    umap = None


def plot_pca_umap(
    z: np.ndarray,
    y: np.ndarray,
    attr_name: str,
    output_prefix: Path,
    title: str | None = None,
    use_umap: bool = True,
    cmap: str = "coolwarm",
) -> Dict[str, str]:
    """
    Reduce embeddings to 2D via PCA (and optionally UMAP) and save scatter plots.

    Args:
        z: (N, D) numpy array of embeddings.
        y: (N,) labels used for coloring (assumed binary but works for multi-class).
        attr_name: label used for axis/colorbar titles.
        output_prefix: base path (without suffix). We'll append `_pca.png`/`_umap.png`.
        title: optional custom title (defaults to describing attr_name).
        use_umap: whether to attempt a UMAP projection (requires umap-learn).
    Returns:
        dict with keys "pca" and optionally "umap" pointing to the saved file paths.
    """
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    results: Dict[str, str] = {}

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(z)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        cmap=cmap,
        s=6,
        alpha=0.6,
    )
    plt.colorbar(scatter, label=f"{attr_name}")
    plt.title(title or f"PCA – colored by {attr_name}")
    plt.tight_layout()
    pca_path = Path(f"{output_prefix}_pca.png")
    plt.savefig(pca_path, dpi=220)
    plt.close()
    results["pca"] = str(pca_path)

    if use_umap and umap is not None:
        reducer = umap.UMAP(n_components=2, random_state=0)
        X_umap = reducer.fit_transform(z)
        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(
            X_umap[:, 0],
            X_umap[:, 1],
            c=y,
            cmap=cmap,
            s=6,
            alpha=0.6,
        )
        plt.colorbar(scatter, label=f"{attr_name}")
        plt.title(title or f"UMAP – colored by {attr_name}")
        plt.tight_layout()
        umap_path = Path(f"{output_prefix}_umap.png")
        plt.savefig(umap_path, dpi=220)
        plt.close()
        results["umap"] = str(umap_path)
    elif use_umap:
        print("[WARN] `umap-learn` not installed; skipping UMAP plot.")

    return results


def save_attribute_grid(
    celeba_root: str,
    filenames: list[str],
    attr_values: torch.Tensor,
    attr_name: str,
    output_path: Path,
    num_examples: int = 4,
) -> str:
    """
    Save a grid with num_examples positive samples on the first row and negatives on the second.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    attr_values = attr_values.cpu()
    pos_indices = torch.where(attr_values == 1)[0][:num_examples]
    neg_indices = torch.where(attr_values == 0)[0][:num_examples]

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        raise ValueError(
            f"Not enough samples to build grid for attribute {attr_name}. "
            f"Found positives={len(pos_indices)}, negatives={len(neg_indices)}"
        )

    fig, axes = plt.subplots(2, num_examples, figsize=(2 * num_examples, 4))
    for row, indices in enumerate([pos_indices, neg_indices]):
        label = "pos" if row == 0 else "neg"
        for col, idx in enumerate(indices):
            img_fname = filenames[int(idx)]
            img_path = Path(celeba_root) / "img_align_celeba" / img_fname
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"{attr_name}={1 if label=='pos' else 0}", fontsize=8)
    plt.suptitle(f"{attr_name}: positives (top) vs negatives (bottom)", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.close()
    return str(output_path)


def plot_flow_losses(log_csv_path: str, output_path: Path) -> str:
    """
    Plot total/contrastive/regularizer losses from a CSV log.
    """
    if not os.path.exists(log_csv_path):
        return ""
    with open(log_csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return ""

    epochs = [int(row["epoch"]) for row in rows]
    plt.figure(figsize=(6, 4))
    for key, label in [
        ("total_loss", "total"),
        ("contrastive_loss", "contrastive"),
        ("identity_loss", "identity"),
        ("curl_loss", "curl"),
        ("div_loss", "div"),
    ]:
        values = [float(row.get(key, 0.0)) for row in rows]
        plt.plot(epochs, values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Flow training losses")
    plt.legend()
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()
    return str(output_path)

