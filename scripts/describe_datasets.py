import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
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
from scripts.viz_utils import plot_pca_umap, save_attribute_grid


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def compute_correlations(attr_matrix: torch.Tensor, attr_names: List[str]) -> Dict[str, Dict[str, float]]:
    arr = _to_numpy(attr_matrix).astype(np.float32)
    corr = np.corrcoef(arr, rowvar=False)
    result: Dict[str, Dict[str, float]] = {}
    for i, name_i in enumerate(attr_names):
        result[name_i] = {}
        for j, name_j in enumerate(attr_names):
            result[name_i][name_j] = float(corr[i, j])
    return result


def describe_dataset(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_split, val_split, test_split = load_embeddings_splits(args.embedding_dir)
    attr_indices = get_attribute_indices(train_split.attr_names, args.attr_names)

    stats = {
        "splits": {
            "train": len(train_split.embeddings),
            "val": len(val_split.embeddings),
            "test": len(test_split.embeddings),
        },
        "class_counts": {},
        "correlations": {},
    }

    for split_name, split in [("train", train_split), ("val", val_split), ("test", test_split)]:
        attrs = select_attributes(split, attr_indices)
        split_counts = {
            attr: int(attrs[:, idx].sum().item())
            for idx, attr in enumerate(attr_indices.keys())
        }
        stats["class_counts"][split_name] = split_counts
        print(f"\n[{split_name.upper()}] samples={len(split.embeddings)}")
        for attr, count in split_counts.items():
            print(f"  {attr:12s}: {count:6d} positives ({count/len(split.embeddings):.3f})")

    train_attrs_selected = select_attributes(train_split, attr_indices)
    stats["correlations"] = compute_correlations(train_attrs_selected, list(attr_indices.keys()))
    print("\n[TRAIN] Attribute correlations")
    for attr_i, row in stats["correlations"].items():
        row_str = " ".join(f"{row[attr_j]: .2f}" for attr_j in attr_indices.keys())
        print(f"  {attr_i:12s}: {row_str}")

    stats_path = output_dir / "celeba_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\n[INFO] Saved dataset stats to {stats_path}")

    # PCA/UMAP plots for requested attributes
    if args.plot_attrs:
        for attr_name in args.plot_attrs:
            if attr_name not in train_split.attr_names:
                raise ValueError(f"Attribute {attr_name} not found in CelebA metadata.")
            attr_idx = train_split.attr_names.index(attr_name)
            embeddings = train_split.embeddings
            labels = train_split.attributes[:, attr_idx]
            N = embeddings.shape[0]
            if N > args.max_points:
                idx = torch.randperm(N)[: args.max_points]
                embeddings = embeddings[idx]
                labels = labels[idx]
            plot_pca_umap(
                z=_to_numpy(embeddings),
                y=_to_numpy(labels),
                attr_name=attr_name,
                title=f"Raw CLIP embeddings – {attr_name}",
                output_prefix=output_dir / f"raw_clip_{attr_name.lower()}",
            )

    # Attribute example grid
    if args.example_attr:
        if args.example_attr not in train_split.attr_names:
            raise ValueError(f"Attribute {args.example_attr} not found in CelebA metadata.")
        attr_idx = train_split.attr_names.index(args.example_attr)
        attr_values = train_split.attributes[:, attr_idx]
        grid_path = output_dir / f"celeba_examples_{args.example_attr.lower()}.png"
        save_attribute_grid(
            celeba_root=args.celeba_root,
            filenames=train_split.filenames,
            attr_values=attr_values,
            attr_name=args.example_attr,
            output_path=grid_path,
            num_examples=args.grid_examples,
        )
        print(f"[INFO] Saved attribute grid to {grid_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe CelebA-derived datasets.")
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="data/embeddings",
        help="Directory with train/val/test embedding .pt files.",
    )
    parser.add_argument(
        "--celeba_root",
        type=str,
        default="data/celeba",
        help="Path to raw CelebA images (for grids).",
    )
    parser.add_argument(
        "--attr_names",
        type=str,
        nargs="+",
        default=["Smiling", "Young", "Male", "Eyeglasses", "Mustache", "Heavy_Makeup"],
        help="Attributes to include in the statistics/correlation table.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval/dataset_stats",
        help="Directory to store JSON summaries and figures.",
    )
    parser.add_argument(
        "--plot_attrs",
        type=str,
        nargs="*",
        default=["Eyeglasses", "Smiling", "Young", "Mustache", "Male"],
        help="Attributes to visualize via PCA/UMAP.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=12000,
        help="Max number of points for PCA/UMAP visualizations.",
    )
    parser.add_argument(
        "--example_attr",
        type=str,
        default="Eyeglasses",
        help="Attribute to use when saving the example image grid.",
    )
    parser.add_argument(
        "--grid_examples",
        type=int,
        default=4,
        help="Number of positive/negative examples per row in the grid.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    describe_dataset(parse_args())

