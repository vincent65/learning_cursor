#!/usr/bin/env python3
"""
Comprehensive hyperparameter search and ablation study for FCLF.

Usage:
    python scripts/hp_search.py                    # Run full grid search
    python scripts/hp_search.py --quick            # Quick test (small grid)
    python scripts/hp_search.py --ablation         # Run 30-config ablation study
    python scripts/hp_search.py --summarize        # Show results from previous runs
    python scripts/hp_search.py --eval-best        # Run full eval on best checkpoint
    python scripts/hp_search.py --generate-figures # Generate publication figures

The script:
1. Loads fclf_config.yaml as the base config
2. Generates experiment configs by sweeping over hyperparameter grid
3. Trains each config and saves training curves + checkpoints
4. Evaluates with linear probe on val set
5. Generates publication-quality comparison figures
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from celeba_embeddings import (
    get_attribute_indices,
    load_embeddings_splits,
    select_attributes,
)
from fclf_model import FCLFConfig, ConditionalVectorField, integrate_flow
from scripts.train_fclf import load_yaml_config, train_fclf

# ============================================================================
# Configuration
# ============================================================================

HP_SEARCH_DIR = Path("eval/param_search")
RESULTS_CSV = HP_SEARCH_DIR / "results.csv"
RESULTS_JSON = HP_SEARCH_DIR / "results.json"
RUNS_DIR = HP_SEARCH_DIR / "runs"
FIGURES_DIR = HP_SEARCH_DIR / "figures"

TARGET_ATTRS = ["Smiling", "Young", "Male", "Eyeglasses", "Mustache"]
PRIMARY_ATTR = "Smiling"  # Use this for quick evaluation metric

# Publication-quality figure settings
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
})


# ============================================================================
# Hyperparameter Grid
# ============================================================================

@dataclass
class HyperparameterGrid:
    """
    Define the hyperparameter search space.
    Each field is a list of values to try.
    """
    hidden_dim: List[int] = field(default_factory=lambda: [256])
    learning_rate: List[float] = field(default_factory=lambda: [1e-4, 5e-5])
    alpha: List[float] = field(default_factory=lambda: [0.05, 0.1])
    num_epochs: List[int] = field(default_factory=lambda: [50])
    temperature: List[float] = field(default_factory=lambda: [0.1, 0.2])
    lambda_contrastive: List[float] = field(default_factory=lambda: [0.5, 0.7, 1.0])
    lambda_identity: List[float] = field(default_factory=lambda: [0.2, 0.3])
    lambda_curl: List[float] = field(default_factory=lambda: [0.0, 0.01])
    lambda_div: List[float] = field(default_factory=lambda: [0.0, 0.01])
    num_flow_steps: List[int] = field(default_factory=lambda: [10, 20])
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        keys = [
            "hidden_dim", "learning_rate", "alpha", "num_epochs",
            "temperature", "lambda_contrastive", "lambda_identity",
            "lambda_curl", "lambda_div", "num_flow_steps"
        ]
        values = [getattr(self, k) for k in keys]
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))
    
    def __len__(self) -> int:
        keys = [
            "hidden_dim", "learning_rate", "alpha", "num_epochs",
            "temperature", "lambda_contrastive", "lambda_identity",
            "lambda_curl", "lambda_div", "num_flow_steps"
        ]
        total = 1
        for k in keys:
            total *= len(getattr(self, k))
        return total


def get_quick_grid() -> HyperparameterGrid:
    """Small grid for quick testing."""
    return HyperparameterGrid(
        hidden_dim=[256],
        learning_rate=[1e-4],
        alpha=[0.1],
        num_epochs=[20],
        temperature=[0.1],
        lambda_contrastive=[0.7],
        lambda_identity=[0.3],
        lambda_curl=[0.01],
        lambda_div=[0.01],
        num_flow_steps=[10],
    )


def get_full_grid() -> HyperparameterGrid:
    """Full grid for comprehensive search."""
    return HyperparameterGrid(
        hidden_dim=[256, 512],
        learning_rate=[1e-4, 5e-5],
        alpha=[0.05, 0.1, 0.15],
        num_epochs=[50],
        temperature=[0.1, 0.15, 0.2],
        lambda_contrastive=[0.5, 0.7, 1.0],
        lambda_identity=[0.2, 0.3, 0.5],
        lambda_curl=[0.0, 0.01, 0.02],
        lambda_div=[0.0, 0.01],
        num_flow_steps=[10, 20],
    )


# ============================================================================
# Ablation Configuration (30 configs)
# ============================================================================

@dataclass
class AblationConfig:
    """Configuration for ablation studies."""
    name: str
    description: str  # For paper tables
    category: str  # For grouping in figures
    use_contrastive: bool = True
    use_curl: bool = True
    use_div: bool = True
    use_label_condition: bool = True
    # Override hyperparameters
    hidden_dim: Optional[int] = None
    learning_rate: Optional[float] = None
    alpha: Optional[float] = None
    num_epochs: Optional[int] = None
    temperature: Optional[float] = None
    lambda_contrastive: Optional[float] = None
    lambda_identity: Optional[float] = None
    lambda_curl: Optional[float] = None
    lambda_div: Optional[float] = None
    num_flow_steps: Optional[int] = None


def get_ablation_configs() -> List[AblationConfig]:
    """
    30 ablation configurations organized by category for research paper.
    
    Categories:
    - baseline: Reference configurations
    - loss_components: Testing individual loss terms
    - loss_weights: Sensitivity to loss weights
    - architecture: Model capacity
    - flow_dynamics: Step size and flow steps
    - temperature: Contrastive temperature sensitivity
    - regularization: Vector field regularization
    - combined: Multi-factor ablations
    """
    configs = []
    
    # ==========================================================================
    # Category 1: Baseline configurations (3 configs)
    # ==========================================================================
    configs.extend([
        AblationConfig(
            name="baseline_full",
            description="Full model (all components)",
            category="baseline",
        ),
        AblationConfig(
            name="baseline_minimal",
            description="Minimal: contrastive only",
            category="baseline",
            lambda_identity=0.0,
            lambda_curl=0.0,
            lambda_div=0.0,
        ),
        AblationConfig(
            name="baseline_no_flow",
            description="No flow (K=0)",
            category="baseline",
            num_flow_steps=0,
        ),
    ])
    
    # ==========================================================================
    # Category 2: Loss component ablations (5 configs)
    # ==========================================================================
    configs.extend([
        AblationConfig(
            name="no_contrastive",
            description="Without contrastive loss",
            category="loss_components",
            use_contrastive=False,
            lambda_contrastive=0.0,
        ),
        AblationConfig(
            name="no_identity",
            description="Without identity loss",
            category="loss_components",
            lambda_identity=0.0,
        ),
        AblationConfig(
            name="no_curl",
            description="Without curl regularization",
            category="loss_components",
            use_curl=False,
            lambda_curl=0.0,
        ),
        AblationConfig(
            name="no_div",
            description="Without divergence regularization",
            category="loss_components",
            use_div=False,
            lambda_div=0.0,
        ),
        AblationConfig(
            name="no_vf_reg",
            description="Without any VF regularization",
            category="loss_components",
            use_curl=False,
            use_div=False,
            lambda_curl=0.0,
            lambda_div=0.0,
        ),
    ])
    
    # ==========================================================================
    # Category 3: Contrastive loss weight sensitivity (5 configs)
    # ==========================================================================
    configs.extend([
        AblationConfig(
            name="lc_0.1",
            description="λ_con = 0.1 (very weak)",
            category="loss_weights",
            lambda_contrastive=0.1,
        ),
        AblationConfig(
            name="lc_0.3",
            description="λ_con = 0.3 (weak)",
            category="loss_weights",
            lambda_contrastive=0.3,
        ),
        AblationConfig(
            name="lc_0.5",
            description="λ_con = 0.5 (moderate)",
            category="loss_weights",
            lambda_contrastive=0.5,
        ),
        AblationConfig(
            name="lc_1.0",
            description="λ_con = 1.0 (strong)",
            category="loss_weights",
            lambda_contrastive=1.0,
        ),
        AblationConfig(
            name="lc_1.5",
            description="λ_con = 1.5 (very strong)",
            category="loss_weights",
            lambda_contrastive=1.5,
        ),
    ])
    
    # ==========================================================================
    # Category 4: Identity loss weight sensitivity (4 configs)
    # ==========================================================================
    configs.extend([
        AblationConfig(
            name="li_0.1",
            description="λ_id = 0.1 (weak)",
            category="loss_weights",
            lambda_identity=0.1,
        ),
        AblationConfig(
            name="li_0.5",
            description="λ_id = 0.5 (strong)",
            category="loss_weights",
            lambda_identity=0.5,
        ),
        AblationConfig(
            name="li_0.7",
            description="λ_id = 0.7 (very strong)",
            category="loss_weights",
            lambda_identity=0.7,
        ),
        AblationConfig(
            name="li_1.0",
            description="λ_id = 1.0 (dominant)",
            category="loss_weights",
            lambda_identity=1.0,
        ),
    ])
    
    # ==========================================================================
    # Category 5: Temperature sensitivity (4 configs)
    # ==========================================================================
    configs.extend([
        AblationConfig(
            name="tau_0.05",
            description="τ = 0.05 (sharp)",
            category="temperature",
            temperature=0.05,
        ),
        AblationConfig(
            name="tau_0.1",
            description="τ = 0.1 (default)",
            category="temperature",
            temperature=0.1,
        ),
        AblationConfig(
            name="tau_0.2",
            description="τ = 0.2 (soft)",
            category="temperature",
            temperature=0.2,
        ),
        AblationConfig(
            name="tau_0.5",
            description="τ = 0.5 (very soft)",
            category="temperature",
            temperature=0.5,
        ),
    ])
    
    # ==========================================================================
    # Category 6: Flow dynamics (4 configs)
    # ==========================================================================
    configs.extend([
        AblationConfig(
            name="alpha_0.01",
            description="α = 0.01 (tiny steps)",
            category="flow_dynamics",
            alpha=0.01,
        ),
        AblationConfig(
            name="alpha_0.05",
            description="α = 0.05 (small steps)",
            category="flow_dynamics",
            alpha=0.05,
        ),
        AblationConfig(
            name="alpha_0.15",
            description="α = 0.15 (large steps)",
            category="flow_dynamics",
            alpha=0.15,
        ),
        AblationConfig(
            name="alpha_0.2",
            description="α = 0.2 (very large steps)",
            category="flow_dynamics",
            alpha=0.2,
        ),
    ])
    
    # ==========================================================================
    # Category 7: Number of flow steps (3 configs)
    # ==========================================================================
    configs.extend([
        AblationConfig(
            name="K_5",
            description="K = 5 flow steps",
            category="flow_dynamics",
            num_flow_steps=5,
        ),
        AblationConfig(
            name="K_20",
            description="K = 20 flow steps",
            category="flow_dynamics",
            num_flow_steps=20,
        ),
        AblationConfig(
            name="K_30",
            description="K = 30 flow steps",
            category="flow_dynamics",
            num_flow_steps=30,
        ),
    ])
    
    # ==========================================================================
    # Category 8: Regularization strength (2 configs)
    # ==========================================================================
    configs.extend([
        AblationConfig(
            name="strong_reg",
            description="Strong VF regularization",
            category="regularization",
            lambda_curl=0.05,
            lambda_div=0.05,
        ),
        AblationConfig(
            name="very_strong_reg",
            description="Very strong VF regularization",
            category="regularization",
            lambda_curl=0.1,
            lambda_div=0.1,
        ),
    ])
    
    # ==========================================================================
    # Category 9: Special configurations (extra to make 30)
    # ==========================================================================
    configs.extend([
        AblationConfig(
            name="no_label_cond",
            description="Unconditional flow",
            category="combined",
            use_label_condition=False,
        ),
    ])
    
    return configs


def get_default_hp() -> Dict[str, Any]:
    """Get default hyperparameters for ablation study."""
    return {
        "hidden_dim": 256,
        "learning_rate": 1e-4,
        "alpha": 0.1,
        "num_epochs": 50,
        "temperature": 0.1,
        "lambda_contrastive": 0.7,
        "lambda_identity": 0.3,
        "lambda_curl": 0.01,
        "lambda_div": 0.01,
        "num_flow_steps": 10,
    }


# ============================================================================
# Experiment Utilities
# ============================================================================

def make_run_id(hp: Dict[str, Any], ablation: Optional[AblationConfig] = None) -> str:
    """Create a unique run ID from hyperparameters."""
    if ablation is not None:
        return ablation.name
    
    parts = [
        f"h{hp['hidden_dim']}",
        f"lr{hp['learning_rate']:.0e}",
        f"a{hp['alpha']}",
        f"t{hp['temperature']}",
        f"lc{hp['lambda_contrastive']}",
        f"li{hp['lambda_identity']}",
        f"curl{hp['lambda_curl']}",
        f"div{hp['lambda_div']}",
        f"K{hp['num_flow_steps']}",
    ]
    return "_".join(parts)


def apply_hp_to_config(
    base_cfg: Dict,
    hp: Dict[str, Any],
    ablation: Optional[AblationConfig] = None,
) -> Dict:
    """Apply hyperparameter overrides to a base config."""
    cfg = deepcopy(base_cfg)
    
    # Model
    cfg.setdefault("model", {})
    cfg["model"]["hidden_dim"] = hp["hidden_dim"]
    
    # Training
    cfg.setdefault("training", {})
    cfg["training"]["learning_rate"] = hp["learning_rate"]
    cfg["training"]["alpha"] = hp["alpha"]
    cfg["training"]["num_epochs"] = hp["num_epochs"]
    
    # Loss
    cfg.setdefault("loss", {})
    cfg["loss"]["temperature"] = hp["temperature"]
    cfg["loss"]["lambda_contrastive"] = hp["lambda_contrastive"]
    cfg["loss"]["lambda_identity"] = hp["lambda_identity"]
    cfg["loss"]["lambda_curl"] = hp["lambda_curl"]
    cfg["loss"]["lambda_div"] = hp["lambda_div"]
    
    # Inference
    cfg.setdefault("inference", {})
    cfg["inference"]["num_flow_steps"] = hp["num_flow_steps"]
    cfg["inference"]["step_size"] = hp["alpha"]
    
    # Ablation settings
    cfg.setdefault("ablation", {})
    if ablation is not None:
        cfg["ablation"]["use_contrastive"] = ablation.use_contrastive
        cfg["ablation"]["use_curl"] = ablation.use_curl
        cfg["ablation"]["use_div"] = ablation.use_div
        cfg["ablation"]["use_label_condition"] = ablation.use_label_condition
        
        # Override individual params if specified
        if ablation.hidden_dim is not None:
            cfg["model"]["hidden_dim"] = ablation.hidden_dim
        if ablation.learning_rate is not None:
            cfg["training"]["learning_rate"] = ablation.learning_rate
        if ablation.alpha is not None:
            cfg["training"]["alpha"] = ablation.alpha
            cfg["inference"]["step_size"] = ablation.alpha
        if ablation.num_epochs is not None:
            cfg["training"]["num_epochs"] = ablation.num_epochs
        if ablation.temperature is not None:
            cfg["loss"]["temperature"] = ablation.temperature
        if ablation.lambda_contrastive is not None:
            cfg["loss"]["lambda_contrastive"] = ablation.lambda_contrastive
        if ablation.lambda_identity is not None:
            cfg["loss"]["lambda_identity"] = ablation.lambda_identity
        if ablation.lambda_curl is not None:
            cfg["loss"]["lambda_curl"] = ablation.lambda_curl
        if ablation.lambda_div is not None:
            cfg["loss"]["lambda_div"] = ablation.lambda_div
        if ablation.num_flow_steps is not None:
            cfg["inference"]["num_flow_steps"] = ablation.num_flow_steps
    
    return cfg


def load_model_from_checkpoint(
    ckpt_path: str,
    fallback_cfg: Optional[FCLFConfig] = None,
) -> ConditionalVectorField:
    """Load a trained model from checkpoint."""
    if fallback_cfg is None:
        fallback_cfg = FCLFConfig()
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    
    cfg = FCLFConfig(
        embedding_dim=cfg_dict.get("embedding_dim", fallback_cfg.embedding_dim),
        num_attributes=cfg_dict.get("num_attributes", fallback_cfg.num_attributes),
        hidden_dim=cfg_dict.get("hidden_dim", fallback_cfg.hidden_dim),
        projection_radius=cfg_dict.get("projection_radius", fallback_cfg.projection_radius),
        alpha=cfg_dict.get("alpha", fallback_cfg.alpha),
    )
    
    model = ConditionalVectorField(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def evaluate_checkpoint(
    ckpt_path: str,
    embedding_dir: str,
    num_flow_steps: int,
    device: torch.device,
    target_attr: str = PRIMARY_ATTR,
) -> Dict[str, float]:
    """Evaluate a checkpoint with linear probe."""
    train_split, val_split, _ = load_embeddings_splits(embedding_dir)
    attr_indices = get_attribute_indices(train_split.attr_names, TARGET_ATTRS)
    
    train_attrs = select_attributes(train_split, attr_indices)
    val_attrs = select_attributes(val_split, attr_indices)
    
    attr_col = TARGET_ATTRS.index(target_attr)
    y_train = train_attrs[:, attr_col].numpy()
    y_val = val_attrs[:, attr_col].numpy()
    
    model = load_model_from_checkpoint(ckpt_path).to(device)
    model.eval()
    
    batch_size = 512
    
    def flow_embeddings(embeddings: torch.Tensor, attrs: torch.Tensor) -> np.ndarray:
        chunks = []
        n = embeddings.shape[0]
        with torch.no_grad():
            for i in range(0, n, batch_size):
                z_batch = embeddings[i:i+batch_size].to(device)
                y_batch = attrs[i:i+batch_size].to(device)
                z_flow, _ = integrate_flow(model, z_batch, y_batch, num_steps=num_flow_steps)
                chunks.append(z_flow.cpu())
        return torch.cat(chunks, dim=0).numpy()
    
    X_train_flow = flow_embeddings(train_split.embeddings, train_attrs)
    X_val_flow = flow_embeddings(val_split.embeddings, val_attrs)
    
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train_flow, y_train)
    
    train_pred = clf.predict(X_train_flow)
    val_pred = clf.predict(X_val_flow)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    if len(np.unique(y_val)) >= 2:
        val_probs = clf.predict_proba(X_val_flow)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)
    else:
        val_auc = float("nan")
    
    return {
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "val_auc": float(val_auc),
    }


# ============================================================================
# Training Curve Plotting
# ============================================================================

def plot_training_curves(
    loss_log_path: str,
    output_dir: Path,
    run_id: str,
) -> Dict[str, str]:
    """
    Plot training curves from CSV log.
    Returns dict of saved figure paths.
    """
    if not os.path.exists(loss_log_path):
        return {}
    
    with open(loss_log_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return {}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = [int(row["epoch"]) for row in rows]
    
    paths = {}
    
    # Plot 1: Combined loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    
    loss_keys = [
        ("total_loss", "Total Loss", "#2E86AB"),
        ("contrastive_loss", "Contrastive", "#A23B72"),
        ("identity_loss", "Identity", "#F18F01"),
        ("curl_loss", "Curl Reg", "#C73E1D"),
        ("div_loss", "Div Reg", "#3B1F2B"),
    ]
    
    for key, label, color in loss_keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        if max(values) > 0:  # Only plot if non-zero
            ax.plot(epochs, values, label=label, color=color, linewidth=1.5)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Losses: {run_id}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    combined_path = output_dir / f"loss_curves_{run_id}.png"
    fig.tight_layout()
    fig.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["combined"] = str(combined_path)
    
    # Plot 2: Log-scale loss (for better visualization of small values)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for key, label, color in loss_keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        if max(values) > 0:
            # Add small epsilon to avoid log(0)
            values = [max(v, 1e-8) for v in values]
            ax.semilogy(epochs, values, label=label, color=color, linewidth=1.5)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(f"Training Losses (Log Scale): {run_id}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    log_path = output_dir / f"loss_curves_log_{run_id}.png"
    fig.tight_layout()
    fig.savefig(log_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths["log_scale"] = str(log_path)
    
    return paths


# ============================================================================
# Results Management
# ============================================================================

@dataclass
class ExperimentResult:
    run_id: str
    hyperparams: Dict[str, Any]
    checkpoint_path: str
    run_dir: str
    loss_log_path: Optional[str]
    train_acc: float
    val_acc: float
    val_auc: float
    ablation_name: Optional[str] = None
    ablation_category: Optional[str] = None
    ablation_description: Optional[str] = None
    loss_plot_paths: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            **self.hyperparams,
            "checkpoint_path": self.checkpoint_path,
            "run_dir": self.run_dir,
            "loss_log_path": self.loss_log_path,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "val_auc": self.val_auc,
            "ablation_name": self.ablation_name,
            "ablation_category": self.ablation_category,
            "ablation_description": self.ablation_description,
        }


def run_single_experiment(
    hp: Dict[str, Any],
    base_config_path: str,
    embedding_dir: str,
    seed: Optional[int] = None,
    ablation: Optional[AblationConfig] = None,
) -> ExperimentResult:
    """Train a single experiment and return results."""
    run_id = make_run_id(hp, ablation)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config for reproducibility
    config_path = run_dir / "config.json"
    
    # Load and apply config
    base_cfg = load_yaml_config(base_config_path)
    train_cfg = apply_hp_to_config(base_cfg, hp, ablation=ablation)
    
    # Determine effective num_flow_steps for evaluation
    effective_flow_steps = train_cfg["inference"]["num_flow_steps"]
    
    # Save the full config
    with open(config_path, "w") as f:
        json.dump({
            "base_hp": hp,
            "ablation": ablation.name if ablation else None,
            "train_cfg": train_cfg,
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"[HP Search] Starting run: {run_id}")
    if ablation:
        print(f"  Ablation: {ablation.name}")
        print(f"  Description: {ablation.description}")
        print(f"  Category: {ablation.category}")
    print(f"  hidden_dim={train_cfg['model']['hidden_dim']}, lr={train_cfg['training']['learning_rate']:.0e}")
    print(f"  alpha={train_cfg['training']['alpha']}, tau={train_cfg['loss']['temperature']}")
    print(f"  λ_con={train_cfg['loss']['lambda_contrastive']}, λ_id={train_cfg['loss']['lambda_identity']}")
    print(f"  λ_curl={train_cfg['loss']['lambda_curl']}, λ_div={train_cfg['loss']['lambda_div']}")
    print(f"  K={effective_flow_steps}")
    print(f"  Output: {run_dir}")
    print(f"{'='*70}\n")
    
    # Train
    train_metadata = train_fclf(
        cfg_yaml=train_cfg,
        embedding_dir=embedding_dir,
        output_dir=str(run_dir),
        checkpoint_path=None,
        resume=False,
        save_every=10,  # Save checkpoints every 10 epochs
        seed=seed,
    )
    
    checkpoint_path = train_metadata["checkpoint_path"]
    loss_log_path = train_metadata.get("loss_log_path")
    
    # Plot training curves
    loss_plot_paths = {}
    if loss_log_path:
        loss_plot_paths = plot_training_curves(loss_log_path, plots_dir, run_id)
    
    # Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_checkpoint(
        ckpt_path=checkpoint_path,
        embedding_dir=embedding_dir,
        num_flow_steps=effective_flow_steps,
        device=device,
    )
    
    # Save metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[HP Search] Run {run_id} complete:")
    print(f"  train_acc={metrics['train_acc']:.4f}")
    print(f"  val_acc={metrics['val_acc']:.4f}")
    print(f"  val_auc={metrics['val_auc']:.4f}")
    if loss_plot_paths:
        print(f"  Loss plots saved to: {plots_dir}")
    
    # Build hyperparams dict for logging
    logged_hp = dict(hp)
    if ablation:
        # Update with ablation overrides for accurate logging
        if ablation.temperature is not None:
            logged_hp["temperature"] = ablation.temperature
        if ablation.lambda_contrastive is not None:
            logged_hp["lambda_contrastive"] = ablation.lambda_contrastive
        if ablation.lambda_identity is not None:
            logged_hp["lambda_identity"] = ablation.lambda_identity
        if ablation.lambda_curl is not None:
            logged_hp["lambda_curl"] = ablation.lambda_curl
        if ablation.lambda_div is not None:
            logged_hp["lambda_div"] = ablation.lambda_div
        if ablation.alpha is not None:
            logged_hp["alpha"] = ablation.alpha
        if ablation.num_flow_steps is not None:
            logged_hp["num_flow_steps"] = ablation.num_flow_steps
    
    return ExperimentResult(
        run_id=run_id,
        hyperparams=logged_hp,
        checkpoint_path=checkpoint_path,
        run_dir=str(run_dir),
        loss_log_path=loss_log_path,
        train_acc=metrics["train_acc"],
        val_acc=metrics["val_acc"],
        val_auc=metrics["val_auc"],
        ablation_name=ablation.name if ablation else None,
        ablation_category=ablation.category if ablation else None,
        ablation_description=ablation.description if ablation else None,
        loss_plot_paths=loss_plot_paths,
    )


def save_results(results: List[ExperimentResult]) -> None:
    """Save all results to CSV and JSON."""
    HP_SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    
    if not results:
        return
    
    # CSV with comprehensive fields
    fieldnames = [
        "run_id", "ablation_name", "ablation_category", "ablation_description",
        "val_acc", "val_auc", "train_acc",
        "hidden_dim", "learning_rate", "alpha", "num_epochs",
        "temperature", "lambda_contrastive", "lambda_identity",
        "lambda_curl", "lambda_div", "num_flow_steps",
        "checkpoint_path", "run_dir", "loss_log_path",
    ]
    
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = r.to_dict()
            writer.writerow(row)
    
    # JSON
    RESULTS_JSON.write_text(json.dumps([r.to_dict() for r in results], indent=2))
    
    print(f"\n[HP Search] Results saved to:")
    print(f"  {RESULTS_CSV}")
    print(f"  {RESULTS_JSON}")


def load_results() -> List[ExperimentResult]:
    """Load results from JSON if available."""
    if not RESULTS_JSON.exists():
        return []
    
    data = json.loads(RESULTS_JSON.read_text())
    results = []
    for d in data:
        hp_keys = [
            "hidden_dim", "learning_rate", "alpha", "num_epochs",
            "temperature", "lambda_contrastive", "lambda_identity",
            "lambda_curl", "lambda_div", "num_flow_steps"
        ]
        hp = {k: d.get(k) for k in hp_keys if d.get(k) is not None}
        results.append(ExperimentResult(
            run_id=d["run_id"],
            hyperparams=hp,
            checkpoint_path=d["checkpoint_path"],
            run_dir=d.get("run_dir", ""),
            loss_log_path=d.get("loss_log_path"),
            train_acc=d["train_acc"],
            val_acc=d["val_acc"],
            val_auc=d["val_auc"],
            ablation_name=d.get("ablation_name"),
            ablation_category=d.get("ablation_category"),
            ablation_description=d.get("ablation_description"),
        ))
    return results


def summarize_results(results: List[ExperimentResult], top_k: int = 10) -> None:
    """Print summary of results, sorted by val_acc."""
    if not results:
        print("[HP Search] No results found.")
        return
    
    sorted_results = sorted(results, key=lambda r: r.val_acc, reverse=True)
    
    print(f"\n{'='*90}")
    print(f"HP Search Results ({len(results)} experiments)")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'Val Acc':<10} {'Val AUC':<10} {'Category':<15} {'Run ID':<45}")
    print("-" * 90)
    
    for i, r in enumerate(sorted_results[:top_k], 1):
        cat = r.ablation_category or "grid"
        print(f"{i:<5} {r.val_acc:<10.4f} {r.val_auc:<10.4f} {cat:<15} {r.run_id:<45}")
    
    print("-" * 90)
    
    best = sorted_results[0]
    print(f"\nBest configuration:")
    print(f"  Run ID: {best.run_id}")
    print(f"  Checkpoint: {best.checkpoint_path}")
    print(f"  Val Accuracy: {best.val_acc:.4f}")
    print(f"  Val AUC: {best.val_auc:.4f}")
    if best.ablation_description:
        print(f"  Description: {best.ablation_description}")


# ============================================================================
# Publication Figures
# ============================================================================

def generate_ablation_comparison_figure(results: List[ExperimentResult]) -> str:
    """
    Generate bar chart comparing all ablation configurations.
    Returns path to saved figure.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Filter to ablation results and sort by val_acc
    ablation_results = [r for r in results if r.ablation_name]
    if not ablation_results:
        print("[Figures] No ablation results to plot.")
        return ""
    
    sorted_results = sorted(ablation_results, key=lambda r: r.val_acc, reverse=True)
    
    # Color by category
    category_colors = {
        "baseline": "#2E86AB",
        "loss_components": "#A23B72",
        "loss_weights": "#F18F01",
        "temperature": "#C73E1D",
        "flow_dynamics": "#3B1F2B",
        "regularization": "#28A745",
        "combined": "#6C757D",
    }
    
    names = [r.run_id for r in sorted_results]
    accuracies = [r.val_acc for r in sorted_results]
    colors = [category_colors.get(r.ablation_category, "#888888") for r in sorted_results]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars = ax.barh(range(len(names)), accuracies, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Validation Accuracy")
    ax.set_title("Ablation Study: Validation Accuracy Comparison", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 0.005, i, f"{acc:.3f}", va='center', fontsize=8)
    
    # Legend
    legend_handles = [
        mpatches.Patch(color=color, label=cat.replace("_", " ").title())
        for cat, color in category_colors.items()
        if any(r.ablation_category == cat for r in sorted_results)
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=9)
    
    fig.tight_layout()
    path = FIGURES_DIR / "ablation_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[Figures] Saved ablation comparison: {path}")
    return str(path)


def generate_category_grouped_figure(results: List[ExperimentResult]) -> str:
    """
    Generate grouped bar chart by category.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    ablation_results = [r for r in results if r.ablation_name]
    if not ablation_results:
        return ""
    
    # Group by category
    categories = {}
    for r in ablation_results:
        cat = r.ablation_category or "other"
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    # Sort categories
    cat_order = ["baseline", "loss_components", "loss_weights", "temperature", 
                 "flow_dynamics", "regularization", "combined"]
    ordered_cats = [c for c in cat_order if c in categories]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    category_colors = {
        "baseline": "#2E86AB",
        "loss_components": "#A23B72",
        "loss_weights": "#F18F01",
        "temperature": "#C73E1D",
        "flow_dynamics": "#3B1F2B",
        "regularization": "#28A745",
        "combined": "#6C757D",
    }
    
    for idx, cat in enumerate(ordered_cats):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        cat_results = sorted(categories[cat], key=lambda r: r.val_acc, reverse=True)
        
        names = [r.run_id.replace(f"{cat}_", "").replace("_", "\n") for r in cat_results]
        accs = [r.val_acc for r in cat_results]
        
        color = category_colors.get(cat, "#888888")
        bars = ax.bar(range(len(names)), accs, color=color, edgecolor='white')
        
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7, rotation=45, ha='right')
        ax.set_ylabel("Val Acc")
        ax.set_title(cat.replace("_", " ").title(), fontsize=11, fontweight='bold')
        ax.set_ylim(bottom=min(accs) - 0.05 if accs else 0, top=max(accs) + 0.02 if accs else 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, acc + 0.005, f"{acc:.3f}", 
                   ha='center', va='bottom', fontsize=7)
    
    # Hide unused axes
    for idx in range(len(ordered_cats), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("Ablation Study: Results by Category", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    path = FIGURES_DIR / "ablation_by_category.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[Figures] Saved category comparison: {path}")
    return str(path)


def generate_loss_weight_sensitivity_figure(results: List[ExperimentResult]) -> str:
    """
    Generate line plots showing sensitivity to loss weights.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    ablation_results = [r for r in results if r.ablation_name]
    if not ablation_results:
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: λ_contrastive sensitivity
    lc_results = [r for r in ablation_results if r.run_id.startswith("lc_")]
    if lc_results:
        lc_results = sorted(lc_results, key=lambda r: r.hyperparams.get("lambda_contrastive", 0))
        lc_vals = [r.hyperparams.get("lambda_contrastive", 0) for r in lc_results]
        lc_accs = [r.val_acc for r in lc_results]
        
        axes[0].plot(lc_vals, lc_accs, 'o-', color='#A23B72', linewidth=2, markersize=8)
        axes[0].set_xlabel("λ_contrastive")
        axes[0].set_ylabel("Validation Accuracy")
        axes[0].set_title("Contrastive Loss Weight Sensitivity")
        axes[0].grid(True, alpha=0.3)
        
        for x, y in zip(lc_vals, lc_accs):
            axes[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points", 
                           xytext=(0, 10), ha='center', fontsize=9)
    
    # Plot 2: λ_identity sensitivity
    li_results = [r for r in ablation_results if r.run_id.startswith("li_")]
    if li_results:
        li_results = sorted(li_results, key=lambda r: r.hyperparams.get("lambda_identity", 0))
        li_vals = [r.hyperparams.get("lambda_identity", 0) for r in li_results]
        li_accs = [r.val_acc for r in li_results]
        
        axes[1].plot(li_vals, li_accs, 's-', color='#F18F01', linewidth=2, markersize=8)
        axes[1].set_xlabel("λ_identity")
        axes[1].set_ylabel("Validation Accuracy")
        axes[1].set_title("Identity Loss Weight Sensitivity")
        axes[1].grid(True, alpha=0.3)
        
        for x, y in zip(li_vals, li_accs):
            axes[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points", 
                           xytext=(0, 10), ha='center', fontsize=9)
    
    fig.suptitle("Loss Weight Sensitivity Analysis", fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    path = FIGURES_DIR / "loss_weight_sensitivity.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[Figures] Saved loss weight sensitivity: {path}")
    return str(path)


def generate_flow_dynamics_figure(results: List[ExperimentResult]) -> str:
    """
    Generate plots for flow dynamics (step size, num steps).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    ablation_results = [r for r in results if r.ablation_name]
    if not ablation_results:
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Step size (alpha) sensitivity
    alpha_results = [r for r in ablation_results if r.run_id.startswith("alpha_")]
    if alpha_results:
        alpha_results = sorted(alpha_results, key=lambda r: r.hyperparams.get("alpha", 0))
        alpha_vals = [r.hyperparams.get("alpha", 0) for r in alpha_results]
        alpha_accs = [r.val_acc for r in alpha_results]
        
        axes[0].plot(alpha_vals, alpha_accs, '^-', color='#C73E1D', linewidth=2, markersize=10)
        axes[0].set_xlabel("Step Size (α)")
        axes[0].set_ylabel("Validation Accuracy")
        axes[0].set_title("Step Size Sensitivity")
        axes[0].grid(True, alpha=0.3)
        
        for x, y in zip(alpha_vals, alpha_accs):
            axes[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points", 
                           xytext=(0, 10), ha='center', fontsize=9)
    
    # Plot 2: Number of flow steps
    k_results = [r for r in ablation_results if r.run_id.startswith("K_") or r.run_id == "baseline_no_flow"]
    if k_results:
        k_results = sorted(k_results, key=lambda r: r.hyperparams.get("num_flow_steps", 0))
        k_vals = [r.hyperparams.get("num_flow_steps", 0) for r in k_results]
        k_accs = [r.val_acc for r in k_results]
        
        axes[1].plot(k_vals, k_accs, 'D-', color='#3B1F2B', linewidth=2, markersize=10)
        axes[1].set_xlabel("Number of Flow Steps (K)")
        axes[1].set_ylabel("Validation Accuracy")
        axes[1].set_title("Flow Steps Sensitivity")
        axes[1].grid(True, alpha=0.3)
        
        for x, y in zip(k_vals, k_accs):
            axes[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points", 
                           xytext=(0, 10), ha='center', fontsize=9)
    
    fig.suptitle("Flow Dynamics Analysis", fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    path = FIGURES_DIR / "flow_dynamics.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[Figures] Saved flow dynamics: {path}")
    return str(path)


def generate_ablation_table_latex(results: List[ExperimentResult]) -> str:
    """
    Generate LaTeX table for research paper.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    ablation_results = [r for r in results if r.ablation_name]
    if not ablation_results:
        return ""
    
    sorted_results = sorted(ablation_results, key=lambda r: r.val_acc, reverse=True)
    
    # Generate LaTeX
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation Study Results}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Category & Configuration & Val Acc & Val AUC & Description \\",
        r"\midrule",
    ]
    
    for r in sorted_results:
        cat = r.ablation_category or "-"
        name = r.run_id.replace("_", r"\_")
        desc = (r.ablation_description or "-").replace("_", r"\_")
        lines.append(
            f"{cat} & {name} & {r.val_acc:.4f} & {r.val_auc:.4f} & {desc} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    latex_content = "\n".join(lines)
    
    path = FIGURES_DIR / "ablation_table.tex"
    with open(path, "w") as f:
        f.write(latex_content)
    
    print(f"[Figures] Saved LaTeX table: {path}")
    return str(path)


def generate_all_figures(results: List[ExperimentResult]) -> None:
    """Generate all publication figures."""
    print("\n[Figures] Generating publication figures...")
    
    generate_ablation_comparison_figure(results)
    generate_category_grouped_figure(results)
    generate_loss_weight_sensitivity_figure(results)
    generate_flow_dynamics_figure(results)
    generate_ablation_table_latex(results)
    
    # Generate combined training curves figure
    generate_combined_training_curves(results)
    
    print(f"\n[Figures] All figures saved to: {FIGURES_DIR}")


def generate_combined_training_curves(results: List[ExperimentResult]) -> str:
    """
    Generate a figure comparing training curves of key configurations.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Select key configs to compare
    key_configs = ["baseline_full", "no_contrastive", "no_identity", "no_vf_reg", "baseline_no_flow"]
    
    selected = [r for r in results if r.ablation_name in key_configs and r.loss_log_path]
    
    if not selected:
        print("[Figures] No training logs found for key configs.")
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))
    
    for idx, r in enumerate(selected):
        if not r.loss_log_path or not os.path.exists(r.loss_log_path):
            continue
        
        with open(r.loss_log_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            continue
        
        epochs = [int(row["epoch"]) for row in rows]
        total_loss = [float(row.get("total_loss", 0)) for row in rows]
        con_loss = [float(row.get("contrastive_loss", 0)) for row in rows]
        
        label = r.ablation_name.replace("_", " ").title()
        
        axes[0].plot(epochs, total_loss, label=label, color=colors[idx], linewidth=1.5)
        axes[1].plot(epochs, con_loss, label=label, color=colors[idx], linewidth=1.5)
    
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss Comparison")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Contrastive Loss")
    axes[1].set_title("Contrastive Loss Comparison")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle("Training Curves: Key Ablation Configurations", fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    path = FIGURES_DIR / "training_curves_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[Figures] Saved training curves comparison: {path}")
    return str(path)


# ============================================================================
# Main Search Logic
# ============================================================================

def run_hp_search(
    grid: HyperparameterGrid,
    base_config_path: str,
    embedding_dir: str,
    seed: Optional[int] = None,
) -> List[ExperimentResult]:
    """Run the full hyperparameter search."""
    HP_SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    total = len(grid)
    print(f"\n[HP Search] Starting search over {total} configurations")
    print(f"[HP Search] Results will be saved to {HP_SEARCH_DIR}")
    
    results: List[ExperimentResult] = []
    
    for i, hp in enumerate(grid, 1):
        print(f"\n[HP Search] Experiment {i}/{total}")
        
        try:
            result = run_single_experiment(
                hp=hp,
                base_config_path=base_config_path,
                embedding_dir=embedding_dir,
                seed=seed,
            )
            results.append(result)
            save_results(results)
            
        except Exception as e:
            print(f"[HP Search] Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def run_ablation_study(
    base_hp: Dict[str, Any],
    base_config_path: str,
    embedding_dir: str,
    seed: Optional[int] = None,
) -> List[ExperimentResult]:
    """Run ablation study with all 30 configurations."""
    HP_SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    ablations = get_ablation_configs()
    total = len(ablations)
    
    print(f"\n[Ablation Study] Starting study with {total} configurations")
    print(f"[Ablation Study] Base hyperparameters:")
    for k, v in base_hp.items():
        print(f"  {k}: {v}")
    print(f"[Ablation Study] Results will be saved to {HP_SEARCH_DIR}")
    
    results: List[ExperimentResult] = []
    
    for i, ablation in enumerate(ablations, 1):
        print(f"\n[Ablation Study] Experiment {i}/{total}: {ablation.name}")
        
        try:
            result = run_single_experiment(
                hp=base_hp,
                base_config_path=base_config_path,
                embedding_dir=embedding_dir,
                seed=seed,
                ablation=ablation,
            )
            results.append(result)
            save_results(results)
            
        except Exception as e:
            print(f"[Ablation Study] Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate figures after completing all experiments
    generate_all_figures(results)
    
    return results


def run_full_eval_on_best(embedding_dir: str) -> None:
    """Run eval_fclf_metrics.py on the best checkpoint."""
    results = load_results()
    if not results:
        print("[HP Search] No results found. Run search first.")
        return
    
    best = max(results, key=lambda r: r.val_acc)
    ckpt_path = best.checkpoint_path
    
    if not os.path.exists(ckpt_path):
        print(f"[HP Search] Checkpoint not found: {ckpt_path}")
        return
    
    print(f"\n[HP Search] Running full evaluation on best checkpoint:")
    print(f"  {ckpt_path}")
    print(f"  val_acc={best.val_acc:.4f}")
    
    output_dir = HP_SEARCH_DIR / "best_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_steps = best.hyperparams.get("num_flow_steps", 10)
    
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_fclf_metrics.py"),
        "--checkpoint_path", ckpt_path,
        "--embedding_dir", embedding_dir,
        "--num_steps_flow", str(num_steps),
        "--output_dir", str(output_dir),
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive hyperparameter search and ablation study for FCLF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/hp_search.py                    # Full grid search
    python scripts/hp_search.py --quick            # Quick test run
    python scripts/hp_search.py --ablation         # Run 30-config ablation study
    python scripts/hp_search.py --summarize        # Show results
    python scripts/hp_search.py --eval-best        # Run full eval on best
    python scripts/hp_search.py --generate-figures # Generate publication figures
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fclf_config.yaml",
        help="Base YAML config path.",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="data/embeddings",
        help="Directory with precomputed embeddings.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a minimal grid for quick testing.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run 30-config ablation study.",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Summarize existing results without running new experiments.",
    )
    parser.add_argument(
        "--eval-best",
        action="store_true",
        help="Run eval_fclf_metrics.py on the best checkpoint.",
    )
    parser.add_argument(
        "--generate-figures",
        action="store_true",
        help="Generate publication-quality figures from existing results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.summarize:
        results = load_results()
        summarize_results(results)
        return
    
    if args.generate_figures:
        results = load_results()
        if not results:
            print("[HP Search] No results found. Run search first.")
            return
        generate_all_figures(results)
        return
    
    if args.eval_best:
        run_full_eval_on_best(args.embedding_dir)
        return
    
    if args.ablation:
        base_hp = get_default_hp()
        results = run_ablation_study(
            base_hp=base_hp,
            base_config_path=args.config,
            embedding_dir=args.embedding_dir,
            seed=args.seed,
        )
        summarize_results(results)
        print("\n[Ablation Study] Complete!")
        print(f"  To view results:        python scripts/hp_search.py --summarize")
        print(f"  To run full eval:       python scripts/hp_search.py --eval-best")
        print(f"  To regenerate figures:  python scripts/hp_search.py --generate-figures")
        return
    
    # Run grid search
    grid = get_quick_grid() if args.quick else get_full_grid()
    
    results = run_hp_search(
        grid=grid,
        base_config_path=args.config,
        embedding_dir=args.embedding_dir,
        seed=args.seed,
    )
    
    summarize_results(results)
    generate_all_figures(results)
    
    print("\n[HP Search] Complete!")
    print(f"  To view results:        python scripts/hp_search.py --summarize")
    print(f"  To run full eval:       python scripts/hp_search.py --eval-best")
    print(f"  To regenerate figures:  python scripts/hp_search.py --generate-figures")


if __name__ == "__main__":
    main()
