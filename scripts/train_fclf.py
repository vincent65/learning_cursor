import argparse
import csv
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import yaml

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
from fclf_model import FCLFConfig, ConditionalVectorField, integrate_flow, tangent_sphere_step


class EmbeddingDataset(Dataset):
    """
    Simple dataset over precomputed CLIP embeddings and selected attributes.
    """

    def __init__(self, embeddings: torch.Tensor, attrs: torch.Tensor):
        assert embeddings.shape[0] == attrs.shape[0]
        self.embeddings = embeddings
        self.attrs = attrs

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.attrs[idx]


def supervised_contrastive_loss_multi_label(
    z: torch.Tensor,
    y: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Multi-label supervised contrastive loss on flowed embeddings.

    z: (B, D) – flowed embeddings
    y: (B, K) – binary attribute vectors {0,1}
    """
    z = nn.functional.normalize(z, dim=-1)
    B = z.size(0)

    # Similarity matrix
    sim = torch.matmul(z, z.t()) / temperature  # (B, B)

    # Mask out self-similarities
    logits_mask = torch.ones_like(sim) - torch.eye(B, device=sim.device)
    sim = sim * logits_mask

    # Positive mask: samples with identical attribute vectors
    y_i = y.unsqueeze(1)  # (B, 1, K)
    y_j = y.unsqueeze(0)  # (1, B, K)
    pos_mask = (y_i == y_j).all(dim=-1).float() * logits_mask  # (B, B)

    # Compute log-probabilities
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    pos_count = pos_mask.sum(dim=1)  # (B,)
    # Avoid division by zero; samples without positives do not contribute
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

    valid = pos_count > 0
    if valid.sum() == 0:
        # Degenerate minibatch (all labels unique) – return zero to avoid NaNs
        return torch.zeros((), device=z.device)

    loss = -(mean_log_prob_pos[valid]).mean()
    return loss


def train_one_epoch(
    model: ConditionalVectorField,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float,
    temperature: float,
    lambda_contrastive: float,
    lambda_identity: float,
    lambda_curl: float,
    lambda_div: float,
    use_contrastive: bool,
    use_curl: bool,
    use_div: bool,
    use_label_condition: bool,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_con = 0.0
    total_id = 0.0
    total_curl = 0.0
    total_div = 0.0
    n_samples = 0

    for z_batch, y_batch in dataloader:
        z_batch = z_batch.to(device)
        y_batch = y_batch.to(device)

        y_input = y_batch if use_label_condition else torch.zeros_like(y_batch)
        z_next, v_tan = tangent_sphere_step(model, z_batch, y_input, alpha=alpha)

        L_con = supervised_contrastive_loss_multi_label(
            z_next, y_batch, temperature=temperature
        )
        if not use_contrastive:
            L_con = torch.zeros((), device=z_batch.device)

        L_id = (z_next - z_batch).pow(2).sum(dim=-1).mean()
        L_curl = v_tan.pow(2).sum(dim=-1).mean()
        L_div = (v_tan.sum(dim=-1)).pow(2).mean()

        loss = lambda_identity * L_id
        if use_contrastive:
            loss = loss + lambda_contrastive * L_con
        if use_curl:
            loss = loss + lambda_curl * L_curl
        if use_div:
            loss = loss + lambda_div * L_div

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = z_batch.size(0)
        total_loss += loss.item() * bs
        total_con += L_con.item() * bs
        total_id += L_id.item() * bs
        total_curl += L_curl.item() * bs
        total_div += L_div.item() * bs
        n_samples += bs

    mean_loss = total_loss / n_samples
    mean_con = total_con / n_samples
    mean_id = total_id / n_samples
    mean_curl = total_curl / n_samples
    mean_div = total_div / n_samples
    mean_reg = mean_id
    if use_curl:
        mean_reg += mean_curl
    if use_div:
        mean_reg += mean_div
    return {
        "loss": mean_loss,
        "contrastive": mean_con,
        "identity": mean_id,
        "curl": mean_curl,
        "div": mean_div,
        "reg": mean_reg,
    }


def evaluate_identity_shift(
    model: ConditionalVectorField,
    dataloader: DataLoader,
    device: torch.device,
    alpha: float,
    use_label_condition: bool = True,
) -> float:
    """
    Simple stability diagnostic: average squared shift ||z_next - z||^2.
    """
    model.eval()
    total_shift = 0.0
    n_samples = 0

    with torch.no_grad():
        for z_batch, y_batch in dataloader:
            z_batch = z_batch.to(device)
            y_batch = y_batch.to(device)
            y_input = y_batch if use_label_condition else torch.zeros_like(y_batch)
            z_next, _ = tangent_sphere_step(model, z_batch, y_input, alpha=alpha)
            shift = (z_next - z_batch).pow(2).sum(dim=-1).mean()
            bs = z_batch.size(0)
            total_shift += shift.item() * bs
            n_samples += bs

    return total_shift / n_samples


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _append_csv_row(path: str, fieldnames: List[str], row: Dict) -> None:
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _log_classifier_metrics(
    epoch: int,
    model: ConditionalVectorField,
    device: torch.device,
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    train_attrs: torch.Tensor,
    val_attrs: torch.Tensor,
    attr_names: List[str],
    selected_attrs: List[str],
    max_points: int,
    num_steps: int,
    csv_path: str,
) -> None:
    if max_points <= 0 or not selected_attrs:
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    model.eval()
    train_idx = torch.randperm(train_embeddings.shape[0])[: min(max_points, train_embeddings.shape[0])]
    val_idx = torch.randperm(val_embeddings.shape[0])[: min(max_points, val_embeddings.shape[0])]

    train_subset = train_embeddings[train_idx]
    val_subset = val_embeddings[val_idx]
    train_attrs_subset = train_attrs[train_idx]
    val_attrs_subset = val_attrs[val_idx]

    with torch.no_grad():
        z_train_flow, _ = integrate_flow(
            model=model,
            z0=train_subset.to(device),
            y=train_attrs_subset.to(device),
            num_steps=num_steps,
        )
        z_val_flow, _ = integrate_flow(
            model=model,
            z0=val_subset.to(device),
            y=val_attrs_subset.to(device),
            num_steps=num_steps,
        )

    X_train_raw = train_subset.cpu().numpy()
    X_val_raw = val_subset.cpu().numpy()
    X_train_flow = z_train_flow.cpu().numpy()
    X_val_flow = z_val_flow.cpu().numpy()

    for attr in selected_attrs:
        if attr not in attr_names:
            continue
        attr_idx = attr_names.index(attr)
        y_train = train_attrs_subset[:, attr_idx].cpu().numpy()
        y_val = val_attrs_subset[:, attr_idx].cpu().numpy()
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue

        def _train_and_log(mode: str, X_tr: np.ndarray, X_va: np.ndarray) -> Dict:
            clf = LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                solver="lbfgs",
            )
            clf.fit(X_tr, y_train)
            train_pred = clf.predict(X_tr)
            val_pred = clf.predict(X_va)
            row = {
                "epoch": epoch,
                "attr": attr,
                "mode": mode,
                "train_acc": float(accuracy_score(y_train, train_pred)),
                "val_acc": float(accuracy_score(y_val, val_pred)),
                "val_auc": float("nan"),
            }
            if len(np.unique(y_val)) == 2:
                try:
                    val_probs = clf.predict_proba(X_va)[:, 1]
                    row["val_auc"] = float(roc_auc_score(y_val, val_probs))
                except ValueError:
                    row["val_auc"] = float("nan")
            return row

        raw_row = _train_and_log("raw", X_train_raw, X_val_raw)
        flow_row = _train_and_log("flow", X_train_flow, X_val_flow)
        _append_csv_row(
            csv_path,
            ["epoch", "attr", "mode", "train_acc", "val_acc", "val_auc"],
            raw_row,
        )
        _append_csv_row(
            csv_path,
            ["epoch", "attr", "mode", "train_acc", "val_acc", "val_auc"],
            flow_row,
        )


def save_checkpoint(
    path: str,
    model: ConditionalVectorField,
    optimizer: torch.optim.Optimizer,
    cfg: FCLFConfig,
    epoch: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.__dict__,
            "epoch": epoch,
        },
        path,
    )


def maybe_resume_training(
    checkpoint_path: str,
    model: ConditionalVectorField,
    optimizer: torch.optim.Optimizer,
) -> int:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_path} not found; cannot resume."
        )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt.get("epoch", 0)
    print(f"[INFO] Resuming training from epoch {start_epoch} using {checkpoint_path}")
    return start_epoch


def train_fclf(
    cfg_yaml: Dict,
    embedding_dir: str,
    output_dir: str,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    save_every: int = 5,
    seed: Optional[int] = None,
) -> Dict:
    """
    Programmatic entry point for training. Returns a metadata dict with checkpoint info.
    """

    _set_seed(seed)

    model_cfg = cfg_yaml.get("model", {})
    training_cfg = cfg_yaml.get("training", {})
    loss_cfg = cfg_yaml.get("loss", {})
    inference_cfg = cfg_yaml.get("inference", {})
    logging_cfg = cfg_yaml.get("logging", {})

    embedding_dim = model_cfg.get("embedding_dim", 512)
    num_attributes = model_cfg.get("num_attributes", 5)
    hidden_dim = model_cfg.get("hidden_dim", 256)
    projection_radius = model_cfg.get("projection_radius", 1.0)

    num_epochs = training_cfg.get("num_epochs", 50)
    batch_size = training_cfg.get("batch_size", 512)
    learning_rate = training_cfg.get("learning_rate", 1e-4)
    alpha = training_cfg.get("alpha", 0.1)

    temperature = loss_cfg.get("temperature", 0.1)
    lambda_contrastive = loss_cfg.get("lambda_contrastive", 0.7)
    lambda_identity = loss_cfg.get("lambda_identity", 0.3)

    lambda_curl = loss_cfg.get("lambda_curl", 0.01)
    lambda_div = loss_cfg.get("lambda_div", 0.0)

    ablation_cfg = cfg_yaml.get("ablation", {})
    use_contrastive = ablation_cfg.get("use_contrastive", True)
    use_curl = ablation_cfg.get("use_curl", True)
    use_div = ablation_cfg.get("use_div", True)
    use_label_condition = ablation_cfg.get("use_label_condition", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Loading precomputed embeddings...")
    train_split, val_split, _ = load_embeddings_splits(embedding_dir)

    target_attrs = ["Smiling", "Young", "Male", "Eyeglasses", "Mustache"]
    attr_indices = get_attribute_indices(train_split.attr_names, target_attrs)

    train_attrs = select_attributes(train_split, attr_indices)
    val_attrs = select_attributes(val_split, attr_indices)

    train_ds = EmbeddingDataset(train_split.embeddings, train_attrs)
    val_ds = EmbeddingDataset(val_split.embeddings, val_attrs)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print("[INFO] Building vector field model...")
    fclf_cfg = FCLFConfig(
        embedding_dim=embedding_dim,
        num_attributes=num_attributes,
        hidden_dim=hidden_dim,
        projection_radius=projection_radius,
        alpha=alpha,
    )
    model = ConditionalVectorField(fclf_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs(output_dir, exist_ok=True)
    if checkpoint_path is None:
        checkpoint_path = os.path.join(output_dir, "fclf_last.pt")
    loss_log_path = os.path.join(output_dir, logging_cfg.get("loss_log_name", "flow_losses.csv"))
    classifier_log_path = os.path.join(
        output_dir,
        logging_cfg.get("classifier_log_name", "classifier_curves.csv"),
    )
    classifier_interval = logging_cfg.get("classifier_interval", 0)
    classifier_attrs = logging_cfg.get("classifier_attrs", ["Smiling"])
    classifier_max_points = logging_cfg.get("classifier_max_points", 2000)
    classifier_flow_steps = logging_cfg.get(
        "classifier_flow_steps",
        inference_cfg.get("num_flow_steps", 10),
    )
    if os.path.exists(loss_log_path):
        os.remove(loss_log_path)
    if classifier_interval > 0 and os.path.exists(classifier_log_path):
        os.remove(classifier_log_path)

    start_epoch = 0
    if resume:
        start_epoch = maybe_resume_training(checkpoint_path, model, optimizer)
        if start_epoch >= num_epochs:
            print(
                f"[INFO] Checkpoint already at epoch {start_epoch} >= num_epochs; nothing to do."
            )
            return {
                "checkpoint_path": checkpoint_path,
                "output_dir": output_dir,
                "epochs_trained": 0,
            }

    print(
        f"[INFO] Training for {num_epochs - start_epoch} additional epochs "
        f"(total target {num_epochs}) on device {device} with batch_size={batch_size}"
    )
    loss_fieldnames = [
        "epoch",
        "total_loss",
        "contrastive_loss",
        "identity_loss",
        "curl_loss",
        "div_loss",
    ]
    for epoch in range(start_epoch + 1, num_epochs + 1):
        metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha=alpha,
            temperature=temperature,
            lambda_contrastive=lambda_contrastive,
            lambda_identity=lambda_identity,
            lambda_curl=lambda_curl,
            lambda_div=lambda_div,
            use_contrastive=use_contrastive,
            use_curl=use_curl,
            use_div=use_div,
            use_label_condition=use_label_condition,
        )
        mean_loss = metrics["loss"]
        mean_con = metrics["contrastive"]
        mean_reg = metrics["reg"]

        avg_shift = evaluate_identity_shift(
            model=model,
            dataloader=val_loader,
            device=device,
            alpha=alpha,
            use_label_condition=use_label_condition,
        )

        print(
            f"[Epoch {epoch:03d}/{num_epochs}] "
            f"loss={mean_loss:.4f} | con={mean_con:.4f} | reg={mean_reg:.4f} | "
            f"avg_val_shift={avg_shift:.4f}"
        )
        _append_csv_row(
            loss_log_path,
            loss_fieldnames,
            {
                "epoch": epoch,
                "total_loss": mean_loss,
                "contrastive_loss": metrics["contrastive"],
                "identity_loss": metrics["identity"],
                "curl_loss": metrics["curl"],
                "div_loss": metrics["div"],
            },
        )
        if classifier_interval > 0 and epoch % classifier_interval == 0:
            _log_classifier_metrics(
                epoch=epoch,
                model=model,
                device=device,
                train_embeddings=train_split.embeddings,
                val_embeddings=val_split.embeddings,
                train_attrs=train_attrs,
                val_attrs=val_attrs,
                attr_names=target_attrs,
                selected_attrs=classifier_attrs,
                max_points=classifier_max_points,
                num_steps=classifier_flow_steps,
                csv_path=classifier_log_path,
            )

        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            cfg=fclf_cfg,
            epoch=epoch,
        )

        if save_every > 0 and epoch % save_every == 0:
            epoch_ckpt = os.path.join(
                output_dir,
                f"fclf_epoch_{epoch:03d}.pt",
            )
            save_checkpoint(
                path=epoch_ckpt,
                model=model,
                optimizer=optimizer,
                cfg=fclf_cfg,
                epoch=epoch,
            )

    return {
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir,
        "epochs_trained": num_epochs - start_epoch,
        "final_epoch": num_epochs,
        "loss_log_path": loss_log_path,
        "classifier_log_path": classifier_log_path if classifier_interval > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FCLF on CelebA CLIP embeddings.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fclf_config.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="data/embeddings",
        help="Directory with train/val/test embedding .pt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/default",
        help="Directory to store checkpoints for this run.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Full path to the latest-checkpoint file. "
        "If omitted, defaults to <output_dir>/fclf_last.pt",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from --checkpoint_path if it exists.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save an additional checkpoint every N epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    args = parser.parse_args()

    cfg_yaml = load_yaml_config(args.config)
    train_fclf(
        cfg_yaml=cfg_yaml,
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        resume=args.resume,
        save_every=args.save_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


