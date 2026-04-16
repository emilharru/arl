#!/usr/bin/env python3
"""Train a joint baseline model for each dataset under data/.

This script trains one end-to-end multiclass model per dataset, where all signals
(`X_*.npy`) are trained jointly at once instead of one expert at a time.

Outputs per dataset:
- best model checkpoint (selected by lowest validation loss)
- training history
- validation/test metrics (accuracy, kappa, confusion matrix)
- TensorBoard logs
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from cycle_preprocessing import apply_preprocessing

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as exc:  # pragma: no cover - runtime dependency check
    SummaryWriter = None
    _TENSORBOARD_IMPORT_ERROR = exc
else:
    _TENSORBOARD_IMPORT_ERROR = None


@dataclass
class DatasetBundle:
    dataset_name: str
    signal_files: List[str]
    class_labels: List[str]
    x_train: List[np.ndarray]
    y_train: np.ndarray
    x_val: List[np.ndarray]
    y_val: np.ndarray
    x_test: Optional[List[np.ndarray]]
    y_test: Optional[np.ndarray]


class SignalBranch(nn.Module):
    """Small per-signal encoder that supports variable sequence lengths."""

    def __init__(self, in_channels: int, width: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(x))


class JointBaselineModel(nn.Module):
    """Joint multi-signal baseline: one branch per signal, trained end-to-end."""

    def __init__(
        self,
        in_channels_per_signal: Sequence[int],
        num_classes: int,
        branch_width: int = 32,
        embedding_dim: int = 32,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if not in_channels_per_signal:
            raise ValueError("JointBaselineModel requires at least one signal")

        self.branches = nn.ModuleList(
            [
                SignalBranch(
                    in_channels=max(1, int(in_ch)),
                    width=max(4, int(branch_width)),
                    embedding_dim=max(4, int(embedding_dim)),
                    dropout=float(dropout),
                )
                for in_ch in in_channels_per_signal
            ]
        )

        concat_dim = max(4, len(in_channels_per_signal) * max(4, int(embedding_dim)))
        hidden_dim = max(16, concat_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, int(num_classes)),
        )

    def forward(self, x_list: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(x_list) != len(self.branches):
            raise ValueError(
                f"Expected {len(self.branches)} input signals, received {len(x_list)}"
            )
        embeddings = [branch(x) for branch, x in zip(self.branches, x_list)]
        if len(embeddings) == 1:
            concat = embeddings[0]
        else:
            concat = torch.cat(embeddings, dim=1)
        return self.classifier(concat)


def to_model_input(x: np.ndarray, signal_name: str = "signal") -> np.ndarray:
    """Coerce arrays to (N, C, T) float32 format for model consumption."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(
            f"Expected preprocessed {signal_name} with at least 2 dims (N,...), got shape={arr.shape}"
        )

    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.ndim == 2:
        return arr[:, None, :]

    sample_count = arr.shape[0]
    time_steps = arr.shape[-1]
    channel_count = int(np.prod(arr.shape[1:-1]))
    if channel_count < 1 or time_steps < 1:
        raise ValueError(f"Invalid preprocessed {signal_name} shape={arr.shape}")

    return arr.reshape(sample_count, channel_count, time_steps).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one joint multiclass baseline per dataset using all signals at once, "
            "with validation-loss early stopping and TensorBoard logs."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root folder containing dataset directories")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "joint_baseline_all_datasets",
        help="Root folder for output runs",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run subdirectory name (default: UTC timestamp)",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to include (default: all datasets under data-root)",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early-stop patience on validation loss")
    parser.add_argument("--batch-size", type=int, default=128, help="Initial batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--branch-width", type=int, default=32, help="Conv width per signal branch")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding width per signal branch")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout in projection/classifier")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sorted_signal_files(split_dir: Path) -> List[str]:
    return sorted([p.name for p in split_dir.glob("X_*.npy")], key=str)


def _load_split_signals(
    split_dir: Path,
    signal_files: Sequence[str],
    dataset_name: str,
    split_name: str,
) -> List[np.ndarray]:
    loaded: List[np.ndarray] = []
    for file_name in signal_files:
        path = split_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing signal file {path}")
        arr = np.load(path)
        arr = apply_preprocessing(arr)
        arr = to_model_input(arr, signal_name=f"{dataset_name}:{split_name}:{file_name}")
        loaded.append(arr.astype(np.float32, copy=False))
    return loaded


def _assert_split_consistency(
    x_list: Sequence[np.ndarray],
    y: np.ndarray,
    dataset_name: str,
    split_name: str,
) -> None:
    sample_count = int(np.asarray(y).reshape(-1).shape[0])
    if sample_count <= 0:
        raise ValueError(f"{dataset_name} {split_name} split has no labels")
    for idx, arr in enumerate(x_list):
        if arr.shape[0] != sample_count:
            raise ValueError(
                f"{dataset_name} {split_name} sample mismatch in signal index {idx}: "
                f"X={arr.shape[0]} Y={sample_count}"
            )


def _fit_label_map(label_arrays: Sequence[np.ndarray]) -> Tuple[Dict[str, int], List[str]]:
    unique_labels = sorted(
        {str(v) for arr in label_arrays for v in np.asarray(arr).reshape(-1).tolist()},
        key=str,
    )
    if not unique_labels:
        raise ValueError("No class labels discovered")
    mapping = {label: i for i, label in enumerate(unique_labels)}
    return mapping, unique_labels


def _map_labels(y: np.ndarray, label_to_idx: Dict[str, int]) -> np.ndarray:
    y_flat = np.asarray(y).reshape(-1)
    mapped = np.empty(y_flat.shape[0], dtype=np.int64)
    for idx, value in enumerate(y_flat.tolist()):
        key = str(value)
        if key not in label_to_idx:
            raise KeyError(f"Unknown label '{value}' not present in training mapping")
        mapped[idx] = label_to_idx[key]
    return mapped


def load_dataset_bundle(dataset_dir: Path) -> DatasetBundle:
    dataset_name = dataset_dir.name
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "validate"
    test_dir = dataset_dir / "test"

    if not (train_dir / "y.npy").exists():
        raise FileNotFoundError(f"{dataset_name}: missing train/y.npy")
    if not (val_dir / "y.npy").exists():
        raise FileNotFoundError(f"{dataset_name}: missing validate/y.npy")

    signal_files = _sorted_signal_files(train_dir)
    if not signal_files:
        raise FileNotFoundError(f"{dataset_name}: no X_*.npy files in train split")

    train_signal_set = set(signal_files)
    val_signal_set = set(_sorted_signal_files(val_dir))
    if train_signal_set != val_signal_set:
        raise ValueError(
            f"{dataset_name}: train/validate signal mismatch: "
            f"train={sorted(train_signal_set)} validate={sorted(val_signal_set)}"
        )

    y_train_raw = np.load(train_dir / "y.npy", allow_pickle=True).reshape(-1)
    y_val_raw = np.load(val_dir / "y.npy", allow_pickle=True).reshape(-1)

    has_test = (test_dir / "y.npy").exists()
    y_test_raw = np.load(test_dir / "y.npy", allow_pickle=True).reshape(-1) if has_test else None

    label_arrays = [y_train_raw, y_val_raw]
    if y_test_raw is not None:
        label_arrays.append(y_test_raw)

    label_to_idx, class_labels = _fit_label_map(label_arrays)
    y_train = _map_labels(y_train_raw, label_to_idx)
    y_val = _map_labels(y_val_raw, label_to_idx)
    y_test = _map_labels(y_test_raw, label_to_idx) if y_test_raw is not None else None

    x_train = _load_split_signals(train_dir, signal_files, dataset_name, "train")
    x_val = _load_split_signals(val_dir, signal_files, dataset_name, "validate")

    x_test: Optional[List[np.ndarray]] = None
    if has_test:
        test_signal_set = set(_sorted_signal_files(test_dir))
        if test_signal_set != train_signal_set:
            raise ValueError(
                f"{dataset_name}: train/test signal mismatch: "
                f"train={sorted(train_signal_set)} test={sorted(test_signal_set)}"
            )
        x_test = _load_split_signals(test_dir, signal_files, dataset_name, "test")

    _assert_split_consistency(x_train, y_train, dataset_name, "train")
    _assert_split_consistency(x_val, y_val, dataset_name, "validate")
    if x_test is not None and y_test is not None:
        _assert_split_consistency(x_test, y_test, dataset_name, "test")


    for idx, name in enumerate(signal_files):
        train_shape = x_train[idx].shape[1:]
        val_shape = x_val[idx].shape[1:]
        if train_shape != val_shape:
            raise ValueError(
                f"{dataset_name}:{name} train/validate shape mismatch: {train_shape} vs {val_shape}"
            )
        if x_test is not None:
            test_shape = x_test[idx].shape[1:]
            if train_shape != test_shape:
                raise ValueError(
                    f"{dataset_name}:{name} train/test shape mismatch: {train_shape} vs {test_shape}"
                )

    return DatasetBundle(
        dataset_name=dataset_name,
        signal_files=signal_files,
        class_labels=class_labels,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, Any]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "kappa": 0.0,
            "confusion_matrix": [],
            "evaluated_samples": 0,
        }

    accuracy = float(np.mean(y_true == y_pred))
    kappa_raw = cohen_kappa_score(y_true, y_pred) if np.unique(y_true).size > 1 else 0.0
    kappa = float(kappa_raw) if np.isfinite(kappa_raw) else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(int(num_classes)))).tolist()
    return {
        "accuracy": accuracy,
        "kappa": kappa,
        "confusion_matrix": cm,
        "evaluated_samples": int(y_true.size),
    }


def _evaluate(
    model: nn.Module,
    x_list: Sequence[torch.Tensor],
    y: torch.Tensor,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    pred_chunks: List[np.ndarray] = []
    true_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, int(y.size(0)), int(batch_size)):
            end = min(start + int(batch_size), int(y.size(0)))
            batch_x = [x[start:end].to(device) for x in x_list]
            batch_y = y[start:end].to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            count = int(batch_y.size(0))
            total_loss += float(loss.item()) * count
            total_samples += count

            pred = torch.argmax(logits, dim=1).detach().cpu().numpy().reshape(-1)
            true = batch_y.detach().cpu().numpy().reshape(-1)
            pred_chunks.append(pred)
            true_chunks.append(true)

    if pred_chunks:
        y_pred = np.concatenate(pred_chunks)
        y_true = np.concatenate(true_chunks)
    else:
        y_pred = np.array([], dtype=np.int64)
        y_true = np.array([], dtype=np.int64)

    avg_loss = float(total_loss / max(total_samples, 1))
    metrics = _compute_metrics(y_true, y_pred, num_classes=int(model.classifier[-1].out_features))
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def _train_single_dataset(
    bundle: DatasetBundle,
    dataset_out_dir: Path,
    tensorboard_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    dataset_out_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    x_train_t = [torch.tensor(arr, dtype=torch.float32) for arr in bundle.x_train]
    x_val_t = [torch.tensor(arr, dtype=torch.float32) for arr in bundle.x_val]
    x_test_t = [torch.tensor(arr, dtype=torch.float32) for arr in bundle.x_test] if bundle.x_test is not None else None

    y_train_t = torch.tensor(bundle.y_train, dtype=torch.long)
    y_val_t = torch.tensor(bundle.y_val, dtype=torch.long)
    y_test_t = torch.tensor(bundle.y_test, dtype=torch.long) if bundle.y_test is not None else None

    in_channels_per_signal = [int(arr.shape[1]) for arr in bundle.x_train]
    num_classes = int(len(bundle.class_labels))

    model = JointBaselineModel(
        in_channels_per_signal=in_channels_per_signal,
        num_classes=num_classes,
        branch_width=args.branch_width,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best_model_path = dataset_out_dir / "best_model.pt"
    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    stopped_early = False

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text("config/dataset", bundle.dataset_name)
    writer.add_text("config/signals", ", ".join(bundle.signal_files))

    effective_batch_size = max(1, int(args.batch_size))


    while True:
        try:
            for epoch in range(1, int(args.epochs) + 1):
                model.train()
                permutation = torch.randperm(y_train_t.size(0))

                train_loss_sum = 0.0
                train_count = 0
                train_preds: List[np.ndarray] = []
                train_true: List[np.ndarray] = []

                for start in range(0, int(y_train_t.size(0)), int(effective_batch_size)):
                    batch_idx = permutation[start : start + int(effective_batch_size)]
                    batch_x = [x[batch_idx].to(device) for x in x_train_t]
                    batch_y = y_train_t[batch_idx].to(device)

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()

                    batch_size_now = int(batch_y.size(0))
                    train_loss_sum += float(loss.item()) * batch_size_now
                    train_count += batch_size_now

                    pred = torch.argmax(logits.detach(), dim=1).cpu().numpy().reshape(-1)
                    true = batch_y.detach().cpu().numpy().reshape(-1)
                    train_preds.append(pred)
                    train_true.append(true)

                train_avg_loss = float(train_loss_sum / max(train_count, 1))
                train_pred_np = np.concatenate(train_preds) if train_preds else np.array([], dtype=np.int64)
                train_true_np = np.concatenate(train_true) if train_true else np.array([], dtype=np.int64)
                train_metrics = _compute_metrics(train_true_np, train_pred_np, num_classes=num_classes)

                val_loss, val_metrics = _evaluate(
                    model=model,
                    x_list=x_val_t,
                    y=y_val_t,
                    criterion=criterion,
                    batch_size=effective_batch_size,
                    device=device,
                )

                writer.add_scalar("loss/train", train_avg_loss, epoch)
                writer.add_scalar("loss/validate", float(val_loss), epoch)
                writer.add_scalar("metrics/train_accuracy", float(train_metrics["accuracy"]), epoch)
                writer.add_scalar("metrics/validate_accuracy", float(val_metrics["accuracy"]), epoch)
                writer.add_scalar("metrics/validate_kappa", float(val_metrics["kappa"]), epoch)

                epoch_row = {
                    "epoch": int(epoch),
                    "train_loss": float(train_avg_loss),
                    "val_loss": float(val_loss),
                    "train_accuracy": float(train_metrics["accuracy"]),
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "val_kappa": float(val_metrics["kappa"]),
                }
                history.append(epoch_row)

                improved = float(val_loss) < float(best_val_loss)
                if improved:
                    best_val_loss = float(val_loss)
                    best_epoch = int(epoch)
                    patience_counter = 0

                    checkpoint = {
                        "dataset": bundle.dataset_name,
                        "epoch": int(epoch),
                        "best_val_loss": float(best_val_loss),
                        "model_state_dict": model.state_dict(),
                        "model_config": {
                            "in_channels_per_signal": in_channels_per_signal,
                            "num_classes": num_classes,
                            "branch_width": int(args.branch_width),
                            "embedding_dim": int(args.embedding_dim),
                            "dropout": float(args.dropout),
                        },
                        "signal_files": bundle.signal_files,
                        "class_labels": bundle.class_labels,
                    }
                    torch.save(checkpoint, best_model_path)
                else:
                    patience_counter += 1

                print(
                    f"[{bundle.dataset_name}] epoch {epoch:03d} "
                    f"train_loss={train_avg_loss:.6f} "
                    f"val_loss={val_loss:.6f} "
                    f"val_acc={val_metrics['accuracy']:.4f} "
                    f"val_kappa={val_metrics['kappa']:.4f}"
                )

                if patience_counter >= int(args.patience):
                    stopped_early = True
                    print(
                        f"[{bundle.dataset_name}] early stopping at epoch {epoch} "
                        f"(patience={args.patience}, best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f})"
                    )
                    break
            break

        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            if effective_batch_size <= 1:
                raise RuntimeError(
                    f"{bundle.dataset_name}: CUDA OOM even with batch size 1"
                )
            effective_batch_size = max(1, effective_batch_size // 2)
            print(
                f"[{bundle.dataset_name}] CUDA OOM, retrying training with batch size {effective_batch_size}"
            )
            model = JointBaselineModel(
                in_channels_per_signal=in_channels_per_signal,
                num_classes=num_classes,
                branch_width=args.branch_width,
                embedding_dim=args.embedding_dim,
                dropout=args.dropout,
            ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
            )
            history = []
            best_val_loss = float("inf")
            best_epoch = 0
            patience_counter = 0
            stopped_early = False

    writer.flush()
    writer.close()

    if not best_model_path.exists():
        raise RuntimeError(f"{bundle.dataset_name}: no best checkpoint was saved")

    best_ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    val_loss, val_metrics = _evaluate(
        model=model,
        x_list=x_val_t,
        y=y_val_t,
        criterion=criterion,
        batch_size=effective_batch_size,
        device=device,
    )

    test_metrics: Optional[Dict[str, Any]] = None
    if x_test_t is not None and y_test_t is not None:
        _, test_metrics = _evaluate(
            model=model,
            x_list=x_test_t,
            y=y_test_t,
            criterion=criterion,
            batch_size=effective_batch_size,
            device=device,
        )

    return {
        "status": "success",
        "dataset": bundle.dataset_name,
        "signal_count": len(bundle.signal_files),
        "class_count": int(num_classes),
        "class_labels": bundle.class_labels,
        "signal_files": bundle.signal_files,
        "train_samples": int(y_train_t.size(0)),
        "validate_samples": int(y_val_t.size(0)),
        "test_samples": int(y_test_t.size(0)) if y_test_t is not None else 0,
        "epochs_completed": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "early_stopped": bool(stopped_early),
        "patience": int(args.patience),
        "effective_batch_size": int(effective_batch_size),
        "metrics_validate": {
            "accuracy": float(val_metrics["accuracy"]),
            "kappa": float(val_metrics["kappa"]),
            "confusion_matrix": val_metrics["confusion_matrix"],
            "loss": float(val_loss),
            "evaluated_samples": int(val_metrics["evaluated_samples"]),
        },
        "metrics_test": test_metrics,
        "best_model_path": str(best_model_path),
        "tensorboard_log_dir": str(tensorboard_dir),
        "history": history,
    }


def _select_dataset_dirs(data_root: Path, include: Optional[Sequence[str]]) -> List[Path]:
    include_set = {name.strip() for name in (include or []) if name and name.strip()}
    dataset_dirs = []
    for child in sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        if include_set and child.name not in include_set:
            continue
        if not (child / "train").exists():
            continue
        dataset_dirs.append(child)
    return dataset_dirs


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    args = parse_args()

    if SummaryWriter is None:
        raise RuntimeError(
            "TensorBoard writer unavailable. Install the dependency first: pip install tensorboard "
            f"(import error: {_TENSORBOARD_IMPORT_ERROR})"
        )

    if int(args.epochs) < 1:
        raise ValueError("--epochs must be >= 1")
    if int(args.patience) < 1:
        raise ValueError("--patience must be >= 1")
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be >= 1")

    set_seed(int(args.seed))

    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    device = resolve_device(str(args.device).strip().lower())
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_name = args.run_name or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = (args.output_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_root = run_dir / "tensorboard"
    tensorboard_root.mkdir(parents=True, exist_ok=True)

    dataset_dirs = _select_dataset_dirs(data_root, args.datasets)
    if not dataset_dirs:
        raise RuntimeError("No datasets selected/found under data root")

    print(f"Selected {len(dataset_dirs)} datasets from {data_root}")
    print(f"Output run directory: {run_dir}")
    print(f"Device: {device}")

    all_results: List[Dict[str, Any]] = []
    started_at = time.time()

    for idx, dataset_dir in enumerate(dataset_dirs, start=1):
        dataset_name = dataset_dir.name
        print(f"\n[{idx}/{len(dataset_dirs)}] Training joint baseline for {dataset_name}")

        dataset_out_dir = run_dir / "datasets" / dataset_name
        dataset_tb_dir = tensorboard_root / dataset_name

        t0 = time.time()
        try:
            bundle = load_dataset_bundle(dataset_dir)
            result = _train_single_dataset(
                bundle=bundle,
                dataset_out_dir=dataset_out_dir,
                tensorboard_dir=dataset_tb_dir,
                args=args,
                device=device,
            )
            result["runtime_seconds"] = float(time.time() - t0)
            _save_json(dataset_out_dir / "metrics.json", result)

            history_payload = {
                "dataset": dataset_name,
                "history": result.get("history", []),
                "best_epoch": result.get("best_epoch"),
                "best_val_loss": result.get("best_val_loss"),
            }
            _save_json(dataset_out_dir / "history.json", history_payload)


            compact_result = dict(result)
            compact_result.pop("history", None)
            all_results.append(compact_result)

            val_acc = result["metrics_validate"]["accuracy"]
            val_kappa = result["metrics_validate"]["kappa"]
            test_metrics = result.get("metrics_test")
            if isinstance(test_metrics, dict):
                test_acc = float(test_metrics.get("accuracy", 0.0))
                test_kappa = float(test_metrics.get("kappa", 0.0))
                print(
                    f"[{dataset_name}] done in {result['runtime_seconds']:.1f}s "
                    f"best_val_loss={result['best_val_loss']:.6f} "
                    f"val_acc={val_acc:.4f} val_kappa={val_kappa:.4f} "
                    f"test_acc={test_acc:.4f} test_kappa={test_kappa:.4f}"
                )
            else:
                print(
                    f"[{dataset_name}] done in {result['runtime_seconds']:.1f}s "
                    f"best_val_loss={result['best_val_loss']:.6f} "
                    f"val_acc={val_acc:.4f} val_kappa={val_kappa:.4f} "
                    "test_split=unavailable"
                )

        except Exception as exc:
            error_result = {
                "status": "failed",
                "dataset": dataset_name,
                "error": str(exc),
                "runtime_seconds": float(time.time() - t0),
            }
            all_results.append(error_result)
            _save_json(dataset_out_dir / "error.json", error_result)
            print(f"[{dataset_name}] failed: {exc}")

    elapsed = float(time.time() - started_at)
    success_rows = [r for r in all_results if r.get("status") == "success"]

    test_acc_values = [
        float(r.get("metrics_test", {}).get("accuracy"))
        for r in success_rows
        if isinstance(r.get("metrics_test"), dict)
    ]
    test_kappa_values = [
        float(r.get("metrics_test", {}).get("kappa"))
        for r in success_rows
        if isinstance(r.get("metrics_test"), dict)
    ]

    validate_acc_values = [
        float(r.get("metrics_validate", {}).get("accuracy", 0.0))
        for r in success_rows
    ]
    validate_kappa_values = [
        float(r.get("metrics_validate", {}).get("kappa", 0.0))
        for r in success_rows
    ]

    summary = {
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "run_dir": str(run_dir),
        "device": str(device),
        "hyperparameters": {
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "branch_width": int(args.branch_width),
            "embedding_dim": int(args.embedding_dim),
            "dropout": float(args.dropout),
            "seed": int(args.seed),
        },
        "dataset_count": int(len(dataset_dirs)),
        "success_count": int(len(success_rows)),
        "failure_count": int(len(dataset_dirs) - len(success_rows)),
        "runtime_seconds": elapsed,
        "mean_validate_accuracy": float(np.mean(validate_acc_values)) if validate_acc_values else None,
        "mean_validate_kappa": float(np.mean(validate_kappa_values)) if validate_kappa_values else None,
        "mean_test_accuracy": float(np.mean(test_acc_values)) if test_acc_values else None,
        "mean_test_kappa": float(np.mean(test_kappa_values)) if test_kappa_values else None,
        "datasets": all_results,
        "tensorboard_log_root": str(tensorboard_root),
        "tensorboard_hint": f"tensorboard --logdir {tensorboard_root}",
    }

    _save_json(run_dir / "summary.json", summary)

    print("\nRun complete")
    print(f"Summary: {run_dir / 'summary.json'}")
    print(f"TensorBoard logs: {tensorboard_root}")
    print(f"Launch TensorBoard: tensorboard --logdir {tensorboard_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
