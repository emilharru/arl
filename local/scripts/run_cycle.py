#!/usr/bin/env python3
import os
import sys
import json
import inspect
import yaml
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from binary_expert_model import BinaryExpertModel
from cycle_preprocessing import apply_preprocessing


def load_and_preprocess_signal_split(train_dir, val_dir, x_filename, train_indices=None):
    """Load train/val arrays and run preprocessing without split-aware preprocessing logic."""
    x_tr_path = train_dir / x_filename
    x_vl_path = val_dir / x_filename if val_dir.exists() else x_tr_path

    x_train = np.load(x_tr_path)
    if train_indices is not None:
        x_train = x_train[train_indices]

    x_val = np.load(x_vl_path)
    return apply_preprocessing(x_train), apply_preprocessing(x_val)


def to_model_input(x, signal_name="signal"):
    """Coerce preprocessed arrays to (N, C, T) for model consumption."""
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


def normalize_ensemble_architecture(value: Any) -> str:
    text = str(value or "default").strip().lower()
    if text in {"default", "simple"}:
        return text
    return "default"

class BaselineEnsemble(nn.Module):
    def __init__(
        self,
        experts: List[nn.Module],
        input_map: List[int],
        num_classes: int,
        expert_dim: int = 16,
        expert_dims: Optional[List[int]] = None,
        dropout_p: float = 0.35,
        architecture: str = "default",
    ):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.input_map = input_map

        if len(self.input_map) != len(self.experts):
            raise ValueError(
                f"input_map length {len(self.input_map)} must match experts length {len(self.experts)}"
            )

        if expert_dims is not None and len(expert_dims) == len(experts):
            self.expert_dims = [max(1, int(dim)) for dim in expert_dims]
        else:
            self.expert_dims = [max(1, int(expert_dim)) for _ in experts]

        try:
            self.dropout_p = float(dropout_p)
        except (TypeError, ValueError):
            self.dropout_p = 0.35
        self.dropout_p = min(0.8, max(0.0, self.dropout_p))
        self.architecture = normalize_ensemble_architecture(architecture)


        for param in self.experts.parameters():
            param.requires_grad = False

        if self.architecture == "simple":

            concat_dim = max(1, len(self.experts))
            self.expert_representation = "binary_logit"
        else:
            concat_dim = max(4, int(sum(self.expert_dims)))
            self.expert_representation = "embedding"

        hidden_1 = max(2, concat_dim // 2)
        hidden_2 = max(2, concat_dim // 4)
        if self.architecture == "simple":
            self.mlp = nn.Sequential(
                nn.Linear(concat_dim, hidden_1),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(hidden_1, num_classes),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(concat_dim, hidden_1),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(hidden_1, hidden_2),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(hidden_2, num_classes),
            )

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:

        expert_outputs = []
        with torch.no_grad():
            for i, expert in enumerate(self.experts):
                channel_idx = self.input_map[i]
                x_c = x_list[channel_idx]
                if x_c.dim() == 2:
                    x_c = x_c.unsqueeze(1)
                elif x_c.dim() > 3:
                    x_c = x_c.reshape(x_c.size(0), -1, x_c.size(-1))

                if self.architecture == "simple":

                    try:
                        expert_out = expert(x_c)
                    except TypeError:
                        lengths = torch.full(
                            (x_c.size(0),),
                            x_c.size(-1),
                            dtype=torch.long,
                            device=x_c.device,
                        )
                        expert_out = expert(x_c, lengths=lengths)

                    if isinstance(expert_out, (tuple, list)):
                        expert_out = expert_out[0]

                    if expert_out.dim() == 1:
                        expert_out = expert_out.unsqueeze(1)
                    elif expert_out.dim() > 2:
                        expert_out = expert_out.reshape(expert_out.size(0), -1)

                    if expert_out.size(1) != 1:
                        expert_out = expert_out[:, :1]
                    expert_outputs.append(expert_out)
                else:
                    feat = expert.extract_features(x_c)
                    expert_outputs.append(feat)

        concat_emb = torch.cat(expert_outputs, dim=1)
        return self.mlp(concat_emb)


def _curve_summary_from_history(history, best_epoch=None, early_stopped=False):
    if not history:
        return {
            "epochs_completed": 0,
            "best_epoch": best_epoch,
            "early_stopped": bool(early_stopped),
            "trend": None,
            "best_val_loss": None,
        }

    val_losses = [float(h.get("val_loss", 0.0)) for h in history if "val_loss" in h]
    best_val_loss = min(val_losses) if val_losses else None
    trend = None
    if len(val_losses) >= 2:
        trend = "improving" if val_losses[-1] <= val_losses[0] else "degrading"

    return {
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "early_stopped": bool(early_stopped),
        "trend": trend,
        "best_val_loss": best_val_loss,
    }


def _build_binary_expert_model(model_cls, in_ch, n_classes, min_seq_len, model_init_overrides=None):
    """Instantiate an expert model class while tolerating varying init signatures."""
    base_kwargs = {
        "in_ch": in_ch,
        "n_classes": n_classes,
        "fs": 100.0,
        "min_seq_len": min_seq_len,
        "dts": (0.05, 0.15, 0.5, 1.5),
        "k_min": 7,
        "k_max_cap": 129,
        "width": 16,
        "depth": 2,
        "dropout": 0.1,
    }
    if isinstance(model_init_overrides, dict):
        for key, value in model_init_overrides.items():
            if value is not None:
                base_kwargs[str(key)] = value

    try:
        signature = inspect.signature(model_cls.__init__)
        params = [p for p in signature.parameters.values() if p.name != "self"]
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

        model_kwargs = {}
        missing_required = []
        for param in params:
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.name in base_kwargs:
                model_kwargs[param.name] = base_kwargs[param.name]
            elif param.default is inspect._empty:
                missing_required.append(param.name)

        if missing_required:
            raise TypeError(
                f"Cannot instantiate {getattr(model_cls, '__name__', model_cls)}: "
                f"missing required init args {missing_required}"
            )

        if accepts_var_kwargs:
            for key, value in base_kwargs.items():
                model_kwargs.setdefault(key, value)

        return model_cls(**model_kwargs)
    except TypeError:

        return model_cls(**base_kwargs)


def _balanced_binary_undersample_indices(y_train, class_label):
    """Build undersampled indices with a 1:2 positive-to-negative split and class-balanced negatives."""
    y_arr = np.asarray(y_train).reshape(-1)
    full_indices = np.arange(y_arr.shape[0], dtype=np.int64)
    if y_arr.size == 0:
        return full_indices, {
            "applied": False,
            "reason": "empty_training_set",
        }

    pos_indices = full_indices[y_arr == class_label]
    neg_indices = full_indices[y_arr != class_label]

    if pos_indices.size == 0 or neg_indices.size == 0:
        return full_indices, {
            "applied": False,
            "reason": "missing_positive_or_negative_class",
            "target_class": str(class_label),
            "original_positive": int(pos_indices.size),
            "original_negative": int(neg_indices.size),
        }

    rng = np.random.default_rng()


    target_positive = int(pos_indices.size)
    target_negative = int(min(neg_indices.size, max(1, target_positive * 2)))
    pos_selected = rng.permutation(pos_indices)[:target_positive]

    neg_labels = y_arr[neg_indices]
    neg_classes = list(np.unique(neg_labels))
    class_pools = []
    for cls in neg_classes:
        pool = neg_indices[neg_labels == cls]
        pool = rng.permutation(pool)
        class_pools.append(pool.tolist())

    selected_neg = []
    active = [i for i, pool in enumerate(class_pools) if pool]
    while len(selected_neg) < target_negative and active:
        rng.shuffle(active)
        progressed = False
        next_active = []
        for idx in active:
            if len(selected_neg) >= target_negative:
                break
            pool = class_pools[idx]
            if pool:
                selected_neg.append(int(pool.pop()))
                progressed = True
            if pool:
                next_active.append(idx)
        active = next_active
        if not progressed:
            break

    if len(selected_neg) < target_negative:
        remaining = []
        for pool in class_pools:
            if pool:
                remaining.extend(pool)
        need = target_negative - len(selected_neg)
        if need > 0 and remaining:
            fill = rng.permutation(np.asarray(remaining, dtype=np.int64))[:need]
            selected_neg.extend(fill.tolist())

    selected_neg_arr = np.asarray(selected_neg[:target_negative], dtype=np.int64)
    selected_indices = np.concatenate([pos_selected.astype(np.int64), selected_neg_arr], axis=0)
    selected_indices = rng.permutation(selected_indices)

    neg_selected_labels = y_arr[selected_neg_arr]
    neg_unique, neg_counts = np.unique(neg_selected_labels, return_counts=True)
    neg_distribution = {str(lbl): int(cnt) for lbl, cnt in zip(neg_unique, neg_counts)}

    return selected_indices.astype(np.int64), {
        "applied": True,
        "target_class": str(class_label),
        "original_total": int(y_arr.size),
        "original_positive": int(pos_indices.size),
        "original_negative": int(neg_indices.size),
        "used_total": int(selected_indices.size),
        "used_positive": int(target_positive),
        "used_negative": int(selected_neg_arr.size),
        "target_negative_to_positive_ratio": 2.0,
        "negative_distribution": neg_distribution,
        "negative_classes": int(len(neg_classes)),
    }


def train_baseline(
    x_train,
    y_train,
    x_val,
    y_val,
    class_label,
    epochs=50,
    patience=3,
    min_epochs=7,
    lr=1e-3,
    project_root=None,
    model_name="Baseline",
    model_cls=BinaryExpertModel,
    model_init_overrides=None,
    stop_on_perfect_f1=1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train = to_model_input(x_train, signal_name=f"{model_name}:train")
    x_val = to_model_input(x_val, signal_name=f"{model_name}:val")
    if x_train.shape[1] != x_val.shape[1] or x_train.shape[-1] != x_val.shape[-1]:
        raise ValueError(
            f"Train/val shape mismatch for {model_name}: train={x_train.shape}, val={x_val.shape}"
        )

    y_train = np.asarray(y_train).reshape(-1)
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Train sample mismatch for {model_name}: X={x_train.shape[0]} Y={y_train.shape[0]}"
        )

    undersample_indices, undersample_summary = _balanced_binary_undersample_indices(
        y_train=y_train,
        class_label=class_label,
    )
    if undersample_summary.get("applied"):
        x_train = x_train[undersample_indices]
        y_train = y_train[undersample_indices]

    y_tr = (y_train == class_label).astype(np.float32)
    y_vl = (y_val == class_label).astype(np.float32)

    epochs = max(1, int(epochs))
    min_epochs = max(1, int(min_epochs))
    effective_epochs = max(epochs, min_epochs)

    num_pos = y_tr.sum()
    num_neg = len(y_tr) - num_pos

    if num_pos == 0 or num_neg == 0:
        empty_summary = {
            "epochs_completed": 0,
            "best_epoch": None,
            "early_stopped": False,
            "trend": None,
            "best_val_loss": None,
            "status": "skipped_missing_binary_class",
            "train_sampling": undersample_summary,
        }
        return 0.0, 0.0, 0.0, 0.0, None, [], empty_summary

    pos_weight_val = num_neg / max(num_pos, 1)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    x_tr_t = torch.tensor(x_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    x_vl_t = torch.tensor(x_val, dtype=torch.float32)
    y_vl_t = torch.tensor(y_vl, dtype=torch.float32).unsqueeze(1)

    in_ch = x_train.shape[1]
    T = x_train.shape[-1]
    configured_batch_size = max(1, int(os.environ.get("ARL_EXPERT_BATCH_SIZE", 256)))
    batch_size = max(1, min(len(x_train), configured_batch_size))


    requested_cv_folds = 3

    try:
        target_f1 = float(stop_on_perfect_f1)
    except (TypeError, ValueError):
        target_f1 = 1.0
    target_f1 = min(1.0, max(0.0, target_f1))

    try:
        cv_random_seed = int(os.environ.get("ARL_EXPERT_CV_SEED", 42))
    except (TypeError, ValueError):
        cv_random_seed = 42

    def _compute_binary_metrics(preds_np, targets_np):
        preds_np = np.asarray(preds_np).reshape(-1)
        targets_np = np.asarray(targets_np).reshape(-1)
        if targets_np.size == 0:
            return {
                "accuracy": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        return {
            "accuracy": float(np.mean(preds_np == targets_np)),
            "f1": float(f1_score(targets_np, preds_np, zero_division=0)),
            "precision": float(precision_score(targets_np, preds_np, zero_division=0)),
            "recall": float(recall_score(targets_np, preds_np, zero_division=0)),
        }

    def _evaluate_binary_model(model, x_eval_t, y_eval_t, criterion, eval_batch_size):
        model.eval()
        loss_sum = 0.0
        batches = 0
        preds_list = []

        with torch.no_grad():
            for i in range(0, x_eval_t.size(0), eval_batch_size):
                batch_x = x_eval_t[i:i+eval_batch_size].to(device)
                batch_y = y_eval_t[i:i+eval_batch_size].to(device)
                lengths = torch.full((batch_x.size(0),), T, dtype=torch.long, device=device)
                logits = model(batch_x, lengths=lengths)
                loss = criterion(logits, batch_y)
                loss_sum += loss.item()
                batches += 1
                preds_list.append((torch.sigmoid(logits) >= 0.5).float().cpu().numpy().reshape(-1))

        preds_np = np.concatenate(preds_list) if preds_list else np.array([], dtype=np.float32)
        targets_np = y_eval_t.numpy().reshape(-1)
        metrics = _compute_binary_metrics(preds_np, targets_np)
        metrics["loss"] = float(loss_sum / max(batches, 1))
        return metrics

    def _train_with_holdout(
        x_train_fold_t,
        y_train_fold_t,
        x_holdout_t,
        y_holdout_t,
        *,
        epochs_to_run,
        allow_early_stopping,
    ):
        model = _build_binary_expert_model(
            model_cls=model_cls,
            in_ch=in_ch,
            n_classes=1,
            min_seq_len=T,
            model_init_overrides=model_init_overrides,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        history = []
        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = None
        patience_counter = 0
        early_stopped = False
        early_stop_reason = None

        for epoch in range(max(1, int(epochs_to_run))):
            model.train()
            permutation = torch.randperm(x_train_fold_t.size(0))
            epoch_loss = 0.0
            batches = 0
            train_preds_batches = []
            train_targets_batches = []

            for i in range(0, x_train_fold_t.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x = x_train_fold_t[indices].to(device)
                batch_y = y_train_fold_t[indices].to(device)
                lengths = torch.full((batch_x.size(0),), T, dtype=torch.long, device=device)

                optimizer.zero_grad()
                logits = model(batch_x, lengths=lengths)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_preds_batches.append(
                    (torch.sigmoid(logits.detach()) >= 0.5).float().cpu().numpy().reshape(-1)
                )
                train_targets_batches.append(batch_y.detach().cpu().numpy().reshape(-1))
                epoch_loss += loss.item()
                batches += 1

            train_preds_epoch = (
                np.concatenate(train_preds_batches)
                if train_preds_batches
                else np.array([], dtype=np.float32)
            )
            train_targets_epoch = (
                np.concatenate(train_targets_batches)
                if train_targets_batches
                else np.array([], dtype=np.float32)
            )
            train_metrics = _compute_binary_metrics(train_preds_epoch, train_targets_epoch)

            holdout_metrics = _evaluate_binary_model(
                model,
                x_holdout_t,
                y_holdout_t,
                criterion,
                batch_size,
            )

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(epoch_loss / max(batches, 1)),
                    "val_loss": float(holdout_metrics.get("loss", 0.0)),
                    "accuracy": float(holdout_metrics.get("accuracy", 0.0)),
                    "f1": float(holdout_metrics.get("f1", 0.0)),
                    "precision": float(holdout_metrics.get("precision", 0.0)),
                    "recall": float(holdout_metrics.get("recall", 0.0)),
                    "train_accuracy": float(train_metrics.get("accuracy", 0.0)),
                    "train_f1": float(train_metrics.get("f1", 0.0)),
                    "train_precision": float(train_metrics.get("precision", 0.0)),
                    "train_recall": float(train_metrics.get("recall", 0.0)),
                    "val_accuracy": float(holdout_metrics.get("accuracy", 0.0)),
                    "val_f1": float(holdout_metrics.get("f1", 0.0)),
                    "val_precision": float(holdout_metrics.get("precision", 0.0)),
                    "val_recall": float(holdout_metrics.get("recall", 0.0)),
                }
            )

            current_val_loss = float(holdout_metrics.get("loss", 0.0))
            current_val_f1 = float(holdout_metrics.get("f1", 0.0))


            if current_val_f1 >= (target_f1 - 1e-12):
                best_val_loss = min(best_val_loss, current_val_loss)
                best_epoch = epoch + 1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                early_stopped = True
                early_stop_reason = f"target_f1_reached_{target_f1:.6f}"
                break

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                best_epoch = epoch + 1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if allow_early_stopping and (epoch + 1) >= min_epochs and patience_counter >= patience:
                early_stopped = True
                early_stop_reason = "patience_exhausted"
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        final_holdout_metrics = _evaluate_binary_model(
            model,
            x_holdout_t,
            y_holdout_t,
            criterion,
            batch_size,
        )

        return {
            "model": model,
            "history": history,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else None,
            "early_stopped": bool(early_stopped),
            "early_stop_reason": early_stop_reason,
            "holdout_metrics": final_holdout_metrics,
        }

    while True:
        try:
            y_tr_binary = y_tr.astype(np.int64)
            class_counts = np.bincount(y_tr_binary, minlength=2)
            nonzero_counts = [int(v) for v in class_counts.tolist() if int(v) > 0]
            min_class_count = min(nonzero_counts) if nonzero_counts else 0

            cv_folds = min(requested_cv_folds, min_class_count)
            cv_fallback_reason = None
            if cv_folds < 2:
                cv_folds = 1
                cv_fallback_reason = "insufficient_samples_per_class_for_3fold"

            cv_runs = []
            if cv_folds >= 2:
                splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_random_seed)
                for fold_idx, (train_idx, holdout_idx) in enumerate(
                    splitter.split(np.zeros_like(y_tr_binary), y_tr_binary),
                    start=1,
                ):
                    train_idx_t = torch.tensor(np.asarray(train_idx, dtype=np.int64), dtype=torch.long)
                    holdout_idx_t = torch.tensor(np.asarray(holdout_idx, dtype=np.int64), dtype=torch.long)

                    fold_run = _train_with_holdout(
                        x_train_fold_t=x_tr_t[train_idx_t],
                        y_train_fold_t=y_tr_t[train_idx_t],
                        x_holdout_t=x_tr_t[holdout_idx_t],
                        y_holdout_t=y_tr_t[holdout_idx_t],
                        epochs_to_run=effective_epochs,
                        allow_early_stopping=True,
                    )
                    fold_run["fold"] = int(fold_idx)
                    fold_run["train_samples"] = int(train_idx_t.numel())
                    fold_run["holdout_samples"] = int(holdout_idx_t.numel())
                    cv_runs.append(fold_run)
            else:
                full_idx_t = torch.arange(x_tr_t.size(0), dtype=torch.long)
                fold_run = _train_with_holdout(
                    x_train_fold_t=x_tr_t[full_idx_t],
                    y_train_fold_t=y_tr_t[full_idx_t],
                    x_holdout_t=x_tr_t[full_idx_t],
                    y_holdout_t=y_tr_t[full_idx_t],
                    epochs_to_run=effective_epochs,
                    allow_early_stopping=True,
                )
                fold_run["fold"] = 1
                fold_run["train_samples"] = int(full_idx_t.numel())
                fold_run["holdout_samples"] = int(full_idx_t.numel())
                cv_runs.append(fold_run)

            cv_histories = [run.get("history", []) for run in cv_runs if isinstance(run.get("history"), list)]
            max_cv_epochs = max((len(hist) for hist in cv_histories), default=0)
            live_log = []
            metric_keys = [
                "train_loss",
                "val_loss",
                "accuracy",
                "f1",
                "precision",
                "recall",
                "train_accuracy",
                "train_f1",
                "train_precision",
                "train_recall",
                "val_accuracy",
                "val_f1",
                "val_precision",
                "val_recall",
            ]

            for epoch_idx in range(max_cv_epochs):
                rows = [hist[epoch_idx] for hist in cv_histories if epoch_idx < len(hist)]
                if not rows:
                    continue

                row = {"epoch": epoch_idx + 1}
                for key in metric_keys:
                    vals = [float(r.get(key, 0.0)) for r in rows]
                    row[key] = float(np.mean(vals)) if vals else 0.0
                live_log.append(row)

            cv_best_epochs = [
                int(run.get("best_epoch"))
                for run in cv_runs
                if run.get("best_epoch") is not None
            ]
            if cv_best_epochs:
                selected_epoch = int(round(float(np.mean(cv_best_epochs))))
            else:
                selected_epoch = int(effective_epochs)
            selected_epoch = max(min_epochs, min(effective_epochs, max(1, selected_epoch)))

            final_run = _train_with_holdout(
                x_train_fold_t=x_tr_t,
                y_train_fold_t=y_tr_t,
                x_holdout_t=x_tr_t,
                y_holdout_t=y_tr_t,
                epochs_to_run=selected_epoch,
                allow_early_stopping=False,
            )
            model = final_run.get("model")

            final_eval_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            validation_metrics = _evaluate_binary_model(
                model,
                x_vl_t,
                y_vl_t,
                final_eval_criterion,
                batch_size,
            )

            fold_metric_names = ["accuracy", "f1", "precision", "recall", "loss"]
            cv_metrics = {
                "fold_count": int(cv_folds),
                "target_fold_count": int(requested_cv_folds),
                "fallback_reason": cv_fallback_reason,
            }
            for metric_name in fold_metric_names:
                metric_values = [
                    float((run.get("holdout_metrics") or {}).get(metric_name, 0.0))
                    for run in cv_runs
                ]
                if metric_values:
                    cv_metrics[f"{metric_name}_mean"] = float(np.mean(metric_values))
                    cv_metrics[f"{metric_name}_std"] = float(np.std(metric_values))
                else:
                    cv_metrics[f"{metric_name}_mean"] = 0.0
                    cv_metrics[f"{metric_name}_std"] = 0.0

            cv_fold_details = []
            for run in cv_runs:
                holdout_metrics = run.get("holdout_metrics") or {}
                cv_fold_details.append(
                    {
                        "fold": int(run.get("fold", 0)),
                        "train_samples": int(run.get("train_samples", 0)),
                        "holdout_samples": int(run.get("holdout_samples", 0)),
                        "best_epoch": int(run.get("best_epoch") or 0) if run.get("best_epoch") is not None else None,
                        "early_stopped": bool(run.get("early_stopped", False)),
                        "early_stop_reason": run.get("early_stop_reason"),
                        "metrics": {
                            "accuracy": float(holdout_metrics.get("accuracy", 0.0)),
                            "f1": float(holdout_metrics.get("f1", 0.0)),
                            "precision": float(holdout_metrics.get("precision", 0.0)),
                            "recall": float(holdout_metrics.get("recall", 0.0)),
                            "loss": float(holdout_metrics.get("loss", 0.0)),
                        },
                    }
                )

            cv_early_stopped_folds = sum(1 for run in cv_runs if bool(run.get("early_stopped", False)))
            cv_perfect_f1_stopped_folds = sum(
                1
                for run in cv_runs
                if str(run.get("early_stop_reason") or "").startswith("target_f1_reached_")
            )
            curve_summary = _curve_summary_from_history(
                live_log,
                best_epoch=selected_epoch,
                early_stopped=cv_early_stopped_folds > 0,
            )
            curve_summary["train_sampling"] = undersample_summary
            curve_summary["min_epochs"] = int(min_epochs)
            curve_summary["configured_epochs"] = int(epochs)
            curve_summary["effective_epochs"] = int(effective_epochs)
            curve_summary["batch_size"] = int(batch_size)
            curve_summary["train_samples"] = int(x_train.shape[0])
            curve_summary["steps_per_epoch"] = int(max(1, (x_train.shape[0] + batch_size - 1) // batch_size))
            curve_summary["early_stopping_basis"] = "cross_validation_train_split"
            curve_summary["cv_metrics"] = cv_metrics
            curve_summary["cv_fold_details"] = cv_fold_details
            curve_summary["cv_best_epochs"] = [int(v) for v in cv_best_epochs]
            curve_summary["cv_selected_epoch"] = int(selected_epoch)
            curve_summary["cv_early_stopped_folds"] = int(cv_early_stopped_folds)
            curve_summary["cv_perfect_f1_stopped_folds"] = int(cv_perfect_f1_stopped_folds)
            curve_summary["stop_on_perfect_f1"] = float(target_f1)
            curve_summary["final_train_early_stopped"] = bool(final_run.get("early_stopped", False))
            if final_run.get("early_stop_reason"):
                curve_summary["final_train_early_stop_reason"] = str(final_run.get("early_stop_reason"))
            curve_summary["validation_metrics"] = {
                "accuracy": float(validation_metrics.get("accuracy", 0.0)),
                "f1": float(validation_metrics.get("f1", 0.0)),
                "precision": float(validation_metrics.get("precision", 0.0)),
                "recall": float(validation_metrics.get("recall", 0.0)),
                "loss": float(validation_metrics.get("loss", 0.0)),
                "split": "validate",
            }

            if project_root:
                log_file = project_root / "state" / "live_training.json"
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "w") as f:
                    json.dump(
                        {
                            "model_name": model_name,
                            "history": live_log,
                            "validation_source": "cross_validation_train_split",
                        },
                        f,
                    )

            return (
                float(validation_metrics.get("f1", 0.0)),
                float(validation_metrics.get("accuracy", 0.0)),
                float(validation_metrics.get("precision", 0.0)),
                float(validation_metrics.get("recall", 0.0)),
                model,
                live_log,
                curve_summary,
            )

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size = batch_size // 2
            if batch_size == 0:
                raise RuntimeError("CUDA out of memory even with batch size 1 in train_baseline")
            print(f"CUDA OutOfMemoryError in train_baseline. Reducing batch size to {batch_size}")

def train_ensemble(
    model,
    x_train_list,
    y_train,
    x_val_list,
    y_val,
    epochs=50,
    patience=5,
    lr=1e-3,
    batch_size=64,
    project_root=None,
    stop_on_perfect_f1=1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_tr_t_list = [torch.tensor(x, dtype=torch.float32) for x in x_train_list]
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    x_vl_t_list = [torch.tensor(x, dtype=torch.float32) for x in x_val_list]
    y_vl_t = torch.tensor(y_val, dtype=torch.long)

    model = model.to(device)

    epochs = max(1, int(epochs))
    patience = max(1, int(patience))
    batch_size = max(1, int(batch_size))


    requested_cv_folds = 3
    try:
        cv_random_seed = int(os.environ.get("ARL_ENSEMBLE_CV_SEED", 42))
    except (TypeError, ValueError):
        cv_random_seed = 42

    try:
        target_f1 = float(stop_on_perfect_f1)
    except (TypeError, ValueError):
        target_f1 = 1.0
    target_f1 = min(1.0, max(0.0, target_f1))

    def reset_mlp(target_model):
        for module in target_model.mlp.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    simple_mode = normalize_ensemble_architecture(getattr(model, "architecture", "default")) == "simple"
    if simple_mode:
        expected_concat_dim = max(1, len(model.experts))
    else:
        expert_dims = getattr(model, "expert_dims", None)
        if isinstance(expert_dims, list) and expert_dims:
            expected_concat_dim = max(4, int(sum(expert_dims)))
        else:
            expected_concat_dim = max(4, int(len(model.experts)) * 16)

    def _compute_multiclass_metrics(preds_np, targets_np):
        preds_np = np.asarray(preds_np).reshape(-1)
        targets_np = np.asarray(targets_np).reshape(-1)
        if targets_np.size == 0:
            return {
                "accuracy": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        return {
            "accuracy": float(np.mean(preds_np == targets_np)),
            "f1": float(f1_score(targets_np, preds_np, average="macro", zero_division=0)),
            "precision": float(precision_score(targets_np, preds_np, average="macro", zero_division=0)),
            "recall": float(recall_score(targets_np, preds_np, average="macro", zero_division=0)),
        }

    def _extract_concat_expert_outputs(x_source_list_t, sample_indices_t):
        if sample_indices_t.numel() <= 0:
            return np.zeros((0, expected_concat_dim), dtype=np.float32)

        model.experts.eval()
        batch_outputs = []

        with torch.no_grad():
            for i in range(0, sample_indices_t.numel(), batch_size):
                batch_idx = sample_indices_t[i:i + batch_size]
                batch_x_list = [x[batch_idx].to(device) for x in x_source_list_t]
                expert_outputs = []

                for expert_idx, expert in enumerate(model.experts):
                    channel_idx = model.input_map[expert_idx]
                    x_c = batch_x_list[channel_idx]
                    if x_c.dim() == 2:
                        x_c = x_c.unsqueeze(1)
                    elif x_c.dim() > 3:
                        x_c = x_c.reshape(x_c.size(0), -1, x_c.size(-1))

                    if simple_mode:
                        try:
                            expert_out = expert(x_c)
                        except TypeError:
                            lengths = torch.full(
                                (x_c.size(0),),
                                x_c.size(-1),
                                dtype=torch.long,
                                device=x_c.device,
                            )
                            expert_out = expert(x_c, lengths=lengths)

                        if isinstance(expert_out, (tuple, list)):
                            expert_out = expert_out[0]
                        if not torch.is_tensor(expert_out):
                            expert_out = torch.as_tensor(expert_out, device=x_c.device)
                        if expert_out.dim() == 1:
                            expert_out = expert_out.unsqueeze(1)
                        elif expert_out.dim() > 2:
                            expert_out = expert_out.reshape(expert_out.size(0), -1)
                        if expert_out.size(1) != 1:
                            expert_out = expert_out[:, :1]
                        expert_outputs.append(expert_out.detach().cpu())
                    else:
                        feat = expert.extract_features(x_c)
                        if feat.dim() == 1:
                            feat = feat.unsqueeze(1)
                        elif feat.dim() > 2:
                            feat = feat.reshape(feat.size(0), -1)
                        expert_outputs.append(feat.detach().cpu())

                if expert_outputs:
                    batch_concat = torch.cat(expert_outputs, dim=1)
                else:
                    batch_concat = torch.zeros((batch_idx.numel(), expected_concat_dim), dtype=torch.float32)
                batch_outputs.append(batch_concat)

        if not batch_outputs:
            return np.zeros((0, expected_concat_dim), dtype=np.float32)

        concat = torch.cat(batch_outputs, dim=0)
        return concat.numpy().astype(np.float32, copy=False)

    def _evaluate_mlp_model(feature_t, target_t, criterion, eval_batch_size):
        model.mlp.eval()
        pred_batches = []
        loss_sum = 0.0
        batches = 0

        with torch.no_grad():
            for i in range(0, target_t.size(0), eval_batch_size):
                batch_x = feature_t[i:i + eval_batch_size].to(device)
                batch_y = target_t[i:i + eval_batch_size].to(device)
                logits = model.mlp(batch_x)
                loss = criterion(logits, batch_y)
                loss_sum += loss.item()
                batches += 1
                pred_batches.append(torch.argmax(logits, dim=1).detach().cpu().numpy().reshape(-1))

        preds_np = np.concatenate(pred_batches) if pred_batches else np.array([], dtype=np.int64)
        targets_np = target_t.numpy().reshape(-1)
        metrics = _compute_multiclass_metrics(preds_np, targets_np)
        metrics["loss"] = float(loss_sum / max(batches, 1))
        return metrics, targets_np, preds_np

    def _train_mlp_with_validate(
        train_feature_t,
        train_target_t,
        val_feature_t,
        val_target_t,
        *,
        epochs_to_run,
        allow_early_stopping,
    ):
        reset_mlp(model)
        optimizer = torch.optim.Adam(model.mlp.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        history = []
        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = None
        patience_counter = 0
        early_stopped = False
        early_stop_reason = None

        for epoch in range(max(1, int(epochs_to_run))):
            model.mlp.train()
            permutation = torch.randperm(train_target_t.size(0))
            epoch_loss = 0.0
            batches = 0
            train_preds_batches = []
            train_targets_batches = []

            for i in range(0, train_target_t.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_x = train_feature_t[indices].to(device)
                batch_y = train_target_t[indices].to(device)

                optimizer.zero_grad()
                logits = model.mlp(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                batch_preds = torch.argmax(logits.detach(), dim=1).cpu().numpy().reshape(-1)
                batch_targets = batch_y.detach().cpu().numpy().reshape(-1)
                train_preds_batches.append(batch_preds)
                train_targets_batches.append(batch_targets)

                epoch_loss += loss.item()
                batches += 1

            train_preds_np = (
                np.concatenate(train_preds_batches)
                if train_preds_batches
                else np.array([], dtype=np.int64)
            )
            train_targets_np = (
                np.concatenate(train_targets_batches)
                if train_targets_batches
                else np.array([], dtype=np.int64)
            )
            train_metrics = _compute_multiclass_metrics(train_preds_np, train_targets_np)

            holdout_metrics, _, _ = _evaluate_mlp_model(
                val_feature_t,
                val_target_t,
                criterion,
                batch_size,
            )

            current_val_loss = float(holdout_metrics.get("loss", 0.0))
            current_val_f1 = float(holdout_metrics.get("f1", 0.0))

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(epoch_loss / max(batches, 1)),
                    "val_loss": current_val_loss,
                    "accuracy": float(holdout_metrics.get("accuracy", 0.0)),
                    "f1": current_val_f1,
                    "precision": float(holdout_metrics.get("precision", 0.0)),
                    "recall": float(holdout_metrics.get("recall", 0.0)),
                    "train_accuracy": float(train_metrics.get("accuracy", 0.0)),
                    "train_f1": float(train_metrics.get("f1", 0.0)),
                    "train_precision": float(train_metrics.get("precision", 0.0)),
                    "train_recall": float(train_metrics.get("recall", 0.0)),
                    "val_accuracy": float(holdout_metrics.get("accuracy", 0.0)),
                    "val_f1": current_val_f1,
                    "val_precision": float(holdout_metrics.get("precision", 0.0)),
                    "val_recall": float(holdout_metrics.get("recall", 0.0)),
                }
            )

            if current_val_f1 >= (target_f1 - 1e-12):
                best_val_loss = min(best_val_loss, current_val_loss)
                best_epoch = epoch + 1
                best_model_state = {k: v.cpu().clone() for k, v in model.mlp.state_dict().items()}
                early_stopped = True
                early_stop_reason = f"target_f1_reached_{target_f1:.6f}"
                break

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                best_epoch = epoch + 1
                best_model_state = {k: v.cpu().clone() for k, v in model.mlp.state_dict().items()}
            else:
                patience_counter += 1

            if allow_early_stopping and patience_counter >= patience:
                early_stopped = True
                early_stop_reason = "patience_exhausted"
                break

        if best_model_state is not None:
            model.mlp.load_state_dict(best_model_state)

        final_holdout_metrics, holdout_targets_np, holdout_preds_np = _evaluate_mlp_model(
            val_feature_t,
            val_target_t,
            criterion,
            batch_size,
        )

        return {
            "history": history,
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else None,
            "early_stopped": bool(early_stopped),
            "early_stop_reason": early_stop_reason,
            "holdout_metrics": final_holdout_metrics,
            "holdout_targets": holdout_targets_np,
            "holdout_preds": holdout_preds_np,
        }

    while True:
        try:
            criterion = nn.CrossEntropyLoss()

            log_file = None
            if project_root:
                log_file = project_root / "state" / "live_training.json"
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "w") as f:
                    json.dump(
                        {
                            "model_name": "Ensemble Model",
                            "history": [],
                            "validation_source": "validate_split",
                            "feature_source": "out_of_fold_train_features",
                        },
                        f,
                    )

            y_tr_np = y_tr_t.numpy().astype(np.int64).reshape(-1)
            if y_tr_np.size == 0:
                raise RuntimeError("Ensemble training received an empty training label array")

            if y_vl_t.numel() == 0:
                raise RuntimeError("Ensemble validation split is empty; cannot tune ensemble hyperparameters")

            class_counts = np.bincount(y_tr_np)
            nonzero_counts = [int(v) for v in class_counts.tolist() if int(v) > 0]
            min_class_count = min(nonzero_counts) if nonzero_counts else 0

            cv_folds = min(requested_cv_folds, min_class_count)
            cv_fallback_reason = None
            if cv_folds < 2:
                cv_folds = 1
                cv_fallback_reason = "insufficient_samples_per_class_for_3fold"

            train_count = int(y_tr_t.size(0))
            oof_features = np.zeros((train_count, expected_concat_dim), dtype=np.float32)
            oof_covered = np.zeros(train_count, dtype=np.bool_)
            oof_fold_details = []

            if cv_folds >= 2:
                splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_random_seed)
                for fold_idx, (train_idx, holdout_idx) in enumerate(
                    splitter.split(np.zeros_like(y_tr_np), y_tr_np),
                    start=1,
                ):
                    holdout_idx_arr = np.asarray(holdout_idx, dtype=np.int64)
                    holdout_idx_t = torch.tensor(holdout_idx_arr, dtype=torch.long)
                    holdout_features = _extract_concat_expert_outputs(x_tr_t_list, holdout_idx_t)

                    if holdout_features.shape[1] != oof_features.shape[1]:
                        oof_features = np.zeros((train_count, holdout_features.shape[1]), dtype=np.float32)

                    oof_features[holdout_idx_arr] = holdout_features
                    oof_covered[holdout_idx_arr] = True
                    oof_fold_details.append(
                        {
                            "fold": int(fold_idx),
                            "train_samples": int(len(train_idx)),
                            "holdout_samples": int(len(holdout_idx_arr)),
                        }
                    )
            else:
                full_idx_t = torch.arange(y_tr_t.size(0), dtype=torch.long)
                all_features = _extract_concat_expert_outputs(x_tr_t_list, full_idx_t)
                oof_features = all_features
                oof_covered[:] = True
                oof_fold_details.append(
                    {
                        "fold": 1,
                        "train_samples": int(full_idx_t.numel()),
                        "holdout_samples": int(full_idx_t.numel()),
                    }
                )

            oof_fill_count = 0
            if not bool(np.all(oof_covered)):
                missing_idx = np.where(~oof_covered)[0].astype(np.int64)
                missing_idx_t = torch.tensor(missing_idx, dtype=torch.long)
                fill_features = _extract_concat_expert_outputs(x_tr_t_list, missing_idx_t)
                oof_features[missing_idx] = fill_features
                oof_covered[missing_idx] = True
                oof_fill_count = int(missing_idx.size)

            val_all_idx_t = torch.arange(y_vl_t.size(0), dtype=torch.long)
            val_features = _extract_concat_expert_outputs(x_vl_t_list, val_all_idx_t)

            train_feature_t = torch.tensor(oof_features, dtype=torch.float32)
            val_feature_t = torch.tensor(val_features, dtype=torch.float32)

            final_run = _train_mlp_with_validate(
                train_feature_t=train_feature_t,
                train_target_t=y_tr_t,
                val_feature_t=val_feature_t,
                val_target_t=y_vl_t,
                epochs_to_run=epochs,
                allow_early_stopping=True,
            )

            live_log = final_run.get("history", []) if isinstance(final_run.get("history"), list) else []
            selected_epoch = final_run.get("best_epoch")
            if selected_epoch is None:
                selected_epoch = len(live_log) if live_log else int(epochs)
            selected_epoch = max(1, int(selected_epoch))

            eval_metrics = final_run.get("holdout_metrics") or {}
            y_vl_np = np.asarray(final_run.get("holdout_targets", np.array([], dtype=np.int64))).reshape(-1)
            preds = np.asarray(final_run.get("holdout_preds", np.array([], dtype=np.int64))).reshape(-1)

            if y_vl_np.size <= 0 or preds.size <= 0:
                eval_metrics, y_vl_np, preds = _evaluate_mlp_model(
                    val_feature_t,
                    y_vl_t,
                    criterion,
                    batch_size,
                )

            report = classification_report(y_vl_np, preds, output_dict=True, zero_division=0)
            conf_matrix = confusion_matrix(y_vl_np, preds).tolist()
            kappa_raw = cohen_kappa_score(y_vl_np, preds) if np.unique(y_vl_np).size > 1 else 0.0
            kappa = float(kappa_raw) if np.isfinite(kappa_raw) else 0.0
            metrics = {
                "accuracy": float(eval_metrics.get("accuracy", 0.0)),
                "kappa": kappa,
                "classification_report": report,
                "confusion_matrix": conf_matrix,
            }

            fold_metric_names = ["accuracy", "f1", "precision", "recall", "loss"]
            cv_metrics = {
                "fold_count": int(cv_folds),
                "target_fold_count": int(requested_cv_folds),
                "fallback_reason": cv_fallback_reason,
                "oof_samples_total": int(train_count),
                "oof_samples_covered": int(np.sum(oof_covered)),
                "oof_fill_count": int(oof_fill_count),
            }
            for metric_name in fold_metric_names:
                metric_val = float(eval_metrics.get(metric_name, 0.0))
                cv_metrics[f"{metric_name}_mean"] = metric_val
                cv_metrics[f"{metric_name}_std"] = 0.0

            cv_fold_details = oof_fold_details

            cv_early_stopped_folds = 1 if bool(final_run.get("early_stopped", False)) else 0
            cv_perfect_f1_stopped_folds = 1 if str(final_run.get("early_stop_reason") or "").startswith("target_f1_reached_") else 0

            curve_summary = _curve_summary_from_history(
                live_log,
                best_epoch=selected_epoch,
                early_stopped=cv_early_stopped_folds > 0,
            )
            curve_summary["early_stopping_basis"] = "validation_split"
            curve_summary["feature_source"] = "out_of_fold_train_features"
            curve_summary["cv_metrics"] = cv_metrics
            curve_summary["cv_fold_details"] = cv_fold_details
            curve_summary["cv_best_epochs"] = [int(selected_epoch)]
            curve_summary["cv_selected_epoch"] = int(selected_epoch)
            curve_summary["cv_early_stopped_folds"] = int(cv_early_stopped_folds)
            curve_summary["cv_perfect_f1_stopped_folds"] = int(cv_perfect_f1_stopped_folds)
            curve_summary["stop_on_perfect_f1"] = float(target_f1)
            curve_summary["final_train_early_stopped"] = bool(final_run.get("early_stopped", False))
            if final_run.get("early_stop_reason"):
                curve_summary["final_train_early_stop_reason"] = str(final_run.get("early_stop_reason"))

            if log_file:
                with open(log_file, "w") as f:
                    json.dump(
                        {
                            "model_name": "Ensemble Model",
                            "history": live_log,
                            "validation_source": "validate_split",
                            "feature_source": "out_of_fold_train_features",
                        },
                        f,
                    )

            return metrics, model, live_log, curve_summary

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size = batch_size // 2
            if batch_size == 0:
                raise RuntimeError("CUDA out of memory even with batch size 1 in train_ensemble")
            print(f"CUDA OutOfMemoryError in train_ensemble. Reducing batch size to {batch_size}")

def main():
    project_root_str = os.environ.get("ARL_PROJECT_ROOT")
    if not project_root_str:
        raise ValueError("ARL_PROJECT_ROOT not set")
    project_root = Path(project_root_str)

    config_path = project_root / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ensemble_architecture = normalize_ensemble_architecture(config.get("ensemble_architecture", "default"))

    dataset_path = Path(config["dataset_path"])
    expert_matrix_file = project_root / "artifacts" / "expert_matrix.json"

    if expert_matrix_file.exists():
        with open(expert_matrix_file, "r") as f:
            expert_matrix = json.load(f)
    else:
        expert_matrix = {}

    cycle_history_dir = project_root / "artifacts" / "cycle_history" / "cycle_0000"
    cycle_history_dir.mkdir(parents=True, exist_ok=True)

    training_curves_payload = {
        "schema_version": "1.0",
        "cycle_id": "0",
        "project_id": config.get("project_name", project_root.name),
        "experts": [],
        "ensemble": None,
    }

    models_root_dir = project_root / "models"
    models_root_dir.mkdir(parents=True, exist_ok=True)

    train_dir = dataset_path / "train"
    val_dir = dataset_path / "validate"

    if not (train_dir / "y.npy").exists():
        print("No training labels found. Skipping local baseline.")
        return

    y_train = np.load(train_dir / "y.npy")
    y_val = np.load(val_dir / "y.npy") if (val_dir / "y.npy").exists() else y_train

    train_prop = float(config.get("train_proportion", 1.0))
    train_indices = None
    if train_prop < 1.0:
        subset_len = max(1, int(len(y_train) * train_prop))
        train_indices = np.random.permutation(len(y_train))[:subset_len]
        y_train = y_train[train_indices]

    classes = np.unique(y_train)

    signals = []
    for x_file in sorted(train_dir.glob("X_*.npy"), key=lambda p: p.name):
        sig = x_file.stem[len("X_"):]
        signals.append((sig, x_file.name))

    for sig, x_filename in signals:
        if sig not in expert_matrix:
            expert_matrix[sig] = {}

        x_train, x_val = load_and_preprocess_signal_split(
            train_dir=train_dir,
            val_dir=val_dir,
            x_filename=x_filename,
            train_indices=train_indices,
        )


        modality_name = sig

        for cls in classes:
            cls_str = str(cls)
            if cls_str in expert_matrix[sig]:
                continue

            print(f"Training Cycle 0 Baseline for {sig} vs Class {cls_str}")
            f1, acc, prec, rec, model, training_history, training_summary = train_baseline(
                x_train, y_train, x_val, y_val, cls, project_root=project_root, model_name=f"Baseline {sig} vs Class {cls_str}"
            )

            candidate_id = f"baseline_{sig}_{cls_str}"
            if model is not None:
                model_dir = models_root_dir / f"{sig}_{cls_str}"
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / f"{candidate_id}.pt"
                torch.save(model.state_dict(), model_path)

            cv_metrics = (
                training_summary.get("cv_metrics")
                if isinstance(training_summary, dict) and isinstance(training_summary.get("cv_metrics"), dict)
                else {}
            )
            validation_metrics = (
                training_summary.get("validation_metrics")
                if isinstance(training_summary, dict) and isinstance(training_summary.get("validation_metrics"), dict)
                else {}
            )
            if not validation_metrics:
                validation_metrics = {
                    "accuracy": float(acc),
                    "f1": float(f1),
                    "precision": float(prec),
                    "recall": float(rec),
                    "split": "validate",
                }

            expert_matrix[sig][cls_str] = {
                "modality": modality_name,
                "class_label": cls_str,
                "candidate_id": candidate_id,
                "f1": float(f1),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "is_best": True,
                "preprocessing_code_ref": "local/scripts/cycle_preprocessing.py",
                "final_model_py_ref": "local/scripts/binary_expert_model.py",
                "cv_metrics": cv_metrics,
                "validation_metrics": validation_metrics,
                "cv_accuracy_mean": float(cv_metrics.get("accuracy_mean", 0.0)),
                "cv_f1_mean": float(cv_metrics.get("f1_mean", 0.0)),
                "cv_precision_mean": float(cv_metrics.get("precision_mean", 0.0)),
                "cv_recall_mean": float(cv_metrics.get("recall_mean", 0.0)),
                "validation_accuracy": float(validation_metrics.get("accuracy", acc)),
                "validation_f1": float(validation_metrics.get("f1", f1)),
                "validation_precision": float(validation_metrics.get("precision", prec)),
                "validation_recall": float(validation_metrics.get("recall", rec)),
            }

            training_curves_payload["experts"].append(
                {
                    "modality": modality_name,
                    "class_label": cls_str,
                    "candidate_id": candidate_id,
                    "history": training_history,
                    "summary": training_summary,
                }
            )

            with open(expert_matrix_file, "w") as f:
                json.dump(expert_matrix, f, indent=2)


            with open(cycle_history_dir / "expert_matrix.json", "w") as f:
                json.dump(expert_matrix, f, indent=2)


    print("Training Cycle 0 Ensemble Model...")


    x_train_all = []
    x_val_all = []


    signal_indices = {}
    for i, (sig, x_filename) in enumerate(signals):
        signal_indices[sig] = i
        x_train_sig, x_val_sig = load_and_preprocess_signal_split(
            train_dir=train_dir,
            val_dir=val_dir,
            x_filename=x_filename,
            train_indices=train_indices,
        )

        x_train_all.append(to_model_input(x_train_sig, signal_name=f"{sig}:train"))
        x_val_all.append(to_model_input(x_val_sig, signal_name=f"{sig}:val"))


    class_map = {c: i for i, c in enumerate(classes)}
    y_train_mapped = np.array([class_map[c] for c in y_train])
    y_val_mapped = np.array([class_map[c] for c in y_val])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experts_list = []
    input_map = []

    for sig, x_filename in signals:
        idx = signal_indices[sig]
        in_ch_sig = x_train_all[idx].shape[1]
        T_sig = x_train_all[idx].shape[-1]

        for cls in classes:
            cls_str = str(cls)
            candidate_id = f"baseline_{sig}_{cls_str}"
            model_path = models_root_dir / f"{sig}_{cls_str}" / f"{candidate_id}.pt"

            if not model_path.exists():
                print(f"Warning: Expert piece {model_path} missing.")
                continue

            model = BinaryExpertModel(
                in_ch=in_ch_sig, n_classes=1, fs=100.0, min_seq_len=T_sig,
                dts=(0.05, 0.15, 0.5, 1.5), k_min=7, k_max_cap=129,
                width=16, depth=2, dropout=0.1
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            experts_list.append(model)
            input_map.append(signal_indices[sig])

    if experts_list:
        ensemble_model = BaselineEnsemble(
            experts_list,
            input_map,
            num_classes=len(classes),
            expert_dim=16,
            architecture=ensemble_architecture,
        )
        ensemble_metrics, ensemble_model_trained, ensemble_history, ensemble_summary = train_ensemble(

            ensemble_model,
            x_train_all,
            y_train_mapped,
            x_val_all,
            y_val_mapped,
            project_root=project_root,
        )
        print(f"Ensemble Accuracy: {ensemble_metrics['accuracy']:.4f}")

        ensemble_metrics["training_history"] = ensemble_history
        ensemble_metrics["training_summary"] = ensemble_summary
        ensemble_metrics["ensemble_architecture"] = ensemble_architecture
        ensemble_metrics["class_labels"] = [str(c) for c in classes]
        ensemble_metrics["evaluation_split"] = "validate"
        ensemble_metrics["training_monitor_split"] = "validate"
        ensemble_metrics["training_fit_split"] = "train_oof"

        training_curves_payload["ensemble"] = {
            "candidate_id": "baseline_ensemble",
            "history": ensemble_history,
            "summary": ensemble_summary,
        }

        ensemble_model_path = models_root_dir / "baseline_ensemble.pt"
        torch.save(ensemble_model_trained.state_dict(), ensemble_model_path)

        with open(project_root / "artifacts" / "baseline_ensemble_metrics.json", "w") as f:
            json.dump(ensemble_metrics, f, indent=2)

    training_curves_path = project_root / "artifacts" / "training_curves_cycle0.json"
    with open(training_curves_path, "w") as f:
        json.dump(training_curves_payload, f, indent=2)

    with open(cycle_history_dir / "training_curves.json", "w") as f:
        json.dump(training_curves_payload, f, indent=2)


    live_log_path = project_root / "state" / "live_training.json"
    if live_log_path.exists():
        live_log_path.unlink()

if __name__ == "__main__":
    main()
