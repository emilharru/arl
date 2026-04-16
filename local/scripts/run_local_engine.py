#!/usr/bin/env python3
"""Cycle >0 local engine execution.

This runner consumes the latest directive and director-generated model/preprocessing
files, resolves one target modality into one or more concrete dimensions, trains
binary experts per dimension, updates expert_matrix.json, and writes structured
results to shared/outbound.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from run_cycle import train_baseline, to_model_input
from binary_expert_model import BinaryExpertModel as BaselineBinaryExpertModel
from cycle_preprocessing import apply_preprocessing as baseline_apply_preprocessing
from run_cycle import BaselineEnsemble, _build_binary_expert_model, normalize_ensemble_architecture, train_ensemble


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def to_project_ref(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root)).replace(os.sep, "/")
    except ValueError:
        return str(path)


def read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def sorted_key(value: Any):
    text = str(value)
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def normalize_class_label(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return text
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return text


def sanitize_token(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned or "unknown"


UNKNOWN_MODALITY_TOKENS = {
    "",
    "unspecified",
    "unknown",
    "n/a",
    "na",
    "none",
    "null",
}


def normalize_modality_name(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in UNKNOWN_MODALITY_TOKENS:
        return ""
    return text


def parse_cycle_folder_number(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None

    match = re.match(r"^cycle_(\d+)$", text, flags=re.IGNORECASE)
    if not match:
        return None

    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def cycle_payload_has_binary_improvement(results_payload: Dict[str, Any]) -> bool:
    if not isinstance(results_payload, dict):
        return False

    execution_summary = (
        results_payload.get("execution_summary")
        if isinstance(results_payload.get("execution_summary"), dict)
        else {}
    )
    if bool(execution_summary.get("binary_expert_improved", False)):
        return True

    updates = results_payload.get("expert_matrix_updates")
    if not isinstance(updates, list):
        return False

    for row in updates:
        if not isinstance(row, dict):
            continue
        if bool(row.get("improved", False)):
            return True

        if "improved" not in row and bool(row.get("is_best", False)):
            return True

    return False


def cycle_payload_has_successful_ensemble_eval(results_payload: Dict[str, Any]) -> bool:
    if not isinstance(results_payload, dict):
        return False

    ensemble_eval = (
        results_payload.get("ensemble_evaluation")
        if isinstance(results_payload.get("ensemble_evaluation"), dict)
        else {}
    )
    status = str(ensemble_eval.get("status", "")).strip().lower()
    ran = bool(ensemble_eval.get("ran", False))
    return ran and status == "success"


def build_ensemble_retrain_policy(
    *,
    project_root: Path,
    current_cycle: int,
    current_cycle_improved: bool,
    required_successful_improvement_cycles: int = 5,
) -> Dict[str, Any]:


    required = 1

    cycle_history_root = project_root / "artifacts" / "cycle_history"
    improved_cycles: List[int] = []
    successful_ensemble_cycles: List[int] = []

    if cycle_history_root.exists() and cycle_history_root.is_dir():
        for entry in cycle_history_root.iterdir():
            if not entry.is_dir():
                continue

            cycle_num = parse_cycle_folder_number(entry.name)
            if cycle_num is None or cycle_num >= int(current_cycle):
                continue

            results_path = entry / "results.json"
            payload = read_json_file(results_path, None)
            if not isinstance(payload, dict):
                continue

            if cycle_payload_has_binary_improvement(payload):
                improved_cycles.append(int(cycle_num))
            if cycle_payload_has_successful_ensemble_eval(payload):
                successful_ensemble_cycles.append(int(cycle_num))

    improved_cycles = sorted(set(improved_cycles))
    successful_ensemble_cycles = sorted(set(successful_ensemble_cycles))
    last_successful_ensemble_cycle = max(successful_ensemble_cycles) if successful_ensemble_cycles else None

    if last_successful_ensemble_cycle is None:
        improvements_since_last = list(improved_cycles)
    else:
        improvements_since_last = [
            cycle for cycle in improved_cycles
            if cycle > int(last_successful_ensemble_cycle)
        ]

    prior_count = len(improvements_since_last)
    including_current = prior_count + (1 if current_cycle_improved else 0)
    should_run = True
    remaining = 0
    reason = "run_every_cycle"

    return {
        "policy_name": "ensemble_every_cycle",
        "required_successful_improvement_cycles": int(required),
        "last_successful_ensemble_cycle": int(last_successful_ensemble_cycle)
        if last_successful_ensemble_cycle is not None
        else None,
        "successful_improvement_cycles_since_last_ensemble": [int(v) for v in improvements_since_last],
        "successful_improvement_count_since_last_ensemble": int(prior_count),
        "successful_improvement_count_including_current": int(including_current),
        "current_cycle_improved": bool(current_cycle_improved),
        "should_run_ensemble_this_cycle": bool(should_run),
        "remaining_improvements_until_next_ensemble": int(remaining),
        "decision_reason": reason,
    }


def evaluate_ensemble_on_split(
    *,
    model: torch.nn.Module,
    x_eval_list: List[np.ndarray],
    y_eval: np.ndarray,
    batch_size: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    y_eval_arr = np.asarray(y_eval).reshape(-1)
    if y_eval_arr.size == 0:
        metrics_empty = {
            "accuracy": 0.0,
            "kappa": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "classification_report": {},
            "confusion_matrix": [],
        }
        return metrics_empty, y_eval_arr, np.asarray([], dtype=np.int64)

    x_eval_tensors = [torch.tensor(x, dtype=torch.float32) for x in x_eval_list]
    if not x_eval_tensors:
        raise ValueError("No ensemble input blocks were provided for evaluation.")

    for idx, x_tensor in enumerate(x_eval_tensors):
        if int(x_tensor.shape[0]) != int(y_eval_arr.size):
            raise ValueError(
                f"Ensemble eval sample mismatch for block {idx}: X={x_tensor.shape[0]} Y={y_eval_arr.size}"
            )

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    if hasattr(model, "experts"):
        model.experts.eval()

    preds_batches = []
    eval_batch_size = max(1, int(batch_size))

    with torch.no_grad():
        for i in range(0, y_eval_arr.size, eval_batch_size):
            batch_x_list = [x[i:i+eval_batch_size].to(device) for x in x_eval_tensors]
            logits = model(batch_x_list)
            preds_batches.append(torch.argmax(logits, dim=1).cpu().numpy().reshape(-1))

    preds = np.concatenate(preds_batches) if preds_batches else np.array([], dtype=np.int64)
    if preds.size != y_eval_arr.size:
        raise ValueError(
            f"Ensemble eval prediction count mismatch: preds={preds.size} y={y_eval_arr.size}"
        )

    accuracy = float(np.mean(preds == y_eval_arr)) if y_eval_arr.size > 0 else 0.0
    precision_macro = float(precision_score(y_eval_arr, preds, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_eval_arr, preds, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_eval_arr, preds, average="macro", zero_division=0))
    kappa_raw = cohen_kappa_score(y_eval_arr, preds) if y_eval_arr.size > 1 else 0.0
    kappa = float(kappa_raw) if np.isfinite(kappa_raw) else 0.0
    report = classification_report(y_eval_arr, preds, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_eval_arr, preds).tolist()

    metrics = {
        "accuracy": accuracy,
        "kappa": kappa,
        "precision": precision_macro,
        "recall": recall_macro,
        "f1": f1_macro,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
    }
    return metrics, y_eval_arr, preds


def _to_optional_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return float(number)


def _build_stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y_arr = np.asarray(y_true).reshape(-1)
    if y_arr.size <= 0:
        return np.asarray([], dtype=np.int64)

    class_to_indices: Dict[Any, np.ndarray] = {}
    for idx, value in enumerate(y_arr):
        class_to_indices.setdefault(value, []).append(idx)

    sampled_indices: List[int] = []
    for value in sorted(class_to_indices.keys(), key=sorted_key):
        pool = np.asarray(class_to_indices[value], dtype=np.int64)
        if pool.size <= 0:
            continue
        sampled = rng.choice(pool, size=pool.size, replace=True)
        sampled_indices.extend(int(v) for v in sampled.tolist())

    out = np.asarray(sampled_indices, dtype=np.int64)
    if out.size > 1:
        rng.shuffle(out)
    return out


def _summarize_metric_samples(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def build_bootstrap_validation_summary(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_pred_arr = np.asarray(y_pred).reshape(-1)

    repeats = max(1, int(repeats))
    if y_true_arr.size <= 0 or y_pred_arr.size != y_true_arr.size:
        return {
            "method": "stratified_bootstrap_validation",
            "repeats": int(repeats),
            "seed": int(seed),
            "metrics": {
                "accuracy": _summarize_metric_samples([]),
                "kappa": _summarize_metric_samples([]),
                "precision": _summarize_metric_samples([]),
                "recall": _summarize_metric_samples([]),
                "f1": _summarize_metric_samples([]),
            },
            "notes": "Bootstrap skipped due empty/mismatched predictions.",
        }

    rng = np.random.default_rng(int(seed))
    metric_samples: Dict[str, List[float]] = {
        "accuracy": [],
        "kappa": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    for _ in range(repeats):
        idx = _build_stratified_bootstrap_indices(y_true_arr, rng)
        if idx.size <= 0:
            continue

        y_b = y_true_arr[idx]
        p_b = y_pred_arr[idx]

        metric_samples["accuracy"].append(float(np.mean(p_b == y_b)))
        metric_samples["precision"].append(
            float(precision_score(y_b, p_b, average="macro", zero_division=0))
        )
        metric_samples["recall"].append(
            float(recall_score(y_b, p_b, average="macro", zero_division=0))
        )
        metric_samples["f1"].append(
            float(f1_score(y_b, p_b, average="macro", zero_division=0))
        )

        kappa_raw = cohen_kappa_score(y_b, p_b) if np.unique(y_b).size > 1 else 0.0
        kappa_val = float(kappa_raw) if np.isfinite(kappa_raw) else 0.0
        metric_samples["kappa"].append(kappa_val)

    return {
        "method": "stratified_bootstrap_validation",
        "repeats": int(repeats),
        "seed": int(seed),
        "metrics": {
            name: _summarize_metric_samples(vals)
            for name, vals in metric_samples.items()
        },
        "notes": "Scores are computed on stratified bootstrap resamples of validation predictions.",
    }


def resolve_previous_ensemble_acceptance_score(project_root: Path) -> Dict[str, Any]:
    metrics_path = project_root / "artifacts" / "baseline_ensemble_metrics.json"
    payload = read_json_file(metrics_path, None)
    if not isinstance(payload, dict):
        return {
            "score": None,
            "source": "none",
        }

    acceptance_payload = payload.get("validation_acceptance")
    if isinstance(acceptance_payload, dict):
        bootstrap_payload = acceptance_payload.get("bootstrap_summary")
        if isinstance(bootstrap_payload, dict):
            metrics = bootstrap_payload.get("metrics")
            if isinstance(metrics, dict):
                kappa_metric = metrics.get("kappa")
                if isinstance(kappa_metric, dict):
                    score = _to_optional_float(kappa_metric.get("mean"))
                    if score is not None:
                        return {
                            "score": score,
                            "source": "validation_acceptance.bootstrap.kappa_mean",
                        }

        score = _to_optional_float(acceptance_payload.get("selected_score"))
        if score is not None:
            return {
                "score": score,
                "source": "validation_acceptance.selected_score",
            }

    bootstrap_summary = payload.get("bootstrap_validation_summary")
    if isinstance(bootstrap_summary, dict):
        metrics = bootstrap_summary.get("metrics")
        if isinstance(metrics, dict):
            kappa_metric = metrics.get("kappa")
            if isinstance(kappa_metric, dict):
                score = _to_optional_float(kappa_metric.get("mean"))
                if score is not None:
                    return {
                        "score": score,
                        "source": "bootstrap_validation_summary.kappa_mean",
                    }

    score = _to_optional_float(payload.get("kappa"))
    return {
        "score": score,
        "source": "kappa" if score is not None else "none",
    }


def snapshot_candidate_code(
    project_root: Path,
    cycle_id: str,
    candidate_id: str,
    runtime_model_path: Path,
    runtime_preprocessing_path: Path,
    model_meta_path: Optional[Path] = None,
) -> Dict[str, str]:
    snapshot_dir = (
        project_root
        / "artifacts"
        / "candidate_snapshots"
        / f"cycle_{safe_int(cycle_id, 0):04d}"
        / sanitize_token(candidate_id)
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    snapshot_model_path = snapshot_dir / "model.py"
    snapshot_preprocessing_path = snapshot_dir / "preprocessing.py"

    runtime_model_resolved = runtime_model_path.resolve()
    snapshot_model_resolved = snapshot_model_path.resolve()
    if runtime_model_resolved != snapshot_model_resolved:
        shutil.copyfile(runtime_model_path, snapshot_model_path)
    elif not snapshot_model_path.exists():
        raise FileNotFoundError(f"Snapshot model path missing: {snapshot_model_path}")

    runtime_pre_resolved = runtime_preprocessing_path.resolve()
    snapshot_pre_resolved = snapshot_preprocessing_path.resolve()
    if runtime_pre_resolved != snapshot_pre_resolved:
        shutil.copyfile(runtime_preprocessing_path, snapshot_preprocessing_path)
    elif not snapshot_preprocessing_path.exists():
        raise FileNotFoundError(f"Snapshot preprocessing path missing: {snapshot_preprocessing_path}")

    refs: Dict[str, str] = {
        "model_py_ref": to_project_ref(snapshot_model_path, project_root),
        "preprocessing_py_ref": to_project_ref(snapshot_preprocessing_path, project_root),
    }

    if model_meta_path is not None and model_meta_path.exists():
        snapshot_model_meta_path = snapshot_dir / "model.meta.json"
        if model_meta_path.resolve() != snapshot_model_meta_path.resolve():
            shutil.copyfile(model_meta_path, snapshot_model_meta_path)
        elif not snapshot_model_meta_path.exists():
            raise FileNotFoundError(f"Snapshot model meta path missing: {snapshot_model_meta_path}")
        refs["model_meta_ref"] = to_project_ref(snapshot_model_meta_path, project_root)

    return refs


def load_python_module(module_path: Path, module_name: str):
    if not module_path.exists():
        raise FileNotFoundError(f"Missing module: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_code_ref_path(ref: Any, project_root: Path, repo_root: Path) -> Optional[Path]:
    if ref is None:
        return None

    text = str(ref).strip()
    if not text:
        return None

    candidate = Path(text)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    project_candidate = project_root / candidate
    if project_candidate.exists():
        return project_candidate

    repo_candidate = repo_root / candidate
    if repo_candidate.exists():
        return repo_candidate

    return None


def load_model_class_from_ref(
    model_ref: Any,
    project_root: Path,
    repo_root: Path,
    cache: Dict[str, Any],
):
    path = resolve_code_ref_path(model_ref, project_root=project_root, repo_root=repo_root)
    if path is None:
        return BaselineBinaryExpertModel

    key = str(path)
    module = cache.get(key)
    if module is None:
        module = load_python_module(path, f"arl_model_ref_{abs(hash(key))}")
        cache[key] = module

    cls = getattr(module, "BinaryExpertModel", None)
    if cls is None:
        raise AttributeError(f"Model module missing BinaryExpertModel: {path}")
    return cls


def load_preprocessing_callable_from_ref(
    preprocessing_ref: Any,
    project_root: Path,
    repo_root: Path,
    cache: Dict[str, Any],
):
    path = resolve_code_ref_path(preprocessing_ref, project_root=project_root, repo_root=repo_root)
    if path is None:
        return baseline_apply_preprocessing

    key = str(path)
    module = cache.get(key)
    if module is None:
        module = load_python_module(path, f"arl_pre_ref_{abs(hash(key))}")
        cache[key] = module

    fn = getattr(module, "apply_preprocessing", None)
    if fn is None:
        raise AttributeError(f"Preprocessing module missing apply_preprocessing: {path}")
    return fn


def infer_expert_feature_dim(model: Any, x_input: np.ndarray, default: int = 16) -> int:
    attr_dim = getattr(model, "embedding_dim", None)
    if isinstance(attr_dim, (int, float)) and int(attr_dim) > 0:
        return int(attr_dim)

    try:
        with torch.no_grad():
            sample = torch.tensor(x_input[:1], dtype=torch.float32)
            if sample.dim() == 2:
                sample = sample.unsqueeze(1)
            elif sample.dim() > 3:
                sample = sample.reshape(sample.size(0), -1, sample.size(-1))

            features = model.extract_features(sample)
            if features.dim() == 1:
                return 1
            if features.dim() >= 2 and int(features.shape[1]) > 0:
                return int(features.shape[1])
    except Exception:
        pass

    return max(1, int(default))


def build_secondary_curves_from_history(history):
    train_loss_curve = []
    val_loss_curve = []
    train_metric_curves = {
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }
    val_metric_curves = {
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }

    if not isinstance(history, list):
        history = []

    for row in history:
        if not isinstance(row, dict):
            continue

        train_loss_curve.append(safe_float(row.get("train_loss", 0.0)))
        val_loss_curve.append(safe_float(row.get("val_loss", 0.0)))

        train_metric_curves["accuracy"].append(
            safe_float(row.get("train_accuracy", row.get("accuracy", 0.0)))
        )
        train_metric_curves["f1"].append(
            safe_float(row.get("train_f1", row.get("f1", 0.0)))
        )
        train_metric_curves["precision"].append(
            safe_float(row.get("train_precision", row.get("precision", 0.0)))
        )
        train_metric_curves["recall"].append(
            safe_float(row.get("train_recall", row.get("recall", 0.0)))
        )

        val_metric_curves["accuracy"].append(
            safe_float(row.get("val_accuracy", row.get("accuracy", 0.0)))
        )
        val_metric_curves["f1"].append(
            safe_float(row.get("val_f1", row.get("f1", 0.0)))
        )
        val_metric_curves["precision"].append(
            safe_float(row.get("val_precision", row.get("precision", 0.0)))
        )
        val_metric_curves["recall"].append(
            safe_float(row.get("val_recall", row.get("recall", 0.0)))
        )

    return {
        "train_loss_curve": train_loss_curve,
        "val_loss_curve": val_loss_curve,
        "train_metric_curves": train_metric_curves,
        "val_metric_curves": val_metric_curves,
    }


def summarize_history_for_results(history, raw_summary=None):
    summary = raw_summary if isinstance(raw_summary, dict) else {}
    if not isinstance(history, list):
        history = []

    epochs_completed = safe_int(summary.get("epochs_completed", len(history)), len(history))
    best_epoch_raw = summary.get("best_epoch")
    best_epoch = safe_int(best_epoch_raw, 0) if best_epoch_raw is not None else None
    early_stopped = bool(summary.get("early_stopped", False))

    trend = summary.get("trend")
    if trend is None and len(history) >= 2:
        first_val = safe_float(history[0].get("val_loss", 0.0)) if isinstance(history[0], dict) else 0.0
        last_val = safe_float(history[-1].get("val_loss", 0.0)) if isinstance(history[-1], dict) else 0.0
        trend = "improving" if last_val <= first_val else "degrading"

    return {
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "early_stopped": early_stopped,
        "trend": trend,
    }


def coerce_target_class_value(class_label: str, y_values: np.ndarray):
    text = str(class_label).strip()
    arr = np.asarray(y_values)

    candidates = []
    try:
        as_float = float(text)
        candidates.append(as_float)
        if as_float.is_integer():
            candidates.insert(0, int(as_float))
    except ValueError:
        pass
    candidates.append(text)

    for candidate in candidates:
        try:
            if np.any(arr == candidate):
                return candidate
        except Exception:
            continue

    if arr.dtype.kind in ("i", "u"):
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return text

    if arr.dtype.kind == "f":
        try:
            return float(text)
        except (TypeError, ValueError):
            return text

    return text


def _format_class_count_map(counts: Dict[str, int]) -> str:
    if not isinstance(counts, dict) or not counts:
        return "{}"
    items = sorted(((str(k), int(v)) for k, v in counts.items()), key=lambda kv: sorted_key(kv[0]))
    return "{" + ", ".join(f"{k}: {v}" for k, v in items) + "}"


def build_stratified_train_subset_indices(
    y_values: np.ndarray,
    train_fraction: float,
    min_per_class: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    y_arr = np.asarray(y_values).reshape(-1)
    total = int(y_arr.size)
    min_per_class = max(1, int(min_per_class))

    if total <= 1:
        indices = np.arange(total, dtype=np.int64)
        return indices, {
            "applied": False,
            "reason": "insufficient_samples",
            "total_samples": total,
            "selected_samples": int(indices.size),
            "requested_fraction": float(train_fraction),
            "effective_fraction": 1.0 if total == 1 else 0.0,
            "min_per_class": min_per_class,
            "class_counts_total": {},
            "class_counts_selected": {},
            "expanded_for_class_minimum": False,
        }

    fraction = max(0.0, min(1.0, float(train_fraction)))
    if fraction >= 1.0:
        indices = np.arange(total, dtype=np.int64)
        class_counts_total = {}
        for value in y_arr:
            label = normalize_class_label(value)
            class_counts_total[label] = class_counts_total.get(label, 0) + 1
        return indices, {
            "applied": False,
            "reason": "full_dataset",
            "total_samples": total,
            "selected_samples": int(indices.size),
            "requested_fraction": fraction,
            "effective_fraction": 1.0,
            "min_per_class": min_per_class,
            "class_counts_total": class_counts_total,
            "class_counts_selected": dict(class_counts_total),
            "expanded_for_class_minimum": False,
        }

    class_to_indices: Dict[str, np.ndarray] = {}
    for idx, value in enumerate(y_arr):
        label = normalize_class_label(value)
        class_to_indices.setdefault(label, []).append(idx)

    class_to_indices = {
        label: np.asarray(idxs, dtype=np.int64)
        for label, idxs in class_to_indices.items()
        if len(idxs) > 0
    }

    if not class_to_indices:
        indices = np.arange(total, dtype=np.int64)
        return indices, {
            "applied": False,
            "reason": "no_classes_detected",
            "total_samples": total,
            "selected_samples": int(indices.size),
            "requested_fraction": fraction,
            "effective_fraction": 1.0,
            "min_per_class": min_per_class,
            "class_counts_total": {},
            "class_counts_selected": {},
            "expanded_for_class_minimum": False,
        }

    target_total = max(1, int(round(total * fraction)))
    class_counts_total = {label: int(indices.size) for label, indices in class_to_indices.items()}

    desired_counts: Dict[str, int] = {}
    for label, indices in class_to_indices.items():
        class_total = int(indices.size)
        proportional = max(1, int(round(class_total * fraction)))
        required = min(class_total, max(proportional, min_per_class))
        desired_counts[label] = required

    min_required_total = int(sum(desired_counts.values()))
    expanded_for_minimum = min_required_total > target_total
    target_total = max(target_total, min_required_total)

    selected_counts = dict(desired_counts)
    remaining = target_total - int(sum(selected_counts.values()))


    label_order = sorted(class_to_indices.keys(), key=sorted_key)
    while remaining > 0:
        progressed = False
        for label in label_order:
            capacity = class_counts_total[label] - selected_counts[label]
            if capacity <= 0:
                continue
            selected_counts[label] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break

    rng = np.random.default_rng()
    selected_indices_list: List[int] = []
    for label in label_order:
        per_class_indices = class_to_indices[label]
        count = max(0, min(int(selected_counts.get(label, 0)), int(per_class_indices.size)))
        if count <= 0:
            selected_counts[label] = 0
            continue
        chosen = rng.choice(per_class_indices, size=count, replace=False)
        selected_indices_list.extend(int(x) for x in chosen.tolist())
        selected_counts[label] = count

    selected_indices = np.asarray(selected_indices_list, dtype=np.int64)
    if selected_indices.size > 0:
        rng.shuffle(selected_indices)

    selected_total = int(selected_indices.size)
    effective_fraction = float(selected_total / total) if total > 0 else 0.0
    summary = {
        "applied": True,
        "reason": "stratified",
        "total_samples": total,
        "selected_samples": selected_total,
        "requested_fraction": fraction,
        "effective_fraction": effective_fraction,
        "min_per_class": min_per_class,
        "class_counts_total": class_counts_total,
        "class_counts_selected": selected_counts,
        "expanded_for_class_minimum": bool(expanded_for_minimum),
    }
    return selected_indices, summary


def load_directive_job(directive: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    jobs = directive.get("jobs") if isinstance(directive.get("jobs"), list) else []
    if not jobs or not isinstance(jobs[0], dict):
        raise ValueError("Directive must contain at least one valid job")

    job = jobs[0]
    target = job.get("target") if isinstance(job.get("target"), dict) else {}
    candidate = job.get("candidate") if isinstance(job.get("candidate"), dict) else {}
    requested_outputs = (
        job.get("requested_outputs") if isinstance(job.get("requested_outputs"), dict) else {}
    )

    return job, target, candidate, requested_outputs


def discover_dataset_dimensions(dataset_path: Path) -> List[str]:
    train_dir = dataset_path / "train"
    if not train_dir.exists():
        return []

    return sorted([p.stem[2:] for p in train_dir.glob("X_*.npy")], key=sorted_key)


def parse_signal_modality_map(project_root: Path, dimensions: List[str]) -> Dict[str, str]:

    mapping = {dim: dim for dim in dimensions}

    context_path = project_root / "shared" / "context" / "data_context.md"
    if not context_path.exists():
        return mapping

    try:
        content = context_path.read_text(encoding="utf-8")
    except Exception:
        return mapping

    known = set(dimensions)
    for raw_line in content.splitlines():
        line = raw_line.strip()
        match = re.match(r"^- \*\*(.+?)\*\*: \[Modality: (.+?)\]", line)
        if not match:
            continue

        signal_name, raw_modality = match.groups()
        signal_name = signal_name.strip()
        if signal_name not in known:
            continue

        mapping[signal_name] = normalize_modality_name(raw_modality) or signal_name

    return mapping


def resolve_target_dimensions(
    project_root: Path,
    dataset_path: Path,
    requested_modality: str,
) -> Tuple[List[str], Dict[str, str]]:
    dimensions = discover_dataset_dimensions(dataset_path)
    if not dimensions:
        raise FileNotFoundError(f"No training dimensions found under {dataset_path / 'train'} (expected X_*.npy)")

    signal_to_modality = parse_signal_modality_map(project_root, dimensions)

    modality_to_dims: Dict[str, List[str]] = {}
    for dim in dimensions:
        modality_name = signal_to_modality.get(dim, dim) or dim
        modality_to_dims.setdefault(modality_name, []).append(dim)

    for modality_name in modality_to_dims:
        modality_to_dims[modality_name] = sorted(modality_to_dims[modality_name], key=sorted_key)

    requested_raw = str(requested_modality or "").strip()
    requested_norm = normalize_modality_name(requested_raw)

    if requested_raw in modality_to_dims:
        return list(modality_to_dims[requested_raw]), signal_to_modality

    if requested_norm and requested_norm in modality_to_dims:
        return list(modality_to_dims[requested_norm]), signal_to_modality

    lower_to_modality = {name.lower(): name for name in modality_to_dims}
    if requested_raw.lower() in lower_to_modality:
        key = lower_to_modality[requested_raw.lower()]
        return list(modality_to_dims[key]), signal_to_modality

    if requested_raw in dimensions:
        return [requested_raw], signal_to_modality

    lower_to_dim = {dim.lower(): dim for dim in dimensions}
    if requested_raw.lower() in lower_to_dim:
        return [lower_to_dim[requested_raw.lower()]], signal_to_modality

    if not requested_norm:

        return list(dimensions), signal_to_modality

    available_modalities = sorted(modality_to_dims.keys(), key=sorted_key)
    raise ValueError(
        "Directive target modality could not be resolved. "
        f"Requested='{requested_raw}'. Available modalities={available_modalities}. "
        f"Available dimensions={dimensions}."
    )


def load_training_data(
    dataset_path: Path,
    modality: str,
    train_indices: Optional[np.ndarray] = None,
    val_indices: Optional[np.ndarray] = None,
):
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "validate"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train split at {train_dir}")

    x_file = f"X_{modality}.npy"
    x_train_path = train_dir / x_file
    x_val_path = val_dir / x_file if val_dir.exists() else x_train_path

    if not x_train_path.exists():
        raise FileNotFoundError(f"Missing modality training data: {x_train_path}")
    if not x_val_path.exists():
        raise FileNotFoundError(f"Missing modality validation data: {x_val_path}")

    y_train_path = train_dir / "y.npy"
    y_val_path = val_dir / "y.npy" if val_dir.exists() else y_train_path
    if not y_train_path.exists():
        raise FileNotFoundError(f"Missing y.npy in train split: {y_train_path}")
    if not y_val_path.exists():
        raise FileNotFoundError(f"Missing y.npy in validation split: {y_val_path}")

    x_train = np.load(x_train_path)
    y_train = np.asarray(np.load(y_train_path)).reshape(-1)
    x_val = np.load(x_val_path)
    y_val = np.asarray(np.load(y_val_path)).reshape(-1)

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Train sample mismatch for {modality}: X={x_train.shape[0]} Y={y_train.shape[0]}"
        )
    if x_val.shape[0] != y_val.shape[0]:
        raise ValueError(
            f"Validation sample mismatch for {modality}: X={x_val.shape[0]} Y={y_val.shape[0]}"
        )

    if train_indices is not None:
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]

    if val_indices is not None:
        x_val = x_val[val_indices]
        y_val = y_val[val_indices]

    return x_train, y_train, x_val, y_val


def build_ensemble_evaluation_payload(
    project_root: Path,
    dataset_path: Path,
    directive: Dict[str, Any],
    run_eval: bool,
    expert_matrix: Dict[str, Any],
    train_indices: Optional[np.ndarray],
    training_overrides: Dict[str, Any],
):
    subset_fraction = None

    if not run_eval:
        return {
            "ran": False,
            "subset_fraction": subset_fraction,
            "status": "skipped",
            "candidate_id_used": None,
            "notes": "Skipped by ensemble retrain policy for this cycle.",
        }

    if not isinstance(expert_matrix, dict) or not expert_matrix:
        return {
            "ran": True,
            "subset_fraction": subset_fraction,
            "status": "failed",
            "candidate_id_used": None,
            "notes": "expert_matrix.json was empty or invalid; cannot retrain ensemble.",
        }

    train_dir = dataset_path / "train"
    val_dir = dataset_path / "validate"

    y_train_path = train_dir / "y.npy"
    y_val_path = val_dir / "y.npy" if val_dir.exists() else y_train_path
    if not y_train_path.exists() or not y_val_path.exists():
        return {
            "ran": True,
            "subset_fraction": subset_fraction,
            "status": "failed",
            "candidate_id_used": None,
            "notes": "Missing train/validate labels for ensemble retraining.",
        }

    y_train = np.asarray(np.load(y_train_path)).reshape(-1)
    y_val = np.asarray(np.load(y_val_path)).reshape(-1)
    if train_indices is not None:
        y_train = y_train[train_indices]

    combined_labels = [normalize_class_label(v) for v in np.concatenate([y_train, y_val])]
    class_labels = sorted({lbl for lbl in combined_labels if lbl}, key=sorted_key)
    if not class_labels:
        return {
            "ran": True,
            "subset_fraction": subset_fraction,
            "status": "failed",
            "candidate_id_used": None,
            "notes": "Unable to infer class labels for ensemble retraining.",
        }

    class_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    y_train_mapped = np.array([class_to_idx[normalize_class_label(v)] for v in y_train], dtype=np.int64)
    y_val_mapped = np.array([class_to_idx[normalize_class_label(v)] for v in y_val], dtype=np.int64)

    y_val_eval = y_val_mapped

    repo_root = Path(__file__).resolve().parents[2]
    model_module_cache: Dict[str, Any] = {}
    pre_module_cache: Dict[str, Any] = {}
    raw_modality_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    processed_input_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

    x_train_blocks = []
    x_val_blocks = []
    block_index_map: Dict[Tuple[str, str], int] = {}

    experts = []
    input_map = []
    expert_dims = []
    warnings = []
    expected_experts = 0

    def load_modality_raw(modality_name: str):
        cached = raw_modality_cache.get(modality_name)
        if cached is not None:
            return cached

        x_train_path = train_dir / f"X_{modality_name}.npy"
        x_val_path = val_dir / f"X_{modality_name}.npy" if val_dir.exists() else x_train_path
        if not x_train_path.exists() or not x_val_path.exists():
            raise FileNotFoundError(
                f"Missing modality files for '{modality_name}': {x_train_path} / {x_val_path}"
            )

        x_train_raw = np.load(x_train_path)
        x_val_raw = np.load(x_val_path)
        if train_indices is not None:
            x_train_raw = x_train_raw[train_indices]

        raw_modality_cache[modality_name] = (x_train_raw, x_val_raw)
        return x_train_raw, x_val_raw

    for modality in sorted(expert_matrix.keys(), key=sorted_key):
        by_class = expert_matrix.get(modality)
        if not isinstance(by_class, dict):
            continue

        modality_name = str(modality)
        for class_label in sorted(by_class.keys(), key=sorted_key):
            rec = by_class.get(class_label)
            if not isinstance(rec, dict):
                continue

            expected_experts += 1
            class_label_norm = normalize_class_label(class_label)
            candidate_id = str(rec.get("candidate_id", "")).strip()
            if not candidate_id:
                warnings.append(f"{modality_name}/{class_label_norm}: missing candidate_id")
                continue

            model_weights_path = (
                project_root / "models" / f"{modality_name}_{class_label_norm}" / f"{candidate_id}.pt"
            )
            if not model_weights_path.exists():
                warnings.append(f"{modality_name}/{class_label_norm}: missing weights {model_weights_path.name}")
                continue

            model_ref = rec.get("final_model_py_ref")
            preprocessing_ref = rec.get("preprocessing_code_ref")

            try:
                model_cls = load_model_class_from_ref(
                    model_ref=model_ref,
                    project_root=project_root,
                    repo_root=repo_root,
                    cache=model_module_cache,
                )
            except Exception as exc:
                warnings.append(f"{modality_name}/{class_label_norm}: model ref load failed ({exc})")
                continue

            try:
                preprocess_fn = load_preprocessing_callable_from_ref(
                    preprocessing_ref=preprocessing_ref,
                    project_root=project_root,
                    repo_root=repo_root,
                    cache=pre_module_cache,
                )
            except Exception as exc:
                warnings.append(f"{modality_name}/{class_label_norm}: preprocessing ref load failed ({exc})")
                continue

            preprocess_key = str(resolve_code_ref_path(preprocessing_ref, project_root, repo_root) or "__baseline__")
            cache_key = (modality_name, preprocess_key)
            cached_inputs = processed_input_cache.get(cache_key)
            if cached_inputs is None:
                try:
                    x_train_raw, x_val_raw = load_modality_raw(modality_name)
                    x_train_processed = preprocess_fn(x_train_raw)
                    x_val_processed = preprocess_fn(x_val_raw)
                    x_train_input = to_model_input(
                        x_train_processed,
                        signal_name=f"{modality_name}:{class_label_norm}:train",
                    )
                    x_val_input = to_model_input(
                        x_val_processed,
                        signal_name=f"{modality_name}:{class_label_norm}:val",
                    )
                    processed_input_cache[cache_key] = (x_train_input, x_val_input)
                    cached_inputs = (x_train_input, x_val_input)
                except Exception as exc:
                    warnings.append(f"{modality_name}/{class_label_norm}: preprocessing failed ({exc})")
                    continue

            x_train_input, x_val_input = cached_inputs
            if x_train_input.shape[0] != y_train_mapped.shape[0]:
                warnings.append(
                    f"{modality_name}/{class_label_norm}: train sample mismatch X={x_train_input.shape[0]} Y={y_train_mapped.shape[0]}"
                )
                continue
            if x_val_input.shape[0] != y_val_mapped.shape[0]:
                warnings.append(
                    f"{modality_name}/{class_label_norm}: val sample mismatch X={x_val_input.shape[0]} Y={y_val_mapped.shape[0]}"
                )
                continue

            try:
                expert_model = _build_binary_expert_model(
                    model_cls=model_cls,
                    in_ch=int(x_train_input.shape[1]),
                    n_classes=1,
                    min_seq_len=int(x_train_input.shape[-1]),
                    model_init_overrides={},
                )
                expert_model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
                expert_model.eval()
            except Exception as exc:
                warnings.append(f"{modality_name}/{class_label_norm}: weights load failed ({exc})")
                continue

            block_idx = block_index_map.get(cache_key)
            if block_idx is None:
                block_idx = len(x_train_blocks)
                x_train_blocks.append(x_train_input)
                x_val_blocks.append(x_val_input)
                block_index_map[cache_key] = block_idx

            experts.append(expert_model)
            input_map.append(block_idx)
            expert_dims.append(infer_expert_feature_dim(expert_model, x_train_input, default=16))

    if not experts:
        note = "No expert models were loadable for ensemble retraining."
        if warnings:
            note = f"{note} Sample issues: {', '.join(warnings[:3])}"
        return {
            "ran": True,
            "subset_fraction": subset_fraction,
            "status": "failed",
            "candidate_id_used": None,
            "notes": note,
            "warnings": warnings[:10],
        }

    x_val_eval_blocks = x_val_blocks

    ensemble_epochs = max(
        1,
        safe_int(
            training_overrides.get("ensemble_epochs", training_overrides.get("epochs", 50)),
            50,
        ),
    )
    ensemble_patience = max(
        1,
        safe_int(
            training_overrides.get("ensemble_patience", training_overrides.get("patience", 5)),
            5,
        ),
    )
    ensemble_lr = safe_float(
        training_overrides.get("ensemble_lr", training_overrides.get("lr", 1e-3)),
        1e-3,
    )
    ensemble_batch_size = max(1, safe_int(training_overrides.get("ensemble_batch_size", 64), 64))

    ensemble_architecture = "default"
    config_path = project_root / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                project_config = yaml.safe_load(f) or {}
            ensemble_architecture = normalize_ensemble_architecture(
                project_config.get("ensemble_architecture", "default")
            )
        except Exception:
            ensemble_architecture = "default"
    if "ensemble_architecture" in training_overrides:
        ensemble_architecture = normalize_ensemble_architecture(training_overrides.get("ensemble_architecture"))

    try:
        default_expert_dim = expert_dims[0] if expert_dims else 16
        dims_arg = expert_dims if len(set(expert_dims)) > 1 else None
        ensemble_model = BaselineEnsemble(
            experts=experts,
            input_map=input_map,
            num_classes=len(class_labels),
            expert_dim=int(default_expert_dim),
            expert_dims=dims_arg,
            architecture=ensemble_architecture,
        )

        ensemble_metrics, ensemble_model_trained, ensemble_history, ensemble_summary = train_ensemble(
            model=ensemble_model,

            x_train_list=x_train_blocks,
            y_train=y_train_mapped,
            x_val_list=x_val_blocks,
            y_val=y_val_mapped,
            epochs=ensemble_epochs,
            patience=ensemble_patience,
            lr=ensemble_lr,
            batch_size=ensemble_batch_size,
            project_root=project_root,
        )

        ensemble_eval_metrics_raw, y_val_true, y_val_pred = evaluate_ensemble_on_split(
            model=ensemble_model_trained,
            x_eval_list=x_val_eval_blocks,
            y_eval=y_val_eval,
            batch_size=ensemble_batch_size,
        )

        bootstrap_repeats = max(
            20,
            safe_int(os.environ.get("ARL_ENSEMBLE_BOOTSTRAP_REPEATS", 200), 200),
        )
        bootstrap_seed = safe_int(os.environ.get("ARL_ENSEMBLE_BOOTSTRAP_SEED", 42), 42)
        bootstrap_summary = build_bootstrap_validation_summary(
            y_true=y_val_true,
            y_pred=y_val_pred,
            repeats=bootstrap_repeats,
            seed=bootstrap_seed,
        )

        acceptance_metric = str(os.environ.get("ARL_ENSEMBLE_ACCEPT_METRIC", "kappa")).strip().lower() or "kappa"
        if acceptance_metric not in {"accuracy", "kappa", "precision", "recall", "f1"}:
            acceptance_metric = "kappa"

        previous_score_info = resolve_previous_ensemble_acceptance_score(project_root)
        previous_score = _to_optional_float(previous_score_info.get("score"))
        min_delta = safe_float(os.environ.get("ARL_ENSEMBLE_ACCEPT_MIN_DELTA", 0.0), 0.0)

        bootstrap_metrics = bootstrap_summary.get("metrics") if isinstance(bootstrap_summary, dict) else {}
        candidate_metric_summary = (
            bootstrap_metrics.get(acceptance_metric)
            if isinstance(bootstrap_metrics, dict)
            else None
        )
        candidate_score = None
        if isinstance(candidate_metric_summary, dict):
            candidate_score = _to_optional_float(candidate_metric_summary.get("mean"))
        if candidate_score is None:
            candidate_score = _to_optional_float(ensemble_eval_metrics_raw.get(acceptance_metric))

        accepted = True
        if candidate_score is not None and previous_score is not None:
            accepted = bool(candidate_score >= (previous_score + float(min_delta)))

        validation_acceptance = {
            "method": "stratified_bootstrap_validation",
            "metric": acceptance_metric,
            "selected_score": candidate_score,
            "previous_score": previous_score,
            "previous_score_source": previous_score_info.get("source"),
            "min_delta": float(min_delta),
            "accepted": bool(accepted),
            "bootstrap_summary": bootstrap_summary,
        }

        if accepted:
            ensemble_model_path = project_root / "models" / "baseline_ensemble.pt"
            ensemble_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ensemble_model_trained.state_dict(), ensemble_model_path)

            metrics_payload = dict(ensemble_eval_metrics_raw)
            metrics_payload["training_history"] = ensemble_history
            metrics_payload["training_summary"] = ensemble_summary
            metrics_payload["training_monitor_metrics"] = ensemble_metrics
            metrics_payload["ensemble_architecture"] = ensemble_architecture
            metrics_payload["evaluation_split"] = "validate"
            metrics_payload["training_monitor_split"] = "validate"
            metrics_payload["training_fit_split"] = "train_oof"
            metrics_payload["bootstrap_validation_summary"] = bootstrap_summary
            metrics_payload["validation_acceptance"] = validation_acceptance
            metrics_payload["class_labels"] = class_labels
            metrics_payload["expert_count"] = len(experts)
            metrics_payload["expected_expert_count"] = expected_experts
            metrics_payload["input_block_count"] = len(x_train_blocks)
            metrics_payload["updated_at"] = utc_now_iso()
            write_json(project_root / "artifacts" / "baseline_ensemble_metrics.json", metrics_payload)

        def _bootstrap_mean(metric_name: str, fallback_value: float) -> float:
            metric_payload = bootstrap_metrics.get(metric_name) if isinstance(bootstrap_metrics, dict) else None
            if isinstance(metric_payload, dict):
                mean_val = _to_optional_float(metric_payload.get("mean"))
                if mean_val is not None:
                    return float(mean_val)
            return float(fallback_value)

        metric_rows = []
        metric_rows.append(
            {
                "name": "accuracy",
                "value": _bootstrap_mean("accuracy", safe_float(ensemble_eval_metrics_raw.get("accuracy"), 0.0)),
            }
        )
        metric_rows.append(
            {
                "name": "kappa",
                "value": _bootstrap_mean("kappa", safe_float(ensemble_eval_metrics_raw.get("kappa"), 0.0)),
            }
        )
        metric_rows.append(
            {
                "name": "macro_precision",
                "value": _bootstrap_mean("precision", safe_float(ensemble_eval_metrics_raw.get("precision"), 0.0)),
            }
        )
        metric_rows.append(
            {
                "name": "macro_recall",
                "value": _bootstrap_mean("recall", safe_float(ensemble_eval_metrics_raw.get("recall"), 0.0)),
            }
        )
        metric_rows.append(
            {
                "name": "macro_f1",
                "value": _bootstrap_mean("f1", safe_float(ensemble_eval_metrics_raw.get("f1"), 0.0)),
            }
        )

        report = ensemble_eval_metrics_raw.get("classification_report")

        curves = build_secondary_curves_from_history(ensemble_history)
        notes = (
            f"Retrained ensemble using {len(experts)} experts "
            f"({len(x_train_blocks)} unique input blocks, architecture={ensemble_architecture}) "
            "with train OOF feature fitting and validation monitoring; "
            "validation acceptance uses stratified bootstrap scoring."
        )
        if accepted:
            notes += " Candidate accepted and promoted as active baseline ensemble."
        else:
            notes += " Candidate not accepted; previous baseline ensemble remains active."
        if warnings:
            notes += f" Skipped {len(warnings)} expert slots with load/preprocess issues."

        selected_candidate_id = "baseline_ensemble"

        return {
            "ran": True,
            "subset_fraction": subset_fraction,
            "status": "success" if accepted else "not_accepted",
            "candidate_id_used": selected_candidate_id,
            "accepted": bool(accepted),
            "metrics": metric_rows,
            "classification_report": report if isinstance(report, dict) else {},
            "confusion_matrix": ensemble_eval_metrics_raw.get("confusion_matrix", []),
            "training_summary": summarize_history_for_results(ensemble_history, ensemble_summary),
            "train_loss_curve": curves.get("train_loss_curve", []),
            "val_loss_curve": curves.get("val_loss_curve", []),
            "train_metric_curves": curves.get("train_metric_curves", {}),
            "val_metric_curves": curves.get("val_metric_curves", {}),
            "bootstrap_validation_summary": bootstrap_summary,
            "validation_acceptance": validation_acceptance,
            "notes": notes,
            "warnings": warnings[:10],
        }
    except Exception as exc:
        note = f"Ensemble retraining failed: {exc}"
        if warnings:
            note += f" | sample issues: {', '.join(warnings[:3])}"
        return {
            "ran": True,
            "subset_fraction": subset_fraction,
            "status": "failed",
            "candidate_id_used": None,
            "notes": note,
            "warnings": warnings[:10],
        }


def update_expert_matrix(
    project_root: Path,
    modality: str,
    modality_group: str,
    class_label: str,
    candidate_id: str,
    f1: float,
    acc: float,
    prec: float,
    rec: float,
    is_success: bool,
    model_py_ref: str,
    preprocessing_ref: str,
    cv_metrics: Optional[Dict[str, Any]] = None,
    validation_metrics: Optional[Dict[str, Any]] = None,
):
    expert_matrix_path = project_root / "artifacts" / "expert_matrix.json"
    matrix = read_json_file(expert_matrix_path, {})
    if not isinstance(matrix, dict):
        matrix = {}

    per_modality = matrix.setdefault(modality, {})
    if not isinstance(per_modality, dict):
        per_modality = {}
        matrix[modality] = per_modality

    key = str(class_label)
    existing = per_modality.get(key)
    if not isinstance(existing, dict):
        existing = {}

    history = existing.get("history") if isinstance(existing.get("history"), list) else []
    cv_metrics_payload = cv_metrics if isinstance(cv_metrics, dict) else {}
    validation_metrics_payload = validation_metrics if isinstance(validation_metrics, dict) else {}
    if not validation_metrics_payload:
        validation_metrics_payload = {
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "split": "validate",
        }

    history_row = {
        "candidate_id": candidate_id,
        "f1": float(f1),
        "accuracy": float(acc),
        "recall": float(rec),
        "precision": float(prec),
        "cv_metrics": cv_metrics_payload,
        "validation_metrics": validation_metrics_payload,
    }
    history = list(history)
    history.append(history_row)
    history = history[-30:]

    prev_best_f1 = safe_float(existing.get("f1"), -1.0)
    improved = bool(is_success and (not existing or float(f1) > prev_best_f1 + 1e-12))
    is_best = bool(is_success and (not existing or float(f1) >= prev_best_f1))

    if is_best or not existing:
        updated = {
            "modality": modality,
            "modality_group": modality_group,
            "class_label": key,
            "candidate_id": candidate_id,
            "f1": float(f1),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "is_best": bool(is_success),
            "improved": bool(improved),
            "preprocessing_code_ref": preprocessing_ref,
            "final_model_py_ref": model_py_ref,
            "cv_metrics": cv_metrics_payload,
            "validation_metrics": validation_metrics_payload,
            "cv_accuracy_mean": safe_float(cv_metrics_payload.get("accuracy_mean"), 0.0),
            "cv_f1_mean": safe_float(cv_metrics_payload.get("f1_mean"), 0.0),
            "cv_precision_mean": safe_float(cv_metrics_payload.get("precision_mean"), 0.0),
            "cv_recall_mean": safe_float(cv_metrics_payload.get("recall_mean"), 0.0),
            "validation_accuracy": safe_float(validation_metrics_payload.get("accuracy"), float(acc)),
            "validation_f1": safe_float(validation_metrics_payload.get("f1"), float(f1)),
            "validation_precision": safe_float(validation_metrics_payload.get("precision"), float(prec)),
            "validation_recall": safe_float(validation_metrics_payload.get("recall"), float(rec)),
            "history": history,
        }
    else:
        updated = dict(existing)
        updated["improved"] = bool(improved)
        if cv_metrics_payload:
            updated["cv_metrics"] = cv_metrics_payload
            updated["cv_accuracy_mean"] = safe_float(cv_metrics_payload.get("accuracy_mean"), 0.0)
            updated["cv_f1_mean"] = safe_float(cv_metrics_payload.get("f1_mean"), 0.0)
            updated["cv_precision_mean"] = safe_float(cv_metrics_payload.get("precision_mean"), 0.0)
            updated["cv_recall_mean"] = safe_float(cv_metrics_payload.get("recall_mean"), 0.0)
        if validation_metrics_payload:
            updated["validation_metrics"] = validation_metrics_payload
            updated["validation_accuracy"] = safe_float(validation_metrics_payload.get("accuracy"), float(acc))
            updated["validation_f1"] = safe_float(validation_metrics_payload.get("f1"), float(f1))
            updated["validation_precision"] = safe_float(validation_metrics_payload.get("precision"), float(prec))
            updated["validation_recall"] = safe_float(validation_metrics_payload.get("recall"), float(rec))
        updated["history"] = history

    per_modality[key] = updated
    matrix[modality] = per_modality
    write_json(expert_matrix_path, matrix)

    update_payload = {
        "modality": modality,
        "modality_group": modality_group,
        "class_label": key,
        "candidate_id": candidate_id,
        "f1": float(f1),
        "accuracy": float(acc),
        "recall": float(rec),
        "precision": float(prec),
        "is_best": is_best,
        "improved": bool(improved),
        "preprocessing_code_ref": preprocessing_ref,
        "final_model_py_ref": model_py_ref,
        "cv_metrics": cv_metrics_payload,
        "validation_metrics": validation_metrics_payload,
        "cv_accuracy_mean": safe_float(cv_metrics_payload.get("accuracy_mean"), 0.0),
        "cv_f1_mean": safe_float(cv_metrics_payload.get("f1_mean"), 0.0),
        "cv_precision_mean": safe_float(cv_metrics_payload.get("precision_mean"), 0.0),
        "cv_recall_mean": safe_float(cv_metrics_payload.get("recall_mean"), 0.0),
        "validation_accuracy": safe_float(validation_metrics_payload.get("accuracy"), float(acc)),
        "validation_f1": safe_float(validation_metrics_payload.get("f1"), float(f1)),
        "validation_precision": safe_float(validation_metrics_payload.get("precision"), float(prec)),
        "validation_recall": safe_float(validation_metrics_payload.get("recall"), float(rec)),
        "history": [history_row],
    }

    return update_payload, matrix


def build_results_markdown(
    cycle_id: str,
    project_id: str,
    requested_modality: str,
    class_label: str,
    candidate_id: str,
    jobs: List[Dict[str, Any]],
    expert_updates: List[Dict[str, Any]],
    ensemble_eval: Dict[str, Any],
):
    succeeded = 0
    failed = 0
    skipped = 0
    for job in jobs:
        if not isinstance(job, dict):
            continue
        status = str(job.get("status", "")).lower()
        if status == "success":
            succeeded += 1
        elif status == "failed":
            failed += 1
        else:
            skipped += 1

    changed_dimensions = sorted(
        {
            str(row.get("modality", "")).strip()
            for row in expert_updates
            if isinstance(row, dict) and str(row.get("modality", "")).strip()
        },
        key=sorted_key,
    )

    lines = [
        f"# Cycle {cycle_id} Local Engine Results",
        "",
        f"- Project: `{project_id}`",
        f"- Candidate: `{candidate_id}`",
        f"- Requested target: modality `{requested_modality}` class `{class_label}`",
        f"- Dimensions evaluated: `{len(jobs)}` (success={succeeded}, skipped={skipped}, failed={failed})",
        f"- Dimensions changed in expert matrix: `{', '.join(changed_dimensions) if changed_dimensions else 'none'}`",
        "",
        "## Per-Dimension Expert Metrics",
    ]

    for job in jobs:
        if not isinstance(job, dict):
            continue

        target = job.get("target") if isinstance(job.get("target"), dict) else {}
        candidate = job.get("candidate") if isinstance(job.get("candidate"), dict) else {}
        metrics = job.get("metrics") if isinstance(job.get("metrics"), dict) else {}
        secondary = metrics.get("secondary_metrics") if isinstance(metrics.get("secondary_metrics"), dict) else {}
        primary = metrics.get("primary_metric") if isinstance(metrics.get("primary_metric"), dict) else {}
        cv_metrics = (
            secondary.get("cross_validation_metrics")
            if isinstance(secondary.get("cross_validation_metrics"), dict)
            else {}
        )
        validation_metrics = (
            secondary.get("validation_set_metrics")
            if isinstance(secondary.get("validation_set_metrics"), dict)
            else {}
        )

        val_f1 = safe_float(validation_metrics.get("f1", primary.get("value", 0.0)), 0.0)
        val_acc = safe_float(validation_metrics.get("accuracy", secondary.get("accuracy", 0.0)), 0.0)
        val_prec = safe_float(validation_metrics.get("precision", secondary.get("precision", 0.0)), 0.0)
        val_rec = safe_float(validation_metrics.get("recall", secondary.get("recall", 0.0)), 0.0)

        dim_name = str(target.get("modality", "unknown"))
        group_name = str(target.get("requested_modality", requested_modality))
        status = str(job.get("status", "unknown"))
        weights_ref = str(candidate.get("best_weights_ref") or "n/a")

        lines.extend(
            [
                "",
                f"### Dimension `{dim_name}`",
                f"- Requested modality group: `{group_name}`",
                f"- Status: `{status}`",
                f"- Weights: `{weights_ref}`",
                f"- Validation F1: `{val_f1:.4f}`",
                f"- Validation Accuracy: `{val_acc:.4f}`",
                f"- Validation Precision: `{val_prec:.4f}`",
                f"- Validation Recall: `{val_rec:.4f}`",
            ]
        )

        if cv_metrics:
            lines.extend(
                [
                    f"- CV Fold Count: `{safe_int(cv_metrics.get('fold_count', 0), 0)}`",
                    f"- CV F1 Mean/Std: `{safe_float(cv_metrics.get('f1_mean', 0.0), 0.0):.4f}` / `{safe_float(cv_metrics.get('f1_std', 0.0), 0.0):.4f}`",
                    f"- CV Accuracy Mean/Std: `{safe_float(cv_metrics.get('accuracy_mean', 0.0), 0.0):.4f}` / `{safe_float(cv_metrics.get('accuracy_std', 0.0), 0.0):.4f}`",
                    f"- CV Precision Mean/Std: `{safe_float(cv_metrics.get('precision_mean', 0.0), 0.0):.4f}` / `{safe_float(cv_metrics.get('precision_std', 0.0), 0.0):.4f}`",
                    f"- CV Recall Mean/Std: `{safe_float(cv_metrics.get('recall_mean', 0.0), 0.0):.4f}` / `{safe_float(cv_metrics.get('recall_std', 0.0), 0.0):.4f}`",
                ]
            )

        errors = job.get("errors") if isinstance(job.get("errors"), list) else []
        if errors:
            lines.append(f"- Errors: {errors[0]}")

    lines.extend(
        [
            "",
            "## Ensemble Evaluation",
            f"- Ran: `{bool(ensemble_eval.get('ran', False))}`",
            f"- Status: `{ensemble_eval.get('status', 'unknown')}`",
            f"- Candidate Used: `{ensemble_eval.get('candidate_id_used') or 'n/a'}`",
            f"- Notes: {ensemble_eval.get('notes', 'n/a')}",
        ]
    )

    policy = ensemble_eval.get("policy") if isinstance(ensemble_eval.get("policy"), dict) else {}
    if policy:
        policy_name = str(policy.get("policy_name", "")).strip().lower()
        if policy_name == "ensemble_every_cycle":
            lines.append("- Ensemble Policy: run after every cycle")
        else:
            lines.append(
                f"- Ensemble Policy: run every `{safe_int(policy.get('required_successful_improvement_cycles', 5), 5)}` successful binary-expert improvement cycles"
            )
        lines.extend(
            [
                f"- Successful improvements since last ensemble: `{safe_int(policy.get('successful_improvement_count_since_last_ensemble', 0), 0)}`",
                f"- Successful improvements including current cycle: `{safe_int(policy.get('successful_improvement_count_including_current', 0), 0)}`",
                f"- Policy decision reason: `{policy.get('decision_reason', 'n/a')}`",
            ]
        )

    metrics = ensemble_eval.get("metrics")
    if isinstance(metrics, list) and metrics:
        lines.append("")
        lines.append("### Ensemble Metrics")
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            name = str(metric.get("name", "metric"))
            value = metric.get("value")
            try:
                lines.append(f"- {name}: `{float(value):.4f}`")
            except (TypeError, ValueError):
                lines.append(f"- {name}: `{value}`")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    started_at = utc_now_iso()
    wall_start = time.time()

    project_root_env = os.environ.get("ARL_PROJECT_ROOT")
    if not project_root_env:
        raise ValueError("ARL_PROJECT_ROOT is required for local engine execution")

    project_root = Path(project_root_env).resolve()
    project_id = os.environ.get("ARL_PROJECT") or os.environ.get("ARL_PROJECT_ID") or project_root.name
    cycle_id = str(os.environ.get("ARL_CYCLE_ID", "1"))

    shared_root = project_root / "shared"
    directive_path = Path(os.environ.get("ARL_DIRECTIVE_PATH", str(shared_root / "inbound" / "directive.json")))
    model_py_path = Path(os.environ.get("ARL_MODEL_PY_PATH", str(shared_root / "models" / "model.py")))
    preprocessing_py_path = Path(
        os.environ.get("ARL_PREPROCESSING_PY_PATH", str(shared_root / "models" / "preprocessing.py"))
    )
    model_meta_path = Path(os.environ.get("ARL_MODEL_META_PATH", str(shared_root / "models" / "model.meta.json")))
    results_json_path = Path(os.environ.get("ARL_RESULTS_JSON_PATH", str(shared_root / "outbound" / "results.json")))
    results_md_path = results_json_path.parent / "results.md"

    job_result = None
    results_payload = None

    live_log_path = project_root / "state" / "live_training.json"

    try:
        config_path = project_root / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing project config: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        dataset_path = Path(str(config.get("dataset_path", "")))
        if not dataset_path.exists():
            raise FileNotFoundError(f"Configured dataset path does not exist: {dataset_path}")

        directive = read_json_file(directive_path, {})
        if not isinstance(directive, dict):
            raise ValueError(f"Invalid directive JSON at {directive_path}")

        job, target, candidate, requested_outputs = load_directive_job(directive)

        modality = str(target.get("modality", "unknown")).strip() or "unknown"
        class_label = normalize_class_label(target.get("class_label", "unknown"))
        candidate_id = str(candidate.get("candidate_id", f"cycle_{cycle_id}_candidate")).strip() or f"cycle_{cycle_id}_candidate"
        candidate_origin = str(candidate.get("origin", "director"))
        candidate_model_ref = str(candidate.get("model_py_ref", "shared/models/model.py")).strip() or "shared/models/model.py"
        candidate_preprocessing_ref = (
            str(candidate.get("preprocessing_py_ref", "shared/models/preprocessing.py")).strip()
            or "shared/models/preprocessing.py"
        )

        train_fraction = safe_float(directive.get("train_fraction", config.get("train_proportion", 1.0)), 1.0)
        train_fraction = max(0.01, min(1.0, train_fraction))
        train_subset_min_per_class = max(
            1,
            safe_int(os.environ.get("ARL_TRAIN_SUBSET_MIN_PER_CLASS", 64), 64),
        )

        train_indices = None
        train_subset_summary: Dict[str, Any] = {
            "applied": False,
            "reason": "full_dataset",
            "requested_fraction": train_fraction,
            "effective_fraction": 1.0,
            "min_per_class": train_subset_min_per_class,
            "class_counts_total": {},
            "class_counts_selected": {},
            "expanded_for_class_minimum": False,
        }
        y_train_full_path = dataset_path / "train" / "y.npy"
        if 0.0 < train_fraction < 1.0 and y_train_full_path.exists():
            y_train_full = np.asarray(np.load(y_train_full_path)).reshape(-1)
            if len(y_train_full) > 1:
                train_indices, train_subset_summary = build_stratified_train_subset_indices(
                    y_values=y_train_full,
                    train_fraction=train_fraction,
                    min_per_class=train_subset_min_per_class,
                )

        val_indices = None
        val_subset_summary: Dict[str, Any] = {
            "applied": False,
            "reason": "full_dataset",
            "requested_fraction": train_fraction,
            "effective_fraction": 1.0,
            "min_per_class": train_subset_min_per_class,
            "class_counts_total": {},
            "class_counts_selected": {},
            "expanded_for_class_minimum": False,
        }
        y_val_full_path = dataset_path / "validate" / "y.npy"
        if not y_val_full_path.exists():
            y_val_full_path = dataset_path / "train" / "y.npy"

        if 0.0 < train_fraction < 1.0 and y_val_full_path.exists():
            y_val_full = np.asarray(np.load(y_val_full_path)).reshape(-1)
            if len(y_val_full) > 1:
                val_indices, val_subset_summary = build_stratified_train_subset_indices(
                    y_values=y_val_full,
                    train_fraction=train_fraction,
                    min_per_class=train_subset_min_per_class,
                )

        if train_subset_summary.get("applied"):
            print(
                "[Local Engine] Train subset selection: "
                f"requested_fraction={safe_float(train_subset_summary.get('requested_fraction'), train_fraction):.4f}, "
                f"effective_fraction={safe_float(train_subset_summary.get('effective_fraction'), train_fraction):.4f}, "
                f"selected={safe_int(train_subset_summary.get('selected_samples'), 0)}/"
                f"{safe_int(train_subset_summary.get('total_samples'), 0)}, "
                f"min_per_class={safe_int(train_subset_summary.get('min_per_class'), train_subset_min_per_class)}, "
                f"expanded_for_class_minimum={bool(train_subset_summary.get('expanded_for_class_minimum', False))}"
            )
            print(
                "[Local Engine] Train subset class counts: "
                f"total={_format_class_count_map(train_subset_summary.get('class_counts_total', {}))}, "
                f"selected={_format_class_count_map(train_subset_summary.get('class_counts_selected', {}))}"
            )

        if val_subset_summary.get("applied"):
            print(
                "[Local Engine] Validation subset selection: "
                f"requested_fraction={safe_float(val_subset_summary.get('requested_fraction'), train_fraction):.4f}, "
                f"effective_fraction={safe_float(val_subset_summary.get('effective_fraction'), train_fraction):.4f}, "
                f"selected={safe_int(val_subset_summary.get('selected_samples'), 0)}/"
                f"{safe_int(val_subset_summary.get('total_samples'), 0)}, "
                f"min_per_class={safe_int(val_subset_summary.get('min_per_class'), train_subset_min_per_class)}, "
                f"expanded_for_class_minimum={bool(val_subset_summary.get('expanded_for_class_minimum', False))}"
            )
            print(
                "[Local Engine] Validation subset class counts: "
                f"total={_format_class_count_map(val_subset_summary.get('class_counts_total', {}))}, "
                f"selected={_format_class_count_map(val_subset_summary.get('class_counts_selected', {}))}"
            )

        training_overrides = job.get("training_overrides") if isinstance(job.get("training_overrides"), dict) else {}
        epochs = max(1, safe_int(training_overrides.get("epochs", 50), 50))
        patience = max(1, safe_int(training_overrides.get("patience", 5), 5))
        lr = safe_float(training_overrides.get("lr", 1e-3), 1e-3)
        model_init_overrides = (
            training_overrides.get("model_init") if isinstance(training_overrides.get("model_init"), dict) else {}
        )

        repo_root = Path(__file__).resolve().parents[2]
        runtime_model_path = resolve_code_ref_path(
            candidate_model_ref,
            project_root=project_root,
            repo_root=repo_root,
        ) or model_py_path
        runtime_preprocessing_path = resolve_code_ref_path(
            candidate_preprocessing_ref,
            project_root=project_root,
            repo_root=repo_root,
        ) or preprocessing_py_path

        runtime_model_module = load_python_module(runtime_model_path, "arl_runtime_model")
        runtime_pre_module = load_python_module(runtime_preprocessing_path, "arl_runtime_preprocessing")

        if not hasattr(runtime_model_module, "BinaryExpertModel"):
            raise AttributeError("Generated model module must define BinaryExpertModel")
        if not hasattr(runtime_pre_module, "apply_preprocessing"):
            raise AttributeError("Generated preprocessing module must define apply_preprocessing")

        model_cls = runtime_model_module.BinaryExpertModel

        snapshot_refs = snapshot_candidate_code(
            project_root=project_root,
            cycle_id=cycle_id,
            candidate_id=candidate_id,
            runtime_model_path=runtime_model_path,
            runtime_preprocessing_path=runtime_preprocessing_path,
            model_meta_path=model_meta_path,
        )

        model_py_ref = snapshot_refs.get("model_py_ref", candidate_model_ref)
        preprocessing_ref = snapshot_refs.get("preprocessing_py_ref", candidate_preprocessing_ref)

        if isinstance(job.get("candidate"), dict):
            job["candidate"]["model_py_ref"] = model_py_ref
            if "model_meta_ref" in snapshot_refs:
                job["candidate"]["model_meta_ref"] = snapshot_refs["model_meta_ref"]
            job["candidate"]["preprocessing_py_ref"] = preprocessing_ref
            directive["jobs"] = [job]
            write_json(directive_path, directive)

        target_dimensions, signal_to_modality = resolve_target_dimensions(
            project_root=project_root,
            dataset_path=dataset_path,
            requested_modality=modality,
        )

        output_contract = read_json_file(model_meta_path, {}).get("output_contract", {})
        embedding_dim = None
        if isinstance(output_contract, dict):
            embedding_dim = output_contract.get("embedding_dim")

        job_results: List[Dict[str, Any]] = []
        expert_updates: List[Dict[str, Any]] = []

        updated_matrix = read_json_file(project_root / "artifacts" / "expert_matrix.json", {})
        if not isinstance(updated_matrix, dict):
            updated_matrix = {}

        for dim_idx, dimension_name in enumerate(target_dimensions, start=1):
            f1 = 0.0
            acc = 0.0
            prec = 0.0
            rec = 0.0
            model = None
            parameter_count = None
            training_history = []
            training_summary = {}
            cv_metrics_summary: Dict[str, Any] = {}
            validation_metrics_summary: Dict[str, Any] = {}
            weights_ref = None
            runtime_status = "success"
            job_status = "success"
            errors: List[str] = []
            dim_started = time.time()

            print(
                f"[Local Engine] Training binary expert {dim_idx}/{len(target_dimensions)}: "
                f"dimension='{dimension_name}', requested_modality='{modality}', "
                f"class_label='{class_label}', candidate_id='{candidate_id}'."
            )

            try:

                runtime_pre_module = load_python_module(
                    runtime_preprocessing_path,
                    f"arl_runtime_preprocessing_{sanitize_token(dimension_name)}_{dim_idx}",
                )
                if not hasattr(runtime_pre_module, "apply_preprocessing"):
                    raise AttributeError("Generated preprocessing module must define apply_preprocessing")
                apply_preprocessing = runtime_pre_module.apply_preprocessing

                x_train_raw, y_train, x_val_raw, y_val = load_training_data(
                    dataset_path=dataset_path,
                    modality=dimension_name,
                    train_indices=train_indices,
                    val_indices=val_indices,
                )

                x_train_processed = apply_preprocessing(x_train_raw)
                x_val_processed = apply_preprocessing(x_val_raw)

                x_train_input = to_model_input(x_train_processed, signal_name=f"{dimension_name}:train")
                x_val_input = to_model_input(x_val_processed, signal_name=f"{dimension_name}:val")

                class_value = coerce_target_class_value(class_label, y_train)

                f1, acc, prec, rec, model, training_history, training_summary = train_baseline(
                    x_train=x_train_input,
                    y_train=y_train,
                    x_val=x_val_input,
                    y_val=y_val,
                    class_label=class_value,
                    epochs=epochs,
                    patience=patience,
                    lr=lr,
                    project_root=project_root,
                    model_name=f"Cycle {cycle_id} {dimension_name} vs Class {class_label}",
                    model_cls=model_cls,
                    model_init_overrides=model_init_overrides,
                )

                if isinstance(training_summary, dict):
                    cv_metrics_summary = (
                        training_summary.get("cv_metrics")
                        if isinstance(training_summary.get("cv_metrics"), dict)
                        else {}
                    )
                    validation_metrics_summary = (
                        training_summary.get("validation_metrics")
                        if isinstance(training_summary.get("validation_metrics"), dict)
                        else {}
                    )

                if not validation_metrics_summary:
                    validation_metrics_summary = {
                        "accuracy": float(acc),
                        "f1": float(f1),
                        "precision": float(prec),
                        "recall": float(rec),
                        "split": "validate",
                    }

                if model is not None:
                    model_dir = project_root / "models" / f"{dimension_name}_{class_label}"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    model_path = model_dir / f"{candidate_id}.pt"
                    torch.save(model.state_dict(), model_path)
                    weights_ref = to_project_ref(model_path, project_root)

                    try:
                        parameter_count = int(sum(p.numel() for p in model.parameters()))
                    except Exception:
                        parameter_count = None
                else:
                    job_status = "skipped"
                    runtime_status = "skipped"
                    errors.append(
                        "Training skipped because no positive samples were available in train or validation split."
                    )
            except Exception as exc:
                job_status = "failed"
                runtime_status = "failed"
                failure_message = f"Training failed for dimension '{dimension_name}': {exc}"
                errors.append(failure_message)
                tb_text = traceback.format_exc().strip()
                if tb_text:
                    errors.append(f"Traceback:\n{tb_text}")

                print(failure_message)
                if tb_text:
                    print(tb_text)
                model = None
                training_history = []
                training_summary = {}

            train_seconds = float(time.time() - dim_started)
            secondary_curves = build_secondary_curves_from_history(training_history)
            learning_curve_summary = summarize_history_for_results(training_history, training_summary)

            if job_status != "failed":
                modality_group = signal_to_modality.get(dimension_name, dimension_name)
                update_payload, updated_matrix = update_expert_matrix(
                    project_root=project_root,
                    modality=dimension_name,
                    modality_group=modality_group,
                    class_label=class_label,
                    candidate_id=candidate_id,
                    f1=f1,
                    acc=acc,
                    prec=prec,
                    rec=rec,
                    is_success=(job_status == "success"),
                    model_py_ref=model_py_ref,
                    preprocessing_ref=preprocessing_ref,
                    cv_metrics=cv_metrics_summary,
                    validation_metrics=validation_metrics_summary,
                )
                expert_updates.append(update_payload)

            secondary_metrics = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "validation_split": "validate",
                "architecture_code_ref": model_py_ref,
                "preprocessing_code_ref": preprocessing_ref,
                "cross_validation_metrics": cv_metrics_summary,
                "validation_set_metrics": validation_metrics_summary,
                "cv_accuracy_mean": safe_float(cv_metrics_summary.get("accuracy_mean"), 0.0),
                "cv_f1_mean": safe_float(cv_metrics_summary.get("f1_mean"), 0.0),
                "cv_precision_mean": safe_float(cv_metrics_summary.get("precision_mean"), 0.0),
                "cv_recall_mean": safe_float(cv_metrics_summary.get("recall_mean"), 0.0),
                "validation_accuracy": safe_float(validation_metrics_summary.get("accuracy"), float(acc)),
                "validation_f1": safe_float(validation_metrics_summary.get("f1"), float(f1)),
                "validation_precision": safe_float(validation_metrics_summary.get("precision"), float(prec)),
                "validation_recall": safe_float(validation_metrics_summary.get("recall"), float(rec)),
            }
            secondary_metrics.update(secondary_curves)

            modality_group = signal_to_modality.get(dimension_name, dimension_name)
            job_result = {
                "job_id": str(job.get("job_id", f"cycle_{cycle_id}_train_expert")) + f"_{dim_idx:03d}",
                "job_type": "train_expert",
                "target": {
                    "modality": dimension_name,
                    "requested_modality": modality,
                    "modality_group": modality_group,
                    "class_label": class_label,
                },
                "candidate": {
                    "candidate_id": candidate_id,
                    "origin": candidate_origin,
                    "final_model_py_ref": model_py_ref,
                    "best_weights_ref": weights_ref,
                },
                "status": job_status,
                "repair": {
                    "attempted": False,
                    "attempt_count": 0,
                    "final_outcome": "not_needed",
                },
                "compile_status": "success" if job_status != "failed" else "failed",
                "runtime_status": runtime_status,
                "metrics": {
                    "primary_metric": {
                        "name": "f1",
                        "value": float(f1),
                    },
                    "secondary_metrics": secondary_metrics,
                },
                "learning_curve_summary": learning_curve_summary,
                "model_summary": {
                    "parameter_count": parameter_count,
                    "penultimate_dim": embedding_dim,
                },
                "runtime": {
                    "train_seconds": train_seconds,
                    "peak_vram_gb": None,
                },
                "errors": errors,
                "artifacts": {
                    "train_log_ref": None,
                    "metrics_ref": "artifacts/expert_matrix.json",
                    "failure_trace_ref": None,
                },
            }
            job_results.append(job_result)

            print(
                f"[Local Engine] Dimension '{dimension_name}' finished with status={job_status}, "
                f"f1={float(f1):.4f}, accuracy={float(acc):.4f}, "
                f"precision={float(prec):.4f}, recall={float(rec):.4f}."
            )

            if job_status == "failed":
                print(
                    f"Per-dimension job failed for '{dimension_name}'. "
                    "Skipping remaining dimensions and ensemble evaluation to trigger immediate repair."
                )
                if errors:
                    print("Per-dimension failure details:")
                    for err in errors:
                        if str(err).strip():
                            print(f"- {err}")
                break

        jobs_total = len(job_results)
        jobs_succeeded = sum(1 for row in job_results if str(row.get("status", "")).lower() == "success")
        jobs_failed = sum(1 for row in job_results if str(row.get("status", "")).lower() == "failed")
        jobs_skipped = sum(1 for row in job_results if str(row.get("status", "")).lower() == "skipped")

        improved_dimensions = sum(
            1
            for row in expert_updates
            if isinstance(row, dict) and bool(row.get("improved", False))
        )
        current_cycle_improved = improved_dimensions > 0
        ensemble_policy = build_ensemble_retrain_policy(
            project_root=project_root,
            current_cycle=safe_int(cycle_id, 0),
            current_cycle_improved=current_cycle_improved,
            required_successful_improvement_cycles=1,
        )

        if jobs_failed > 0:
            ensemble_eval = {
                "ran": False,
                "subset_fraction": None,
                "status": "skipped",
                "candidate_id_used": None,
                "notes": (
                    "Skipped because one or more per-dimension jobs failed. "
                    "Immediate repair should run before ensemble evaluation."
                ),
                "policy": ensemble_policy,
            }
        else:
            run_ensemble_eval = bool(ensemble_policy.get("should_run_ensemble_this_cycle", False))
            if run_ensemble_eval:
                print(
                    "[Local Engine] Starting end-of-cycle ensemble retraining/evaluation "
                    "because policy is to retrain the ensemble every cycle."
                )
            else:
                print(
                    "[Local Engine] Skipping end-of-cycle ensemble retraining/evaluation "
                    f"because policy decision was '{ensemble_policy.get('decision_reason', 'n/a')}'."
                )
            ensemble_eval = build_ensemble_evaluation_payload(
                project_root=project_root,
                dataset_path=dataset_path,
                directive=directive,
                run_eval=run_ensemble_eval,
                expert_matrix=updated_matrix,
                train_indices=train_indices,
                training_overrides=training_overrides,
            )
            if isinstance(ensemble_eval, dict):
                ensemble_eval["policy"] = ensemble_policy

        if jobs_total <= 0:
            raise RuntimeError(
                f"No dimensions were resolved for target modality '{modality}' and class '{class_label}'."
            )

        if jobs_failed > 0:
            overall_status = "failure"
        elif jobs_succeeded == jobs_total and jobs_failed == 0 and jobs_skipped == 0:
            overall_status = "success"
        elif jobs_succeeded > 0:
            overall_status = "partial_success"
        elif jobs_failed == jobs_total:
            overall_status = "failure"
        else:
            overall_status = "partial_success"

        finished_at = utc_now_iso()
        wall_seconds = float(time.time() - wall_start)

        results_payload = {
            "schema_version": "1.0",
            "directive_id": str(directive.get("directive_id", f"cycle_{cycle_id}_directive")),
            "cycle_id": str(cycle_id),
            "project_id": str(directive.get("project_id", project_id)),
            "started_at": started_at,
            "finished_at": finished_at,
            "overall_status": overall_status,
            "execution_summary": {
                "jobs_total": jobs_total,
                "jobs_succeeded": jobs_succeeded,
                "jobs_failed": jobs_failed,
                "jobs_skipped": jobs_skipped,
                "jobs_repaired": 0,
                "binary_expert_improved": bool(current_cycle_improved),
                "binary_expert_improved_dimensions": int(improved_dimensions),
                "ensemble_policy_required_successful_improvement_cycles": 1,
                "ensemble_policy_improvements_since_last_successful_run": safe_int(
                    ensemble_policy.get("successful_improvement_count_since_last_ensemble", 0),
                    0,
                ),
                "ensemble_policy_improvements_including_current": safe_int(
                    ensemble_policy.get("successful_improvement_count_including_current", 0),
                    0,
                ),
                "wall_time_seconds": wall_seconds,
            },
            "jobs": job_results,
            "ensemble_evaluation": ensemble_eval,
            "expert_matrix_updates": expert_updates,
            "sanitization": {
                "checked": False,
                "status": "approved",
                "redactions_applied": [],
                "reviewed_files": [
                    "shared/outbound/results.json",
                    "shared/outbound/results.md",
                ],
            },
            "notes": (
                "Cycle local engine executed modality-class expansion using director-generated "
                "model/preprocessing modules across resolved dimensions."
            ),
        }

        write_json(results_json_path, results_payload)
        results_md_path.write_text(
            build_results_markdown(
                cycle_id=cycle_id,
                project_id=project_id,
                requested_modality=modality,
                class_label=class_label,
                candidate_id=candidate_id,
                jobs=job_results,
                expert_updates=expert_updates,
                ensemble_eval=ensemble_eval,
            ),
            encoding="utf-8",
        )

        cycle_dir = project_root / "artifacts" / "cycle_history" / f"cycle_{safe_int(cycle_id, 0):04d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        write_json(cycle_dir / "results.json", results_payload)
        (cycle_dir / "results.md").write_text(results_md_path.read_text(encoding="utf-8"), encoding="utf-8")

        print(f"Wrote local engine results: {results_json_path}")
        print(f"Wrote local engine summary: {results_md_path}")
        if jobs_failed > 0:
            print(
                "Per-dimension failure detected. Returning non-zero to trigger immediate upstream repair."
            )
            return 1
        return 0

    except Exception as exc:
        traceback.print_exc()
        error_msg = f"Local engine failed for cycle {cycle_id}: {exc}"
        print(error_msg)

        finished_at = utc_now_iso()
        wall_seconds = float(time.time() - wall_start)

        failure_job = {
            "job_id": f"cycle_{cycle_id}_train_expert_001",
            "job_type": "train_expert",
            "target": {
                "modality": "unknown",
                "class_label": "unknown",
            },
            "candidate": {
                "candidate_id": f"cycle_{cycle_id}_candidate",
                "origin": "director",
            },
            "status": "failed",
            "repair": {
                "attempted": False,
                "attempt_count": 0,
                "final_outcome": "not_attempted",
            },
            "compile_status": "failed",
            "runtime_status": "failed",
            "metrics": {
                "primary_metric": {
                    "name": "f1",
                    "value": 0.0,
                },
                "secondary_metrics": {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                },
            },
            "learning_curve_summary": {
                "best_epoch": None,
                "epochs_completed": 0,
                "early_stopped": False,
                "trend": None,
            },
            "runtime": {
                "train_seconds": 0.0,
                "peak_vram_gb": None,
            },
            "errors": [error_msg],
            "artifacts": {
                "train_log_ref": None,
                "metrics_ref": None,
                "failure_trace_ref": None,
            },
        }

        results_payload = {
            "schema_version": "1.0",
            "directive_id": f"cycle_{cycle_id}_directive",
            "cycle_id": str(cycle_id),
            "project_id": str(project_id),
            "started_at": started_at,
            "finished_at": finished_at,
            "overall_status": "failure",
            "execution_summary": {
                "jobs_total": 1,
                "jobs_succeeded": 0,
                "jobs_failed": 1,
                "jobs_repaired": 0,
                "wall_time_seconds": wall_seconds,
            },
            "jobs": [failure_job],
            "ensemble_evaluation": {
                "ran": False,
                "subset_fraction": None,
                "status": "skipped",
                "candidate_id_used": None,
                "notes": "Skipped due to local engine failure.",
            },
            "expert_matrix_updates": [],
            "sanitization": {
                "checked": False,
                "status": "approved",
                "redactions_applied": [],
                "reviewed_files": [
                    "shared/outbound/results.json",
                    "shared/outbound/results.md",
                ],
            },
            "notes": "Local engine failure payload.",
        }

        try:
            write_json(results_json_path, results_payload)
            results_md_path.parent.mkdir(parents=True, exist_ok=True)
            results_md_path.write_text(
                "# Local Engine Failure\n\n"
                f"- Cycle: `{cycle_id}`\n"
                f"- Error: {error_msg}\n",
                encoding="utf-8",
            )
        except Exception:
            traceback.print_exc()

        return 1
    finally:
        if live_log_path.exists():
            try:
                live_log_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
