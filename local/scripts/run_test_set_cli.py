#!/usr/bin/env python3
"""Memory-safe CLI test-set evaluator for ELD-NAS ensemble projects.

This script avoids loading the full test tensors into memory by streaming
signal data in mini-batches from numpy memmaps. It can run on CPU or GPU.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import cohen_kappa_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app as app_mod  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ELD-NAS test-set evaluation from CLI")
    parser.add_argument("--project", required=True, help="Project name under projects/")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["auto", "cpu", "cuda"],
        help="Inference device for ensemble test evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for streamed inference",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path (default: projects/<project>/artifacts/test_set_ensemble_results_cli.json)",
    )
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=0,
        help="If >0 and using CUDA, run empty_cache every N batches",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N batches (<=0 disables progress output)",
    )
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    whole = int(round(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_progress(
    *,
    batch_idx: int,
    total_batches: int,
    sample_end: int,
    total_samples: int,
    started_at: float,
) -> None:
    elapsed = max(0.0, time.perf_counter() - started_at)
    batch_rate = (float(batch_idx) / elapsed) if elapsed > 0 else 0.0
    remaining_batches = max(0, int(total_batches) - int(batch_idx))
    eta = (remaining_batches / batch_rate) if batch_rate > 0 else float("inf")
    pct = (100.0 * float(batch_idx) / float(total_batches)) if total_batches > 0 else 100.0
    print(
        "[progress] "
        f"batches={batch_idx}/{total_batches} "
        f"({pct:5.1f}%) "
        f"samples={sample_end}/{total_samples} "
        f"elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta)}",
        file=sys.stderr,
        flush=True,
    )


def resolve_device(requested: str) -> torch.device:
    req = str(requested or "auto").strip().lower()
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False")
        return torch.device("cuda")


    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_project_paths(project: str) -> Tuple[Path, Path, Path, Path, Dict[str, Any]]:
    project_root = REPO_ROOT / "projects" / project
    config_path = project_root / "config.yaml"
    if not project_root.exists():
        raise FileNotFoundError(f"Project not found: {project_root}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.yaml: {config_path}")

    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset_path = Path(str(config_data.get("dataset_path", "")).strip())
    if not dataset_path.exists():
        raise FileNotFoundError(f"Configured dataset path does not exist: {dataset_path}")

    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Dataset must contain train/ and test/ splits")

    return project_root, dataset_path, train_dir, test_dir, config_data


def resolve_class_labels(project_root: Path, expert_matrix: Dict[str, Any], train_dir: Path) -> List[str]:
    class_labels = app_mod.resolve_ensemble_class_labels(project_root)
    if class_labels:
        return class_labels

    labels = set()
    for by_class in expert_matrix.values():
        if not isinstance(by_class, dict):
            continue
        for raw in by_class.keys():
            label = app_mod.normalize_class_label(raw)
            if label:
                labels.add(label)

    if not labels:
        y_train_path = train_dir / "y.npy"
        if y_train_path.exists():
            y_train = np.asarray(np.load(y_train_path, mmap_mode="r")).reshape(-1)
            labels = {app_mod.normalize_class_label(v) for v in np.unique(y_train)}
            labels = {x for x in labels if x}

    return sorted(labels, key=app_mod.label_sort_key)


def load_baseline_preprocess_fn() -> Callable[[np.ndarray], np.ndarray]:
    baseline_fn = lambda arr: arr
    baseline_pre_path = (REPO_ROOT / "local" / "scripts" / "cycle_preprocessing.py").resolve()
    if not baseline_pre_path.exists():
        return baseline_fn

    try:
        module = app_mod.load_python_module(
            baseline_pre_path,
            "arl_cli_baseline_preprocessing",
            cache_key="arl_cli_baseline_preprocessing",
        )
        fn = getattr(module, "apply_preprocessing", None)
        if callable(fn):
            return fn
    except Exception:
        pass
    return baseline_fn


def ensure_writable_array(arr: np.ndarray) -> np.ndarray:
    """Return a writable ndarray for preprocessing functions that mutate inputs."""
    np_arr = np.asarray(arr)
    if np_arr.flags.writeable:
        return np_arr
    return np.array(np_arr, copy=True)


def build_block_and_expert_specs(
    *,
    project_root: Path,
    test_dir: Path,
    expert_matrix: Dict[str, Any],
    runtime: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    block_specs: List[Dict[str, Any]] = []
    block_index_map: Dict[Tuple[str, str], int] = {}
    expert_specs: List[Dict[str, Any]] = []

    baseline_preprocess_fn = load_baseline_preprocess_fn()

    signal_order = sorted(
        [str(sig) for sig in expert_matrix.keys() if (test_dir / f"X_{sig}.npy").exists()],
        key=app_mod.label_sort_key,
    )

    for sig in signal_order:
        by_class = expert_matrix.get(sig)
        if not isinstance(by_class, dict):
            continue

        labels = sorted(
            [app_mod.normalize_class_label(v) for v in by_class.keys() if app_mod.normalize_class_label(v)],
            key=app_mod.label_sort_key,
        )

        for class_label in labels:
            rec = by_class.get(class_label)
            if not isinstance(rec, dict):
                rec = by_class.get(str(class_label))
            if not isinstance(rec, dict):
                continue

            candidate_id = str(rec.get("candidate_id", "")).strip()
            if not candidate_id:
                continue

            expert_weights = project_root / "models" / f"{sig}_{class_label}" / f"{candidate_id}.pt"
            if not expert_weights.exists():
                continue

            model_cls = runtime.BinaryExpertModel
            model_ref_path = app_mod.resolve_code_ref_path(rec.get("final_model_py_ref"), project_root)
            if model_ref_path is not None:
                try:
                    model_module = app_mod.load_python_module(
                        model_ref_path,
                        f"arl_cli_model_{abs(hash(str(model_ref_path)))}",
                        cache_key=f"cli_model::{model_ref_path}",
                    )
                    candidate_cls = getattr(model_module, "BinaryExpertModel", None)
                    if candidate_cls is not None:
                        model_cls = candidate_cls
                except Exception:
                    pass

            preprocess_fn = baseline_preprocess_fn
            pre_ref_path = app_mod.resolve_code_ref_path(rec.get("preprocessing_code_ref"), project_root)
            pre_key = "__baseline__"
            if pre_ref_path is not None:
                pre_key = str(pre_ref_path)
                try:
                    pre_module = app_mod.load_python_module(
                        pre_ref_path,
                        f"arl_cli_pre_{abs(hash(str(pre_ref_path)))}",
                        cache_key=f"cli_pre::{pre_ref_path}",
                    )
                    candidate_fn = getattr(pre_module, "apply_preprocessing", None)
                    if callable(candidate_fn):
                        preprocess_fn = candidate_fn
                except Exception:
                    pass

            block_key = (str(sig), pre_key)
            block_idx = block_index_map.get(block_key)
            if block_idx is None:
                raw_path = test_dir / f"X_{sig}.npy"
                if not raw_path.exists():
                    continue
                raw_memmap = np.load(raw_path, mmap_mode="r")
                block_idx = len(block_specs)
                block_index_map[block_key] = block_idx
                block_specs.append(
                    {
                        "signal": str(sig),
                        "preprocess_key": pre_key,
                        "raw": raw_memmap,
                        "preprocess_fn": preprocess_fn,
                    }
                )

            expert_specs.append(
                {
                    "signal": str(sig),
                    "class_label": str(class_label),
                    "candidate_id": candidate_id,
                    "weights_path": expert_weights,
                    "model_cls": model_cls,
                    "block_idx": int(block_idx),
                }
            )

    return block_specs, expert_specs


def instantiate_experts(
    *,
    expert_specs: List[Dict[str, Any]],
    block_specs: List[Dict[str, Any]],
    device: torch.device,
) -> Tuple[List[torch.nn.Module], List[int], List[int]]:
    experts: List[torch.nn.Module] = []
    input_map: List[int] = []
    expert_dims: List[int] = []

    for spec in expert_specs:
        block = block_specs[int(spec["block_idx"])]
        raw_probe = ensure_writable_array(block["raw"][0:1])
        processed_probe = block["preprocess_fn"](raw_probe)
        x_probe = app_mod.to_model_input(
            processed_probe,
            signal_name=f"{spec['signal']}:{spec['class_label']}:probe",
        )

        model = app_mod.build_binary_expert_model(
            model_cls=spec["model_cls"],
            in_ch=int(x_probe.shape[1]),
            n_classes=1,
            min_seq_len=int(x_probe.shape[-1]),
        ).to(device)

        state_dict = torch.load(spec["weights_path"], map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        experts.append(model)
        input_map.append(int(spec["block_idx"]))
        expert_dims.append(app_mod.infer_expert_feature_dim(model, x_probe, torch, default=16))

        del raw_probe, processed_probe, x_probe

    return experts, input_map, expert_dims


def build_ensemble_model(
    *,
    runtime: Any,
    experts: List[torch.nn.Module],
    input_map: List[int],
    class_count: int,
    expert_dims: List[int],
    ensemble_architecture: str,
    device: torch.device,
) -> torch.nn.Module:
    init_sig = inspect.signature(runtime.BaselineEnsemble.__init__)
    kwargs: Dict[str, Any] = {
        "experts": experts,
        "input_map": input_map,
        "num_classes": int(class_count),
        "expert_dim": int(expert_dims[0]) if expert_dims else 16,
    }
    if "expert_dims" in init_sig.parameters and len(set(expert_dims)) > 1:
        kwargs["expert_dims"] = expert_dims
    if "architecture" in init_sig.parameters:
        kwargs["architecture"] = str(ensemble_architecture)

    ensemble = runtime.BaselineEnsemble(**kwargs).to(device)
    ensemble.eval()
    return ensemble


def run_streaming_inference(
    *,
    ensemble: torch.nn.Module,
    block_specs: List[Dict[str, Any]],
    y_idx: np.ndarray,
    valid_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
    empty_cache_every: int,
    progress_every: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = int(y_idx.shape[0])
    current_batch_size = max(1, int(batch_size))
    progress_stride = max(1, int(progress_every)) if int(progress_every) > 0 else 0
    show_progress = progress_stride > 0

    while True:
        preds_chunks: List[np.ndarray] = []
        true_chunks: List[np.ndarray] = []
        total_batches = (n_samples + current_batch_size - 1) // current_batch_size
        batches_seen = 0
        started_at = time.perf_counter()

        if show_progress:
            print(
                f"[progress] starting inference: samples={n_samples} batches={total_batches} "
                f"device={device} batch_size={current_batch_size}",
                file=sys.stderr,
                flush=True,
            )

        try:
            with torch.inference_mode():
                for batch_idx, start in enumerate(range(0, n_samples, current_batch_size), start=1):
                    end = min(start + current_batch_size, n_samples)
                    x_batch_list: List[torch.Tensor] = []

                    for block in block_specs:
                        raw_batch = ensure_writable_array(block["raw"][start:end])
                        processed_batch = block["preprocess_fn"](raw_batch)
                        x_batch = app_mod.to_model_input(
                            processed_batch,
                            signal_name=f"{block['signal']}:batch",
                        )
                        x_batch_list.append(torch.tensor(x_batch, dtype=torch.float32, device=device))

                        del raw_batch, processed_batch, x_batch

                    logits = ensemble(x_batch_list)
                    pred_batch = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)

                    batch_valid = valid_mask[start:end]
                    if np.any(batch_valid):
                        preds_chunks.append(pred_batch[batch_valid])
                        true_chunks.append(y_idx[start:end][batch_valid])

                    del x_batch_list, logits, pred_batch, batch_valid
                    gc.collect()

                    batches_seen = batch_idx
                    if show_progress and (
                        batch_idx == 1 or batch_idx == total_batches or (batch_idx % progress_stride == 0)
                    ):
                        print_progress(
                            batch_idx=batch_idx,
                            total_batches=total_batches,
                            sample_end=end,
                            total_samples=n_samples,
                            started_at=started_at,
                        )

                    if empty_cache_every > 0 and device.type == "cuda" and (batches_seen % empty_cache_every == 0):
                        torch.cuda.empty_cache()

            preds = np.concatenate(preds_chunks) if preds_chunks else np.asarray([], dtype=np.int64)
            true = np.concatenate(true_chunks) if true_chunks else np.asarray([], dtype=np.int64)
            return true, preds

        except torch.OutOfMemoryError:
            if device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            gc.collect()

            if current_batch_size <= 1:
                raise RuntimeError("CUDA out of memory even with inference batch size 1")

            next_batch_size = max(1, current_batch_size // 2)
            print(
                f"[warning] CUDA OOM during test inference; retrying with batch_size={next_batch_size}",
                file=sys.stderr,
                flush=True,
            )
            current_batch_size = next_batch_size


def main() -> int:
    args = parse_args()
    project = str(args.project).strip()
    if not project:
        raise ValueError("--project is required")

    project_root, _, train_dir, test_dir, config_data = load_project_paths(project)
    device = resolve_device(args.device)
    ensemble_architecture = app_mod.normalize_ensemble_architecture(config_data.get("ensemble_architecture", "default"))

    y_test_path = test_dir / "y.npy"
    if not y_test_path.exists():
        raise FileNotFoundError(f"Missing test labels: {y_test_path}")
    y_test = np.asarray(np.load(y_test_path, mmap_mode="r")).reshape(-1)

    expert_matrix_path = project_root / "artifacts" / "expert_matrix.json"
    if not expert_matrix_path.exists():
        raise FileNotFoundError(f"Missing expert matrix: {expert_matrix_path}")
    expert_matrix = json.loads(expert_matrix_path.read_text(encoding="utf-8"))
    if not isinstance(expert_matrix, dict):
        raise RuntimeError("Invalid expert_matrix.json format")

    class_labels = resolve_class_labels(project_root, expert_matrix, train_dir)
    if not class_labels:
        raise RuntimeError("Could not resolve class labels for ensemble evaluation")

    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    y_norm = np.array([app_mod.normalize_class_label(v) for v in y_test], dtype=object)
    y_idx = np.array([label_to_idx.get(lbl, -1) for lbl in y_norm], dtype=np.int64)
    valid_mask = y_idx >= 0
    if int(np.sum(valid_mask)) <= 0:
        raise RuntimeError("No valid test labels matched the ensemble class mapping")

    runtime = app_mod.load_cycle0_runtime_module()
    block_specs, expert_specs = build_block_and_expert_specs(
        project_root=project_root,
        test_dir=test_dir,
        expert_matrix=expert_matrix,
        runtime=runtime,
    )
    if not expert_specs:
        raise RuntimeError("No expert checkpoints were found for test-set evaluation")

    experts = []
    ensemble = None
    try:
        experts, input_map, expert_dims = instantiate_experts(
            expert_specs=expert_specs,
            block_specs=block_specs,
            device=device,
        )

        candidate_id, ensemble_weights = app_mod.resolve_best_ensemble_weights(project_root)
        if ensemble_weights is None:
            raise RuntimeError("No ensemble checkpoint found (expected models/baseline_ensemble.pt)")

        ensemble = build_ensemble_model(
            runtime=runtime,
            experts=experts,
            input_map=input_map,
            class_count=len(class_labels),
            expert_dims=expert_dims,
            ensemble_architecture=ensemble_architecture,
            device=device,
        )
        app_mod.load_ensemble_mlp_weights_from_checkpoint(ensemble, ensemble_weights, torch)

        true, pred = run_streaming_inference(
            ensemble=ensemble,
            block_specs=block_specs,
            y_idx=y_idx,
            valid_mask=valid_mask,
            batch_size=max(1, int(args.batch_size)),
            device=device,
            empty_cache_every=max(0, int(args.empty_cache_every)),
            progress_every=int(args.progress_every),
        )
        if pred.size <= 0:
            raise RuntimeError("No valid predictions produced during test-set evaluation")

        accuracy = float(np.mean(pred == true))
        kappa_raw = cohen_kappa_score(true, pred) if np.unique(true).size > 1 else 0.0
        kappa = float(kappa_raw) if np.isfinite(kappa_raw) else 0.0

        payload = {
            "status": "success",
            "project": project,
            "ensemble_candidate_id": str(candidate_id or "baseline_ensemble"),
            "evaluation_device": str(device),
            "ensemble_architecture": ensemble_architecture,
            "batch_size": int(args.batch_size),
            "evaluated_samples": int(pred.size),
            "ignored_samples": int(len(y_idx) - int(np.sum(valid_mask))),
            "metrics": {
                "accuracy": accuracy,
                "kappa": kappa,
            },
        }

        output_path = (
            Path(args.output).expanduser().resolve()
            if str(args.output).strip()
            else project_root / "artifacts" / "test_set_ensemble_results_cli.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        print(f"project={project}")
        print(f"ensemble_candidate={payload['ensemble_candidate_id']}")
        print(f"evaluation_device={payload['evaluation_device']}")
        print(f"ensemble_architecture={payload['ensemble_architecture']}")
        print(f"evaluated_samples={payload['evaluated_samples']}")
        print(f"accuracy={accuracy:.6f}")
        print(f"kappa={kappa:.6f}")
        print(f"output={output_path}")
        return 0
    finally:

        ensemble = None
        experts = None
        block_specs = None  # type: ignore[assignment]
        gc.collect()
        app_mod.release_cuda_cache(torch)


if __name__ == "__main__":
    raise SystemExit(main())