import os
import sys
import time
import argparse
import subprocess
import shutil
import base64
import yaml
import numpy as np
import torch
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from data_analyzer import compute_channel_statistics
from db import get_engine_and_session, ProjectState, ExecutionLog

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # type: ignore[assignment]


def _load_env_file() -> None:
    if load_dotenv is None:
        return

    repo_root = Path(__file__).resolve().parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv(override=False)


_load_env_file()


LLM_ROLE_NAMES = ("director",)
DEFAULT_LLM_ROLE_EXECUTION = {role: "remote" for role in LLM_ROLE_NAMES}
STEP_TO_LLM_ROLE = {
    "Director": "director",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def sorted_key(value):
    text = str(value)
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def to_project_ref(path: Path, project_root: Path):
    try:
        return str(path.relative_to(project_root)).replace(os.sep, "/")
    except ValueError:
        return str(path)


def format_cycle_label(cycle):
    try:
        return f"cycle_{int(cycle):04d}"
    except (TypeError, ValueError):
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(cycle)).strip("_")
        return f"cycle_{safe or 'unknown'}"


def manual_verification_state_path(project_root: Path) -> Path:
    return project_root / "state" / "manual_verification.json"


def load_manual_verification_state(project_root: Path):
    default_state = {
        "enabled": False,
        "confirmed_log_ids": [],
    }

    path = manual_verification_state_path(project_root)
    if not path.exists():
        return default_state

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return default_state
        confirmed_ids = sorted({int(x) for x in raw.get("confirmed_log_ids", [])})
        return {
            "enabled": bool(raw.get("enabled", False)),
            "confirmed_log_ids": confirmed_ids,
        }
    except Exception:
        return default_state


def wait_for_manual_confirmation(project_name, project_root, session, cycle, step_name, log_id):
    if not (project_name and project_root and session and log_id is not None):
        return

    state = load_manual_verification_state(project_root)
    if not state.get("enabled", False):
        return

    proj_state = session.get(ProjectState, project_name)
    if proj_state:
        proj_state.current_step = f"Waiting for confirmation: {step_name}"
        session.commit()

    print(
        f"Manual verification enabled. Waiting for confirmation of {step_name} "
        f"(cycle {cycle}, log_id={log_id})."
    )

    while True:
        state = load_manual_verification_state(project_root)
        if not state.get("enabled", False):
            print("Manual verification disabled while waiting. Continuing.")
            return

        confirmed_ids = set(state.get("confirmed_log_ids", []))
        if int(log_id) in confirmed_ids:
            print(f"Step confirmed for {step_name} (log_id={log_id}).")
            return

        time.sleep(1)


def honor_pause_or_stop_signal(session, project_name, cycle):
    """Honor UI control state changes between orchestration steps."""
    if not (session and project_name):
        return "continue"

    pause_announced = False
    while True:
        proj_state = session.get(ProjectState, project_name)
        if not proj_state:
            return "continue"

        target_status = str(getattr(proj_state, "target_status", "") or "").strip()
        if target_status == "Stopped":
            print("\nReceived Stop signal from UI. Exiting gracefully.")
            proj_state.status = "Stopped"
            proj_state.current_step = "Idle"
            proj_state.pid = None
            session.commit()
            return "stopped"

        if target_status == "Paused":
            if not pause_announced:
                print("\nPause requested from UI. Waiting for Start to resume.")
                pause_announced = True

            if str(getattr(proj_state, "status", "") or "") != "Paused":
                proj_state.status = "Paused"
                proj_state.current_step = "Paused"
                proj_state.pid = os.getpid()
                session.commit()

            time.sleep(1)
            session.expire_all()
            continue

        if pause_announced:
            print("Resuming run after pause.")
        if str(getattr(proj_state, "status", "") or "") == "Paused":
            proj_state.status = "Running"
            if str(getattr(proj_state, "current_step", "") or "").strip().lower() == "paused":
                proj_state.current_step = "Resuming"
            proj_state.pid = os.getpid()
            session.commit()

        return "continue"


def parse_iso_datetime(value):
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def coerce_utc_datetime(value):
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def read_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        return loaded
    except Exception:
        return default


def normalize_llm_backend(value):
    text = str(value or "").strip().lower()
    return "local" if text == "local" else "remote"


def normalize_llm_role_execution(value):
    normalized = dict(DEFAULT_LLM_ROLE_EXECUTION)
    if not isinstance(value, dict):
        return normalized

    lowered = {str(k).strip().lower(): v for k, v in value.items()}
    for role in LLM_ROLE_NAMES:
        if role in lowered:
            normalized[role] = normalize_llm_backend(lowered.get(role))
    return normalized


def load_project_llm_role_execution(project_root: Path):
    settings_payload = read_json_file(project_root / "state" / "project_settings.json", {})
    if not isinstance(settings_payload, dict):
        return dict(DEFAULT_LLM_ROLE_EXECUTION)
    return normalize_llm_role_execution(settings_payload.get("llm_role_execution"))


def first_env_value(env: dict, keys, default=None):
    for key in keys:
        raw = env.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            return text
    return default


def apply_llm_profile_env(env: dict, backend: str):
    backend_norm = normalize_llm_backend(backend)

    if backend_norm == "local":
        api_url = first_env_value(
            env,
            ["ARL_LOCAL_LLM_API_URL", "ARL_LOCAL_LLM_API_BASE_URL"],
            default="http://localhost:11434/v1/responses",
        )
        api_key = first_env_value(
            env,
            ["ARL_LOCAL_LLM_API_KEY", "ARL_LLM_API_KEY"],
            default="local-placeholder-key",
        )
        model = first_env_value(
            env,
            ["ARL_LOCAL_LLM_MODEL", "ARL_OLLAMA_MODEL", "ARL_LLM_MODEL"],
            default="",
        )
        temperature = first_env_value(env, ["ARL_LOCAL_LLM_TEMPERATURE", "ARL_LLM_TEMPERATURE"])
        max_tokens = first_env_value(env, ["ARL_LOCAL_LLM_MAX_TOKENS", "ARL_LLM_MAX_TOKENS"])
        reasoning_effort = first_env_value(env, ["ARL_LOCAL_REASONING_EFFORT", "ARL_REASONING_EFFORT"])
    else:
        api_url = first_env_value(
            env,
            ["ARL_REMOTE_LLM_API_URL", "ARL_LLM_API_URL"],
            default="https://api.openai.com/v1/responses",
        )
        api_key = first_env_value(env, ["ARL_REMOTE_LLM_API_KEY", "ARL_LLM_API_KEY"], default="")
        model = first_env_value(env, ["ARL_REMOTE_LLM_MODEL", "ARL_LLM_MODEL"], default="")
        temperature = first_env_value(env, ["ARL_REMOTE_LLM_TEMPERATURE", "ARL_LLM_TEMPERATURE"])
        max_tokens = first_env_value(env, ["ARL_REMOTE_LLM_MAX_TOKENS", "ARL_LLM_MAX_TOKENS"])
        reasoning_effort = first_env_value(env, ["ARL_REMOTE_REASONING_EFFORT", "ARL_REASONING_EFFORT"])

    if api_url is not None:
        env["ARL_LLM_API_URL"] = str(api_url)
    if api_key is not None:
        env["ARL_LLM_API_KEY"] = str(api_key)
    if model is not None:
        env["ARL_LLM_MODEL"] = str(model)
    if temperature is not None:
        env["ARL_LLM_TEMPERATURE"] = str(temperature)
    if max_tokens is not None:
        env["ARL_LLM_MAX_TOKENS"] = str(max_tokens)
    if reasoning_effort is not None:
        env["ARL_REASONING_EFFORT"] = str(reasoning_effort)

    env["ARL_LLM_EXECUTION_BACKEND"] = backend_norm

    return {
        "backend": backend_norm,
        "api_url": str(api_url or ""),
        "model": str(model or ""),
        "has_api_key": bool(str(api_key or "").strip()),
    }


def read_yaml_file(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        return loaded if loaded is not None else default
    except Exception:
        return default


def first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict, tuple)) and len(value) == 0:
            continue
        return value
    return None


def get_path_value(data, path):
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        if key not in current:
            return None
        current = current.get(key)
    return current


def pick_first_path(data, paths):
    for path in paths:
        value = get_path_value(data, path)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict, tuple)) and len(value) == 0:
            continue
        return value
    return None


def extract_job_curves(job):
    secondary = job.get("metrics", {}).get("secondary_metrics", {})
    if not isinstance(secondary, dict):
        secondary = {}
    learning_summary = job.get("learning_curve_summary", {})
    if not isinstance(learning_summary, dict):
        learning_summary = {}

    train_loss = first_non_empty(
        pick_first_path(
            secondary,
            [
                ("train_loss_curve",),
                ("training_loss_curve",),
                ("train_loss",),
                ("curves", "train_loss"),
                ("learning_curves", "train_loss"),
            ],
        ),
        pick_first_path(learning_summary, [("train_loss_curve",), ("train_loss",)]),
    )
    val_loss = first_non_empty(
        pick_first_path(
            secondary,
            [
                ("val_loss_curve",),
                ("validation_loss_curve",),
                ("valid_loss_curve",),
                ("val_loss",),
                ("curves", "val_loss"),
                ("learning_curves", "val_loss"),
            ],
        ),
        pick_first_path(learning_summary, [("val_loss_curve",), ("val_loss",)]),
    )
    train_metrics = pick_first_path(
        secondary,
        [
            ("train_metric_curves",),
            ("train_metrics",),
            ("curves", "train_metrics"),
            ("learning_curves", "train_metrics"),
        ],
    )
    val_metrics = pick_first_path(
        secondary,
        [
            ("val_metric_curves",),
            ("validation_metric_curves",),
            ("val_metrics",),
            ("curves", "val_metrics"),
            ("learning_curves", "val_metrics"),
        ],
    )

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


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

        train_metric_curves["accuracy"].append(safe_float(first_non_empty(row.get("train_accuracy"), row.get("accuracy"), 0.0)))
        train_metric_curves["f1"].append(safe_float(first_non_empty(row.get("train_f1"), row.get("f1"), 0.0)))
        train_metric_curves["precision"].append(safe_float(first_non_empty(row.get("train_precision"), row.get("precision"), 0.0)))
        train_metric_curves["recall"].append(safe_float(first_non_empty(row.get("train_recall"), row.get("recall"), 0.0)))

        val_metric_curves["accuracy"].append(safe_float(first_non_empty(row.get("val_accuracy"), row.get("accuracy"), 0.0)))
        val_metric_curves["f1"].append(safe_float(first_non_empty(row.get("val_f1"), row.get("f1"), 0.0)))
        val_metric_curves["precision"].append(safe_float(first_non_empty(row.get("val_precision"), row.get("precision"), 0.0)))
        val_metric_curves["recall"].append(safe_float(first_non_empty(row.get("val_recall"), row.get("recall"), 0.0)))

    return {
        "train_loss_curve": train_loss_curve,
        "val_loss_curve": val_loss_curve,
        "train_metric_curves": train_metric_curves,
        "val_metric_curves": val_metric_curves,
    }


def reflection_curve_view_from_secondary(curves_payload):
    if not isinstance(curves_payload, dict):
        return {}
    return {
        "train_loss": curves_payload.get("train_loss_curve"),
        "val_loss": curves_payload.get("val_loss_curve"),
        "train_metrics": curves_payload.get("train_metric_curves"),
        "val_metrics": curves_payload.get("val_metric_curves"),
    }


def summarize_history_for_results(history, raw_summary=None):
    safe_summary = raw_summary if isinstance(raw_summary, dict) else {}
    if not isinstance(history, list):
        history = []

    epochs_completed = safe_int(first_non_empty(safe_summary.get("epochs_completed"), len(history)), 0)
    best_epoch_raw = safe_summary.get("best_epoch")
    best_epoch = safe_int(best_epoch_raw, 0) if best_epoch_raw is not None else None
    early_stopped = bool(safe_summary.get("early_stopped", False))
    trend = safe_summary.get("trend")

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


def extract_section(content: str, section_title: str):
    marker = f"## {section_title}"
    start = content.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = content.find("\n## ", start)
    if end < 0:
        end = len(content)
    return content[start:end].strip()


def parse_data_context(project_root: Path):
    out = {
        "objective": "",
        "signals": {},
        "classes": {},
    }

    context_path = project_root / "shared" / "context" / "data_context.md"
    if not context_path.exists():
        return out

    try:
        content = context_path.read_text(encoding="utf-8")
    except Exception:
        return out

    objective = ""
    if "# Project Objective" in content:
        start = content.find("# Project Objective") + len("# Project Objective")
        end = content.find("\n## ", start)
        if end < 0:
            end = len(content)
        objective = content[start:end].strip()
    out["objective"] = objective

    signals_section = extract_section(content, "Signal Features (X)")
    for raw_line in signals_section.splitlines():
        line = raw_line.strip()
        if not line.startswith("- **"):
            continue
        match = re.match(r"^- \*\*(.+?)\*\*: \[Modality: (.+?)\] (.*)$", line)
        if not match:
            continue
        signal_name, modality, description = match.groups()
        signal_name = signal_name.strip()
        modality_text = modality.strip()
        if modality_text.lower() in {"", "unspecified", "unknown", "n/a", "na", "none", "null"}:
            modality_text = signal_name
        out["signals"][signal_name.strip()] = {
            "modality": modality_text,
            "description": description.strip(),
        }

    classes_section = extract_section(content, "Output Labels (y)")
    for raw_line in classes_section.splitlines():
        line = raw_line.strip()
        if not line.startswith("- **Class"):
            continue
        match = re.match(r"^- \*\*Class\s+(.+?)\*\*:\s*(.*)$", line)
        if not match:
            continue
        class_id, description = match.groups()
        out["classes"][str(class_id).strip()] = description.strip()

    return out


def summarize_dataset_structure(dataset_path: Path):
    summary = {
        "signals": [],
        "class_labels": [],
        "sequence_notes": "N/A",
    }

    train_dir = dataset_path / "train"
    if not train_dir.exists():
        return summary

    signal_files = sorted(train_dir.glob("X_*.npy"), key=lambda p: p.name)
    signals = [p.stem[2:] for p in signal_files]
    summary["signals"] = signals

    y_path = train_dir / "y.npy"
    if y_path.exists():
        try:
            y_values = np.load(y_path, mmap_mode="r")
            labels = sorted({str(int(x)) for x in np.unique(y_values)}, key=sorted_key)
            summary["class_labels"] = labels
        except Exception:
            pass

    if signal_files:
        try:
            x0 = np.load(signal_files[0], mmap_mode="r")
            if hasattr(x0, "shape") and len(x0.shape) >= 2:
                sample_count = int(x0.shape[0])
                temporal_shape = "x".join(str(int(v)) for v in x0.shape[1:])
                summary["sequence_notes"] = (
                    f"Train split shape per signal is approximately (N={sample_count}, T={temporal_shape})."
                )
            elif hasattr(x0, "shape"):
                summary["sequence_notes"] = f"Train split shape per signal is approximately {tuple(int(v) for v in x0.shape)}."
        except Exception:
            pass

    return summary


def _normalize_class_id_token(value):
    if isinstance(value, np.generic):
        value = value.item()

    text = str(value).strip()
    if not text:
        return ""

    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass

    return text


def _extract_class_label_text(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, np.generic)):
        return str(value).strip()
    if isinstance(value, dict):
        for key in ("label", "name", "class_label", "class_name", "description", "value"):
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
            if isinstance(raw, (int, float, np.generic)):
                return str(raw).strip()
    return ""


def _load_dataset_class_label_map(dataset_path: Path):
    label_map = {}
    payload = read_json_file(dataset_path / "classes.json", None)

    def upsert(class_id, class_label):
        key = _normalize_class_id_token(class_id)
        label = _extract_class_label_text(class_label)
        if key and label:
            label_map[key] = label

    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            upsert(idx, value)
        return label_map

    if isinstance(payload, dict):
        list_keys = ("labels", "class_labels", "classes")
        for list_key in list_keys:
            list_payload = payload.get(list_key)
            if isinstance(list_payload, list):
                for idx, value in enumerate(list_payload):
                    upsert(idx, value)

        for raw_key, raw_value in payload.items():
            if raw_key in list_keys:
                continue
            upsert(raw_key, raw_value)

    return label_map


def _build_class_id_to_label_map(dataset_path: Path, data_context: dict, class_ids):
    label_map = _load_dataset_class_label_map(dataset_path)

    classes_from_context = data_context.get("classes") if isinstance(data_context, dict) else {}
    if isinstance(classes_from_context, dict):
        for raw_key, raw_value in classes_from_context.items():
            key = _normalize_class_id_token(raw_key)
            label = _extract_class_label_text(raw_value)
            if key and label and key not in label_map:
                label_map[key] = label

    for class_id in class_ids:
        key = _normalize_class_id_token(class_id)
        if key and key not in label_map:
            label_map[key] = key

    return label_map


def flatten_best_binary_experts(project_root: Path, results_payload):
    expert_matrix = read_json_file(project_root / "artifacts" / "expert_matrix.json", {})
    experts = []
    job_curve_map = {}
    fallback_curve_map = {}

    if isinstance(results_payload, dict):
        jobs = results_payload.get("jobs", [])
        if isinstance(jobs, list):
            for job in jobs:
                if not isinstance(job, dict):
                    continue
                target = job.get("target", {})
                if not isinstance(target, dict):
                    continue
                modality = str(target.get("modality", "")).strip()
                class_label = str(target.get("class_label", "")).strip()
                if not modality or not class_label:
                    continue
                job_curve_map[(modality, class_label)] = extract_job_curves(job)

    fallback_payload = read_json_file(project_root / "artifacts" / "training_curves_cycle0.json", {})
    if isinstance(fallback_payload, dict):
        fallback_rows = fallback_payload.get("experts", [])
        if isinstance(fallback_rows, list):
            for row in fallback_rows:
                if not isinstance(row, dict):
                    continue
                modality = str(row.get("modality", "")).strip()
                class_label = str(row.get("class_label", "")).strip()
                if not modality or not class_label:
                    continue
                curves_payload = build_secondary_curves_from_history(row.get("history"))
                fallback_curve_map[(modality, class_label)] = reflection_curve_view_from_secondary(curves_payload)

    if not isinstance(expert_matrix, dict):
        return experts

    for modality in sorted(expert_matrix.keys(), key=sorted_key):
        per_class = expert_matrix.get(modality, {})
        if not isinstance(per_class, dict):
            continue

        for class_label in sorted(per_class.keys(), key=sorted_key):
            rec = per_class.get(class_label, {})
            if not isinstance(rec, dict):
                continue

            row_modality = str(rec.get("modality", modality))
            row_class = str(rec.get("class_label", class_label))
            curves_from_results = job_curve_map.get((row_modality, row_class), {})
            if not curves_from_results:
                curves_from_results = fallback_curve_map.get((row_modality, row_class), {})

            experts.append(
                {
                    "modality": row_modality,
                    "class_label": row_class,
                    "candidate_id": str(rec.get("candidate_id", f"best_{modality}_{class_label}")),
                    "accuracy": safe_float(rec.get("accuracy", 0.0)),
                    "precision": safe_float(rec.get("precision", 0.0)),
                    "recall": safe_float(rec.get("recall", 0.0)),
                    "f1": safe_float(rec.get("f1", 0.0)),
                    "eval_split": "validate",
                    "curves": {
                        "train_loss": first_non_empty(
                            curves_from_results.get("train_loss"),
                            "[TODO: Add full train loss curve from local engine output]",
                        ),
                        "val_loss": first_non_empty(
                            curves_from_results.get("val_loss"),
                            "[TODO: Add full validation loss curve from local engine output]",
                        ),
                        "train_metrics": first_non_empty(
                            curves_from_results.get("train_metrics"),
                            "[TODO: Add full train metric curves from local engine output]",
                        ),
                        "val_metrics": first_non_empty(
                            curves_from_results.get("val_metrics"),
                            "[TODO: Add full validation metric curves from local engine output]",
                        ),
                    },
                }
            )

    return experts


def extract_ensemble_metrics(results_payload, project_root: Path):
    best_ensemble = {
        "name": "N/A",
        "eval_split": "validate",
        "metrics": {},
        "curves": {
            "train_loss": "[TODO: Add full ensemble train loss curve if available]",
            "val_loss": "[TODO: Add full ensemble validation loss curve if available]",
            "train_metrics": "[TODO: Add full ensemble train metric curves if available]",
            "val_metrics": "[TODO: Add full ensemble validation metric curves if available]",
        },
    }

    ensemble_eval = results_payload.get("ensemble_evaluation", {}) if isinstance(results_payload, dict) else {}
    if isinstance(ensemble_eval, dict):
        candidate_id = ensemble_eval.get("candidate_id_used")
        if candidate_id:
            best_ensemble["name"] = str(candidate_id)

        metric_rows = ensemble_eval.get("metrics", [])
        if isinstance(metric_rows, list):
            for metric in metric_rows:
                if not isinstance(metric, dict):
                    continue
                name = metric.get("name")
                if not name:
                    continue
                best_ensemble["metrics"][str(name)] = safe_float(metric.get("value", 0.0))

        best_ensemble["curves"]["train_loss"] = first_non_empty(
            pick_first_path(
                ensemble_eval,
                [
                    ("train_loss_curve",),
                    ("train_loss",),
                    ("curves", "train_loss"),
                    ("learning_curves", "train_loss"),
                ],
            ),
            best_ensemble["curves"]["train_loss"],
        )
        best_ensemble["curves"]["val_loss"] = first_non_empty(
            pick_first_path(
                ensemble_eval,
                [
                    ("val_loss_curve",),
                    ("val_loss",),
                    ("curves", "val_loss"),
                    ("learning_curves", "val_loss"),
                ],
            ),
            best_ensemble["curves"]["val_loss"],
        )
        best_ensemble["curves"]["train_metrics"] = first_non_empty(
            pick_first_path(
                ensemble_eval,
                [
                    ("train_metric_curves",),
                    ("train_metrics",),
                    ("curves", "train_metrics"),
                    ("learning_curves", "train_metrics"),
                ],
            ),
            best_ensemble["curves"]["train_metrics"],
        )
        best_ensemble["curves"]["val_metrics"] = first_non_empty(
            pick_first_path(
                ensemble_eval,
                [
                    ("val_metric_curves",),
                    ("val_metrics",),
                    ("curves", "val_metrics"),
                    ("learning_curves", "val_metrics"),
                ],
            ),
            best_ensemble["curves"]["val_metrics"],
        )

    baseline_path = project_root / "artifacts" / "baseline_ensemble_metrics.json"
    baseline = read_json_file(baseline_path, {})
    if isinstance(baseline, dict):
        if "accuracy" in baseline:
            best_ensemble["metrics"].setdefault("accuracy", safe_float(baseline.get("accuracy", 0.0)))
        if "kappa" in baseline:
            best_ensemble["metrics"].setdefault("kappa", safe_float(baseline.get("kappa", 0.0)))

        report = baseline.get("classification_report")
        if isinstance(report, dict):
            macro = report.get("macro avg")
            if isinstance(macro, dict):
                if "f1-score" in macro:
                    best_ensemble["metrics"].setdefault("macro_f1", safe_float(macro.get("f1-score", 0.0)))
                if "precision" in macro:
                    best_ensemble["metrics"].setdefault("macro_precision", safe_float(macro.get("precision", 0.0)))
                if "recall" in macro:
                    best_ensemble["metrics"].setdefault("macro_recall", safe_float(macro.get("recall", 0.0)))

        baseline_history = baseline.get("training_history")
        curves_from_baseline = build_secondary_curves_from_history(baseline_history)
        curve_view = reflection_curve_view_from_secondary(curves_from_baseline)
        best_ensemble["curves"]["train_loss"] = first_non_empty(
            curve_view.get("train_loss"),
            best_ensemble["curves"]["train_loss"],
        )
        best_ensemble["curves"]["val_loss"] = first_non_empty(
            curve_view.get("val_loss"),
            best_ensemble["curves"]["val_loss"],
        )
        best_ensemble["curves"]["train_metrics"] = first_non_empty(
            curve_view.get("train_metrics"),
            best_ensemble["curves"]["train_metrics"],
        )
        best_ensemble["curves"]["val_metrics"] = first_non_empty(
            curve_view.get("val_metrics"),
            best_ensemble["curves"]["val_metrics"],
        )

    return best_ensemble


def compute_weakest_link_classes(binary_experts, class_descriptions, top_k=3, class_focus_counts=None):
    grouped = {}

    for expert in binary_experts:
        cls = str(expert.get("class_label", "unknown"))
        bucket = grouped.setdefault(
            cls,
            {
                "f1": [],
                "recall": [],
            },
        )
        f1 = safe_float(expert.get("f1", 0.0))
        rec_score = safe_float(expert.get("recall", 0.0))
        bucket["f1"].append(f1)
        bucket["recall"].append(rec_score)

    rows = []
    for class_label, bucket in grouped.items():
        mean_f1 = float(np.mean(bucket["f1"])) if bucket["f1"] else 0.0
        mean_recall = float(np.mean(bucket["recall"])) if bucket["recall"] else 0.0
        class_desc = class_descriptions.get(class_label, "")

        rows.append(
            {
                "class_label": class_label,
                "weakness_summary": f"Lowest mean expert F1 across modalities ({len(bucket['f1'])} experts).",
                "f1": round(mean_f1, 4),
                "recall": round(mean_recall, 4),
                "cycles_focused": (
                    safe_int(class_focus_counts.get(class_label, 0), 0)
                    if isinstance(class_focus_counts, dict)
                    else 0
                ),
                "notes": f"description: {class_desc}" if class_desc else "N/A",
            }
        )

    rows.sort(key=lambda r: (safe_float(r.get("f1", 0.0)), sorted_key(r.get("class_label", ""))))
    return rows[: max(1, int(top_k))]


def extract_focus_classes_from_directive(directive_payload):
    focused = set()
    if not isinstance(directive_payload, dict):
        return focused

    jobs = directive_payload.get("jobs", [])
    if not isinstance(jobs, list):
        return focused

    for job in jobs:
        if not isinstance(job, dict):
            continue
        target = job.get("target", {})
        if not isinstance(target, dict):
            continue
        class_label = target.get("class_label")
        if class_label is None:
            continue
        class_text = str(class_label).strip()
        if class_text:
            focused.add(class_text)

    return focused


def extract_focus_classes_from_manifest(manifest_path: Path, known_class_labels):
    focused = set()
    if not manifest_path.exists():
        return focused

    try:
        text = manifest_path.read_text(encoding="utf-8")
    except Exception:
        return focused

    labels = sorted({str(x).strip() for x in (known_class_labels or []) if str(x).strip()}, key=sorted_key)
    if labels:
        for label in labels:
            escaped = re.escape(label)
            if label.isdigit():
                patterns = [
                    rf"\\bclass\\s+{escaped}\\b",
                    rf"\\bclass[_\\s-]*label\\s*[:=]?\\s*{escaped}\\b",
                    rf"\\(\\s*[^,\\n\\r]+\\s*,\\s*{escaped}\\s*\\)",
                ]
            else:
                patterns = [
                    rf"\\bclass\\s+{escaped}\\b",
                    rf"\\b{escaped}\\b",
                ]

            if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
                focused.add(label)
        return focused

    for match in re.finditer(r"\\bclass\\s+([0-9]+)\\b", text, flags=re.IGNORECASE):
        focused.add(str(match.group(1)))
    return focused


def compute_class_focus_counts(project_root: Path, current_cycle, directive_payload, known_class_labels):
    counts = {
        str(label): 0
        for label in sorted({str(x).strip() for x in (known_class_labels or []) if str(x).strip()}, key=sorted_key)
    }

    cycle_ids = gather_cycle_ids(project_root, current_cycle)
    current_cycle_int = safe_int(current_cycle, 0)

    for cycle_id in cycle_ids:
        focused = set()
        cycle_dir_name = format_cycle_label(cycle_id)

        if cycle_id == current_cycle_int and isinstance(directive_payload, dict):
            focused.update(extract_focus_classes_from_directive(directive_payload))

        candidate_directive_paths = [
            project_root / "artifacts" / "cycle_history" / cycle_dir_name / "directive.json",
            project_root / "artifacts" / "cycle_history" / cycle_dir_name / "directive_snapshot.json",
            project_root / "artifacts" / "cycle_history" / cycle_dir_name / "inbound" / "directive.json",
            project_root / "shared" / "context" / "manifests" / cycle_dir_name / "directive.json",
        ]

        for directive_path in candidate_directive_paths:
            payload = read_json_file(directive_path, None)
            if isinstance(payload, dict):
                focused.update(extract_focus_classes_from_directive(payload))

        if not focused:
            manifest_path = project_root / "shared" / "context" / "manifests" / cycle_dir_name / "manifest.md"
            focused.update(extract_focus_classes_from_manifest(manifest_path, known_class_labels))

        for class_label in focused:
            class_text = str(class_label).strip()
            if not class_text:
                continue
            counts[class_text] = counts.get(class_text, 0) + 1

    return counts


def extract_manifest_focus(manifest_path: Path):
    if not manifest_path.exists():
        return "N/A"

    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return "N/A"

    for idx, line in enumerate(lines):
        if line.strip().startswith("# Cycle") and "Objective" in line:
            for next_line in lines[idx + 1 :]:
                stripped = next_line.strip()
                if stripped and not stripped.startswith("#"):
                    return stripped[2:].strip() if stripped.startswith("- ") else stripped

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[2:].strip() if stripped.startswith("- ") else stripped
    return "N/A"


def gather_cycle_ids(project_root: Path, fallback_cycle):
    cycle_ids = set()
    manifests_root = project_root / "shared" / "context" / "manifests"

    if manifests_root.exists():
        for manifest_file in manifests_root.glob("cycle_*/manifest.md"):
            parent = manifest_file.parent.name
            match = re.match(r"^cycle_(\d+)$", parent)
            if match:
                cycle_ids.add(int(match.group(1)))

    cycle_ids.add(safe_int(fallback_cycle, 0))
    cycle_ids.add(0)
    return sorted(cycle_ids)


def compute_cycles_researched(session, project_name, fallback_cycle):
    if not session or not project_name:
        return safe_int(fallback_cycle, 0) + 1

    try:
        rows = (
            session.query(ExecutionLog.cycle)
            .filter(ExecutionLog.project_name == project_name)
            .all()
        )
        cycle_ids = {safe_int(row[0], 0) for row in rows if row and row[0] is not None}
        if cycle_ids:
            return len(cycle_ids)
    except Exception:
        pass

    return safe_int(fallback_cycle, 0) + 1


def compute_hours_elapsed(session, project_name, results_payload):
    start_time = None

    if session and project_name:
        try:
            first_row = (
                session.query(ExecutionLog)
                .filter(ExecutionLog.project_name == project_name)
                .order_by(ExecutionLog.timestamp.asc())
                .first()
            )
            if first_row and first_row.timestamp:
                start_time = coerce_utc_datetime(first_row.timestamp)
        except Exception:
            pass

    if start_time is None and isinstance(results_payload, dict):
        start_time = parse_iso_datetime(results_payload.get("started_at"))

    if start_time is None:
        return None

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() / 3600.0
    return max(0.0, elapsed)


def has_non_empty_curve_values(value):
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        return any(has_non_empty_curve_values(v) for v in value.values())
    return False


def select_binary_curve_rows_for_reflection(cycle, results_payload):
    if safe_int(cycle, 0) == 0:
        return [], "Binary expert training curves are intentionally omitted for Cycle 0."

    jobs = results_payload.get("jobs", []) if isinstance(results_payload, dict) else []
    if not isinstance(jobs, list):
        return [], None

    ranked = []
    for job in jobs:
        if not isinstance(job, dict):
            continue

        target = job.get("target", {})
        if not isinstance(target, dict):
            continue

        modality = str(target.get("modality", "")).strip()
        class_label = str(target.get("class_label", "")).strip()
        if not modality and not class_label:
            continue

        candidate = job.get("candidate", {})
        if not isinstance(candidate, dict):
            candidate = {}

        candidate_id = str(candidate.get("candidate_id") or job.get("job_id") or "unknown")
        curves = extract_job_curves(job)

        primary_metric = safe_float(
            job.get("metrics", {}).get("primary_metric", {}).get("value", -1e9),
            default=-1e9,
        )
        status = str(job.get("status", "")).lower()
        has_curve_values = any(
            has_non_empty_curve_values(curves.get(key))
            for key in ("train_loss", "val_loss", "train_metrics", "val_metrics")
        )

        ranked.append(
            {
                "modality": modality or "unknown",
                "class_label": class_label or "unknown",
                "candidate_id": candidate_id,
                "curves": curves,
                "primary_metric": primary_metric,
                "status": status,
                "has_curve_values": has_curve_values,
            }
        )

    if not ranked:
        return [], None

    ranked.sort(
        key=lambda row: (
            1 if row.get("has_curve_values") else 0,
            1 if row.get("status") == "success" else 0,
            safe_float(row.get("primary_metric", -1e9), default=-1e9),
        ),
        reverse=True,
    )

    best_row = ranked[0]
    best_curves = best_row.get("curves", {}) if isinstance(best_row.get("curves"), dict) else {}

    selected = {
        "modality": best_row.get("modality", "unknown"),
        "class_label": best_row.get("class_label", "unknown"),
        "candidate_id": best_row.get("candidate_id", "unknown"),
        "curves": {
            "train_loss": first_non_empty(
                best_curves.get("train_loss"),
                "[TODO: Add full train loss curve from local engine output for this cycle]",
            ),
            "val_loss": first_non_empty(
                best_curves.get("val_loss"),
                "[TODO: Add full validation loss curve from local engine output for this cycle]",
            ),
            "train_metrics": first_non_empty(
                best_curves.get("train_metrics"),
                "[TODO: Add full train metric curves from local engine output for this cycle]",
            ),
            "val_metrics": first_non_empty(
                best_curves.get("val_metrics"),
                "[TODO: Add full validation metric curves from local engine output for this cycle]",
            ),
        },
    }
    return [selected], None


def compute_time_budget_hours(project_root: Path, directive_payload, hours_elapsed):
    now_dt = datetime.now(timezone.utc)

    directive_hours_allotted = None
    if isinstance(directive_payload, dict):
        budget = directive_payload.get("resource_budget", {})
        if isinstance(budget, dict):
            raw_minutes = budget.get("wall_time_minutes")
            if isinstance(raw_minutes, (int, float)) and float(raw_minutes) > 0:
                directive_hours_allotted = float(raw_minutes) / 60.0

    settings_payload = read_json_file(project_root / "state" / "project_settings.json", {})
    settings_start = None
    settings_end = None
    if isinstance(settings_payload, dict):
        settings_start = parse_iso_datetime(settings_payload.get("start_time_utc"))
        settings_end = parse_iso_datetime(settings_payload.get("end_time_utc"))

    settings_hours_allotted = None
    if settings_start and settings_end and settings_end > settings_start:
        settings_hours_allotted = (settings_end - settings_start).total_seconds() / 3600.0

    hours_allotted = first_non_empty(directive_hours_allotted, settings_hours_allotted)
    if not isinstance(hours_allotted, (int, float)) or float(hours_allotted) <= 0:
        hours_allotted = 24.0

    if settings_end is not None:
        hours_remaining = max(0.0, (settings_end - now_dt).total_seconds() / 3600.0)
    elif hours_elapsed is not None:
        hours_remaining = max(0.0, float(hours_allotted) - float(hours_elapsed))
    else:
        hours_remaining = float(hours_allotted)

    hours_remaining = min(float(hours_remaining), float(hours_allotted))
    return round(float(hours_allotted), 2), round(float(hours_remaining), 2)


def try_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_ensemble_precision_recall(project_root: Path, results_payload):
    precision = None
    recall = None

    ensemble_eval = results_payload.get("ensemble_evaluation", {}) if isinstance(results_payload, dict) else {}
    if isinstance(ensemble_eval, dict):
        metric_values = {}
        metrics = ensemble_eval.get("metrics", [])
        if isinstance(metrics, list):
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                name = str(metric.get("name", "")).strip().lower()
                value = try_float(metric.get("value"))
                if not name or value is None:
                    continue
                metric_values[name] = value

        precision = first_non_empty(
            metric_values.get("precision"),
            metric_values.get("macro_precision"),
            metric_values.get("precision_macro"),
        )
        recall = first_non_empty(
            metric_values.get("recall"),
            metric_values.get("macro_recall"),
            metric_values.get("recall_macro"),
        )

        report = ensemble_eval.get("classification_report")
        if isinstance(report, dict):
            macro = report.get("macro avg")
            if isinstance(macro, dict):
                precision = first_non_empty(precision, try_float(macro.get("precision")))
                recall = first_non_empty(recall, try_float(macro.get("recall")))

    if precision is None or recall is None:
        baseline = read_json_file(project_root / "artifacts" / "baseline_ensemble_metrics.json", {})
        if isinstance(baseline, dict):
            report = baseline.get("classification_report")
            if isinstance(report, dict):
                macro = report.get("macro avg")
                if isinstance(macro, dict):
                    precision = first_non_empty(precision, try_float(macro.get("precision")))
                    recall = first_non_empty(recall, try_float(macro.get("recall")))

            precision = first_non_empty(
                precision,
                try_float(baseline.get("precision")),
                try_float(baseline.get("macro_precision")),
            )
            recall = first_non_empty(
                recall,
                try_float(baseline.get("recall")),
                try_float(baseline.get("macro_recall")),
            )

    return precision, recall


def should_finish_project_on_ensemble_metrics(project_root: Path, results_payload, threshold=1.0):
    precision, recall = extract_ensemble_precision_recall(project_root, results_payload)
    target = float(threshold)
    epsilon = 1e-9

    is_finished = (
        isinstance(precision, (int, float))
        and isinstance(recall, (int, float))
        and float(precision) >= (target - epsilon)
        and float(recall) >= (target - epsilon)
    )
    return is_finished, precision, recall


def finalize_project_as_finished(session, project_name, cycle, precision, recall):
    if not (session and project_name):
        return

    proj_state = session.get(ProjectState, project_name)
    if not proj_state:
        return

    proj_state.current_cycle = safe_int(cycle, proj_state.current_cycle or 0)
    proj_state.status = "Finished"
    proj_state.current_step = (
        f"Finished: ensemble precision={safe_float(precision, 0.0):.4f}, "
        f"recall={safe_float(recall, 0.0):.4f}"
    )
    proj_state.target_status = "Stopped"
    proj_state.pid = None
    session.commit()


def build_cycles_history(project_root: Path, current_cycle, results_payload, directive_payload, proposal_payload, reflection_summary):
    cycle_ids = gather_cycle_ids(project_root, current_cycle)
    current_cycle_int = safe_int(current_cycle, 0)
    history_rows = []

    for cycle_id in cycle_ids:
        manifest_path = project_root / "shared" / "context" / "manifests" / format_cycle_label(cycle_id) / "manifest.md"
        focus = extract_manifest_focus(manifest_path)

        if cycle_id == 0 and (not focus or focus == "N/A"):
            focus = "Baseline initialization and benchmarking"

        key_changes = "N/A"
        if cycle_id == current_cycle_int:
            changed = proposal_payload.get("changed_factors") if isinstance(proposal_payload, dict) else None
            if isinstance(changed, list) and changed:
                key_changes = "; ".join(str(x) for x in changed)

        outcome = "N/A"
        result_summary = "N/A"
        primary_metric_delta = "N/A"
        preprocessing_summary = "N/A"
        model_architecture_summary = "N/A"

        if cycle_id == 1:
            preprocessing_summary = "no preprocessing"
            model_architecture_summary = (
                "all models are a small, raw-signal, single-modality, "
                "one-vs-rest, InceptionTime-style 1D CNN baseline"
            )

        if cycle_id == current_cycle_int and isinstance(results_payload, dict):
            execution = results_payload.get("execution_summary", {})
            jobs_total = safe_int(execution.get("jobs_total", 0)) if isinstance(execution, dict) else 0
            jobs_succeeded = safe_int(execution.get("jobs_succeeded", 0)) if isinstance(execution, dict) else 0
            jobs_failed = safe_int(execution.get("jobs_failed", 0)) if isinstance(execution, dict) else 0
            outcome = str(results_payload.get("overall_status", "N/A"))
            result_summary = (
                f"overall_status={outcome}; jobs_succeeded={jobs_succeeded}/{jobs_total}; "
                f"jobs_failed={jobs_failed}"
            )

            jobs = results_payload.get("jobs", []) if isinstance(results_payload.get("jobs", []), list) else []
            if jobs:
                primary_values = []
                primary_names = []
                changed_dims = []

                for row in jobs:
                    if not isinstance(row, dict):
                        continue

                    target = row.get("target", {})
                    if isinstance(target, dict):
                        dim_name = str(target.get("modality", "")).strip()
                        if dim_name:
                            changed_dims.append(dim_name)

                    primary_metric = row.get("metrics", {}).get("primary_metric", {})
                    if not isinstance(primary_metric, dict):
                        continue
                    primary_name = str(primary_metric.get("name", "metric"))
                    primary_value = safe_float(primary_metric.get("value", 0.0))
                    primary_names.append(primary_name)
                    primary_values.append(primary_value)

                if primary_values:
                    metric_name = primary_names[0] if len(set(primary_names)) == 1 else "primary_metric"
                    primary_metric_delta = (
                        f"{metric_name}_mean={float(np.mean(primary_values)):.4f}; "
                        f"best={max(primary_values):.4f}"
                    )

                if changed_dims:
                    changed_dims = sorted(set(changed_dims), key=sorted_key)
                    result_summary += f"; dims_changed={','.join(changed_dims)}"

        elif cycle_id == 0:
            baseline = read_json_file(project_root / "artifacts" / "baseline_ensemble_metrics.json", {})
            accuracy = baseline.get("accuracy") if isinstance(baseline, dict) else None
            if isinstance(accuracy, (int, float)):
                result_summary = f"Baseline initialized; ensemble_accuracy={float(accuracy):.4f}"
            else:
                result_summary = "Baseline initialized"
            outcome = str(reflection_summary.get("outcome_category", "baseline_completed"))
            primary_metric_delta = "baseline_reference"

        history_rows.append(
            {
                "cycle_id": cycle_id,
                "focus": focus,
                "key_changes": key_changes,
                "result_summary": result_summary,
                "primary_metric_delta": primary_metric_delta,
                "outcome": outcome,
                "preprocessing_summary": preprocessing_summary,
                "model_architecture_summary": model_architecture_summary,
            }
        )

    return history_rows


def build_problem_section(config_payload, data_context, dataset_summary):
    dataset_path = Path(str(config_payload.get("dataset_path", "N/A")))
    dataset_name = dataset_path.name if str(dataset_path) != "N/A" else "N/A"
    signals = list(dataset_summary.get("signals", []))

    modality_by_signal = {}
    for signal in signals:
        signal_meta = data_context.get("signals", {}).get(signal, {}) if isinstance(data_context, dict) else {}
        modality_name = str(signal_meta.get("modality", "")).strip()
        if modality_name.lower() in {"", "unspecified", "unknown", "n/a", "na", "none", "null"}:
            modality_name = str(signal)
        modality_by_signal[signal] = modality_name

    modalities = sorted({v for v in modality_by_signal.values() if v}, key=sorted_key)

    if len(signals) <= 1:
        modality_structure = "single-modality univariate" if len(modalities) <= 1 else "multi-modality univariate"
    else:
        modality_structure = "single-modality multivariate" if len(modalities) <= 1 else "multi-modality multivariate"

    is_multivariate_text = "Yes" if len(signals) > 1 else "No"
    if signals:
        is_multivariate_text += f" ({len(signals)} parallel signals/channels)"

    objective_text = (data_context.get("objective") or "").strip() if isinstance(data_context, dict) else ""
    statement = objective_text if objective_text else "[TODO: Provide concise problem statement from project context]"

    return {
        "dataset_name": dataset_name,
        "domain": str(config_payload.get("dataset_path", "N/A")),
        "statement": statement,
        "modalities": modalities,
        "modality_structure": modality_structure,
        "is_multivariate_text": is_multivariate_text,
        "sequence_notes": dataset_summary.get("sequence_notes", "N/A"),
    }


def _load_numpy_array(path: Path):
    if not path.exists():
        return None

    try:
        return np.load(path, mmap_mode="r")
    except Exception:
        try:
            return np.load(path, allow_pickle=True)
        except Exception:
            return None


def _extract_sequence_lengths_from_array(array):
    if array is None:
        return []

    try:
        arr = np.asarray(array)
    except Exception:
        return []

    if arr.size == 0:
        return []

    if arr.dtype == object:
        lengths = []
        for item in arr:
            try:
                item_arr = np.asarray(item)
            except Exception:
                continue
            if item_arr.ndim == 0:
                continue
            lengths.append(safe_int(item_arr.shape[-1], 0))
        return lengths

    if arr.ndim >= 2:
        sample_count = safe_int(arr.shape[0], 0)
        seq_len = safe_int(arr.shape[-1], 0)
        return [seq_len for _ in range(max(0, sample_count))]

    return []


def _build_sequence_length_summary(train_lengths, validate_lengths):
    def _summary(values):
        if not values:
            return 0, 0, 0
        arr = np.asarray(values, dtype=np.float64)
        return safe_int(np.min(arr), 0), safe_int(np.median(arr), 0), safe_int(np.max(arr), 0)

    train_min, train_median, train_max = _summary(train_lengths)
    validate_min, validate_median, validate_max = _summary(validate_lengths)
    return {
        "train_min": train_min,
        "train_median": train_median,
        "train_max": train_max,
        "validate_min": validate_min,
        "validate_median": validate_median,
        "validate_max": validate_max,
    }


def _compute_nan_inf_fractions(array):
    if array is None:
        return 0.0, 0.0

    try:
        arr = np.asarray(array)
    except Exception:
        return 0.0, 0.0

    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        return 0.0, 0.0

    arr_float = arr.astype(np.float64, copy=False)
    total = float(arr_float.size)
    if total <= 0:
        return 0.0, 0.0

    nan_fraction = float(np.isnan(arr_float).sum() / total)
    inf_fraction = float(np.isinf(arr_float).sum() / total)
    return nan_fraction, inf_fraction


def _compute_flatline_fraction(array):
    if array is None:
        return 0.0

    try:
        arr = np.asarray(array)
    except Exception:
        return 0.0

    if arr.size == 0 or arr.ndim < 2 or not np.issubdtype(arr.dtype, np.number):
        return 0.0

    arr_float = arr.astype(np.float64, copy=False)
    sample_axes = tuple(range(1, arr_float.ndim))
    if not sample_axes:
        return 0.0

    std_values = np.nanstd(arr_float, axis=sample_axes)
    std_values = np.nan_to_num(std_values, nan=np.inf, posinf=np.inf, neginf=np.inf)
    return float(np.mean(std_values <= 1e-12)) if std_values.size else 0.0


def _resolve_channels_after_load(array):
    if array is None:
        return 1

    try:
        arr = np.asarray(array)
    except Exception:
        return 1

    if arr.dtype == object:
        for item in arr:
            try:
                item_arr = np.asarray(item)
            except Exception:
                continue
            if item_arr.ndim >= 2:
                return max(1, safe_int(item_arr.shape[0], 1))
            if item_arr.ndim == 1:
                return 1
        return 1

    if arr.ndim >= 3:
        return max(1, safe_int(arr.shape[1], 1))

    return 1


def _extract_sampling_rate_hz(stats_payload, default=100.0):
    if not isinstance(stats_payload, dict):
        return float(default)

    for modality in sorted(stats_payload.keys(), key=sorted_key):
        per_class = stats_payload.get(modality, {})
        if not isinstance(per_class, dict):
            continue

        for class_label in sorted(per_class.keys(), key=sorted_key):
            row = per_class.get(class_label, {})
            if not isinstance(row, dict):
                continue
            sampling_rate = row.get("sampling_rate")
            if isinstance(sampling_rate, (int, float)) and float(sampling_rate) > 0:
                return float(sampling_rate)

    return float(default)


def _extract_flatline_fraction_from_stats(stats_payload, modality_name):
    if not isinstance(stats_payload, dict):
        return None

    per_class = stats_payload.get(modality_name)
    if not isinstance(per_class, dict) or not per_class:
        return None

    values = []
    for class_label in sorted(per_class.keys(), key=sorted_key):
        row = per_class.get(class_label, {})
        if not isinstance(row, dict):
            continue
        raw = row.get("clipping_flatline_rate")
        if isinstance(raw, (int, float)):
            values.append(float(raw))

    if not values:
        return None

    return float(np.mean(values))


def _normalize_modality_structure(problem_section):
    raw = str(problem_section.get("modality_structure", "")).strip().lower()
    mapping = {
        "single-modality univariate": "single_modality_univariate",
        "single-modality multivariate": "single_modality_multivariate",
        "multi-modality univariate": "multi_modality_univariate",
        "multi-modality multivariate": "multi_modality_multivariate",
    }
    return mapping.get(raw, "single_modality_multivariate")


def _normalize_modality_label(value):
    text = str(value or "").strip()
    if text.lower() in {"", "unspecified", "unknown", "n/a", "na", "none", "null"}:
        return ""
    return text


def _infer_default_group_name(dataset_name: str, objective_text: str):
    joined = f"{dataset_name} {objective_text}".lower()
    if "trajectory" in joined:
        return "trajectory"
    if "accelerometer" in joined or "accelerometry" in joined or "accel" in joined:
        return "accelerometry"
    if "eeg" in joined:
        return "eeg"
    return "modality"


def _shape_channels(shape, fallback_channels):
    if not isinstance(shape, list):
        return max(1, safe_int(fallback_channels, 1))
    if len(shape) >= 3:
        return max(1, safe_int(shape[1], fallback_channels if fallback_channels is not None else 1))
    return max(1, safe_int(fallback_channels, 1))


def _aggregate_group_shapes(member_rows, split_key):
    shapes = []
    for row in member_rows:
        shape = row.get(split_key)
        if isinstance(shape, list) and len(shape) >= 2:
            shapes.append(shape)

    if not shapes:
        return []

    sample_counts = {safe_int(shape[0], 0) for shape in shapes}
    seq_lens = {safe_int(shape[-1], 0) for shape in shapes}
    if len(sample_counts) != 1 or len(seq_lens) != 1:
        return shapes[0]

    n_samples = next(iter(sample_counts))
    seq_len = next(iter(seq_lens))
    total_channels = 0
    for row in member_rows:
        total_channels += _shape_channels(row.get(split_key), row.get("channels_after_load"))

    if total_channels <= 1:
        return [n_samples, seq_len]

    return [n_samples, total_channels, seq_len]


def _build_modality_groups(signal_rows, data_context, dataset_name):
    row_by_signal = {
        str(row.get("name", "")).strip(): row
        for row in signal_rows
        if str(row.get("name", "")).strip()
    }
    signal_names = sorted(row_by_signal.keys(), key=sorted_key)

    context_signals = {}
    objective_text = ""
    if isinstance(data_context, dict):
        context_signals = data_context.get("signals", {}) if isinstance(data_context.get("signals"), dict) else {}
        objective_text = str(data_context.get("objective", ""))

    explicit_groups = {}
    unassigned = []
    for signal_name in signal_names:
        signal_meta = context_signals.get(signal_name, {})
        modality_label = ""
        if isinstance(signal_meta, dict):
            modality_label = _normalize_modality_label(signal_meta.get("modality"))

        if modality_label:
            explicit_groups.setdefault(modality_label, []).append(signal_name)
        else:
            unassigned.append(signal_name)

    grouped_signals = {}
    if explicit_groups:
        grouped_signals.update(explicit_groups)
        for signal_name in unassigned:
            grouped_signals[signal_name] = [signal_name]
    else:
        if len(signal_names) > 1 and all(re.fullmatch(r"dim\d+", name) for name in signal_names):
            inferred_name = _infer_default_group_name(dataset_name, objective_text)
            grouped_signals[inferred_name] = signal_names
        else:
            for signal_name in signal_names:
                grouped_signals[signal_name] = [signal_name]

    grouped_rows = []
    for group_name in sorted(grouped_signals.keys(), key=sorted_key):
        member_signals = sorted(grouped_signals[group_name], key=sorted_key)
        member_rows = [row_by_signal[sig] for sig in member_signals if sig in row_by_signal]
        channels_after_load = sum(
            max(1, safe_int(row.get("channels_after_load"), 1))
            for row in member_rows
        )

        grouped_rows.append(
            {
                "name": str(group_name),
                "member_signals": member_signals,
                "channels_after_load": channels_after_load,
                "train_shape": _aggregate_group_shapes(member_rows, "train_shape"),
                "validate_shape": _aggregate_group_shapes(member_rows, "validate_shape"),
            }
        )

    return grouped_rows


def _infer_modality_structure_token(modality_groups):
    groups = modality_groups if isinstance(modality_groups, list) else []
    modality_count = len(groups)
    has_multivariate = any(
        isinstance(group, dict)
        and isinstance(group.get("member_signals"), list)
        and len(group.get("member_signals", [])) > 1
        for group in groups
    )

    if modality_count <= 1:
        return "single_modality_multivariate" if has_multivariate else "single_modality_univariate"
    return "multi_modality_multivariate" if has_multivariate else "multi_modality_univariate"


def _infer_problem_type(class_count: int):
    if class_count <= 2:
        return "binary_time_series_classification"
    return "multiclass_time_series_classification"


def _extract_primary_metric_from_results(results_payload):
    if not isinstance(results_payload, dict):
        return "macro_f1", None

    ensemble = results_payload.get("ensemble_evaluation")
    if isinstance(ensemble, dict):
        metric_rows = ensemble.get("metrics")
        metric_map = {}
        if isinstance(metric_rows, list):
            for row in metric_rows:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name", "")).strip().lower()
                value = row.get("value")
                if not name or not isinstance(value, (int, float)):
                    continue
                metric_map[name] = float(value)

        for metric_name in ("macro_f1", "f1", "accuracy"):
            if metric_name in metric_map:
                return metric_name, float(metric_map[metric_name])

    jobs = results_payload.get("jobs")
    if isinstance(jobs, list):
        values = []
        metric_name = "f1"
        for job in jobs:
            if not isinstance(job, dict):
                continue
            primary = job.get("metrics", {}).get("primary_metric", {})
            if not isinstance(primary, dict):
                continue
            if isinstance(primary.get("name"), str) and primary.get("name", "").strip():
                metric_name = str(primary.get("name")).strip()
            value = primary.get("value")
            if isinstance(value, (int, float)):
                values.append(float(value))

        if values:
            return metric_name, float(np.mean(values))

    return "macro_f1", None


def _extract_target_from_directive(directive_payload):
    default_target = ("all", "all")
    if not isinstance(directive_payload, dict):
        return default_target

    jobs = directive_payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        return default_target

    first_job = jobs[0] if isinstance(jobs[0], dict) else {}
    target = first_job.get("target") if isinstance(first_job.get("target"), dict) else {}
    modality = str(target.get("modality", "all")).strip() or "all"
    class_label = str(target.get("class_label", "all")).strip() or "all"
    return modality, class_label


def build_cycle_context_history(project_root: Path, current_cycle, latest_results_payload, latest_directive_payload):
    current_cycle_int = safe_int(current_cycle, 0)
    cycle_ids = set(gather_cycle_ids(project_root, current_cycle_int))
    cycle_ids.add(max(0, current_cycle_int - 1))

    completed_cycles = []
    prev_metric_value = None

    latest_results_cycle = safe_int(
        latest_results_payload.get("cycle_id") if isinstance(latest_results_payload, dict) else None,
        -1,
    )
    latest_directive_cycle = safe_int(
        latest_directive_payload.get("cycle_id") if isinstance(latest_directive_payload, dict) else None,
        -1,
    )

    for cycle_id in sorted(cycle_ids):
        if cycle_id >= current_cycle_int:
            continue

        cycle_dir = project_root / "artifacts" / "cycle_history" / format_cycle_label(cycle_id)
        results_payload = read_json_file(cycle_dir / "results.json", {})
        if not results_payload and cycle_id == latest_results_cycle and isinstance(latest_results_payload, dict):
            results_payload = latest_results_payload

        directive_payload = {}
        directive_candidates = [
            cycle_dir / "directive.json",
            cycle_dir / "inbound" / "directive.json",
            project_root / "shared" / "context" / "manifests" / format_cycle_label(cycle_id) / "directive.json",
        ]
        for candidate in directive_candidates:
            directive_payload = read_json_file(candidate, {})
            if isinstance(directive_payload, dict) and directive_payload:
                break

        if (
            (not directive_payload)
            and cycle_id == latest_directive_cycle
            and isinstance(latest_directive_payload, dict)
        ):
            directive_payload = latest_directive_payload

        target_modality, target_class_label = _extract_target_from_directive(directive_payload)
        if cycle_id == 0:
            target_modality, target_class_label = "all", "all"

        status = "success" if cycle_id == 0 else "unknown"
        if isinstance(results_payload, dict) and results_payload:
            status = str(results_payload.get("overall_status", status))

        primary_metric_name, metric_value = _extract_primary_metric_from_results(results_payload)
        if metric_value is None:
            metric_value = 0.0 if cycle_id == 0 else None

        if prev_metric_value is None or metric_value is None:
            metric_delta = 0.0
        else:
            metric_delta = float(metric_value) - float(prev_metric_value)

        completed_cycles.append(
            {
                "cycle_id": str(cycle_id),
                "target_modality": target_modality,
                "target_class_label": target_class_label,
                "status": status,
                "primary_metric_name": primary_metric_name,
                "primary_metric_delta": round(float(metric_delta), 10),
            }
        )

        if metric_value is not None:
            prev_metric_value = float(metric_value)

    return completed_cycles


def _parse_cycle_int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _load_design_history_entries(project_root: Path):
    payload = read_json_file(project_root / "artifacts" / "design_history.json", {})
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    return [row for row in entries if isinstance(row, dict)]


def _load_successful_cycle_ids(project_root: Path):
    cycle_root = project_root / "artifacts" / "cycle_history"
    out = set()
    if not cycle_root.exists():
        return out

    for cycle_dir in sorted(cycle_root.glob("cycle_*"), key=lambda p: p.name):
        if not cycle_dir.is_dir():
            continue
        payload = read_json_file(cycle_dir / "results.json", {})
        if not isinstance(payload, dict) or not payload:
            continue

        status = str(payload.get("overall_status", "")).strip().lower()
        if status != "success":
            continue

        cycle_id = _parse_cycle_int(payload.get("cycle_id"))
        if cycle_id is None:
            match = re.fullmatch(r"cycle_(\d+)", cycle_dir.name)
            cycle_id = int(match.group(1)) if match else None
        if cycle_id is not None:
            out.add(cycle_id)

    return out


def _normalize_design_context_entry(entry, include_snippets=False):
    if not isinstance(entry, dict):
        return None

    target = entry.get("target") if isinstance(entry.get("target"), dict) else {}
    design = entry.get("design") if isinstance(entry.get("design"), dict) else {}
    model_description = design.get("model_description") if isinstance(design.get("model_description"), dict) else {}
    preprocessing_description = (
        design.get("preprocessing_description")
        if isinstance(design.get("preprocessing_description"), dict)
        else {}
    )
    modality = str(target.get("modality", "")).strip()
    class_label = str(target.get("class_label", "")).strip()
    if not modality or not class_label:
        return None

    out = {
        "cycle_id": str(entry.get("cycle_id", "")).strip() or "0",
        "candidate_id": str(entry.get("candidate_id", "")).strip() or "unknown_candidate",
        "target": {
            "modality": modality,
            "class_label": class_label,
        },
        "model_description": model_description,
        "preprocessing_description": preprocessing_description,
        "snippet_hashes": entry.get("snippet_hashes") if isinstance(entry.get("snippet_hashes"), dict) else {},
        "updated_at": str(entry.get("updated_at", "")).strip(),
    }
    if include_snippets:
        out["snippets"] = entry.get("snippets") if isinstance(entry.get("snippets"), dict) else {}
    return out


def build_current_designs_context(project_root: Path):
    include_snippets = str(os.environ.get("ARL_CURRENT_DESIGNS_INCLUDE_SNIPPETS", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    history_limit = max(0, safe_int(os.environ.get("ARL_CURRENT_DESIGNS_HISTORY_LIMIT", "6"), 6))

    entries = _load_design_history_entries(project_root)
    if not entries:
        return {
            "source": "none",
            "active_family_designs": [],
            "latest_accepted_output": None,
            "recent_description_history": [],
        }

    successful_cycles = _load_successful_cycle_ids(project_root)

    def _entry_sort_key(row):
        cycle_id = _parse_cycle_int(row.get("cycle_id"))
        if cycle_id is None:
            cycle_id = 10**9
        return (cycle_id, str(row.get("candidate_id", "")))

    def _has_required_target_and_design(row):
        if not isinstance(row, dict):
            return False
        target = row.get("target")
        design = row.get("design")
        if not isinstance(target, dict) or not isinstance(design, dict):
            return False
        modality = str(target.get("modality", "")).strip()
        class_label = str(target.get("class_label", "")).strip()
        return bool(modality and class_label)

    valid_entries = [row for row in entries if _has_required_target_and_design(row)]
    if not valid_entries:
        return {
            "source": "none",
            "active_family_designs": [],
            "latest_accepted_output": None,
            "recent_description_history": [],
        }

    accepted_entries = []
    for row in valid_entries:
        cycle_int = _parse_cycle_int(row.get("cycle_id"))
        if cycle_int is not None and cycle_int in successful_cycles:
            accepted_entries.append(row)

    source_entries = accepted_entries if accepted_entries else valid_entries
    source_name = "accepted_designs" if accepted_entries else "all_design_history"
    source_entries = sorted(source_entries, key=_entry_sort_key)

    latest_by_family = {}
    for row in source_entries:
        target = row.get("target") if isinstance(row.get("target"), dict) else {}
        key = (str(target.get("modality", "")).strip(), str(target.get("class_label", "")).strip())
        if not key[0] or not key[1]:
            continue
        latest_by_family[key] = row

    active_family_designs = []
    for key in sorted(latest_by_family.keys(), key=lambda item: (sorted_key(item[0]), sorted_key(item[1]))):
        normalized = _normalize_design_context_entry(latest_by_family[key], include_snippets=include_snippets)
        if normalized is not None:
            active_family_designs.append(normalized)

    latest_accepted_output = None
    if accepted_entries:
        latest_accepted_output = _normalize_design_context_entry(
            sorted(accepted_entries, key=_entry_sort_key)[-1],
            include_snippets=include_snippets,
        )

    recent_description_history = []
    if history_limit > 0:
        for row in source_entries[-history_limit:]:
            normalized = _normalize_design_context_entry(row, include_snippets=include_snippets)
            if normalized is None:
                continue

            model_summary = ""
            preprocessing_summary = ""
            model_payload = normalized.get("model_description")
            preprocessing_payload = normalized.get("preprocessing_description")
            if isinstance(model_payload, dict):
                model_summary = str(model_payload.get("summary", "")).strip()
            if isinstance(preprocessing_payload, dict):
                preprocessing_summary = str(preprocessing_payload.get("summary", "")).strip()

            recent_description_history.append(
                {
                    "cycle_id": normalized.get("cycle_id"),
                    "candidate_id": normalized.get("candidate_id"),
                    "target": normalized.get("target"),
                    "model_summary": model_summary,
                    "preprocessing_summary": preprocessing_summary,
                    "snippet_hashes": normalized.get("snippet_hashes", {}),
                }
            )

    return {
        "source": source_name,
        "active_family_designs": active_family_designs,
        "latest_accepted_output": latest_accepted_output,
        "recent_description_history": recent_description_history,
    }


def write_cycle_context_json(project_name: str, cycle: int, session=None):
    project_root = (Path("projects") / project_name).absolute()
    config_payload = read_yaml_file(project_root / "config.yaml", {})
    results_payload = read_json_file(project_root / "shared" / "outbound" / "results.json", {})
    directive_payload = read_json_file(project_root / "shared" / "inbound" / "directive.json", {})

    dataset_path = Path(str(config_payload.get("dataset_path", ""))).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = (Path.cwd() / dataset_path).resolve()

    dataset_summary = summarize_dataset_structure(dataset_path) if dataset_path.exists() else {
        "signals": [],
        "class_labels": [],
        "sequence_notes": "N/A",
    }
    data_context = parse_data_context(project_root)
    problem_section = build_problem_section(config_payload, data_context, dataset_summary)

    train_dir = dataset_path / "train"
    validate_dir = dataset_path / "validate"

    train_signal_files = sorted(train_dir.glob("X_*.npy"), key=lambda p: sorted_key(p.stem[2:])) if train_dir.exists() else []
    modality_names = [f.stem[2:] for f in train_signal_files]
    if not modality_names:
        modality_names = sorted(
            {str(signal).strip() for signal in dataset_summary.get("signals", []) if str(signal).strip()},
            key=sorted_key,
        )

    y_train = _load_numpy_array(train_dir / "y.npy")
    y_validate = _load_numpy_array(validate_dir / "y.npy")

    label_set = set()
    for y_array in (y_train, y_validate):
        if y_array is None:
            continue
        try:
            flat = np.asarray(y_array).reshape(-1)
        except Exception:
            continue
        if flat.size == 0:
            continue
        for label in np.unique(flat):
            if isinstance(label, np.generic):
                label = label.item()
            if isinstance(label, float) and float(label).is_integer():
                label_set.add(str(int(label)))
            else:
                label_set.add(str(label))

    class_ids = sorted(label_set, key=sorted_key)

    train_sample_count = safe_int(np.asarray(y_train).shape[0], 0) if y_train is not None else 0
    validate_sample_count = safe_int(np.asarray(y_validate).shape[0], 0) if y_validate is not None else 0

    signal_rows = []
    data_integrity_per_signal = {}
    sequence_lengths_train = []
    sequence_lengths_validate = []
    data_statistics_payload = read_json_file(project_root / "shared" / "context" / "data_statistics.json", {})

    for modality_name in sorted(modality_names, key=sorted_key):
        train_file = train_dir / f"X_{modality_name}.npy"
        validate_file = validate_dir / f"X_{modality_name}.npy"
        x_train = _load_numpy_array(train_file)
        x_validate = _load_numpy_array(validate_file)

        train_shape = list(np.asarray(x_train).shape) if x_train is not None else []
        validate_shape = list(np.asarray(x_validate).shape) if x_validate is not None else []

        sequence_lengths_train.extend(_extract_sequence_lengths_from_array(x_train))
        sequence_lengths_validate.extend(_extract_sequence_lengths_from_array(x_validate))

        nan_fraction_train, inf_fraction_train = _compute_nan_inf_fractions(x_train)
        nan_fraction_validate, inf_fraction_validate = _compute_nan_inf_fractions(x_validate)
        stats_flatline = _extract_flatline_fraction_from_stats(data_statistics_payload, modality_name)

        dtype_value = "unknown"
        if x_train is not None:
            dtype_value = str(np.asarray(x_train).dtype)
        elif x_validate is not None:
            dtype_value = str(np.asarray(x_validate).dtype)

        channels_after_load = _resolve_channels_after_load(x_train if x_train is not None else x_validate)

        signal_rows.append(
            {
                "name": modality_name,
                "channels_after_load": channels_after_load,
                "train_shape": train_shape,
                "validate_shape": validate_shape,
            }
        )

        data_integrity_per_signal[modality_name] = {
            "dtype": dtype_value,
            "nan_fraction_train": nan_fraction_train,
            "inf_fraction_train": inf_fraction_train,
            "nan_fraction_validate": nan_fraction_validate,
            "inf_fraction_validate": inf_fraction_validate,
            "flatline_fraction_train": (
                float(stats_flatline)
                if isinstance(stats_flatline, (int, float))
                else _compute_flatline_fraction(x_train)
            ),
        }

    if train_sample_count <= 0 and signal_rows:
        train_sample_count = safe_int(signal_rows[0].get("train_shape", [0])[0], 0)
    if validate_sample_count <= 0 and signal_rows:
        validate_sample_count = safe_int(signal_rows[0].get("validate_shape", [0])[0], 0)

    if not class_ids:
        class_ids = sorted(
            {str(x).strip() for x in (dataset_summary.get("class_labels") or []) if str(x).strip()},
            key=sorted_key,
        )

    if not class_ids:
        classes_from_context = data_context.get("classes") if isinstance(data_context, dict) else {}
        if isinstance(classes_from_context, dict):
            class_ids = sorted(
                {
                    _normalize_class_id_token(key)
                    for key in classes_from_context.keys()
                    if _normalize_class_id_token(key)
                },
                key=sorted_key,
            )

    class_id_to_label = _build_class_id_to_label_map(
        dataset_path=dataset_path,
        data_context=data_context,
        class_ids=class_ids,
    )

    if not class_ids and class_id_to_label:
        class_ids = sorted(class_id_to_label.keys(), key=sorted_key)

    class_labels = [class_id_to_label.get(class_id, class_id) for class_id in class_ids]
    class_count = len(class_ids)

    if not sequence_lengths_train and signal_rows:
        fallback_train_len = safe_int(signal_rows[0].get("train_shape", [0, 0])[-1] if len(signal_rows[0].get("train_shape", [])) >= 2 else 0, 0)
        if fallback_train_len > 0 and train_sample_count > 0:
            sequence_lengths_train = [fallback_train_len for _ in range(train_sample_count)]
    if not sequence_lengths_validate and signal_rows:
        fallback_validate_len = safe_int(signal_rows[0].get("validate_shape", [0, 0])[-1] if len(signal_rows[0].get("validate_shape", [])) >= 2 else 0, 0)
        if fallback_validate_len > 0 and validate_sample_count > 0:
            sequence_lengths_validate = [fallback_validate_len for _ in range(validate_sample_count)]

    modality_groups = _build_modality_groups(
        signal_rows=signal_rows,
        data_context=data_context,
        dataset_name=dataset_path.name if str(dataset_path) else "",
    )


    signal_to_modality_family = {}
    modality_member_order = {}
    for group in modality_groups:
        group_name = str(group.get("name", "")).strip()
        if not group_name:
            continue
        member_signals = [
            str(sig).strip()
            for sig in (group.get("member_signals") if isinstance(group.get("member_signals"), list) else [])
            if str(sig).strip()
        ]
        if not member_signals:
            member_signals = [group_name]

        modality_member_order[group_name] = member_signals
        signal_to_modality_family[group_name] = group_name
        for signal_name in member_signals:
            signal_to_modality_family[signal_name] = group_name

    for signal_name in modality_names:
        signal_to_modality_family.setdefault(signal_name, signal_name)

    data_integrity_per_modality = {}
    for group in modality_groups:
        group_name = str(group.get("name", "")).strip()
        if not group_name:
            continue

        member_signals = modality_member_order.get(group_name, [])
        member_signal_stats = {}
        for signal_name in member_signals:
            stats = data_integrity_per_signal.get(signal_name)
            if isinstance(stats, dict):
                member_signal_stats[signal_name] = stats

        if member_signal_stats:
            data_integrity_per_modality[group_name] = {
                "member_signals": member_signal_stats,
            }

    sampling_rate_hz = _extract_sampling_rate_hz(data_statistics_payload, default=100.0)

    best_ensemble = extract_ensemble_metrics(results_payload, project_root)
    best_ensemble_metrics = best_ensemble.get("metrics", {}) if isinstance(best_ensemble, dict) else {}

    ensemble_curves = best_ensemble.get("curves", {}) if isinstance(best_ensemble, dict) else {}
    train_loss_curve = ensemble_curves.get("train_loss") if isinstance(ensemble_curves, dict) else []
    val_loss_curve = ensemble_curves.get("val_loss") if isinstance(ensemble_curves, dict) else []
    train_loss_curve = train_loss_curve if isinstance(train_loss_curve, list) else []
    val_loss_curve = val_loss_curve if isinstance(val_loss_curve, list) else []

    trend = "stable"
    if len(val_loss_curve) >= 2:
        trend = "improving" if safe_float(val_loss_curve[-1]) <= safe_float(val_loss_curve[0]) else "degrading"
    elif len(train_loss_curve) >= 2:
        trend = "improving" if safe_float(train_loss_curve[-1]) <= safe_float(train_loss_curve[0]) else "degrading"

    best_val_loss = None
    best_epoch = None
    if val_loss_curve:
        numeric_vals = [safe_float(v, 0.0) for v in val_loss_curve]
        best_val_loss = float(np.min(numeric_vals))
        best_epoch = safe_int(np.argmin(numeric_vals) + 1, 1)

    all_binary_experts = flatten_best_binary_experts(project_root, results_payload)
    class_groups = {}
    for expert in all_binary_experts:
        class_label = str(expert.get("class_label", "")).strip()
        if not class_label:
            continue
        class_groups.setdefault(class_label, {"f1": [], "precision": [], "recall": []})
        class_groups[class_label]["f1"].append(safe_float(expert.get("f1", 0.0)))
        class_groups[class_label]["precision"].append(safe_float(expert.get("precision", 0.0)))
        class_groups[class_label]["recall"].append(safe_float(expert.get("recall", 0.0)))

    class_rankings = []
    for class_label, grouped in class_groups.items():
        class_rankings.append(
            {
                "class_label": class_label,
                "class_name": class_id_to_label.get(str(class_label), str(class_label)),
                "mean_expert_f1": round(float(np.mean(grouped["f1"])), 4) if grouped["f1"] else 0.0,
                "mean_expert_precision": round(float(np.mean(grouped["precision"])), 4) if grouped["precision"] else 0.0,
                "mean_expert_recall": round(float(np.mean(grouped["recall"])), 4) if grouped["recall"] else 0.0,
                "expert_count": len(grouped["f1"]),
            }
        )

    class_rankings.sort(key=lambda row: (safe_float(row.get("mean_expert_f1", 0.0)), sorted_key(row.get("class_label", ""))))
    selected_class_labels = [row.get("class_label") for row in class_rankings[:3]]

    binary_expert_families = []
    for class_label in selected_class_labels:
        experts_for_class = [
            row for row in all_binary_experts if str(row.get("class_label", "")).strip() == str(class_label)
        ]
        if not experts_for_class:
            continue

        family_rows = {}
        for row in experts_for_class:
            signal_name = str(row.get("modality", "unknown")).strip() or "unknown"
            family_name = signal_to_modality_family.get(signal_name, signal_name)
            family_rows.setdefault(family_name, {})[signal_name] = row

        for family_name in sorted(family_rows.keys(), key=sorted_key):
            rows_for_family = family_rows.get(family_name, {})
            ordered_signals = list(modality_member_order.get(family_name, []))
            extra_signals = [
                signal_name for signal_name in sorted(rows_for_family.keys(), key=sorted_key)
                if signal_name not in ordered_signals
            ]
            ordered_signals.extend(extra_signals)

            member_signal_metrics = {}
            metric_rows = []
            for signal_name in ordered_signals:
                row = rows_for_family.get(signal_name)
                if not isinstance(row, dict):
                    continue

                metrics_payload = {
                    "accuracy": safe_float(row.get("accuracy", 0.0)),
                    "precision": safe_float(row.get("precision", 0.0)),
                    "recall": safe_float(row.get("recall", 0.0)),
                    "f1": safe_float(row.get("f1", 0.0)),
                }
                metric_rows.append(metrics_payload)
                member_signal_metrics[signal_name] = {
                    "candidate_id": str(row.get("candidate_id", "unknown")),
                    "eval_split": str(row.get("eval_split", "validate")) or "validate",
                    "metrics": metrics_payload,
                }

            if not member_signal_metrics:
                continue

            count = max(1, len(metric_rows))
            aggregated_metrics = {
                "mean_f1": round(sum(safe_float(m.get("f1", 0.0)) for m in metric_rows) / count, 6),
                "mean_precision": round(sum(safe_float(m.get("precision", 0.0)) for m in metric_rows) / count, 6),
                "mean_recall": round(sum(safe_float(m.get("recall", 0.0)) for m in metric_rows) / count, 6),
                "member_signal_count": len(member_signal_metrics),
            }

            binary_expert_families.append(
                {
                    "modality": family_name,
                    "class_label": str(class_label),
                    "class_name": class_id_to_label.get(str(class_label), str(class_label)),
                    "aggregated_metrics": aggregated_metrics,
                    "member_signal_metrics": member_signal_metrics,
                }
            )

    binary_expert_families.sort(
        key=lambda row: (sorted_key(row.get("class_label", "")), sorted_key(row.get("modality", "")))
    )

    class_labels_with_experts = sorted(class_groups.keys(), key=sorted_key)
    binary_expert_families_subset = len(selected_class_labels) < len(class_labels_with_experts)

    hours_elapsed = compute_hours_elapsed(session, project_name, results_payload)
    hours_allotted_raw, hours_remaining_raw = compute_time_budget_hours(
        project_root=project_root,
        directive_payload=directive_payload,
        hours_elapsed=hours_elapsed,
    )
    hours_allotted = round(float(hours_allotted_raw), 2)
    hours_remaining = round(min(float(hours_allotted), max(0.0, float(hours_remaining_raw))), 2)
    hours_used = round(max(0.0, float(hours_allotted) - float(hours_remaining)), 2)
    current_designs = build_current_designs_context(project_root)

    cycle_context_payload = {
        "project": {
            "project_name": project_name,
            "cycle_id": str(safe_int(cycle, 0)),
            "time_budget_hours_total": round(float(hours_allotted), 2),
            "time_budget_hours_used": round(float(hours_used), 2),
            "time_budget_hours_remaining": round(float(hours_remaining), 2),
        },
        "dataset": {
            "dataset_name": dataset_path.name if str(dataset_path) else "unknown",
            "dataset_path": str(dataset_path),
            "problem_type": _infer_problem_type(class_count),
            "modality_structure": _infer_modality_structure_token(modality_groups),
            "modalities": modality_groups,
            "class_ids": class_ids,
            "class_labels": class_labels,
            "class_id_to_label": class_id_to_label,
            "class_count": class_count,
            "train_sample_count": train_sample_count,
            "validate_sample_count": validate_sample_count,
            "sequence_length": _build_sequence_length_summary(sequence_lengths_train, sequence_lengths_validate),
            "sampling_rate_hz": float(sampling_rate_hz),
            "dataset_build_metadata": {
                "resampled_to_hz": None,
                "padding": None,
                "windowing": None,
                "normalization": None,
            },
        },
        "data_integrity": {
            "per_signal": data_integrity_per_signal,
            "per_modality": data_integrity_per_modality,
        },
        "training_results": {
            "best_ensemble": {
                "candidate_id": str(best_ensemble.get("name", "baseline_ensemble")),
                "eval_split": "validate",
                "metrics": {
                    "accuracy": safe_float(best_ensemble_metrics.get("accuracy", 0.0)),
                    "kappa": safe_float(best_ensemble_metrics.get("kappa", 0.0)),
                    "macro_precision": safe_float(best_ensemble_metrics.get("macro_precision", 0.0)),
                    "macro_recall": safe_float(best_ensemble_metrics.get("macro_recall", 0.0)),
                    "macro_f1": safe_float(best_ensemble_metrics.get("macro_f1", 0.0)),
                },
                "curve_summary": {
                    "epochs_completed": max(len(train_loss_curve), len(val_loss_curve)),
                    "best_epoch": best_epoch,
                    "early_stopped": False,
                    "trend": trend,
                    "best_val_loss": best_val_loss,
                },
            },
            "binary_expert_families": binary_expert_families,
            "binary_expert_families_subset": binary_expert_families_subset,
            "binary_expert_families_subset_rule": (
                "only weakest classes by mean expert F1"
                if binary_expert_families_subset
                else "all classes included"
            ),
            "class_rankings": {
                "by_mean_expert_f1_ascending": class_rankings[:3],
            },
        },
        "history": {
            "completed_cycles": build_cycle_context_history(
                project_root=project_root,
                current_cycle=cycle,
                latest_results_payload=results_payload,
                latest_directive_payload=directive_payload,
            ),
        },
        "runtime_contract": {
            "model_template": {
                "class_name": "BinaryExpertModel",
                "constructor_args": [
                    "in_ch",
                    "n_classes",
                    "fs",
                    "min_seq_len",
                    "dts",
                    "k_min",
                    "k_max_cap",
                    "width",
                    "depth",
                    "dropout",
                ],
                "extract_features_output": "(B, D)",
                "forward_output": "(B, n_classes)",
            },
            "preprocessing_template": {
                "function_name": "apply_preprocessing",
                "preserve_sample_axis": True,
                "return_float32": True,
                "finite_only": True,
            },
            "allowed_dependencies": [
                "python_stdlib",
                "numpy",
                "scipy",
                "scipy.signal",
                "torch",
                "torch.nn",
                "torch.nn.functional",
            ],
        },
        "current_designs": current_designs,
    }

    output_path = project_root / "shared" / "outbound" / "cycle_context.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cycle_context_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote deterministic cycle context: {output_path}")


def write_cycle_reflection_document(project_name: str, cycle: int, session=None):

    write_cycle_context_json(project_name=project_name, cycle=cycle, session=session)

def validate_data_dir(data_path):
    """
    Ensures a pre-existing dataset directory conforms to ELD-NAS requirements.
    Must contain 'train', 'validate', and 'test' folders.
    Each folder must contain 'y.npy' and at least one 'X_*.npy'.
    """
    path = Path(data_path)
    if not path.is_absolute():
        path = Path(os.getcwd()) / data_path

    if not path.exists() or not path.is_dir():
        raise ValueError(f"Data directory '{path}' does not exist or is not a directory.")

    required_splits = ['train', 'validate', 'test']
    for split in required_splits:
        split_dir = path / split
        if not split_dir.exists() or not split_dir.is_dir():
            raise ValueError(f"Missing required split directory: {split_dir}")


        if not (split_dir / 'y.npy').exists():
            raise ValueError(f"Missing labels 'y.npy' in {split_dir}")


        x_files = list(split_dir.glob("X_*.npy"))
        if not x_files:
            raise ValueError(f"Missing signal data 'X_*.npy' in {split_dir}")

    return path

def generate_synthetic_data(output_dir):
    """Generates a tiny multimodal time series dataset."""
    N_SAMPLES = 200
    N_VAL = 50
    N_MODALITIES = 4
    LENGTH = 100
    N_CLASSES = 5

    dirs = output_dir / "data"
    dirs.mkdir(parents=True, exist_ok=True)

    def create_batch(n):
        X = np.zeros((n, N_MODALITIES, LENGTH), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        t = np.linspace(0, 4*np.pi, LENGTH)

        for i in range(n):
            label = np.random.randint(0, N_CLASSES)
            y[i] = label


            noise = np.random.normal(0, 0.5, (N_MODALITIES, LENGTH))

            if label == 0:
                for m in range(N_MODALITIES):
                    X[i, m] = np.sin(t) + np.sin(t * 0.5) + noise[m]
            elif label == 1:
                for m in range(N_MODALITIES):
                    X[i, m] = np.sin(t * 3) + np.sin(t * 5) + noise[m]
            elif label == 2:
                for m in range(N_MODALITIES):
                    X[i, m] = (t / 10.0) + (1.0 if i % 2 == 0 else -1.0) + noise[m]
            elif label == 3:
                for m in range(N_MODALITIES):
                    idx = np.random.randint(20, 80)
                    X[i, m] = noise[m]
                    X[i, m, idx:idx+10] += 5.0
            elif label == 4:
                X[i] = np.random.normal(0, 1.0, (N_MODALITIES, LENGTH))

        return torch.tensor(X), torch.tensor(y)

    print("Generating synthetic data...")
    train_x, train_y = create_batch(N_SAMPLES)
    val_x, val_y = create_batch(N_VAL)

    torch.save({'x': train_x, 'y': train_y}, dirs / "train.pt")
    torch.save({'x': val_x, 'y': val_y}, dirs / "val.pt")
    print(f"Saved train.pt ({N_SAMPLES}) and val.pt ({N_VAL}) to {dirs}")

def create_project(project_name: str, dataset_path: str, context_metadata: str = None):

    validate_data_dir(dataset_path)

    base = Path("projects") / project_name
    if base.exists():
        print(f"Project '{project_name}' already exists.")
        return


    for dir_path in [
        base / "artifacts" / "cycle_history",
        base / "artifacts" / "models",
        base / "shared" / "context",
        base / "shared" / "context" / "manifests",
        base / "shared" / "inbound",
        base / "shared" / "models",
        base / "shared" / "outbound",
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

    abs_dataset_path = Path(dataset_path).absolute()

    print(f"Creating project '{project_name}' structure bound to {abs_dataset_path}...")


    config = {
        "project_name": project_name,
        "director_loop_interval": 30,
        "dataset_path": str(abs_dataset_path),
        "ensemble_architecture": "default",
    }
    with open(base / "config.yaml", "w") as f:
        yaml.dump(config, f)

    with open(base / "shared" / "context" / "project_bootstrap.md", "w") as f:
        f.write("# Project Bootstrap\n\nInitial scaffold created. Cycle manifests are stored per cycle.")

    with open(base / "artifacts" / "expert_matrix.json", "w") as f:
        f.write("{}")


    if context_metadata and Path(context_metadata).exists():
        try:
            with open(context_metadata, "r") as f:
                meta = json.load(f)

            train_prop = float(meta.get('train_proportion', 1.0))
            config["train_proportion"] = train_prop

            ensemble_architecture = str(meta.get("ensemble_architecture", "default")).strip().lower()
            if ensemble_architecture not in {"default", "simple"}:
                ensemble_architecture = "default"
            config["ensemble_architecture"] = ensemble_architecture

            with open(base / "config.yaml", "w") as f:
                yaml.dump(config, f)

            context_str = f"# Project Objective\n{meta.get('project_description', 'No description provided.')}\n\n"

            context_str += "## Signal Features (X)\n"
            for sig, desc_data in meta.get('signal_metadata', {}).items():
                if isinstance(desc_data, dict):
                    modality = desc_data.get('modality', 'Unspecified')
                    desc_text = desc_data.get('description', 'No description provided.')
                    context_str += f"- **{sig}**: [Modality: {modality}] {desc_text}\n"
                else:
                    context_str += f"- **{sig}**: {desc_data}\n"

            context_str += "\n## Output Labels (y)\n"
            for cls, desc in meta.get('class_metadata', {}).items():
                context_str += f"- **Class {cls}**: {desc}\n"

            with open(base / "shared" / "context" / "data_context.md", "w") as f:
                f.write(context_str)
        except Exception as e:
            print(f"Warning: Failed to parse and structure context metadata: {e}")

    with open(base / "shared" / "inbound" / "user_inbox.json", "w") as f:
        f.write("[]")


    try:
        y_path = abs_dataset_path / "train" / "y.npy"
        if y_path.exists():
            print("Generating initial data statistics...")
            y_data = np.load(y_path)
            unique_classes = np.unique(y_data)


            class_sample_indices = {}
            for cls in unique_classes:
                cls_indices = np.where(y_data == cls)[0]
                if len(cls_indices) > 10:
                    cls_indices = np.random.choice(cls_indices, 10, replace=False)
                class_sample_indices[str(cls)] = cls_indices

            stats_dict = {}
            for x_file in (abs_dataset_path / "train").glob("X_*.npy"):
                channel_name = x_file.stem[2:]
                x_data = np.load(x_file)

                stats_dict[channel_name] = {}
                for cls_str, cls_indices in class_sample_indices.items():

                    x_sample = x_data[cls_indices]


                    ch_stats = compute_channel_statistics(x_sample, sampling_rate=100.0)
                    stats_dict[channel_name][cls_str] = ch_stats

            stats_out_path = base / "shared" / "context" / "data_statistics.json"
            with open(stats_out_path, "w") as f:
                json.dump(stats_dict, f, indent=2)
            print(f"Data statistics written to {stats_out_path}")

    except Exception as e:
        print(f"Warning: Failed to generate data statistics: {e}")
        traceback.print_exc()


def export_cycle0_shared_artifacts(project_name: str):
    project_root = (Path("projects") / project_name).absolute()
    shared_root = project_root / "shared"
    shared_context = shared_root / "context"
    shared_models = shared_root / "models"
    shared_outbound = shared_root / "outbound"

    shared_context.mkdir(parents=True, exist_ok=True)
    shared_models.mkdir(parents=True, exist_ok=True)
    shared_outbound.mkdir(parents=True, exist_ok=True)

    architecture_src = Path("local/scripts/binary_expert_model.py")
    architecture_dst = shared_models / "cycle0_baseline_models.py"
    if architecture_src.exists():
        shutil.copyfile(architecture_src, architecture_dst)
    elif not architecture_dst.exists():
        architecture_dst.write_text(
            "# Baseline architecture source missing.\n",
            encoding="utf-8",
        )

    preprocessing_src = Path("local/scripts/cycle_preprocessing.py")
    preprocessing_dst = shared_models / "cycle0_preprocessing.py"
    if preprocessing_src.exists():
        shutil.copyfile(preprocessing_src, preprocessing_dst)
    elif not preprocessing_dst.exists():
        preprocessing_dst.write_text(
            "# Baseline preprocessing source missing.\n",
            encoding="utf-8",
        )

    architecture_ref = "shared/models/cycle0_baseline_models.py"
    preprocessing_ref = "shared/models/cycle0_preprocessing.py"

    expert_matrix_path = project_root / "artifacts" / "expert_matrix.json"
    if not expert_matrix_path.exists():
        raise FileNotFoundError(f"Missing Cycle 0 artifact: {expert_matrix_path}")

    with open(expert_matrix_path, "r", encoding="utf-8") as f:
        expert_matrix = json.load(f)

    training_curves_payload = read_json_file(project_root / "artifacts" / "training_curves_cycle0.json", {})
    expert_curve_lookup = {}
    if isinstance(training_curves_payload, dict):
        experts_rows = training_curves_payload.get("experts", [])
        if isinstance(experts_rows, list):
            for row in experts_rows:
                if not isinstance(row, dict):
                    continue
                modality_key = str(row.get("modality", "")).strip()
                class_key = str(row.get("class_label", "")).strip()
                if not modality_key or not class_key:
                    continue
                history = row.get("history", [])
                expert_curve_lookup[(modality_key, class_key)] = {
                    "curves": build_secondary_curves_from_history(history),
                    "summary": summarize_history_for_results(history, row.get("summary")),
                }

    if not isinstance(expert_matrix, dict):
        raise TypeError("Cycle 0 expert matrix must be a JSON object")

    jobs = []
    expert_updates = []

    for modality in sorted(expert_matrix.keys(), key=sorted_key):
        by_class = expert_matrix.get(modality, {})
        if not isinstance(by_class, dict):
            continue

        for class_label in sorted(by_class.keys(), key=sorted_key):
            rec = by_class.get(class_label, {})
            if not isinstance(rec, dict):
                continue

            class_label_str = str(class_label)
            candidate_id = str(rec.get("candidate_id", f"baseline_{modality}_{class_label_str}"))
            f1 = safe_float(rec.get("f1", 0.0))
            acc = safe_float(rec.get("accuracy", 0.0))
            prec = safe_float(rec.get("precision", 0.0))
            rec_score = safe_float(rec.get("recall", 0.0))

            curve_bundle = expert_curve_lookup.get((str(modality), class_label_str), {})
            curve_payload = curve_bundle.get("curves", {}) if isinstance(curve_bundle, dict) else {}
            curve_summary = curve_bundle.get("summary", {}) if isinstance(curve_bundle, dict) else {}

            weights_path = project_root / "models" / f"{modality}_{class_label_str}" / f"{candidate_id}.pt"
            weights_ref = to_project_ref(weights_path, project_root) if weights_path.exists() else None

            secondary_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec_score,
                "f1": f1,
                "validation_split": "validate",
                "architecture_code_ref": architecture_ref,
                "preprocessing_code_ref": preprocessing_ref,
            }
            if isinstance(curve_payload, dict) and curve_payload:
                secondary_metrics.update(curve_payload)

            learning_curve_summary = summarize_history_for_results([], None)
            if isinstance(curve_summary, dict) and curve_summary:
                learning_curve_summary = {
                    "best_epoch": curve_summary.get("best_epoch"),
                    "epochs_completed": safe_int(curve_summary.get("epochs_completed", 0), 0),
                    "early_stopped": bool(curve_summary.get("early_stopped", False)),
                    "trend": curve_summary.get("trend"),
                }

            jobs.append(
                {
                    "job_id": f"cycle0_{modality}_{class_label_str}",
                    "job_type": "train_expert",
                    "target": {
                        "modality": str(modality),
                        "class_label": class_label_str,
                    },
                    "candidate": {
                        "candidate_id": candidate_id,
                        "origin": "cycle0_baseline",
                        "final_model_py_ref": architecture_ref,
                        "best_weights_ref": weights_ref,
                    },
                    "status": "success",
                    "repair": {
                        "attempted": False,
                        "attempt_count": 0,
                        "final_outcome": "not_needed",
                    },
                    "compile_status": "success",
                    "runtime_status": "success",
                    "metrics": {
                        "primary_metric": {
                            "name": "f1",
                            "value": f1,
                        },
                        "secondary_metrics": secondary_metrics,
                    },
                    "learning_curve_summary": learning_curve_summary,
                    "model_summary": {
                        "parameter_count": None,
                        "penultimate_dim": 16,
                    },
                    "runtime": {
                        "train_seconds": 0.0,
                        "peak_vram_gb": None,
                    },
                    "errors": [],
                    "artifacts": {
                        "train_log_ref": None,
                        "metrics_ref": "artifacts/expert_matrix.json",
                        "failure_trace_ref": None,
                    },
                }
            )

            expert_updates.append(
                {
                    "modality": str(modality),
                    "class_label": class_label_str,
                    "candidate_id": candidate_id,
                    "f1": f1,
                    "accuracy": acc,
                    "recall": rec_score,
                    "precision": prec,
                    "is_best": bool(rec.get("is_best", True)),
                    "preprocessing_code_ref": preprocessing_ref,
                    "final_model_py_ref": architecture_ref,
                    "history": [
                        {
                            "candidate_id": candidate_id,
                            "f1": f1,
                            "accuracy": acc,
                            "recall": rec_score,
                            "precision": prec,
                        }
                    ],
                }
            )

    if not jobs:
        jobs.append(
            {
                "job_id": "cycle0_bootstrap_noop",
                "job_type": "train_expert",
                "target": {
                    "modality": "unknown",
                    "class_label": "unknown",
                },
                "candidate": {
                    "candidate_id": "cycle0_bootstrap_noop",
                    "origin": "cycle0_baseline",
                    "final_model_py_ref": architecture_ref,
                    "best_weights_ref": None,
                },
                "status": "skipped",
                "repair": {
                    "attempted": False,
                    "attempt_count": 0,
                    "final_outcome": "not_needed",
                },
                "compile_status": "not_attempted",
                "runtime_status": "not_started",
                "metrics": {
                    "primary_metric": {
                        "name": "f1",
                        "value": 0.0,
                    },
                    "secondary_metrics": {
                        "reason": "Cycle 0 did not produce expert rows.",
                    },
                },
                "runtime": {
                    "train_seconds": 0.0,
                    "peak_vram_gb": None,
                },
                "errors": ["Cycle 0 produced no expert rows in artifacts/expert_matrix.json."],
                "artifacts": {
                    "train_log_ref": None,
                    "metrics_ref": "artifacts/expert_matrix.json",
                    "failure_trace_ref": None,
                },
            }
        )

    ensemble_metrics_path = project_root / "artifacts" / "baseline_ensemble_metrics.json"
    ensemble_metrics = None
    if ensemble_metrics_path.exists():
        with open(ensemble_metrics_path, "r", encoding="utf-8") as f:
            loaded_ensemble = json.load(f)
        if isinstance(loaded_ensemble, dict):
            ensemble_metrics = loaded_ensemble

    if ensemble_metrics:
        ensemble_metric_rows = []
        for metric_name in ("accuracy", "kappa"):
            if metric_name in ensemble_metrics:
                ensemble_metric_rows.append(
                    {
                        "name": metric_name,
                        "value": safe_float(ensemble_metrics.get(metric_name, 0.0)),
                    }
                )

        report = ensemble_metrics.get("classification_report")
        if isinstance(report, dict):
            macro_avg = report.get("macro avg")
            if isinstance(macro_avg, dict):
                if "precision" in macro_avg:
                    ensemble_metric_rows.append(
                        {
                            "name": "macro_precision",
                            "value": safe_float(macro_avg.get("precision", 0.0)),
                        }
                    )
                if "recall" in macro_avg:
                    ensemble_metric_rows.append(
                        {
                            "name": "macro_recall",
                            "value": safe_float(macro_avg.get("recall", 0.0)),
                        }
                    )
                if "f1-score" in macro_avg:
                    ensemble_metric_rows.append(
                        {
                            "name": "macro_f1",
                            "value": safe_float(macro_avg.get("f1-score", 0.0)),
                        }
                    )

        ensemble_evaluation = {
            "ran": True,
            "subset_fraction": 1.0,
            "status": "success",
            "candidate_id_used": "baseline_ensemble",
            "metrics": ensemble_metric_rows,
            "notes": "Cycle 0 baseline ensemble evaluation on the full validation split.",
        }
    else:
        ensemble_evaluation = {
            "ran": False,
            "subset_fraction": None,
            "status": "skipped",
            "candidate_id_used": None,
            "notes": "Cycle 0 ensemble metrics were not found.",
        }

    jobs_total = len(jobs)
    jobs_succeeded = sum(1 for j in jobs if j.get("status") == "success")
    jobs_failed = sum(1 for j in jobs if j.get("status") != "success")
    now_iso = utc_now_iso()
    overall_status = "success" if jobs_failed == 0 else "partial_success"

    results_payload = {
        "schema_version": "1.0",
        "directive_id": "cycle0_bootstrap",
        "cycle_id": "0",
        "project_id": project_name,
        "started_at": now_iso,
        "finished_at": now_iso,
        "overall_status": overall_status,
        "execution_summary": {
            "jobs_total": jobs_total,
            "jobs_succeeded": jobs_succeeded,
            "jobs_failed": jobs_failed,
            "jobs_repaired": 0,
            "wall_time_seconds": 0.0,
        },
        "jobs": jobs,
        "ensemble_evaluation": ensemble_evaluation,
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
            "Cycle 0 baseline export includes binary expert validation metrics, "
            "ensemble validation metrics, and architecture/preprocessing code references."
        ),
    }

    results_json_path = shared_outbound / "results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
        f.write("\n")

    md_lines = [
        "# Cycle 0 Baseline Results",
        "",
        f"- Project: `{project_name}`",
        f"- Binary experts evaluated: `{jobs_total}`",
        f"- Overall status: `{overall_status}`",
        "",
        "## Architecture And Preprocessing Code",
        f"- Expert architecture/training source: `{architecture_ref}`",
        f"- Preprocessing code source: `{preprocessing_ref}`",
        "",
        "## Binary Expert Validation Metrics",
        "| Modality | Class | Candidate | F1 | Accuracy | Precision | Recall | Weights |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]

    for job in jobs:
        target = job.get("target", {})
        candidate = job.get("candidate", {})
        metrics = job.get("metrics", {}).get("secondary_metrics", {})
        md_lines.append(
            "| "
            f"{target.get('modality', 'unknown')} | "
            f"{target.get('class_label', 'unknown')} | "
            f"{candidate.get('candidate_id', 'unknown')} | "
            f"{safe_float(metrics.get('f1', 0.0)):.4f} | "
            f"{safe_float(metrics.get('accuracy', 0.0)):.4f} | "
            f"{safe_float(metrics.get('precision', 0.0)):.4f} | "
            f"{safe_float(metrics.get('recall', 0.0)):.4f} | "
            f"{candidate.get('best_weights_ref') or 'n/a'} |"
        )

    md_lines.extend([
        "",
        "## Ensemble Validation Metrics",
    ])

    if ensemble_metrics:
        md_lines.append(f"- Accuracy: `{safe_float(ensemble_metrics.get('accuracy', 0.0)):.4f}`")
        md_lines.append(f"- Kappa: `{safe_float(ensemble_metrics.get('kappa', 0.0)):.4f}`")
        md_lines.append("- Full classification report:")
        md_lines.append("```json")
        md_lines.append(json.dumps(ensemble_metrics.get("classification_report", {}), indent=2))
        md_lines.append("```")
        md_lines.append("- Confusion matrix:")
        md_lines.append("```json")
        md_lines.append(json.dumps(ensemble_metrics.get("confusion_matrix", []), indent=2))
        md_lines.append("```")
    else:
        md_lines.append("- Ensemble metrics were not available from Cycle 0.")

    results_md_path = shared_outbound / "results.md"
    results_md_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")

    successful_jobs = [j for j in jobs if j.get("status") == "success"]
    best_job = None
    if successful_jobs:
        best_job = max(
            successful_jobs,
            key=lambda j: safe_float(j.get("metrics", {}).get("primary_metric", {}).get("value", 0.0)),
        )

    manifest_lines = [
        "# Project Manifest",
        "",
        "## Baseline Initialization Completed",
        f"- Cycle 0 trained and validated `{jobs_total}` binary experts.",
        "- Director is scheduled next and will use deterministic `shared/outbound/cycle_context.json` context.",
    ]

    if best_job:
        best_target = best_job.get("target", {})
        best_value = safe_float(best_job.get("metrics", {}).get("primary_metric", {}).get("value", 0.0))
        manifest_lines.append(
            "- Best Cycle 0 expert by F1: "
            f"`({best_target.get('modality')}, {best_target.get('class_label')})` "
            f"with `F1={best_value:.4f}`."
        )

    if ensemble_metrics:
        manifest_lines.append(
            "- Baseline ensemble validation accuracy: "
            f"`{safe_float(ensemble_metrics.get('accuracy', 0.0)):.4f}`."
        )

    manifest_lines.extend([
        "",
        "## Shared Artifacts For Cycle 1",
        "- `shared/outbound/results.json`: full machine-readable Cycle 0 results.",
        "- `shared/outbound/results.md`: human-readable summary of Cycle 0 training and validation.",
        "- `shared/models/cycle0_baseline_models.py`: baseline architecture/training code reference.",
        "- `shared/models/cycle0_preprocessing.py`: preprocessing code reference.",
    ])

    manifest_path = shared_context / "manifests" / format_cycle_label(0) / "manifest.md"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(manifest_lines).rstrip() + "\n", encoding="utf-8")


def write_placeholder_local_engine_outputs(project_name: str, cycle: int):
    project_root = (Path("projects") / project_name).absolute()
    shared_root = project_root / "shared"
    inbound_path = shared_root / "inbound" / "directive.json"
    outbound_path = shared_root / "outbound"
    outbound_path.mkdir(parents=True, exist_ok=True)

    target = {
        "modality": "unknown",
        "class_label": "unknown",
    }
    candidate = {
        "candidate_id": f"cycle_{cycle}_candidate",
        "origin": "director",
        "final_model_py_ref": "shared/models/model.py",
        "best_weights_ref": None,
    }
    subset_fraction = None

    if inbound_path.exists():
        try:
            with open(inbound_path, "r", encoding="utf-8") as f:
                directive = json.load(f)

            if isinstance(directive, dict):
                subset_value = directive.get("ensemble_validation_subset_fraction")
                if isinstance(subset_value, (int, float)) and 0 < float(subset_value) <= 1:
                    subset_fraction = float(subset_value)

                jobs = directive.get("jobs")
                if isinstance(jobs, list) and jobs and isinstance(jobs[0], dict):
                    first_job = jobs[0]
                    if isinstance(first_job.get("target"), dict):
                        target = {
                            "modality": str(first_job["target"].get("modality", "unknown")),
                            "class_label": str(first_job["target"].get("class_label", "unknown")),
                        }
                    if isinstance(first_job.get("candidate"), dict):
                        candidate = {
                            "candidate_id": str(first_job["candidate"].get("candidate_id", candidate["candidate_id"])),
                            "origin": str(first_job["candidate"].get("origin", "director")),
                            "final_model_py_ref": str(first_job["candidate"].get("model_py_ref", "shared/models/model.py")),
                            "best_weights_ref": None,
                        }
        except Exception as e:
            print(f"Warning: Failed to parse directive for placeholder local-engine output: {e}")

    now_iso = utc_now_iso()
    results_payload = {
        "schema_version": "1.0",
        "directive_id": f"cycle_{cycle}_placeholder_local_engine",
        "cycle_id": str(cycle),
        "project_id": project_name,
        "started_at": now_iso,
        "finished_at": now_iso,
        "overall_status": "partial_success",
        "execution_summary": {
            "jobs_total": 1,
            "jobs_succeeded": 0,
            "jobs_failed": 0,
            "jobs_repaired": 0,
            "wall_time_seconds": 0.0,
        },
        "jobs": [
            {
                "job_id": f"cycle_{cycle}_local_engine_placeholder",
                "job_type": "train_expert",
                "target": target,
                "candidate": candidate,
                "status": "skipped",
                "repair": {
                    "attempted": False,
                    "attempt_count": 0,
                    "final_outcome": "not_needed",
                },
                "compile_status": "not_attempted",
                "runtime_status": "not_started",
                "metrics": {
                    "primary_metric": {
                        "name": "f1",
                        "value": 0.0,
                    },
                    "secondary_metrics": {
                        "note": "Local engine placeholder active. Real training/evaluation is not implemented.",
                    },
                },
                "runtime": {
                    "train_seconds": 0.0,
                    "peak_vram_gb": None,
                },
                "errors": [
                    "Local engine execution is currently a placeholder in main.py.",
                ],
                "artifacts": {
                    "train_log_ref": None,
                    "metrics_ref": None,
                    "failure_trace_ref": None,
                },
            }
        ],
        "ensemble_evaluation": {
            "ran": False,
            "subset_fraction": subset_fraction,
            "status": "skipped",
            "candidate_id_used": candidate["candidate_id"],
            "notes": "Skipped because local engine is still a placeholder.",
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
        "notes": "Placeholder local-engine output generated to keep director->local-engine orchestration consistent.",
    }

    results_json_path = outbound_path / "results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
        f.write("\n")

    results_md_path = outbound_path / "results.md"
    results_md_path.write_text(
        "# Local Engine Placeholder Result\n\n"
        f"Cycle `{cycle}` local execution is not implemented yet.\n\n"
        "- Job execution: skipped\n"
        "- Reason: placeholder local engine\n"
        "- Director can still consume this file for orchestration continuity\n",
        encoding="utf-8",
    )


def _clip_text_for_prompt(text, max_chars=12000):
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    removed = len(value) - max_chars
    return value[:max_chars] + f"\n...[TRUNCATED: removed {removed} chars]"


def _extract_first_json_object_text(raw_text):
    text = str(raw_text or "").strip()
    if not text:
        return None

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
    return None


def _resolve_path_from_ref(ref_value, project_root: Path, repo_root: Path, fallback_path: Path):
    if ref_value is None:
        return fallback_path

    text = str(ref_value).strip()
    if not text:
        return fallback_path

    candidate = Path(text)
    if candidate.is_absolute():
        return candidate

    project_candidate = project_root / candidate
    if project_candidate.exists():
        return project_candidate

    repo_candidate = repo_root / candidate
    if repo_candidate.exists():
        return repo_candidate

    return project_candidate


def _resolve_repair_target_files(project_root: Path):
    shared_root = project_root / "shared"
    directive_path = shared_root / "inbound" / "directive.json"
    fallback_model_path = shared_root / "models" / "model.py"
    fallback_preprocessing_path = shared_root / "models" / "preprocessing.py"

    directive = read_json_file(directive_path, {})
    if not isinstance(directive, dict):
        return fallback_model_path, fallback_preprocessing_path

    jobs = directive.get("jobs")
    if not isinstance(jobs, list) or not jobs or not isinstance(jobs[0], dict):
        return fallback_model_path, fallback_preprocessing_path

    candidate = jobs[0].get("candidate") if isinstance(jobs[0].get("candidate"), dict) else {}
    model_ref = candidate.get("model_py_ref")
    preprocessing_ref = candidate.get("preprocessing_py_ref")

    repo_root = Path(__file__).resolve().parent
    model_path = _resolve_path_from_ref(model_ref, project_root, repo_root, fallback_model_path)
    preprocessing_path = _resolve_path_from_ref(preprocessing_ref, project_root, repo_root, fallback_preprocessing_path)

    return model_path, preprocessing_path


def _build_ollama_repair_prompt(error_text, model_code, preprocessing_code):
    return f"""
You are fixing two Python files for a time-series binary expert training run.

A runtime error occurred. You must update ONLY these two files:
1) binary expert model file
2) preprocessing file

Return exactly one JSON object using ONE of these formats:
{{
  "model_py": "<full updated model.py content>",
  "preprocessing_py": "<full updated preprocessing.py content>",
  "reasoning": "<very short reason>"
}}
OR
{{
    "model_py_b64": "<base64 UTF-8 full updated model.py>",
    "preprocessing_py_b64": "<base64 UTF-8 full updated preprocessing.py>",
    "reasoning": "<very short reason>"
}}

Rules:
- Return JSON only. No markdown fences.
- Include BOTH code keys (either plain or `_b64`) and keep both non-empty.
- If only one file needs changes, still return the other file unchanged (full source text).
- Keep valid Python syntax.
- Keep class `BinaryExpertModel` and methods `extract_features` and `forward` in model file.
- Keep function `apply_preprocessing(x)` in preprocessing file.
- Do not reference or edit any files other than these two.
- Focus on fixing the provided error.
- If using plain string fields, escape all quotes/backslashes as valid JSON.
- If using plain string fields, avoid triple-double-quoted docstrings (use comments or triple-single-quoted strings).

RUNTIME ERROR
```text
{_clip_text_for_prompt(error_text, max_chars=8000)}
```

CURRENT MODEL FILE
```python
{_clip_text_for_prompt(model_code, max_chars=14000)}
```

CURRENT PREPROCESSING FILE
```python
{_clip_text_for_prompt(preprocessing_code, max_chars=14000)}
```
""".strip()


def _strip_single_code_fence(value: str) -> str:
    text = str(value or "").strip()
    match = re.match(r"^```(?:[A-Za-z0-9_+.-]+)?\s*\n([\s\S]*?)\n```$", text)
    if match:
        return match.group(1).strip()
    return text


def _first_non_empty_value(payload: dict, keys):
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _strip_single_code_fence(value)
    return None


def _decode_base64_utf8(text: str):
    value = str(text or "").strip()
    if not value:
        return None
    try:
        decoded_bytes = base64.b64decode(value, validate=True)
        return decoded_bytes.decode("utf-8")
    except Exception:
        return None


def _decode_jsonish_escaped_string(raw_value: str):
    value = str(raw_value or "")
    if not value:
        return value


    try:
        return json.loads(f'"{value}"')
    except Exception:
        pass


    out = value
    out = out.replace("\\r\\n", "\n")
    out = out.replace("\\n", "\n")
    out = out.replace("\\t", "\t")
    out = out.replace('\\"', '"')
    out = out.replace("\\\\", "\\")
    return out


def _extract_jsonish_field(raw_text: str, field_keys, next_keys=(), terminal_ok: bool = False):
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    if not field_keys:
        return None

    field_pattern = "|".join(re.escape(str(k)) for k in field_keys)

    if next_keys:
        next_pattern = "|".join(re.escape(str(k)) for k in next_keys)
        pattern = re.compile(
            rf'"(?:{field_pattern})"\s*:\s*"(?P<value>[\s\S]*?)"\s*,\s*"(?:{next_pattern})"\s*:',
            flags=re.IGNORECASE,
        )
        match = pattern.search(raw_text)
        if match:
            return _decode_jsonish_escaped_string(match.group("value")).strip()

    if terminal_ok:
        pattern = re.compile(
            rf'"(?:{field_pattern})"\s*:\s*"(?P<value>[\s\S]*?)"\s*,?\s*}}',
            flags=re.IGNORECASE,
        )
        match = pattern.search(raw_text)
        if match:
            return _decode_jsonish_escaped_string(match.group("value")).strip()

    return None


def _salvage_ollama_json_like_output(raw_output):
    model_keys = (
        "model_py",
        "model",
        "model_code",
        "updated_model_py",
        "modelPython",
    )
    preprocessing_keys = (
        "preprocessing_py",
        "preprocessing",
        "preprocessing_code",
        "updated_preprocessing_py",
        "preprocessingPython",
    )
    reasoning_keys = ("reasoning", "reason", "note", "notes")

    raw_text = str(raw_output or "")
    if not raw_text.strip():
        return None, None, "salvage_empty"

    payload_text = _extract_first_json_object_text(raw_text)
    if not payload_text:
        payload_text = _strip_single_code_fence(raw_text)
    payload_text = str(payload_text or "").strip()

    model_code = _extract_jsonish_field(
        payload_text,
        model_keys,
        next_keys=preprocessing_keys + reasoning_keys,
        terminal_ok=True,
    )
    preprocessing_code = _extract_jsonish_field(
        payload_text,
        preprocessing_keys,
        next_keys=reasoning_keys,
        terminal_ok=True,
    )

    if not model_code and not preprocessing_code:

        code_blocks = re.findall(r"```(?:python|py)?\s*([\s\S]*?)```", raw_text, flags=re.IGNORECASE)
        code_blocks = [str(block).strip() for block in code_blocks if str(block).strip()]
        if len(code_blocks) >= 2:
            return code_blocks[0], code_blocks[1], "salvaged_code_fences"
        return None, None, "salvage_failed"

    if model_code and preprocessing_code:
        return model_code, preprocessing_code, "salvaged_json_like"
    if model_code:
        return model_code, None, "salvaged_json_like_missing_preprocessing"
    return None, preprocessing_code, "salvaged_json_like_missing_model"


def _parse_ollama_repair_output(raw_output, current_model_code: str, current_pre_code: str):
    payload_text = _extract_first_json_object_text(raw_output)
    if not payload_text:
        salvaged_model, salvaged_pre, salvage_note = _salvage_ollama_json_like_output(raw_output)
        if salvaged_model is None and salvaged_pre is None:
            return None, None, "no_json_object"

        model_code = salvaged_model if salvaged_model is not None else current_model_code
        preprocessing_code = salvaged_pre if salvaged_pre is not None else current_pre_code
        return model_code, preprocessing_code, salvage_note

    try:
        payload = json.loads(payload_text)
    except Exception:
        salvaged_model, salvaged_pre, salvage_note = _salvage_ollama_json_like_output(raw_output)
        if salvaged_model is None and salvaged_pre is None:
            return None, None, "json_decode_failed"

        model_code = salvaged_model if salvaged_model is not None else current_model_code
        preprocessing_code = salvaged_pre if salvaged_pre is not None else current_pre_code
        return model_code, preprocessing_code, salvage_note

    if not isinstance(payload, dict):
        return None, None, "json_not_object"

    model_code = _first_non_empty_value(
        payload,
        (
            "model_py",
            "model",
            "model_code",
            "updated_model_py",
            "modelPython",
        ),
    )
    preprocessing_code = _first_non_empty_value(
        payload,
        (
            "preprocessing_py",
            "preprocessing",
            "preprocessing_code",
            "updated_preprocessing_py",
            "preprocessingPython",
        ),
    )

    used_b64 = False
    if not model_code:
        model_b64 = _first_non_empty_value(
            payload,
            (
                "model_py_b64",
                "model_b64",
                "updated_model_py_b64",
            ),
        )
        decoded_model = _decode_base64_utf8(model_b64) if model_b64 else None
        if isinstance(decoded_model, str) and decoded_model.strip():
            model_code = decoded_model
            used_b64 = True

    if not preprocessing_code:
        preprocessing_b64 = _first_non_empty_value(
            payload,
            (
                "preprocessing_py_b64",
                "preprocessing_b64",
                "updated_preprocessing_py_b64",
            ),
        )
        decoded_pre = _decode_base64_utf8(preprocessing_b64) if preprocessing_b64 else None
        if isinstance(decoded_pre, str) and decoded_pre.strip():
            preprocessing_code = decoded_pre
            used_b64 = True

    if not isinstance(model_code, str) or not model_code.strip():
        model_code = None
    if not isinstance(preprocessing_code, str) or not preprocessing_code.strip():
        preprocessing_code = None

    if model_code is None and preprocessing_code is None:
        return None, None, "json_missing_patch_fields"

    parse_note = "ok_b64" if used_b64 else "ok"
    if model_code is None:
        model_code = current_model_code
        parse_note = "ok_partial_missing_model"
    if preprocessing_code is None:
        preprocessing_code = current_pre_code
        parse_note = "ok_partial_missing_preprocessing" if parse_note == "ok" else "ok_partial_missing_both_handled"

    return model_code, preprocessing_code, parse_note


def _print_subprocess_output(result):
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)


def _run_subprocess_live(cmd, env=None):
    """Run a subprocess and stream stdout/stderr live while collecting output text."""
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="" if line.endswith("\n") else "\n")
            lines.append(line)
        process.stdout.close()

    return_code = process.wait()
    combined_output = "".join(lines)
    return return_code, combined_output, ""


def _stop_ollama_model(container_name: str, model_name: str, *, timeout_seconds: int = 30, quiet: bool = False) -> bool:
    cmd = ["docker", "exec", container_name, "ollama", "stop", model_name]
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=max(5, int(timeout_seconds)),
        )
    except FileNotFoundError:
        if not quiet:
            print("Warning: docker command not found. Cannot stop Ollama model.")
        return False
    except subprocess.TimeoutExpired:
        if not quiet:
            print(f"Warning: timed out while stopping Ollama model '{model_name}'.")
        return False
    except Exception as exc:
        if not quiet:
            print(f"Warning: failed to stop Ollama model '{model_name}': {exc}")
        return False

    stdout_text = (result.stdout or "").strip()
    stderr_text = (result.stderr or "").strip()
    if result.returncode == 0:
        if not quiet:
            message = stdout_text or f"Stopped Ollama model '{model_name}'."
            print(message)
        return True

    combined = (stdout_text + "\n" + stderr_text).strip().lower()

    benign_tokens = ("not found", "not running", "no such model", "not loaded")
    if any(token in combined for token in benign_tokens):
        if not quiet:
            print(f"Ollama model '{model_name}' was already unloaded.")
        return True

    if not quiet:
        print(
            f"Warning: ollama stop returned exit code {result.returncode} for model '{model_name}'."
        )
        if stdout_text:
            print(stdout_text)
        if stderr_text:
            print(stderr_text)
    return False


def _extract_failed_job_error_details(results_json_path: Path) -> str:
    if not isinstance(results_json_path, Path) or not results_json_path.exists():
        return ""

    payload = read_json_file(results_json_path, {})
    if not isinstance(payload, dict):
        return ""

    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return ""

    failed_jobs = [
        job for job in jobs
        if isinstance(job, dict) and str(job.get("status", "")).lower() == "failed"
    ]
    if not failed_jobs:
        return ""

    lines = []
    cycle_id = str(payload.get("cycle_id", "")).strip()
    if cycle_id:
        lines.append(f"cycle_id={cycle_id}")

    for idx, job in enumerate(failed_jobs[:3], start=1):
        target = job.get("target") if isinstance(job.get("target"), dict) else {}
        modality = str(target.get("modality", "unknown")).strip() or "unknown"
        class_label = str(target.get("class_label", "unknown")).strip() or "unknown"
        runtime_status = str(job.get("runtime_status", "unknown")).strip() or "unknown"
        lines.append(
            f"failed_job[{idx}]: modality={modality}, class_label={class_label}, runtime_status={runtime_status}"
        )

        errors = job.get("errors") if isinstance(job.get("errors"), list) else []
        for err_idx, err in enumerate(errors[:5], start=1):
            err_text = str(err).strip()
            if err_text:
                lines.append(f"failed_job[{idx}]_error[{err_idx}]: {err_text}")

    notes = payload.get("notes")
    if isinstance(notes, str) and notes.strip():
        lines.append(f"results_notes: {notes.strip()}")

    return "\n".join(lines).strip()


def _build_local_engine_error_context(stdout_text: str, stderr_text: str, results_json_path: Path):
    sections = []
    failed_job_details = _extract_failed_job_error_details(results_json_path)
    if failed_job_details:
        sections.append("FAILED JOB DETAILS (results.json)\n" + failed_job_details)

    process_output = ((stdout_text or "") + "\n" + (stderr_text or "")).strip()
    if process_output:
        sections.append("LOCAL ENGINE PROCESS OUTPUT\n" + process_output)

    context_text = "\n\n".join(sections).strip()
    return _clip_text_for_prompt(context_text, max_chars=12000), failed_job_details


def _write_json_pretty(path: Path, payload: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        tmp_path.replace(path)
        return True
    except Exception as exc:
        print(f"Warning: failed to write JSON at {path}: {exc}")
        return False


def _apply_local_repair_report_to_results_payload(results_payload: dict, repair_report: dict) -> dict:
    if not isinstance(results_payload, dict):
        return results_payload
    if not isinstance(repair_report, dict):
        return results_payload

    payload = dict(results_payload)
    attempted = bool(repair_report.get("attempted", False))
    succeeded = bool(repair_report.get("succeeded", False))
    attempt_count = max(0, safe_int(repair_report.get("attempt_count", 0), 0))

    payload["local_repair"] = repair_report

    execution_summary = payload.get("execution_summary") if isinstance(payload.get("execution_summary"), dict) else {}
    execution_summary["local_repair_attempted"] = attempted
    execution_summary["local_repair_attempt_count"] = attempt_count
    execution_summary["local_repair_succeeded"] = succeeded
    if succeeded:
        execution_summary["jobs_repaired"] = max(1, safe_int(execution_summary.get("jobs_repaired", 0), 0))
    payload["execution_summary"] = execution_summary

    jobs = payload.get("jobs") if isinstance(payload.get("jobs"), list) else []
    train_jobs = [
        job for job in jobs
        if isinstance(job, dict) and str(job.get("job_type", "")).strip().lower() == "train_expert"
    ]

    failed_train_jobs = [
        job for job in train_jobs
        if str(job.get("status", "")).strip().lower() == "failed"
    ]

    targets = failed_train_jobs if failed_train_jobs else train_jobs[:1]
    for job in targets:
        repair_payload = job.get("repair") if isinstance(job.get("repair"), dict) else {}
        repair_payload["attempted"] = attempted
        repair_payload["attempt_count"] = attempt_count
        repair_payload["final_outcome"] = "success" if succeeded else ("failed" if attempted else "not_attempted")
        repair_payload["last_updated_at"] = utc_now_iso()
        job["repair"] = repair_payload

    payload["jobs"] = jobs
    return payload


def _persist_local_repair_report(
    *,
    project_root: Path,
    cycle: int,
    results_json_path: Path,
    repair_report: dict,
) -> None:
    if not isinstance(repair_report, dict):
        return

    cycle_label = format_cycle_label(cycle)
    report_out_path = project_root / "artifacts" / "local_repair_logs" / f"{cycle_label}.json"
    _write_json_pretty(report_out_path, repair_report)

    cycle_results_path = project_root / "artifacts" / "cycle_history" / cycle_label / "results.json"
    result_paths = []
    for path in (results_json_path, cycle_results_path):
        if isinstance(path, Path) and path not in result_paths:
            result_paths.append(path)

    for path in result_paths:
        if not path.exists():
            continue
        current_payload = read_json_file(path, None)
        if not isinstance(current_payload, dict):
            continue
        updated_payload = _apply_local_repair_report_to_results_payload(current_payload, repair_report)
        _write_json_pretty(path, updated_payload)


def _attempt_ollama_local_engine_repair(project_root: Path, script_path: Path, env: dict, initial_error_text: str):
    max_attempts = max(1, safe_int(os.environ.get("ARL_OLLAMA_REPAIR_MAX_ATTEMPTS", 10), 10))
    container_name = os.environ.get("ARL_OLLAMA_CONTAINER", "auto_research_lab-ollama-1")
    model_name = os.environ.get("ARL_OLLAMA_MODEL", "qwen3.5:9b")
    timeout_seconds = max(30, safe_int(os.environ.get("ARL_OLLAMA_REPAIR_TIMEOUT_SECONDS", 240), 240))

    repair_report = {
        "schema_version": "1.0",
        "triggered": True,
        "attempted": False,
        "succeeded": False,
        "attempt_count": 0,
        "max_attempts": int(max_attempts),
        "container_name": str(container_name),
        "model_name": str(model_name),
        "timeout_seconds": int(timeout_seconds),
        "started_at": utc_now_iso(),
        "finished_at": None,
        "final_status": "not_started",
        "initial_error_excerpt": _clip_text_for_prompt(initial_error_text, max_chars=2000),
        "attempts": [],
    }

    def _finish_report(status: str, succeeded: bool = False):
        repair_report["final_status"] = str(status)
        repair_report["succeeded"] = bool(succeeded)
        repair_report["finished_at"] = utc_now_iso()
        return repair_report

    model_path, preprocessing_path = _resolve_repair_target_files(project_root)
    repair_report["target_model_path"] = str(model_path)
    repair_report["target_preprocessing_path"] = str(preprocessing_path)

    if not model_path.exists() or not preprocessing_path.exists():
        print(
            "Warning: Ollama repair skipped because target files are missing: "
            f"model={model_path}, preprocessing={preprocessing_path}"
        )
        return _finish_report("skipped_missing_targets", succeeded=False)

    print(
        "Attempting local Ollama repair for Local Engine failure "
        f"(max {max_attempts} attempts, container={container_name}, model={model_name}, timeout={timeout_seconds}s)."
    )

    results_json_env = str(env.get("ARL_RESULTS_JSON_PATH", "")).strip()
    results_json_path = Path(results_json_env) if results_json_env else (project_root / "shared" / "outbound" / "results.json")
    repair_report["results_json_path"] = str(results_json_path)

    last_error_text = initial_error_text or "Unknown local engine error"

    for attempt in range(1, max_attempts + 1):
        attempt_report = {
            "attempt": int(attempt),
            "started_at": utc_now_iso(),
            "status": "started",
        }
        repair_report["attempted"] = True
        repair_report["attempt_count"] = int(attempt)


        _stop_ollama_model(
            container_name,
            model_name,
            timeout_seconds=min(30, timeout_seconds),
            quiet=True,
        )

        try:
            current_model_code = model_path.read_text(encoding="utf-8")
            current_pre_code = preprocessing_path.read_text(encoding="utf-8")
        except Exception as read_error:
            print(f"Warning: Failed to read repair target files: {read_error}")
            attempt_report["status"] = "failed_read_targets"
            attempt_report["error"] = _clip_text_for_prompt(str(read_error), max_chars=500)
            attempt_report["finished_at"] = utc_now_iso()
            repair_report["attempts"].append(attempt_report)
            return _finish_report("failed_read_targets", succeeded=False)

        prompt = _build_ollama_repair_prompt(
            error_text=last_error_text,
            model_code=current_model_code,
            preprocessing_code=current_pre_code,
        )
        prompt_chars = len(prompt)
        attempt_report["prompt_chars"] = int(prompt_chars)
        attempt_report["model_chars"] = int(len(current_model_code))
        attempt_report["preprocessing_chars"] = int(len(current_pre_code))

        print(
            f"Ollama repair attempt {attempt}/{max_attempts}: "
            f"prompt_chars={prompt_chars}, model_chars={len(current_model_code)}, "
            f"preprocessing_chars={len(current_pre_code)}"
        )

        cmd = [
            "docker",
            "exec",
            container_name,
            "ollama",
            "run",
            model_name,
            prompt,
        ]

        try:
            llm_result = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except FileNotFoundError:
            print("Warning: docker command not found. Cannot run local Ollama repair.")
            _stop_ollama_model(
                container_name,
                model_name,
                timeout_seconds=min(30, timeout_seconds),
                quiet=True,
            )
            attempt_report["status"] = "failed_docker_not_found"
            attempt_report["finished_at"] = utc_now_iso()
            repair_report["attempts"].append(attempt_report)
            return _finish_report("failed_docker_not_found", succeeded=False)
        except subprocess.TimeoutExpired as timeout_error:
            print(
                f"Ollama repair attempt {attempt}/{max_attempts} timed out after {timeout_seconds}s "
                f"(prompt_chars={prompt_chars})."
            )

            attempt_report["status"] = "timeout"
            attempt_report["timeout_seconds"] = int(timeout_seconds)

            partial_stdout = timeout_error.stdout if isinstance(timeout_error.stdout, str) else ""
            partial_stderr = timeout_error.stderr if isinstance(timeout_error.stderr, str) else ""
            if partial_stdout.strip():
                print(
                    "Ollama timeout partial stdout:\n"
                    + _clip_text_for_prompt(partial_stdout, max_chars=1200)
                )
                attempt_report["stdout_excerpt"] = _clip_text_for_prompt(partial_stdout, max_chars=1200)
            if partial_stderr.strip():
                print(
                    "Ollama timeout partial stderr:\n"
                    + _clip_text_for_prompt(partial_stderr, max_chars=1200)
                )
                attempt_report["stderr_excerpt"] = _clip_text_for_prompt(partial_stderr, max_chars=1200)

            _stop_ollama_model(
                container_name,
                model_name,
                timeout_seconds=min(30, timeout_seconds),
                quiet=False,
            )
            attempt_report["finished_at"] = utc_now_iso()
            repair_report["attempts"].append(attempt_report)
            continue
        except Exception as call_error:
            print(f"Warning: Ollama repair attempt {attempt}/{max_attempts} failed to run: {call_error}")
            _stop_ollama_model(
                container_name,
                model_name,
                timeout_seconds=min(30, timeout_seconds),
                quiet=False,
            )
            attempt_report["status"] = "call_error"
            attempt_report["error"] = _clip_text_for_prompt(str(call_error), max_chars=500)
            attempt_report["finished_at"] = utc_now_iso()
            repair_report["attempts"].append(attempt_report)
            continue

        _stop_ollama_model(
            container_name,
            model_name,
            timeout_seconds=min(30, timeout_seconds),
            quiet=False,
        )

        raw_output = (llm_result.stdout or "").strip()
        stderr_output = (llm_result.stderr or "").strip()
        attempt_report["llm_exit_code"] = int(llm_result.returncode)
        if llm_result.returncode != 0:
            print(
                f"Ollama repair attempt {attempt}/{max_attempts} returned non-zero exit code "
                f"{llm_result.returncode}."
            )
            if stderr_output:
                print(stderr_output)
                attempt_report["stderr_excerpt"] = _clip_text_for_prompt(stderr_output, max_chars=1200)

            attempt_report["status"] = "llm_nonzero_exit"
            attempt_report["finished_at"] = utc_now_iso()
            repair_report["attempts"].append(attempt_report)

            lower_err = (stderr_output or "").lower()
            if "no such container" in lower_err or "cannot connect to the docker daemon" in lower_err:
                return _finish_report("failed_container_unavailable", succeeded=False)
            continue

        fixed_model_code, fixed_pre_code, parse_note = _parse_ollama_repair_output(
            raw_output,
            current_model_code=current_model_code,
            current_pre_code=current_pre_code,
        )
        if not fixed_model_code or not fixed_pre_code:
            snippet = _clip_text_for_prompt(raw_output, max_chars=500)
            print(
                f"Ollama repair attempt {attempt}/{max_attempts} did not return valid JSON patches "
                f"(reason={parse_note}). Raw output preview:\n{snippet}"
            )
            attempt_report["status"] = "invalid_patch"
            attempt_report["parse_note"] = str(parse_note)
            attempt_report["raw_output_excerpt"] = snippet
            attempt_report["finished_at"] = utc_now_iso()
            repair_report["attempts"].append(attempt_report)
            continue

        attempt_report["parse_note"] = str(parse_note)
        if parse_note != "ok":
            print(f"Ollama repair attempt {attempt}/{max_attempts}: accepted partial patch ({parse_note}).")

        try:
            model_path.write_text(fixed_model_code.rstrip() + "\n", encoding="utf-8")
            preprocessing_path.write_text(fixed_pre_code.rstrip() + "\n", encoding="utf-8")
        except Exception as write_error:
            print(f"Warning: Failed writing Ollama repair output: {write_error}")
            attempt_report["status"] = "write_failed"
            attempt_report["error"] = _clip_text_for_prompt(str(write_error), max_chars=500)
            attempt_report["finished_at"] = utc_now_iso()
            repair_report["attempts"].append(attempt_report)
            continue

        attempt_report["status"] = "patch_applied"

        print(
            f"Ollama repair attempt {attempt}/{max_attempts}: updated "
            f"{model_path.name} and {preprocessing_path.name}. Re-running Local Engine."
        )

        rerun_code, rerun_stdout, rerun_stderr = _run_subprocess_live(["python", str(script_path)], env=env)
        attempt_report["rerun_exit_code"] = int(rerun_code)

        if rerun_code == 0:
            print(f"Ollama repair succeeded on attempt {attempt}/{max_attempts}.")
            attempt_report["status"] = "repaired_success"
            attempt_report["finished_at"] = utc_now_iso()
            repair_report["attempts"].append(attempt_report)
            return _finish_report("succeeded", succeeded=True)

        rerun_error_context, rerun_failed_job_details = _build_local_engine_error_context(
            stdout_text=rerun_stdout or "",
            stderr_text=rerun_stderr or "",
            results_json_path=results_json_path,
        )
        if rerun_failed_job_details:
            print("Captured failed-job details for next Ollama retry:")
            print(_clip_text_for_prompt(rerun_failed_job_details, max_chars=2000))

        last_error_text = rerun_error_context or _clip_text_for_prompt(
            (rerun_stdout or "") + "\n" + (rerun_stderr or ""),
            max_chars=12000,
        )
        attempt_report["status"] = "rerun_failed"
        attempt_report["rerun_error_excerpt"] = _clip_text_for_prompt(last_error_text, max_chars=2000)
        attempt_report["finished_at"] = utc_now_iso()
        repair_report["attempts"].append(attempt_report)
        print(
            f"Ollama repair attempt {attempt}/{max_attempts} did not fix Local Engine "
            f"(exit code {rerun_code})."
        )

    print(f"Ollama repair exhausted {max_attempts} attempts without success.")
    return _finish_report("exhausted_failed", succeeded=False)


def run_local_engine_step(project_name: str, cycle: int) -> bool:
    print("(Local engine) Running train/eval step from generated model and preprocessing artifacts.")
    try:
        project_root = (Path("projects") / project_name).absolute()
        shared_root = project_root / "shared"
        script_path = Path("local/scripts/run_local_engine.py").resolve()

        if not script_path.exists():
            raise FileNotFoundError(f"Missing local engine runner at {script_path}")

        env = os.environ.copy()
        env["ARL_PROJECT"] = project_name
        env["ARL_PROJECT_ID"] = project_name
        env["ARL_CYCLE_ID"] = str(cycle)
        env["ARL_PROJECT_ROOT"] = str(project_root)
        env["ARL_DIRECTIVE_PATH"] = str(shared_root / "inbound" / "directive.json")
        env["ARL_MODEL_PY_PATH"] = str(shared_root / "models" / "model.py")
        env["ARL_PREPROCESSING_PY_PATH"] = str(shared_root / "models" / "preprocessing.py")
        env["ARL_MODEL_META_PATH"] = str(shared_root / "models" / "model.meta.json")
        env["ARL_RESULTS_JSON_PATH"] = str(shared_root / "outbound" / "results.json")
        env["PYTHONUNBUFFERED"] = "1"

        results_json_path = Path(env["ARL_RESULTS_JSON_PATH"])

        result_code, result_stdout, result_stderr = _run_subprocess_live(["python", str(script_path)], env=env)
        if result_code != 0:
            combined_error, failed_job_details = _build_local_engine_error_context(
                stdout_text=result_stdout or "",
                stderr_text=result_stderr or "",
                results_json_path=results_json_path,
            )

            if failed_job_details:
                print("Failed-job error details captured for Ollama repair prompt:")
                print(_clip_text_for_prompt(failed_job_details, max_chars=2000))

            repair_report = _attempt_ollama_local_engine_repair(
                project_root=project_root,
                script_path=script_path,
                env=env,
                initial_error_text=combined_error,
            )
            _persist_local_repair_report(
                project_root=project_root,
                cycle=cycle,
                results_json_path=results_json_path,
                repair_report=repair_report,
            )

            if bool(repair_report.get("succeeded", False)):
                return True

            print(f"Error: Local engine runner failed for cycle {cycle} with exit code {result_code}")
            return False
        return True
    except Exception as e:
        print(f"Error: Failed local-engine execution for cycle {cycle}: {e}")
        traceback.print_exc()
        return False


def run_step(name, script_path, project_name, session, cycle):
    print(f"\n--- Running {name} ---")
    env = os.environ.copy()
    project_root = None
    llm_profile = None
    if project_name:
        project_root = (Path("projects") / project_name).absolute()
        shared_root = project_root / "shared"

        env["ARL_PROJECT"] = project_name
        env["ARL_PROJECT_ID"] = project_name
        env["ARL_CYCLE_ID"] = str(cycle)
        env["ARL_PROJECT_ROOT"] = str(project_root)


        env["ARL_MANIFEST_PATH"] = str(
            shared_root / "context" / "manifests" / format_cycle_label(cycle) / "manifest.md"
        )
        env["ARL_DIRECTIVE_PATH"] = str(shared_root / "inbound" / "directive.json")
        env["ARL_MODEL_PY_PATH"] = str(shared_root / "models" / "model.py")
        env["ARL_PREPROCESSING_PY_PATH"] = str(shared_root / "models" / "preprocessing.py")
        env["ARL_MODEL_META_PATH"] = str(shared_root / "models" / "model.meta.json")
        env["ARL_PROPOSAL_SUMMARY_PATH"] = str(shared_root / "proposals" / "proposal_summary.json")
        env["ARL_RESULTS_JSON_PATH"] = str(shared_root / "outbound" / "results.json")
        env["ARL_CYCLE_CONTEXT_JSON_PATH"] = str(shared_root / "outbound" / "cycle_context.json")

        llm_role = STEP_TO_LLM_ROLE.get(name)
        if llm_role is not None:
            role_execution = load_project_llm_role_execution(project_root)
            selected_backend = normalize_llm_backend(role_execution.get(llm_role, "remote"))
            llm_profile = apply_llm_profile_env(env, selected_backend)
            print(
                f"{name} LLM backend: {llm_profile.get('backend')} | "
                f"model={llm_profile.get('model') or '<unset>'} | "
                f"api_url={llm_profile.get('api_url') or '<unset>'}"
            )

            if not llm_profile.get("has_api_key"):
                print(
                    f"Warning: {name} backend '{selected_backend}' has no API key configured "
                    "(ARL_LLM_API_KEY / ARL_REMOTE_LLM_API_KEY / ARL_LOCAL_LLM_API_KEY)."
                )
            if not (llm_profile.get("model") or "").strip():
                print(
                    f"Warning: {name} backend '{selected_backend}' has no model configured "
                    "(ARL_LLM_MODEL / ARL_REMOTE_LLM_MODEL / ARL_LOCAL_LLM_MODEL)."
                )


    if session and project_name:
        log_entry = ExecutionLog(
            project_name=project_name,
            cycle=cycle,
            step_name=name,
            status="Started"
        )
        session.add(log_entry)

        proj_state = session.get(ProjectState, project_name)
        if proj_state:
            proj_state.current_step = name
        session.commit()

    cmd = ["python", script_path]
    result = subprocess.run(cmd, env=env)


    outcome_log_id = None
    if session and project_name:
        log_entry = ExecutionLog(
            project_name=project_name,
            cycle=cycle,
            step_name=name,
            status="Completed" if result.returncode == 0 else "Failed"
        )
        session.add(log_entry)
        session.commit()
        outcome_log_id = log_entry.id

    if project_root is not None and session and project_name and outcome_log_id is not None:
        wait_for_manual_confirmation(
            project_name=project_name,
            project_root=project_root,
            session=session,
            cycle=cycle,
            step_name=name,
            log_id=outcome_log_id,
        )

    if result.returncode != 0:
        print(f"Error: {name} failed with exit code {result.returncode}")
        return False
    return True


def determine_start_cycle(session, project_name: str) -> tuple[int, str]:
    """Return the cycle to start from when a run process is launched."""
    if not project_name:
        return 0, "no project specified"

    proj_state = session.get(ProjectState, project_name)
    if proj_state:
        stored_cycle = safe_int(getattr(proj_state, "current_cycle", 0), 0)
        if stored_cycle > 0:
            return stored_cycle, "project_state.current_cycle"

    cycle0_completed_log = (
        session.query(ExecutionLog.id)
        .filter(
            ExecutionLog.project_name == project_name,
            ExecutionLog.cycle == 0,
            ExecutionLog.step_name == "Cycle 0",
            ExecutionLog.status == "Completed",
        )
        .first()
    )
    if cycle0_completed_log is not None:
        return 1, "cycle0 completion log"

    project_root = (Path("projects") / project_name).absolute()
    cycle0_manifest = project_root / "shared" / "context" / "manifests" / "cycle_0000" / "manifest.md"
    cycle0_results = project_root / "shared" / "outbound" / "results.json"
    if cycle0_manifest.exists() and cycle0_results.exists():
        return 1, "cycle0 artifacts"

    return 0, "default"

def cmd_run(args):
    """Execute the core machine learning loop."""
    project = args.project_name or os.getenv("ARL_PROJECT")
    print("Starting ELD-NAS...")
    print(f"Active Project: {project if project else 'Default (Root)'}")

    scripts = {
        "Cycle0": "local/scripts/run_cycle.py",
        "Director": "remote/scripts/run_director.py",
    }

    db_path = os.environ.get("ARL_DB_PATH", "sqlite:///arl_status.db")
    _, SessionLocal = get_engine_and_session(db_path)

    start_cycle = 0
    start_reason = "default"
    with SessionLocal() as session:
        if project:
            start_cycle, start_reason = determine_start_cycle(session, project)

    try:
        if start_cycle <= 0:
            print(f"\n=== Starting Cycle 0 (Baseline) ===")
            with SessionLocal() as session:
                if project:
                    proj_state = session.get(ProjectState, project)
                    if not proj_state:
                        proj_state = ProjectState(project_name=project)
                        session.add(proj_state)
                    proj_state.current_cycle = 0
                    proj_state.status = 'Running'
                    proj_state.target_status = 'Running'
                    proj_state.pid = os.getpid()
                    session.commit()

                    if honor_pause_or_stop_signal(session, project, 0) == "stopped":
                        return

                if not run_step("Cycle 0", scripts["Cycle0"], project, session, 0):
                    if project:
                        proj_state = session.get(ProjectState, project)
                        if proj_state:
                            proj_state.status = 'Error'
                            session.commit()
                    return

                if project:
                    try:
                        export_cycle0_shared_artifacts(project)
                        print("Cycle 0 shared artifacts exported for deterministic context + Director handoff.")
                        write_cycle_context_json(project_name=project, cycle=1, session=session)

                        project_root = (Path("projects") / project).absolute()
                        cycle0_results = read_json_file(project_root / "shared" / "outbound" / "results.json", {})
                        finished, precision, recall = should_finish_project_on_ensemble_metrics(project_root, cycle0_results)
                        if finished:
                            print(
                                "Project completion condition met after Cycle 0: "
                                f"ensemble precision={safe_float(precision, 0.0):.4f}, "
                                f"recall={safe_float(recall, 0.0):.4f}. Stopping run."
                            )
                            finalize_project_as_finished(session, project, 0, precision, recall)
                            return
                    except Exception as e:
                        print(f"Error: Failed to export Cycle 0 shared artifacts: {e}")
                        traceback.print_exc()
                        proj_state = session.get(ProjectState, project)
                        if proj_state:
                            proj_state.status = 'Error'
                            session.commit()
                        return

            cycle = 1
        else:
            cycle = safe_int(start_cycle, 1)
            print(f"\n=== Resuming From Cycle {cycle} ({start_reason}) ===")
            with SessionLocal() as session:
                if project:
                    proj_state = session.get(ProjectState, project)
                    if not proj_state:
                        proj_state = ProjectState(project_name=project)
                        session.add(proj_state)
                    proj_state.current_cycle = cycle
                    proj_state.status = 'Running'
                    proj_state.target_status = 'Running'
                    proj_state.pid = os.getpid()
                    if str(proj_state.current_step or '').strip().lower() in ('paused', 'pause requested', 'booting', 'idle'):
                        proj_state.current_step = 'Resuming'
                    session.commit()

                    if honor_pause_or_stop_signal(session, project, cycle) == "stopped":
                        return

        while True:
            print(f"\n=== Starting Cycle {cycle} ===")

            with SessionLocal() as session:
                if project:
                    proj_state = session.get(ProjectState, project)
                    if not proj_state:
                        proj_state = ProjectState(project_name=project)
                        session.add(proj_state)

                    control_state = honor_pause_or_stop_signal(session, project, cycle)
                    if control_state == "stopped":
                        break

                    proj_state.current_cycle = cycle
                    proj_state.status = 'Running'
                    proj_state.pid = os.getpid()
                    session.commit()

                if project:
                    try:
                        write_cycle_context_json(project_name=project, cycle=cycle, session=session)
                    except Exception as context_error:
                        print(f"Error: Failed to build deterministic cycle context JSON: {context_error}")
                        if project:
                            proj_state.status = 'Error'
                            session.commit()
                        break

                if project and honor_pause_or_stop_signal(session, project, cycle) == "stopped":
                    break

                if project and honor_pause_or_stop_signal(session, project, cycle) == "stopped":
                    break

                if not run_step("Director", scripts["Director"], project, session, cycle):
                    if project:
                        proj_state.status = 'Error'
                        session.commit()
                    break

                if project and honor_pause_or_stop_signal(session, project, cycle) == "stopped":
                    break

                print("\n--- Running Local Engine ---")
                if project:
                    log_entry = ExecutionLog(project_name=project, cycle=cycle, step_name="Local Engine", status="Started")
                    session.add(log_entry)
                    proj_state.current_step = "Local Engine"
                    session.commit()

                local_engine_ok = bool(project) and run_local_engine_step(project, cycle)
                if not local_engine_ok:
                    if project:
                        log_entry = ExecutionLog(project_name=project, cycle=cycle, step_name="Local Engine", status="Failed")
                        session.add(log_entry)
                        session.commit()

                        try:
                            write_cycle_context_json(project_name=project, cycle=cycle + 1, session=session)
                        except Exception as context_error:
                            print(f"Warning: Failed to refresh deterministic cycle context JSON: {context_error}")

                        wait_for_manual_confirmation(
                            project_name=project,
                            project_root=(Path("projects") / project).absolute(),
                            session=session,
                            cycle=cycle,
                            step_name="Local Engine",
                            log_id=log_entry.id,
                        )


                        proj_state.status = 'Running'
                        proj_state.current_step = 'Local Engine Failed (Continuing)'
                        proj_state.pid = os.getpid()
                        session.commit()

                        if getattr(args, "dry_run", False):
                            print("Completed full cycle. Exiting dry run.")
                            proj_state.status = 'Stopped'
                            proj_state.current_step = 'Idle'
                            proj_state.target_status = 'Stopped'
                            proj_state.pid = None
                            session.commit()
                            break

                        print(
                            "Warning: Local engine failed for this cycle. "
                            "Continuing to next cycle so Director can adapt."
                        )
                        cycle += 1
                        continue

                    break

                time.sleep(2)
                if project:
                    log_entry = ExecutionLog(project_name=project, cycle=cycle, step_name="Local Engine", status="Completed")
                    session.add(log_entry)
                    session.commit()

                    try:
                        write_cycle_context_json(project_name=project, cycle=cycle + 1, session=session)
                    except Exception as context_error:
                        print(f"Warning: Failed to refresh deterministic cycle context JSON: {context_error}")

                    wait_for_manual_confirmation(
                        project_name=project,
                        project_root=(Path("projects") / project).absolute(),
                        session=session,
                        cycle=cycle,
                        step_name="Local Engine",
                        log_id=log_entry.id,
                    )

                    project_root = (Path("projects") / project).absolute()
                    cycle_results = read_json_file(project_root / "shared" / "outbound" / "results.json", {})
                    finished, precision, recall = should_finish_project_on_ensemble_metrics(project_root, cycle_results)
                    if finished:
                        print(
                            "Project completion condition met: "
                            f"ensemble precision={safe_float(precision, 0.0):.4f}, "
                            f"recall={safe_float(recall, 0.0):.4f}. Stopping run."
                        )
                        finalize_project_as_finished(session, project, cycle, precision, recall)
                        break

                if getattr(args, "dry_run", False):
                    print("Completed full cycle. Exiting dry run.")
                    if project:
                        proj_state.status = 'Stopped'
                        proj_state.current_step = 'Idle'
                        proj_state.target_status = 'Stopped'
                        proj_state.pid = None
                        session.commit()
                    break

            cycle += 1

    except KeyboardInterrupt:
        print("\nShutting down ELD-NAS...")
        with SessionLocal() as session:
            if project:
                proj_state = session.get(ProjectState, project)
                if proj_state:
                    proj_state.status = 'Stopped'
                    proj_state.current_step = 'Idle'
                    proj_state.target_status = 'Stopped'
                    proj_state.pid = None
                    session.commit()

def main():
    parser = argparse.ArgumentParser(description="ELD-NAS CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")


    init_parser = subparsers.add_parser("init", help="Create a new ELD-NAS project scaffold")
    init_parser.add_argument("project_name", help="Name of the project directory to build")
    init_parser.add_argument("data_path", help="Path to the dataset directory to tightly bind")
    init_parser.add_argument("--context", help="Internal JSON string bridging frontend descriptions", default=None)

    run_parser = subparsers.add_parser("run", help="Start the main event loop")
    run_parser.add_argument("project_name", help="Name of the project to run")


    args = parser.parse_args()

    if args.command == "init":
        create_project(args.project_name, args.data_path, context_metadata=args.context)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
