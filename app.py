import os
import json
import subprocess
import re
import inspect
import importlib.util
import logging
import gc
from datetime import datetime, timedelta, timezone
import numpy as np
import yaml
import shutil
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from db import get_engine_and_session, ProjectState, ExecutionLog

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # type: ignore[assignment]

app = Flask(__name__)


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


class _SuppressPollingGetRequestLogsFilter(logging.Filter):
    """Hide frequent polling GET request logs while keeping other request logs."""

    _patterns = (
        re.compile(r'"GET /api/data(?:\?|\s)'),
        re.compile(r'"GET /api/project/[^/]+/training_status(?:\?|\s)'),
    )

    def filter(self, record):
        message = record.getMessage()
        if '"GET ' not in message:
            return True
        return not any(pattern.search(message) for pattern in self._patterns)


def configure_werkzeug_request_logging():
    """Install a one-time filter for noisy polling endpoints in werkzeug logs."""
    if os.environ.get("ARL_SUPPRESS_POLLING_GET_LOGS", "1") != "1":
        return

    logger = logging.getLogger("werkzeug")
    if getattr(logger, "_arl_polling_filter_installed", False):
        return

    logger.addFilter(_SuppressPollingGetRequestLogsFilter())
    logger._arl_polling_filter_installed = True


configure_werkzeug_request_logging()


DB_PATH = os.environ.get("ARL_DB_PATH", "sqlite:///arl_status.db")
_, SessionLocal = get_engine_and_session(DB_PATH)
_CYCLE0_RUNTIME_MODULE = None
_DYNAMIC_MODULE_CACHE = {}


def pid_is_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def manual_verification_state_path(project):
    return Path("projects") / project / "state" / "manual_verification.json"


def load_manual_verification_state(project):
    path = manual_verification_state_path(project)
    default_state = {
        "enabled": False,
        "confirmed_log_ids": [],
    }

    if not path.exists():
        return default_state

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        enabled = bool(raw.get("enabled", False)) if isinstance(raw, dict) else False
        confirmed_raw = raw.get("confirmed_log_ids", []) if isinstance(raw, dict) else []
        confirmed_ids = sorted({int(x) for x in confirmed_raw})
        return {
            "enabled": enabled,
            "confirmed_log_ids": confirmed_ids,
        }
    except Exception:
        return default_state


def save_manual_verification_state(project, state):
    path = manual_verification_state_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "enabled": bool(state.get("enabled", False)),
        "confirmed_log_ids": sorted({int(x) for x in state.get("confirmed_log_ids", [])}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def project_settings_state_path(project):
    return Path("projects") / project / "state" / "project_settings.json"


def parse_utc_datetime(value):
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
    else:
        parsed = parsed.astimezone(timezone.utc)

    return parsed.replace(microsecond=0)


def default_end_time_utc():
    return (datetime.now(timezone.utc) + timedelta(hours=24)).replace(microsecond=0)


def default_project_window_utc():
    start_dt = datetime.now(timezone.utc).replace(microsecond=0)
    end_dt = (start_dt + timedelta(hours=24)).replace(microsecond=0)
    return start_dt, end_dt


LLM_ROLE_NAMES = ("director",)
DEFAULT_LLM_ROLE_EXECUTION = {role: "remote" for role in LLM_ROLE_NAMES}


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


def parse_llm_role_execution_payload(value):
    if not isinstance(value, dict):
        return None, "llm_role_execution must be an object with an optional 'director' key."

    unknown_keys = [k for k in value.keys() if str(k).strip().lower() not in LLM_ROLE_NAMES]
    if unknown_keys:
        return None, (
            "Unknown llm_role_execution key(s): "
            + ", ".join(sorted(str(k) for k in unknown_keys))
            + ". Expected keys: director."
        )

    normalized = dict(DEFAULT_LLM_ROLE_EXECUTION)
    lower_keys = {str(k).strip().lower() for k in value.keys()}
    for role in LLM_ROLE_NAMES:
        if role not in lower_keys:
            continue

        raw_value = None
        for key, candidate in value.items():
            if str(key).strip().lower() == role:
                raw_value = candidate
                break

        backend = str(raw_value or "").strip().lower()
        if backend not in {"remote", "local"}:
            return None, (
                f"Invalid llm_role_execution value for '{role}': '{raw_value}'. "
                "Expected 'remote' or 'local'."
            )
        normalized[role] = backend

    return normalized, None


def load_project_settings(project):
    path = project_settings_state_path(project)
    default_start_dt, default_end_dt = default_project_window_utc()
    default_settings = {
        "start_time_utc": default_start_dt.isoformat(),
        "end_time_utc": default_end_dt.isoformat(),
        "llm_role_execution": dict(DEFAULT_LLM_ROLE_EXECUTION),
    }

    if not path.exists():
        return default_settings

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return default_settings

    if not isinstance(raw, dict):
        return default_settings

    end_dt = parse_utc_datetime(raw.get("end_time_utc"))
    if end_dt is None:
        return default_settings

    start_dt = parse_utc_datetime(raw.get("start_time_utc"))
    if start_dt is None:
        start_dt = (end_dt - timedelta(hours=24)).replace(microsecond=0)

    if start_dt >= end_dt:
        start_dt = (end_dt - timedelta(hours=24)).replace(microsecond=0)

    return {
        "start_time_utc": start_dt.isoformat(),
        "end_time_utc": end_dt.isoformat(),
        "llm_role_execution": normalize_llm_role_execution(raw.get("llm_role_execution")),
    }


def save_project_settings(project, settings):
    path = project_settings_state_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)

    end_dt = parse_utc_datetime(settings.get("end_time_utc")) if isinstance(settings, dict) else None
    if end_dt is None:
        end_dt = default_end_time_utc()

    start_dt = parse_utc_datetime(settings.get("start_time_utc")) if isinstance(settings, dict) else None
    if start_dt is None:
        start_dt = (end_dt - timedelta(hours=24)).replace(microsecond=0)

    if start_dt >= end_dt:
        start_dt = (end_dt - timedelta(hours=24)).replace(microsecond=0)

    payload = {
        "start_time_utc": start_dt.isoformat(),
        "end_time_utc": end_dt.isoformat(),
        "llm_role_execution": normalize_llm_role_execution(
            settings.get("llm_role_execution") if isinstance(settings, dict) else None
        ),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def ensure_project_settings(project):
    settings = load_project_settings(project)
    path = project_settings_state_path(project)
    if not path.exists():
        save_project_settings(project, settings)
    return settings


def project_exists(project):
    return (Path("projects") / project).exists()


def project_is_finished_state(proj_state):
    if proj_state is None:
        return False
    status_text = str(getattr(proj_state, "status", "") or "").strip().lower()
    step_text = str(getattr(proj_state, "current_step", "") or "").strip().lower()
    return status_text == "finished" or step_text.startswith("finished:")


def label_sort_key(value):
    text = str(value).strip()
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def normalize_class_label(value):
    text = str(value).strip()
    if not text:
        return text
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
    except ValueError:
        pass
    return text


def normalize_ensemble_architecture(value):
    text = str(value or "default").strip().lower()
    if text in {"default", "simple"}:
        return text
    return "default"


def parse_cycle_number(value):
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    match = re.match(r"^cycle_(\d+)$", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    try:
        number = float(text)
    except ValueError:
        return None

    if not number.is_integer():
        return None

    return int(number)


def to_finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    return float(number)


def extract_train_expert_f1(job_payload):
    if not isinstance(job_payload, dict):
        return None

    metrics = job_payload.get("metrics")
    if not isinstance(metrics, dict):
        return None

    secondary = metrics.get("secondary_metrics")
    if isinstance(secondary, dict):
        secondary_f1 = to_finite_float(secondary.get("f1"))
        if secondary_f1 is not None:
            return secondary_f1

    primary = metrics.get("primary_metric")
    if isinstance(primary, dict):
        metric_name = str(primary.get("name") or "").strip().lower()
        if metric_name == "f1":
            primary_f1 = to_finite_float(primary.get("value"))
            if primary_f1 is not None:
                return primary_f1

    return None


def extract_ensemble_kappa(ensemble_payload):
    if not isinstance(ensemble_payload, dict):
        return None

    metrics = ensemble_payload.get("metrics")
    if isinstance(metrics, list):
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            metric_name = str(metric.get("name") or "").strip().lower()
            if metric_name != "kappa":
                continue

            kappa_value = to_finite_float(metric.get("value"))
            if kappa_value is not None:
                return kappa_value

    return to_finite_float(ensemble_payload.get("kappa"))


def upsert_expert_history_point(expert_history_map, modality, class_label, cycle_value, f1_value, candidate_id):
    if f1_value is None:
        return

    modality_text = str(modality or "").strip()
    class_label_text = normalize_class_label(class_label)
    if not modality_text or not class_label_text:
        return

    expert_id = f"{modality_text}::{class_label_text}"
    if expert_id not in expert_history_map:
        expert_history_map[expert_id] = {
            "modality": modality_text,
            "class_label": class_label_text,
            "history_by_cycle": {},
        }

    cycle_history = expert_history_map[expert_id]["history_by_cycle"]
    current_cycle_entry = cycle_history.get(cycle_value)
    current_f1 = to_finite_float(current_cycle_entry.get("f1")) if isinstance(current_cycle_entry, dict) else None
    next_f1 = to_finite_float(f1_value)
    if next_f1 is None:
        return


    if current_f1 is None or next_f1 > current_f1:
        cycle_history[cycle_value] = {
            "f1": next_f1,
            "candidate_id": str(candidate_id or "").strip(),
        }


def extract_cycle0_ensemble_kappa_from_training_curves(training_curves_payload):
    if not isinstance(training_curves_payload, dict):
        return None

    ensemble_payload = training_curves_payload.get("ensemble")
    if not isinstance(ensemble_payload, dict):
        return None


    direct_kappa = extract_ensemble_kappa(ensemble_payload)
    if direct_kappa is not None:
        return direct_kappa

    summary_payload = ensemble_payload.get("summary")
    if isinstance(summary_payload, dict):
        for key in ("kappa", "val_kappa", "cohen_kappa", "val_cohen_kappa"):
            value = to_finite_float(summary_payload.get(key))
            if value is not None:
                return value

    history_payload = ensemble_payload.get("history")
    if not isinstance(history_payload, list):
        history_payload = ensemble_payload.get("training_history")

    if isinstance(history_payload, list):
        for row in reversed(history_payload):
            if not isinstance(row, dict):
                continue
            for key in ("kappa", "val_kappa", "cohen_kappa", "val_cohen_kappa"):
                value = to_finite_float(row.get(key))
                if value is not None:
                    return value

    return None


def load_class_description_map(project):
    context_file = Path("projects") / project / "shared" / "context" / "data_context.md"
    descriptions = {}
    if not context_file.exists():
        return descriptions

    try:
        content = context_file.read_text(encoding="utf-8")
    except Exception:
        return descriptions

    for line in content.splitlines():
        match = re.match(r"^- \*\*(?:Class\s+)?(.+?)\*\*:\s*(.*)$", line.strip())
        if not match:
            continue
        label_raw, desc_raw = match.groups()
        label = normalize_class_label(label_raw)
        desc = str(desc_raw).strip()
        if label:
            descriptions[label] = desc

    return descriptions


def load_cycle0_runtime_module():
    global _CYCLE0_RUNTIME_MODULE
    if _CYCLE0_RUNTIME_MODULE is not None:
        return _CYCLE0_RUNTIME_MODULE

    module_path = (Path("local") / "scripts" / "run_cycle_0.py").resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Missing cycle0 runtime module at {module_path}")

    spec = importlib.util.spec_from_file_location("arl_cycle0_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _CYCLE0_RUNTIME_MODULE = module
    return module


def load_python_module(module_path, module_name, cache_key=None):
    path = Path(module_path).resolve()
    key = str(cache_key or path)
    cached = _DYNAMIC_MODULE_CACHE.get(key)
    if cached is not None:
        return cached

    if not path.exists():
        raise FileNotFoundError(f"Missing module: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _DYNAMIC_MODULE_CACHE[key] = module
    return module


def resolve_code_ref_path(ref, project_root):
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

    repo_candidate = Path(__file__).resolve().parent / candidate
    if repo_candidate.exists():
        return repo_candidate

    return None


def to_model_input(x, signal_name="signal"):
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


def build_binary_expert_model(model_cls, in_ch, n_classes, min_seq_len):
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


def infer_expert_feature_dim(model, x_input, torch_module, default=16):
    attr_dim = getattr(model, "embedding_dim", None)
    if isinstance(attr_dim, (int, float)) and int(attr_dim) > 0:
        return int(attr_dim)

    try:
        with torch_module.no_grad():
            sample = torch_module.tensor(x_input[:1], dtype=torch_module.float32)
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


def resolve_best_ensemble_weights(project_root):
    candidate_ids = []
    results_path = project_root / "shared" / "outbound" / "results.json"
    if results_path.exists():
        try:
            results_payload = json.loads(results_path.read_text(encoding="utf-8"))
            if isinstance(results_payload, dict):
                ensemble_eval = results_payload.get("ensemble_evaluation", {})
                if isinstance(ensemble_eval, dict):
                    candidate_id = ensemble_eval.get("candidate_id_used")
                    if candidate_id:
                        candidate_ids.append(str(candidate_id))
        except Exception:
            pass

    candidate_ids.append("baseline_ensemble")

    seen = set()
    for candidate_id in candidate_ids:
        if candidate_id in seen:
            continue
        seen.add(candidate_id)

        candidate_path = project_root / "models" / f"{candidate_id}.pt"
        if candidate_path.exists():
            return candidate_id, candidate_path

    return None, None


def resolve_ensemble_class_labels(project_root):
    """Return class labels in the exact order used during ensemble training when available."""
    metrics_path = project_root / "artifacts" / "baseline_ensemble_metrics.json"
    if not metrics_path.exists():
        return []

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    raw_labels = payload.get("class_labels") if isinstance(payload, dict) else None
    if not isinstance(raw_labels, list):
        return []

    labels = []
    seen = set()
    for raw in raw_labels:
        label = normalize_class_label(raw)
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def _unwrap_checkpoint_state_dict(checkpoint_payload):
    if isinstance(checkpoint_payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint_payload.get(key)
            if isinstance(nested, dict):
                return nested
        return checkpoint_payload
    return None


def load_ensemble_mlp_weights_from_checkpoint(ensemble_model, checkpoint_path, torch_module):
    checkpoint_payload = torch_module.load(checkpoint_path, map_location="cpu")
    state_dict = _unwrap_checkpoint_state_dict(checkpoint_payload)
    if not isinstance(state_dict, dict):
        raise RuntimeError("Unsupported ensemble checkpoint format (expected state_dict object).")

    mlp_target_keys = set(ensemble_model.mlp.state_dict().keys())

    prefixed_mlp_state = {
        key[len("mlp."):]: value
        for key, value in state_dict.items()
        if isinstance(key, str) and key.startswith("mlp.")
    }
    direct_mlp_state = {
        key: value
        for key, value in state_dict.items()
        if isinstance(key, str) and key in mlp_target_keys
    }

    if prefixed_mlp_state:
        mlp_state = prefixed_mlp_state
    elif direct_mlp_state:
        mlp_state = direct_mlp_state
    else:
        raise RuntimeError(
            "No MLP keys found in ensemble checkpoint. "
            "Expected keys prefixed by 'mlp.' or direct MLP layer keys."
        )

    report = {
        "mode": "mlp_only_strict",
        "missing_keys": [],
        "unexpected_keys": [],
        "warning": None,
    }

    try:
        ensemble_model.mlp.load_state_dict(mlp_state, strict=True)
        return report
    except Exception as strict_error:
        try:
            incompatible = ensemble_model.mlp.load_state_dict(mlp_state, strict=False)
            report["mode"] = "mlp_only_relaxed"
            report["missing_keys"] = list(getattr(incompatible, "missing_keys", []) or [])
            report["unexpected_keys"] = list(getattr(incompatible, "unexpected_keys", []) or [])
            report["warning"] = f"Strict MLP load failed: {strict_error}"
            return report
        except Exception as relaxed_error:
            raise RuntimeError(
                "Failed loading ensemble MLP weights from checkpoint. "
                f"Strict error: {strict_error}; relaxed error: {relaxed_error}"
            ) from relaxed_error


def release_cuda_cache(torch_module):
    """Best-effort cleanup for CUDA allocations in long-lived web workers."""
    try:
        gc.collect()
    except Exception:
        pass

    if torch_module is None:
        return

    try:
        if not torch_module.cuda.is_available():
            return
    except Exception:
        return

    try:
        torch_module.cuda.synchronize()
    except Exception:
        pass

    try:
        torch_module.cuda.empty_cache()
    except Exception:
        pass

    try:
        torch_module.cuda.ipc_collect()
    except Exception:
        pass

@app.route('/')
def index():
    try:
        js_path = (Path("static") / "js" / "main.js").resolve()
        asset_version = str(int(js_path.stat().st_mtime))
    except Exception:
        asset_version = "1"
    return render_template('index.html', js_asset_version=asset_version)

@app.route('/api/create/<project>', methods=['POST'])
def create_project(project):
    payload = request.get_json() or {}
    dataset_path = payload.get('dataset_path')
    if not dataset_path:
        return jsonify({"status": "error", "message": "Missing required dataset_path from creation payload"}), 400

    try:


        import json
        import tempfile
        import os

        fd, temp_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, 'w') as f:
            json.dump(payload, f)


        result = subprocess.run(
            ["python", "main.py", "init", project, dataset_path, "--context", temp_path],
            capture_output=True,
            text=True
        )


        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    if result.returncode != 0:

        error_msg = result.stderr.strip() if result.stderr else "Unknown error"
        return jsonify({"status": "error", "message": error_msg}), 400

    return start_project(project)

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    try:
        data_root = Path("data").absolute()
        if not data_root.exists():
            return jsonify({"status": "success", "datasets": []})

        valid_datasets = []

        for p in data_root.iterdir():
            if p.is_dir():

                if (p / "train" / "y.npy").exists():
                    valid_datasets.append(str(p))

        return jsonify({
            "status": "success",
            "datasets": sorted(valid_datasets)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/dataset_info', methods=['GET'])
def get_dataset_info():
    path_str = request.args.get('dataset_path')
    if not path_str:
        return jsonify({"status": "error", "message": "Missing dataset_path"}), 400

    try:
        data_dir = Path(path_str) / "train"
        if not data_dir.exists():
            return jsonify({"status": "error", "message": "train/ subdirectory not found in dataset path"}), 404

        y_file = data_dir / "y.npy"
        if not y_file.exists():
            return jsonify({"status": "error", "message": "y.npy not found in train/ split"}), 404


        signals = []
        for file in sorted(data_dir.glob("X_*.npy"), key=lambda p: p.name):
            sig_name = file.stem[2:]
            signals.append(sig_name)

        if not signals:
            return jsonify({"status": "error", "message": "No X_*.npy signal files found"}), 404


        y_array = np.load(y_file)
        unique_classes = np.unique(y_array).tolist()

        return jsonify({
            "status": "success",
            "signals": sorted(signals),
            "classes": unique_classes
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/dataset_description', methods=['GET'])
def get_dataset_description():
    path_str = request.args.get('dataset_path')
    if not path_str:
        return jsonify({"status": "error", "message": "Missing dataset_path"}), 400

    try:
        desc_file = Path(path_str) / "tsml_description.json"
        description = ""
        if desc_file.exists():
            with open(desc_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                description = data.get("description", "")

        return jsonify({
            "status": "success",
            "description": description
        })
    except Exception as e:

        return jsonify({
            "status": "success",
            "description": ""
        })

@app.route('/api/project/<project>/explore_data', methods=['GET'])
def explore_data(project):
    with SessionLocal() as session:
        proj_state = session.query(ProjectState).get(project)
        if not proj_state:
            return jsonify({"status": "error", "message": "Project not found"}), 404

    config_path = Path("projects") / project / "config.yaml"
    if not config_path.exists():
        return jsonify({"status": "error", "message": f"Config not found at {config_path}"}), 404

    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        dataset_path_str = config_data.get("dataset_path")
        if not dataset_path_str:
            return jsonify({"status": "error", "message": "dataset_path not found in config.yaml"}), 404

        project_data_dir = Path(dataset_path_str) / "train"
        if not project_data_dir.exists():
            return jsonify({"status": "error", "message": f"Training data not found at {project_data_dir}"}), 404

        y_path = project_data_dir / "y.npy"
        if not y_path.exists():
            return jsonify({"status": "error", "message": "y.npy not found in train split"}), 404

        y_data = np.load(y_path)
        if len(y_data) == 0:
            return jsonify({"status": "error", "message": "y.npy is empty"}), 400

        sample_idx = np.random.randint(0, len(y_data))
        label_int = int(y_data[sample_idx])
        label_str = str(label_int)


        context_file = Path("projects") / project / "shared" / "context" / "data_context.md"
        if context_file.exists():
            import re
            content = context_file.read_text(encoding='utf-8')


            match = re.search(rf"- \*\*(?:Class )?{label_int}\*\*: (.*)", content)
            if match:
                label_str = f"{label_int} ({match.group(1).strip()})"

        signals = {}
        x_files = sorted(project_data_dir.glob("X_*.npy"), key=lambda p: p.name)

        if not x_files:
            return jsonify({"status": "error", "message": "No signal files (X_*.npy) found"}), 404

        for x_file in x_files:

            signal_name = x_file.stem[2:]
            x_data = np.load(x_file)

            if sample_idx < len(x_data):


                sample_signal = x_data[sample_idx]
                signals[signal_name] = sample_signal.tolist()
            else:
                signals[signal_name] = f"Index {sample_idx} out of bounds for array shape {x_data.shape}"

        return jsonify({
            "status": "success",
            "sample_index": int(sample_idx),
            "label": label_str,
            "signals": signals
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/project/<project>/models', methods=['GET'])
def get_models(project):
    config_path = Path("projects") / project / "config.yaml"
    if not config_path.exists():
        return jsonify({"status": "error", "message": f"Config not found at {config_path}"}), 404

    try:
        import json
        import yaml
        import re

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        dataset_path_str = config_data.get("dataset_path")
        if not dataset_path_str:
            return jsonify({"status": "error", "message": "dataset_path not found in config.yaml"}), 404

        data_dir = Path(dataset_path_str) / "train"
        if not data_dir.exists():
            return jsonify({"status": "error", "message": f"Training data not found at {data_dir}"}), 404

        y_file = data_dir / "y.npy"
        if not y_file.exists():
            return jsonify({"status": "error", "message": "y.npy not found in train/ split"}), 404


        signals = []
        for file in sorted(data_dir.glob("X_*.npy"), key=lambda p: p.name):
            sig_name = file.stem[2:]
            signals.append(sig_name)


        y_array = np.load(y_file)
        unique_classes_ints = np.unique(y_array).tolist()


        context_file = Path("projects") / project / "shared" / "context" / "data_context.md"
        class_map = {}
        for c in unique_classes_ints:
            class_map[str(c)] = str(c)

        if context_file.exists():
            content = context_file.read_text(encoding='utf-8')
            for c in unique_classes_ints:
                match = re.search(rf"- \*\*(?:Class )?{c}\*\*: (.*)", content)
                if match:
                    class_map[str(c)] = f"{c} ({match.group(1).strip()})"

        matrix_file = Path("projects") / project / "artifacts" / "expert_matrix.json"
        matrix_data = {}
        if matrix_file.exists():
            with open(matrix_file, "r") as f:
                try:
                    matrix_data = json.load(f)
                except Exception:
                    pass


        ensemble_file = Path("projects") / project / "artifacts" / "baseline_ensemble_metrics.json"
        ensemble_data = None
        if ensemble_file.exists():
            with open(ensemble_file, "r") as f:
                try:
                    ensemble_data = json.load(f)
                except Exception:
                    pass

        return jsonify({
            "status": "success",
            "signals": sorted(signals),
            "classes": unique_classes_ints,
            "class_map": class_map,
            "matrix": matrix_data,
            "ensemble_metrics": ensemble_data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/project/<project>/model_history', methods=['GET'])
def get_model_history(project):
    project_root = Path("projects") / project
    if not project_root.exists():
        return jsonify({"status": "error", "message": "Project not found"}), 404

    cycle_history_root = project_root / "artifacts" / "cycle_history"
    if not cycle_history_root.exists() or not cycle_history_root.is_dir():
        return jsonify({
            "status": "success",
            "project": project,
            "cycles": [],
            "experts": [],
            "ensemble_kappa": [],
        })

    class_descriptions = load_class_description_map(project)
    cycle_dirs = []
    for entry in cycle_history_root.iterdir():
        if not entry.is_dir():
            continue
        cycle_num = parse_cycle_number(entry.name)
        if cycle_num is None:
            continue
        cycle_dirs.append((cycle_num, entry))
    cycle_dirs.sort(key=lambda pair: pair[0])

    cycles_seen = set()
    expert_history_map = {}
    ensemble_kappa_map = {}

    for cycle_num, cycle_dir in cycle_dirs:
        if cycle_num == 0:
            cycles_seen.add(0)


            cycle0_matrix_path = cycle_dir / "expert_matrix.json"
            if cycle0_matrix_path.exists():
                try:
                    cycle0_matrix_payload = json.loads(cycle0_matrix_path.read_text(encoding="utf-8"))
                except Exception:
                    cycle0_matrix_payload = {}

                if isinstance(cycle0_matrix_payload, dict):
                    for raw_modality, raw_per_class in cycle0_matrix_payload.items():
                        modality = str(raw_modality or "").strip()
                        if not modality or not isinstance(raw_per_class, dict):
                            continue

                        for raw_class_label, raw_entry in raw_per_class.items():
                            class_label = normalize_class_label(raw_class_label)
                            if not class_label:
                                continue

                            entry = raw_entry if isinstance(raw_entry, dict) else {}
                            entry_f1 = to_finite_float(entry.get("f1"))
                            entry_candidate_id = str(entry.get("candidate_id") or "").strip()
                            upsert_expert_history_point(
                                expert_history_map=expert_history_map,
                                modality=modality,
                                class_label=class_label,
                                cycle_value=0,
                                f1_value=entry_f1,
                                candidate_id=entry_candidate_id,
                            )


            if 0 not in ensemble_kappa_map:
                cycle0_curves_path = cycle_dir / "training_curves.json"
                if cycle0_curves_path.exists():
                    try:
                        cycle0_curves_payload = json.loads(cycle0_curves_path.read_text(encoding="utf-8"))
                    except Exception:
                        cycle0_curves_payload = {}

                    cycle0_kappa = extract_cycle0_ensemble_kappa_from_training_curves(cycle0_curves_payload)
                    if cycle0_kappa is not None:
                        ensemble_kappa_map[0] = cycle0_kappa

        results_path = cycle_dir / "results.json"
        if not results_path.exists():
            continue

        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        cycle_value = parse_cycle_number(payload.get("cycle_id")) if isinstance(payload, dict) else None
        if cycle_value is None:
            cycle_value = cycle_num
        cycles_seen.add(cycle_value)

        jobs = payload.get("jobs") if isinstance(payload, dict) else None
        if isinstance(jobs, list):
            for job in jobs:
                if not isinstance(job, dict):
                    continue
                if str(job.get("job_type") or "").strip().lower() != "train_expert":
                    continue

                target = job.get("target")
                if not isinstance(target, dict):
                    continue

                modality = str(target.get("modality") or "").strip()
                class_label = normalize_class_label(target.get("class_label"))
                if not modality or not class_label:
                    continue

                f1_value = extract_train_expert_f1(job)
                if f1_value is None:
                    continue

                candidate = job.get("candidate") if isinstance(job.get("candidate"), dict) else {}
                candidate_id = str(candidate.get("candidate_id") or "").strip()
                upsert_expert_history_point(
                    expert_history_map=expert_history_map,
                    modality=modality,
                    class_label=class_label,
                    cycle_value=cycle_value,
                    f1_value=f1_value,
                    candidate_id=candidate_id,
                )

        ensemble_payload = payload.get("ensemble_evaluation") if isinstance(payload, dict) else None
        kappa_value = extract_ensemble_kappa(ensemble_payload)
        if kappa_value is not None:
            ensemble_kappa_map[cycle_value] = kappa_value

    cycles_sorted = sorted(cycles_seen)


    matrix_file = project_root / "artifacts" / "expert_matrix.json"
    matrix_payload = {}
    if matrix_file.exists():
        try:
            matrix_payload = json.loads(matrix_file.read_text(encoding="utf-8"))
        except Exception:
            matrix_payload = {}

    anchor_cycle = min(cycles_sorted) if cycles_sorted else 0
    if isinstance(matrix_payload, dict):
        for raw_modality, raw_per_class in matrix_payload.items():
            modality = str(raw_modality or "").strip()
            if not modality or not isinstance(raw_per_class, dict):
                continue

            for raw_class_label, raw_entry in raw_per_class.items():
                class_label = normalize_class_label(raw_class_label)
                if not class_label:
                    continue

                entry = raw_entry if isinstance(raw_entry, dict) else {}
                entry_f1 = to_finite_float(entry.get("f1"))
                entry_candidate_id = str(entry.get("candidate_id") or "").strip()

                expert_id = f"{modality}::{class_label}"
                if expert_id not in expert_history_map:
                    expert_history_map[expert_id] = {
                        "modality": modality,
                        "class_label": class_label,
                        "history_by_cycle": {},
                    }

                history_by_cycle = expert_history_map[expert_id]["history_by_cycle"]
                if entry_f1 is None:
                    continue


                candidate_cycle = None
                cycle_match = re.search(r"cycle_(\d+)", entry_candidate_id, flags=re.IGNORECASE)
                if cycle_match:
                    candidate_cycle = int(cycle_match.group(1))
                if candidate_cycle is None:
                    candidate_cycle = anchor_cycle

                existing = history_by_cycle.get(candidate_cycle)
                existing_f1 = to_finite_float(existing.get("f1")) if isinstance(existing, dict) else None
                if existing_f1 is None or entry_f1 > existing_f1:
                    history_by_cycle[candidate_cycle] = {
                        "f1": entry_f1,
                        "candidate_id": entry_candidate_id,
                    }
                cycles_seen.add(candidate_cycle)

    cycles_sorted = sorted(cycles_seen)

    experts_payload = []
    for expert_id, expert_data in sorted(
        expert_history_map.items(),
        key=lambda pair: (pair[1].get("modality", "").lower(), label_sort_key(pair[1].get("class_label", ""))),
    ):
        class_label = expert_data.get("class_label", "")
        class_desc = str(class_descriptions.get(class_label) or "").strip()
        class_name = f"{class_label} ({class_desc})" if class_desc else str(class_label)
        history_by_cycle = expert_data.get("history_by_cycle", {})

        history_points = []
        best_f1 = None
        best_candidate_id = ""
        for cycle_id in cycles_sorted:
            cycle_row = history_by_cycle.get(cycle_id)
            cycle_f1 = None
            cycle_candidate_id = ""
            if isinstance(cycle_row, dict):
                cycle_f1 = to_finite_float(cycle_row.get("f1"))
                cycle_candidate_id = str(cycle_row.get("candidate_id") or "").strip()

            model_changed = False


            if cycle_f1 is not None:
                if best_f1 is None:
                    best_f1 = cycle_f1
                    best_candidate_id = cycle_candidate_id

                    model_changed = bool(re.match(r"^cycle_\d+", cycle_candidate_id, flags=re.IGNORECASE))
                elif cycle_f1 > best_f1:
                    best_f1 = cycle_f1
                    best_candidate_id = cycle_candidate_id
                    model_changed = True


            if best_f1 is None:
                continue

            history_points.append(
                {
                    "cycle": int(cycle_id),
                    "f1": float(best_f1),
                    "candidate_id": best_candidate_id or cycle_candidate_id,
                    "model_changed": model_changed,
                    "trained_f1": float(cycle_f1) if cycle_f1 is not None else None,
                    "trained_candidate_id": cycle_candidate_id if cycle_candidate_id else None,
                    "trained_in_cycle": bool(cycle_f1 is not None),
                }
            )

        experts_payload.append(
            {
                "expert_id": expert_id,
                "modality": expert_data.get("modality"),
                "class_label": class_label,
                "class_name": class_name,
                "display_name": f"{expert_data.get('modality')} | class {class_name}",
                "history": history_points,
            }
        )

    ensemble_payload = [
        {
            "cycle": int(cycle_id),
            "kappa": ensemble_kappa_map[cycle_id],
        }
        for cycle_id in sorted(ensemble_kappa_map.keys())
    ]

    return jsonify(
        {
            "status": "success",
            "project": project,
            "cycles": cycles_sorted,
            "experts": experts_payload,
            "ensemble_kappa": ensemble_payload,
        }
    )


@app.route('/api/project/<project>/run_test_set', methods=['POST'])
def run_test_set(project):
    if not project_exists(project):
        return jsonify({"status": "error", "message": "Project not found"}), 404

    with SessionLocal() as session:
        proj_state = session.query(ProjectState).get(project)
        if proj_state is None:
            return jsonify({"status": "error", "message": "Project state not found"}), 404

    project_root = Path("projects") / project
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        return jsonify({"status": "error", "message": f"Config not found: {config_path}"}), 404

    torch = None
    experts = None
    ensemble_model = None
    x_test_tensors = None
    processed_test_blocks = None

    try:
        import torch
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            cohen_kappa_score,
            f1_score,
            precision_score,
            recall_score,
        )

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        dataset_path_str = config_data.get("dataset_path")
        if not dataset_path_str:
            return jsonify({"status": "error", "message": "dataset_path missing in config.yaml"}), 400

        dataset_path = Path(str(dataset_path_str))
        train_dir = dataset_path / "train"
        test_dir = dataset_path / "test"
        if not train_dir.exists() or not test_dir.exists():
            return jsonify({"status": "error", "message": "Dataset must contain train/ and test/ splits."}), 400

        y_test_path = test_dir / "y.npy"
        if not y_test_path.exists():
            return jsonify({"status": "error", "message": f"Missing test labels: {y_test_path}"}), 400

        y_test_raw = np.asarray(np.load(y_test_path)).reshape(-1)
        if y_test_raw.size == 0:
            return jsonify({"status": "error", "message": "Test split has zero labels."}), 400

        expert_matrix_path = project_root / "artifacts" / "expert_matrix.json"
        if not expert_matrix_path.exists():
            return jsonify({"status": "error", "message": f"Missing expert matrix: {expert_matrix_path}"}), 400

        with open(expert_matrix_path, "r", encoding="utf-8") as f:
            expert_matrix = json.load(f)
        if not isinstance(expert_matrix, dict):
            return jsonify({"status": "error", "message": "Invalid expert_matrix.json format."}), 400


        signal_order = sorted([
            str(sig)
            for sig in expert_matrix.keys()
            if (test_dir / f"X_{sig}.npy").exists()
        ], key=label_sort_key)
        if not signal_order:
            signal_order = sorted([p.stem[2:] for p in test_dir.glob("X_*.npy")], key=label_sort_key)
        if not signal_order:
            return jsonify({"status": "error", "message": "No test signal files found (X_*.npy)."}), 400

        x_test_list = []
        signal_to_index = {}
        sample_count = None
        for idx, sig in enumerate(signal_order):
            x_path = test_dir / f"X_{sig}.npy"
            if not x_path.exists():
                return jsonify({"status": "error", "message": f"Missing test signal file: {x_path}"}), 400

            x_arr = np.load(x_path)
            if x_arr.ndim < 2:
                return jsonify({"status": "error", "message": f"Signal file must be at least 2D: {x_path}"}), 400

            if sample_count is None:
                sample_count = int(x_arr.shape[0])
            elif int(x_arr.shape[0]) != int(sample_count):
                return jsonify({"status": "error", "message": f"Mismatched sample count in signal file: {x_path}"}), 400

            x_test_list.append(x_arr)
            signal_to_index[sig] = idx

        class_labels_sorted = resolve_ensemble_class_labels(project_root)

        if not class_labels_sorted:
            class_labels = set()
            for per_signal in expert_matrix.values():
                if not isinstance(per_signal, dict):
                    continue
                for cls in per_signal.keys():
                    class_labels.add(normalize_class_label(cls))

            class_labels = {lbl for lbl in class_labels if lbl}
            class_labels_sorted = sorted(class_labels, key=label_sort_key)

        if not class_labels_sorted:
            y_train_path = train_dir / "y.npy"
            if y_train_path.exists():
                y_train_vals = np.load(y_train_path)
                fallback_class_labels = {normalize_class_label(v) for v in np.unique(y_train_vals)}
                fallback_class_labels = {lbl for lbl in fallback_class_labels if lbl}
                class_labels_sorted = sorted(fallback_class_labels, key=label_sort_key)

        if not class_labels_sorted:
            return jsonify({"status": "error", "message": "Could not infer class labels for ensemble evaluation."}), 400

        label_to_idx = {label: idx for idx, label in enumerate(class_labels_sorted)}
        y_test_norm = np.array([normalize_class_label(v) for v in y_test_raw], dtype=object)
        y_test_idx = np.array([label_to_idx.get(lbl, -1) for lbl in y_test_norm], dtype=np.int64)
        valid_mask = y_test_idx >= 0
        valid_count = int(np.sum(valid_mask))
        ignored_count = int(len(y_test_idx) - valid_count)
        if valid_count <= 0:
            return jsonify({"status": "error", "message": "No test labels matched the ensemble class mapping."}), 400

        runtime = load_cycle0_runtime_module()
        use_cuda_for_test_set = str(os.environ.get("ARL_TEST_SET_USE_CUDA", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        device = torch.device("cuda" if (use_cuda_for_test_set and torch.cuda.is_available()) else "cpu")

        baseline_preprocess_fn = lambda arr: arr
        baseline_pre_path = (Path("local") / "scripts" / "cycle_preprocessing.py").resolve()
        if baseline_pre_path.exists():
            try:
                baseline_pre_module = load_python_module(
                    baseline_pre_path,
                    "arl_cycle0_preprocessing",
                    cache_key="arl_cycle0_preprocessing",
                )
                pre_fn = getattr(baseline_pre_module, "apply_preprocessing", None)
                if callable(pre_fn):
                    baseline_preprocess_fn = pre_fn
            except Exception:
                pass

        raw_test_by_signal = {sig: x_test_list[idx] for sig, idx in signal_to_index.items()}
        processed_test_blocks = []
        block_index_map = {}

        experts = []
        input_map = []
        expert_dims = []
        skipped_experts = []

        for sig in signal_order:
            per_class = expert_matrix.get(sig, {})
            if not isinstance(per_class, dict):
                continue


            modality_labels = sorted(
                [normalize_class_label(v) for v in per_class.keys() if normalize_class_label(v)],
                key=label_sort_key,
            )
            for class_label in modality_labels:
                rec = per_class.get(class_label)
                if not isinstance(rec, dict):
                    rec = per_class.get(str(class_label))
                if not isinstance(rec, dict):
                    continue

                class_key = str(class_label)
                candidate_id = str(rec.get("candidate_id", f"baseline_{sig}_{class_key}"))
                expert_weights = project_root / "models" / f"{sig}_{class_key}" / f"{candidate_id}.pt"
                if not expert_weights.exists():
                    continue

                model_cls = runtime.BinaryExpertModel
                model_ref = rec.get("final_model_py_ref")
                model_ref_path = resolve_code_ref_path(model_ref, project_root)
                if model_ref_path is not None:
                    try:
                        model_module = load_python_module(
                            model_ref_path,
                            f"arl_test_model_{abs(hash(str(model_ref_path)))}",
                            cache_key=f"model::{model_ref_path}",
                        )
                        model_cls_candidate = getattr(model_module, "BinaryExpertModel", None)
                        if model_cls_candidate is not None:
                            model_cls = model_cls_candidate
                    except Exception:
                        pass

                preprocess_fn = baseline_preprocess_fn
                preprocessing_ref = rec.get("preprocessing_code_ref")
                preprocessing_ref_path = resolve_code_ref_path(preprocessing_ref, project_root)
                preprocess_key = "__baseline__"
                if preprocessing_ref_path is not None:
                    preprocess_key = str(preprocessing_ref_path)
                    try:
                        pre_module = load_python_module(
                            preprocessing_ref_path,
                            f"arl_test_pre_{abs(hash(str(preprocessing_ref_path)))}",
                            cache_key=f"pre::{preprocessing_ref_path}",
                        )
                        pre_fn = getattr(pre_module, "apply_preprocessing", None)
                        if callable(pre_fn):
                            preprocess_fn = pre_fn
                    except Exception:
                        pass

                block_key = (str(sig), preprocess_key)
                block_idx = block_index_map.get(block_key)
                if block_idx is None:
                    raw_signal = raw_test_by_signal.get(sig)
                    if raw_signal is None:
                        skipped_experts.append(f"{sig}/{class_key}/{candidate_id}: missing test modality array")
                        continue
                    try:
                        x_processed = preprocess_fn(raw_signal)
                        x_input = to_model_input(x_processed, signal_name=f"{sig}:{class_key}:test")
                        if int(x_input.shape[0]) != int(len(y_test_idx)):
                            raise ValueError(
                                f"sample mismatch X={x_input.shape[0]} labels={len(y_test_idx)}"
                            )
                    except Exception as exc:
                        skipped_experts.append(f"{sig}/{class_key}/{candidate_id}: preprocessing failed ({exc})")
                        continue

                    block_idx = len(processed_test_blocks)
                    block_index_map[block_key] = block_idx
                    processed_test_blocks.append(x_input)

                x_input = processed_test_blocks[block_idx]
                try:
                    expert_model = build_binary_expert_model(
                        model_cls=model_cls,
                        in_ch=int(x_input.shape[1]),
                        n_classes=1,
                        min_seq_len=int(x_input.shape[-1]),
                    ).to(device)
                    expert_model.load_state_dict(torch.load(expert_weights, map_location=device))
                    expert_model.eval()
                except Exception as exc:
                    skipped_experts.append(f"{sig}/{class_key}/{candidate_id}: weight load failed ({exc})")
                    continue

                experts.append(expert_model)
                input_map.append(block_idx)
                expert_dims.append(infer_expert_feature_dim(expert_model, x_input, torch, default=16))

        if not experts:
            hint = "No compatible expert models were found to build the ensemble."
            if skipped_experts:
                hint += " Sample issues: " + "; ".join(skipped_experts[:3])
            return jsonify({"status": "error", "message": hint}), 400

        candidate_id, ensemble_weights = resolve_best_ensemble_weights(project_root)
        if ensemble_weights is None:
            return jsonify({"status": "error", "message": "No ensemble weights file found for this project."}), 400

        default_expert_dim = int(expert_dims[0]) if expert_dims else 16
        ensemble_architecture = normalize_ensemble_architecture(config_data.get("ensemble_architecture", "default"))
        try:
            ensemble_init_sig = inspect.signature(runtime.BaselineEnsemble.__init__)
            ensemble_kwargs = {
                "experts": experts,
                "input_map": input_map,
                "num_classes": len(class_labels_sorted),
                "expert_dim": default_expert_dim,
            }
            if "expert_dims" in ensemble_init_sig.parameters and len(set(expert_dims)) > 1:
                ensemble_kwargs["expert_dims"] = expert_dims
            if "architecture" in ensemble_init_sig.parameters:
                ensemble_kwargs["architecture"] = ensemble_architecture

            ensemble_model = runtime.BaselineEnsemble(**ensemble_kwargs).to(device)
        except Exception as exc:
            return jsonify({"status": "error", "message": f"Failed to build ensemble model: {exc}"}), 500

        try:
            ensemble_load_report = load_ensemble_mlp_weights_from_checkpoint(
                ensemble_model=ensemble_model,
                checkpoint_path=ensemble_weights,
                torch_module=torch,
            )
        except Exception as exc:
            detail = f"Failed to load ensemble weights: {exc}"
            if skipped_experts:
                detail += " | skipped experts during test prep: " + "; ".join(skipped_experts[:3])
            return jsonify({"status": "error", "message": detail}), 500

        ensemble_model.eval()

        x_test_tensors = [torch.tensor(arr, dtype=torch.float32, device=device) for arr in processed_test_blocks]
        with torch.no_grad():
            logits = ensemble_model(x_test_tensors)
            preds_idx = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)

        if len(preds_idx) != len(y_test_idx):
            return jsonify({"status": "error", "message": "Prediction/label length mismatch during test evaluation."}), 500

        y_eval = y_test_idx[valid_mask]
        preds_eval = preds_idx[valid_mask]
        labels_idx = list(range(len(class_labels_sorted)))

        accuracy = float(np.mean(preds_eval == y_eval))
        precision_macro = float(precision_score(y_eval, preds_eval, average="macro", zero_division=0))
        recall_macro = float(recall_score(y_eval, preds_eval, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_eval, preds_eval, average="macro", zero_division=0))
        kappa = float(cohen_kappa_score(y_eval, preds_eval)) if len(np.unique(y_eval)) > 1 else 0.0
        report = classification_report(
            y_eval,
            preds_eval,
            labels=labels_idx,
            output_dict=True,
            zero_division=0,
        )
        conf_matrix = confusion_matrix(y_eval, preds_eval, labels=labels_idx).tolist()

        class_desc = load_class_description_map(project)
        class_map = {}
        for idx, label in enumerate(class_labels_sorted):
            desc = class_desc.get(label, "").strip()
            class_map[str(idx)] = f"{label} ({desc})" if desc else str(label)

        metrics_payload = {
            "accuracy": accuracy,
            "kappa": kappa,
            "precision": precision_macro,
            "recall": recall_macro,
            "f1": f1_macro,
            "classification_report": report,
            "confusion_matrix": conf_matrix,
        }

        result_payload = {
            "status": "success",
            "project": project,
            "ensemble_candidate_id": str(candidate_id or "baseline_ensemble"),
            "ensemble_weights_ref": str(ensemble_weights.relative_to(project_root)).replace(os.sep, "/"),
            "evaluation_device": str(device),
            "ensemble_weight_load_mode": str(ensemble_load_report.get("mode", "unknown")),
            "ensemble_weight_load_missing_keys": list(ensemble_load_report.get("missing_keys", []) or []),
            "ensemble_weight_load_unexpected_keys": list(ensemble_load_report.get("unexpected_keys", []) or []),
            "ensemble_weight_load_warning": ensemble_load_report.get("warning"),
            "ensemble_architecture": ensemble_architecture,
            "loaded_expert_count": int(len(experts)),
            "skipped_expert_count": int(len(skipped_experts)),
            "skipped_expert_examples": skipped_experts[:10],
            "evaluated_samples": valid_count,
            "ignored_samples": ignored_count,
            "class_map": class_map,
            "metrics": metrics_payload,
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }

        out_path = project_root / "artifacts" / "test_set_ensemble_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

        return jsonify(result_payload)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:

        experts = None
        ensemble_model = None
        x_test_tensors = None
        processed_test_blocks = None
        release_cuda_cache(torch)

@app.route('/api/project/<project>/training_status', methods=['GET'])
def get_training_status(project):
    try:
        status_file = Path("projects") / project / "state" / "live_training.json"
        if not status_file.exists():
            return jsonify({"status": "idle", "data": []})

        with open(status_file, "r") as f:
            data = json.load(f)
            return jsonify({"status": "running", "data": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/project/<project>/cycle/<cycle_id>/files', methods=['GET'])
def get_cycle_files(project, cycle_id):
    try:

        cycle_num = int(cycle_id)
        cycle_dir_name = f"cycle_{cycle_num:04d}"

        project_dir = (Path("projects") / project).resolve()
        cycle_dir = (project_dir / "artifacts" / "cycle_history" / cycle_dir_name).resolve()


        if not str(cycle_dir).startswith(str(project_dir)):
            return jsonify({"status": "error", "message": "Access denied"}), 403

        if not cycle_dir.exists() or not cycle_dir.is_dir():
            return jsonify({"status": "error", "message": f"Cycle history directory not found: {cycle_dir_name}"}), 404


        files = [f.name for f in cycle_dir.iterdir() if f.is_file()]
        files.sort()

        return jsonify({
            "status": "success",
            "cycle_root": f"artifacts/cycle_history/{cycle_dir_name}",
            "files": files
        })
    except ValueError:
         return jsonify({"status": "error", "message": "Invalid cycle ID format"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/project/<project>/file', methods=['GET'])
def get_file(project):
    filepath = request.args.get('path')
    if not filepath:
        return jsonify({"status": "error", "message": "Missing path parameter"}), 400

    project_dir = (Path("projects") / project).resolve()
    target_file = (project_dir / filepath).resolve()


    if not str(target_file).startswith(str(project_dir)):
        return jsonify({"status": "error", "message": "Access denied"}), 403

    if not target_file.exists() or not target_file.is_file():
        return jsonify({"status": "error", "message": "File not found"}), 404

    try:
        content = target_file.read_text(encoding='utf-8')
        return jsonify({"status": "success", "content": content})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/project/<project>/manual_verification', methods=['GET'])
def get_manual_verification(project):
    state = load_manual_verification_state(project)
    return jsonify({
        "status": "success",
        "project": project,
        "enabled": bool(state.get("enabled", False)),
    })


@app.route('/api/project/<project>/end_time', methods=['GET'])
def get_project_end_time(project):
    if not project_exists(project):
        return jsonify({"status": "error", "message": "Project not found"}), 404

    settings = ensure_project_settings(project)
    return jsonify({
        "status": "success",
        "project": project,
        "start_time_utc": settings.get("start_time_utc"),
        "end_time_utc": settings.get("end_time_utc"),
    })


@app.route('/api/project/<project>/end_time', methods=['POST'])
def set_project_end_time(project):
    if not project_exists(project):
        return jsonify({"status": "error", "message": "Project not found"}), 404

    payload = request.get_json() or {}
    raw_end_time = payload.get("end_time_utc")
    end_dt = parse_utc_datetime(raw_end_time)
    if end_dt is None:
        return jsonify({"status": "error", "message": "Invalid end_time_utc. Expected ISO datetime string."}), 400

    settings = ensure_project_settings(project)
    settings["end_time_utc"] = end_dt.isoformat()
    save_project_settings(project, settings)

    return jsonify({
        "status": "success",
        "project": project,
        "start_time_utc": settings.get("start_time_utc"),
        "end_time_utc": settings.get("end_time_utc"),
    })


@app.route('/api/project/<project>/llm_execution', methods=['GET'])
def get_project_llm_execution(project):
    if not project_exists(project):
        return jsonify({"status": "error", "message": "Project not found"}), 404

    settings = ensure_project_settings(project)
    return jsonify({
        "status": "success",
        "project": project,
        "llm_role_execution": normalize_llm_role_execution(settings.get("llm_role_execution")),
    })


@app.route('/api/project/<project>/llm_execution', methods=['POST'])
def set_project_llm_execution(project):
    if not project_exists(project):
        return jsonify({"status": "error", "message": "Project not found"}), 404

    payload = request.get_json(silent=True) or {}
    normalized, error_msg = parse_llm_role_execution_payload(payload.get("llm_role_execution"))
    if error_msg:
        return jsonify({"status": "error", "message": error_msg}), 400

    settings = ensure_project_settings(project)
    settings["llm_role_execution"] = normalized
    save_project_settings(project, settings)

    return jsonify({
        "status": "success",
        "project": project,
        "llm_role_execution": settings.get("llm_role_execution"),
    })


@app.route('/api/project/<project>/manual_verification', methods=['POST'])
def set_manual_verification(project):
    payload = request.get_json() or {}
    enabled = bool(payload.get("enabled", False))

    state = load_manual_verification_state(project)
    state["enabled"] = enabled
    save_manual_verification_state(project, state)

    return jsonify({
        "status": "success",
        "project": project,
        "enabled": enabled,
    })


@app.route('/api/project/<project>/confirm_step', methods=['POST'])
def confirm_step(project):
    payload = request.get_json() or {}
    log_id = payload.get("log_id")

    try:
        log_id = int(log_id)
    except Exception:
        return jsonify({"status": "error", "message": "Missing or invalid log_id"}), 400

    with SessionLocal() as session:
        log_row = session.query(ExecutionLog).filter(
            ExecutionLog.id == log_id,
            ExecutionLog.project_name == project,
        ).first()
        if not log_row:
            return jsonify({"status": "error", "message": "Log entry not found for project"}), 404

    state = load_manual_verification_state(project)
    confirmed = set(state.get("confirmed_log_ids", []))
    confirmed.add(log_id)
    state["confirmed_log_ids"] = sorted(confirmed)
    save_manual_verification_state(project, state)

    return jsonify({
        "status": "success",
        "project": project,
        "log_id": log_id,
        "confirmed": True,
    })

@app.route('/api/project/<project>', methods=['DELETE'])
def delete_project(project):
    try:

        stop_project(project)

        with SessionLocal() as session:

            proj_state = session.query(ProjectState).get(project)
            if proj_state:
                session.delete(proj_state)

            session.query(ExecutionLog).filter(ExecutionLog.project_name == project).delete()
            session.commit()


        proj_dir = Path("projects") / project
        if proj_dir.exists() and proj_dir.is_dir():
            shutil.rmtree(proj_dir)

        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error deleting project: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/data')
def api_data():
    with SessionLocal() as session:
        projects = session.query(ProjectState).all()
        manual_states = {
            p.project_name: load_manual_verification_state(p.project_name)
            for p in projects
        }
        project_settings = {
            p.project_name: ensure_project_settings(p.project_name)
            for p in projects
        }


        dirty = False
        for p in projects:
            if p.status in ('Running', 'Paused') and p.pid is not None:
                try:
                    os.kill(p.pid, 0)
                except OSError:
                    p.status = 'Error'
                    p.target_status = 'Stopped'
                    p.current_step = 'Terminated/Crashed'
                    dirty = True

        if dirty:
            session.commit()


        logs = session.query(ExecutionLog).order_by(ExecutionLog.id.desc()).limit(50).all()

        logs_payload = []
        for l in logs:
            state = manual_states.get(l.project_name)
            if state is None:
                state = load_manual_verification_state(l.project_name)
                manual_states[l.project_name] = state

            manual_enabled = bool(state.get("enabled", False))
            confirmed = int(l.id) in set(state.get("confirmed_log_ids", []))
            needs_confirmation = manual_enabled and l.status in ("Completed", "Failed") and not confirmed
            display_status = "Waiting for confirmation" if needs_confirmation else l.status

            logs_payload.append(
                {
                    "id": l.id,
                    "project_name": l.project_name,
                    "cycle": l.cycle,
                    "step_name": l.step_name,
                    "status": l.status,
                    "display_status": display_status,
                    "needs_confirmation": needs_confirmation,
                    "confirmed": confirmed,
                    "timestamp": l.timestamp.isoformat() if l.timestamp else None,
                }
            )

        return jsonify({
            "projects": [
                {
                    "project_name": p.project_name,
                    "current_cycle": p.current_cycle,
                    "current_step": p.current_step,
                    "status": p.status,
                    "target_status": p.target_status,
                    "is_finished": project_is_finished_state(p),
                    "manual_verification_enabled": bool(
                        manual_states.get(p.project_name, {}).get("enabled", False)
                    ),
                    "start_time_utc": project_settings.get(p.project_name, {}).get("start_time_utc"),
                    "end_time_utc": project_settings.get(p.project_name, {}).get("end_time_utc"),
                    "llm_role_execution": normalize_llm_role_execution(
                        project_settings.get(p.project_name, {}).get("llm_role_execution")
                    ),
                    "last_updated": p.last_updated.isoformat() if p.last_updated else None
                } for p in projects
            ],
            "logs": logs_payload
        })

@app.route('/api/start/<project>', methods=['POST'])
def start_project(project):
    if not project_exists(project):
        return jsonify({"status": "error", "message": "Project not found"}), 404

    payload = request.get_json(silent=True) or {}
    settings = ensure_project_settings(project)
    selected_llm_role_execution = payload.get("llm_role_execution")
    if selected_llm_role_execution is not None:
        normalized, error_msg = parse_llm_role_execution_payload(selected_llm_role_execution)
        if error_msg:
            return jsonify({"status": "error", "message": error_msg}), 400
        settings["llm_role_execution"] = normalized
        save_project_settings(project, settings)

    with SessionLocal() as session:
        proj_state = session.query(ProjectState).get(project)
        if not proj_state:
            proj_state = ProjectState(project_name=project)
            session.add(proj_state)
        else:
            if project_is_finished_state(proj_state):
                session.commit()
                return jsonify({"status": "error", "message": "Project is finished. Use 'Run on test set' instead of start."}), 400


            if proj_state.target_status == 'Paused' and pid_is_alive(proj_state.pid):
                proj_state.status = 'Running'
                proj_state.target_status = 'Running'
                if str(proj_state.current_step or '').strip().lower() == 'paused':
                    proj_state.current_step = 'Resuming'
                session.commit()
                return jsonify({"status": "resumed", "project": project})


            if proj_state.target_status == 'Running' and pid_is_alive(proj_state.pid):
                proj_state.status = 'Running'
                session.commit()
                return jsonify({"status": "already_running", "project": project})


            if not pid_is_alive(proj_state.pid):
                proj_state.pid = None

        proj_state.status = 'Starting'
        proj_state.current_step = 'Booting'
        proj_state.target_status = 'Running'
        session.commit()


    subprocess.Popen(["python", "main.py", "run", project])
    return jsonify({"status": "started", "project": project})


@app.route('/api/pause/<project>', methods=['POST'])
def pause_project(project):
    if not project_exists(project):
        return jsonify({"status": "error", "message": "Project not found"}), 404

    with SessionLocal() as session:
        proj_state = session.query(ProjectState).get(project)
        if not proj_state:
            return jsonify({"status": "error", "message": "Project not found"}), 404

        if project_is_finished_state(proj_state):
            return jsonify({"status": "error", "message": "Project is finished and cannot be paused."}), 400

        if proj_state.target_status == 'Paused' and pid_is_alive(proj_state.pid):
            proj_state.status = 'Paused'
            session.commit()
            return jsonify({"status": "already_paused", "project": project})

        if not pid_is_alive(proj_state.pid):
            proj_state.pid = None
            proj_state.target_status = 'Stopped'
            if str(proj_state.status).lower() in ('running', 'paused', 'starting'):
                proj_state.status = 'Stopped'
                proj_state.current_step = 'Idle'
            session.commit()
            return jsonify({"status": "error", "message": "Project is not currently running."}), 400

        proj_state.target_status = 'Paused'
        proj_state.status = 'Paused'
        proj_state.current_step = 'Pause requested'
        session.commit()

    return jsonify({"status": "paused", "project": project})

@app.route('/api/stop/<project>', methods=['POST'])
def stop_project(project):
    if not project_exists(project):
        return jsonify({"status": "error", "message": "Project not found"}), 404

    with SessionLocal() as session:
        proj_state = session.query(ProjectState).get(project)
        if proj_state:
            proj_state.target_status = 'Stopped'

            if project_is_finished_state(proj_state):
                session.commit()
                return jsonify({"status": "stopped", "project": project})

            if pid_is_alive(proj_state.pid):
                proj_state.status = 'Stopping'
                proj_state.current_step = 'Stopping'
            else:
                proj_state.pid = None
                proj_state.status = 'Stopped'
                proj_state.current_step = 'Idle'

            session.commit()

    return jsonify({"status": "stopped", "project": project})

if __name__ == '__main__':
    debug_mode = os.environ.get("ARL_FLASK_DEBUG", "1") == "1"

    use_reloader = os.environ.get("ARL_FLASK_USE_RELOADER", "0") == "1"
    app.run(debug=debug_mode, use_reloader=use_reloader, port=5000)
