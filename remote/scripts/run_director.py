#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request


try:
    import jsonschema  # type: ignore
except Exception:
    jsonschema = None

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # type: ignore[assignment,misc]


VALID_REASONING_EFFORTS = {"low", "medium", "high"}
VALID_TEXT_VERBOSITY = {"low", "medium", "high"}

DIRECTOR_SINGLE_CALL_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["target", "reasoning", "implementation", "design", "snippets", "proposal_notes"],
    "properties": {
        "target": {
            "type": "object",
            "additionalProperties": False,
            "required": ["modality", "class_label", "expected_upside"],
            "properties": {
                "modality": {"type": "string", "minLength": 1},
                "class_label": {"type": "string", "minLength": 1},
                "expected_upside": {"type": "string", "enum": ["low", "moderate", "high"]},
            },
        },
        "reasoning": {
            "type": "object",
            "additionalProperties": False,
            "required": ["why_this_class", "why_this_modality", "difficulty", "success_criterion"],
            "properties": {
                "why_this_class": {"type": "string", "minLength": 1},
                "why_this_modality": {"type": "string", "minLength": 1},
                "difficulty": {"type": "string", "minLength": 1},
                "success_criterion": {"type": "string", "minLength": 1},
            },
        },
        "implementation": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "dependencies_used",
                "requires_dependency_whitelist_match",
                "shared_across_modality",
            ],
            "properties": {
                "dependencies_used": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "requires_dependency_whitelist_match": {"type": "boolean"},
                "shared_across_modality": {"type": "boolean"},
            },
        },
        "design": {
            "type": "object",
            "additionalProperties": False,
            "required": ["model_description", "preprocessing_description"],
            "properties": {
                "model_description": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "family_name",
                        "summary",
                        "input_expectation",
                        "feature_extractor_type",
                        "uses_multiscale_branches",
                        "uses_residual",
                        "uses_batchnorm",
                        "uses_dropout",
                        "global_pooling",
                        "embedding_dim_source",
                        "logits_head_type",
                        "key_hyperparameters",
                        "shape_notes",
                    ],
                    "properties": {
                        "family_name": {"type": "string", "minLength": 1},
                        "summary": {"type": "string", "minLength": 1},
                        "input_expectation": {"type": "string", "minLength": 1},
                        "feature_extractor_type": {"type": "string", "minLength": 1},
                        "uses_multiscale_branches": {"type": "boolean"},
                        "uses_residual": {"type": "boolean"},
                        "uses_batchnorm": {"type": "boolean"},
                        "uses_dropout": {"type": "boolean"},
                        "global_pooling": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                        "embedding_dim_source": {"type": "string", "minLength": 1},
                        "logits_head_type": {"type": "string", "minLength": 1},
                        "key_hyperparameters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["name", "value"],
                                "properties": {
                                    "name": {"type": "string", "minLength": 1},
                                    "value": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                        "shape_notes": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                    },
                },
                "preprocessing_description": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "family_name",
                        "summary",
                        "preserves_sample_axis",
                        "operates_along_time_axis_only",
                        "operations",
                        "key_parameters",
                        "shape_effect",
                        "finite_output_guarantee",
                    ],
                    "properties": {
                        "family_name": {"type": "string", "minLength": 1},
                        "summary": {"type": "string", "minLength": 1},
                        "preserves_sample_axis": {"type": "boolean"},
                        "operates_along_time_axis_only": {"type": "boolean"},
                        "operations": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                        "key_parameters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["name", "value"],
                                "properties": {
                                    "name": {"type": "string", "minLength": 1},
                                    "value": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                        "shape_effect": {"type": "string", "minLength": 1},
                        "finite_output_guarantee": {"type": "boolean"},
                    },
                },
            },
        },
        "snippets": {
            "type": "object",
            "additionalProperties": False,
            "required": ["MODEL_INIT", "EXTRACT_FEATURES", "LOGITS_HEAD", "PREPROCESSING_PIPELINE"],
            "properties": {
                "MODEL_INIT": {"type": "string", "minLength": 1},
                "EXTRACT_FEATURES": {"type": "string", "minLength": 1},
                "LOGITS_HEAD": {"type": "string", "minLength": 1},
                "PREPROCESSING_PIPELINE": {"type": "string", "minLength": 1},
            },
        },
        "proposal_notes": {
            "type": "object",
            "additionalProperties": False,
            "required": ["main_risk", "why_it_might_help", "compatibility_checks"],
            "properties": {
                "main_risk": {"type": "string", "minLength": 1},
                "why_it_might_help": {"type": "string", "minLength": 1},
                "compatibility_checks": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "default": [],
                },
            },
        },
    },
}

REQUIRED_MODEL_SNIPPETS = ("MODEL_INIT", "EXTRACT_FEATURES", "LOGITS_HEAD")
REQUIRED_PREPROCESSING_SNIPPETS = ("PREPROCESSING_PIPELINE",)
VALID_PROPOSAL_MODES = {"explore", "exploit", "ablation", "retry_after_failure"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_text(path: Path) -> Optional[str]:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Could not parse JSON object from model output: response was empty")


    try:
        parsed_full = json.loads(text)
        if isinstance(parsed_full, dict):
            return parsed_full
    except Exception:
        pass

    candidates = []


    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(block for block in fenced_blocks if block and block.strip())


    start = text.find("{")
    found_balanced = False
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
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
                        candidates.append(text[start : i + 1])
                        found_balanced = True
                        break


        if not found_balanced:
            candidates.append(text[start:])


    candidates.append(text)

    last_err: Optional[Exception] = None
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
            last_err = TypeError(f"Parsed JSON was {type(obj).__name__}, expected object")
        except Exception as e:
            last_err = e

    if start == -1:
        raise ValueError("Could not parse JSON object from model output: no '{' found in response")
    if not found_balanced:
        raise ValueError(
            f"Could not parse JSON object from model output: detected '{{' but no balanced '}}' (likely truncated). "
            f"Last error: {last_err}"
        )

    raise ValueError(f"Could not parse JSON object from model output. Last error: {last_err}")


def is_likely_truncated_json_output(text: str) -> bool:
    payload = (text or "").strip()
    if not payload:
        return False

    if payload.count("{") > payload.count("}"):
        return True
    if payload.count("[") > payload.count("]"):
        return True
    if payload.startswith("{") and not payload.endswith("}"):
        return True
    return False


def validate_json(data: Dict[str, Any], schema_path: Path) -> None:
    if jsonschema is None:
        return
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=data, schema=schema)


def compact_context_block(title: str, value: Any) -> str:
    if value is None:
        return f"## {title}\n<missing>\n"
    if isinstance(value, str):
        return f"## {title}\n{value.strip()}\n"
    return f"## {title}\n```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```\n"


def resolve_file_from_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return Path(value)


def to_project_ref(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root)).replace(os.sep, "/")
    except ValueError:
        return str(path)


def sanitize_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "unknown"


def parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def normalize_reasoning_effort(value: Optional[str], *, default: str = "medium") -> str:
    effort = (value or default).strip().lower()
    if effort not in VALID_REASONING_EFFORTS:
        return default
    return effort


def normalize_text_verbosity(value: Optional[str], *, default: str = "low") -> str:
    verbosity = (value or default).strip().lower()
    if verbosity not in VALID_TEXT_VERBOSITY:
        return default
    return verbosity


def is_model_snapshot_pinned(model: str) -> bool:

    return bool(re.search(r"\d{4}-\d{2}-\d{2}$", model.strip()))


def enforce_model_pin_policy(model: str, *, role: str) -> None:
    if is_model_snapshot_pinned(model):
        return

    message = (
        f"{role} is using non-pinned model '{model}'. "
        "Prefer a dated snapshot (for example gpt-5-YYYY-MM-DD) for reproducible cycles."
    )
    if parse_bool_env("ARL_REQUIRE_PINNED_MODEL", default=False):
        raise RuntimeError(message + " Set ARL_REQUIRE_PINNED_MODEL=0 to override.")
    print(f"Warning: {message}")


def cycle_label(cycle_id: str) -> str:
    try:
        return f"cycle_{int(cycle_id):04d}"
    except (TypeError, ValueError):
        return f"cycle_{sanitize_token(cycle_id)}"


def read_latest_manifest(shared_context: Path) -> Optional[str]:
    manifests_root = shared_context / "manifests"
    latest_path: Optional[Path] = None
    latest_cycle = -1

    if manifests_root.exists():
        for candidate in manifests_root.glob("cycle_*/manifest.md"):
            match = re.fullmatch(r"cycle_(\d+)", candidate.parent.name)
            if not match:
                continue
            cycle_num = int(match.group(1))
            if cycle_num > latest_cycle:
                latest_cycle = cycle_num
                latest_path = candidate

    if latest_path is not None:
        return read_text(latest_path)

    return read_text(shared_context / "manifest.md")


def read_cycle_context_json(shared_outbound: Path) -> Optional[Dict[str, Any]]:
    path_env = os.environ.get("ARL_CYCLE_CONTEXT_JSON_PATH")
    cycle_context_path = Path(path_env) if path_env else (shared_outbound / "cycle_context.json")
    return read_json(cycle_context_path)


def trim_text(value: Optional[str], max_chars: int) -> Optional[str]:
    if value is None:
        return None

    text = value.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    removed = len(text) - max_chars
    return text[:max_chars] + f"\n...[TRUNCATED: removed {removed} chars]"


def normalize_fraction(value: Any, *, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default

    if out <= 0.0 or out > 1.0:
        return default
    return out


def ensure_non_empty_str(value: Any, fallback: str, max_chars: Optional[int] = None) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        text = fallback
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def normalize_snippet(text: Any) -> str:
    if not isinstance(text, str):
        return ""

    raw = text.strip()
    if not raw:
        return ""

    fenced = re.findall(r"```(?:python)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = max((block.strip() for block in fenced if block.strip()), key=len, default="")

    lines = raw.splitlines()
    min_indent = None
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        min_indent = indent if min_indent is None else min(min_indent, indent)

    if min_indent and min_indent > 0:
        lines = [line[min_indent:] if len(line) >= min_indent else "" for line in lines]

    return "\n".join(lines).strip("\n")


def replace_snippet_block(template_text: str, marker: str, snippet: str) -> str:
    pattern = re.compile(
        rf"(?P<indent>[ \t]*)# \[PROPOSER_SNIPPET_START:{re.escape(marker)}\](?:\r?\n)"
        rf"(?P<body>.*?)"
        rf"(?P=indent)# \[PROPOSER_SNIPPET_END:{re.escape(marker)}\]",
        flags=re.DOTALL,
    )
    match = pattern.search(template_text)
    if match is None:
        raise ValueError(f"Template marker not found: {marker}")

    indent = match.group("indent")
    normalized = normalize_snippet(snippet)
    if not normalized:
        raise ValueError(f"Snippet for marker {marker} is empty")

    indented_lines = []
    for line in normalized.splitlines():
        if line.strip():
            indented_lines.append(indent + line)
        else:
            indented_lines.append("")
    replacement = "\n".join(indented_lines)

    return template_text[: match.start()] + replacement + template_text[match.end() :]


def extract_marked_model_snippets(model_architecture_code: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    payload = model_architecture_code or ""
    markers_union = "|".join(REQUIRED_MODEL_SNIPPETS)
    for marker in REQUIRED_MODEL_SNIPPETS:
        pattern = re.compile(
            rf"^\s*{re.escape(marker)}\s*:\s*\n?(.*?)"
            rf"(?=^\s*(?:{markers_union})\s*:|\Z)",
            flags=re.DOTALL | re.MULTILINE,
        )
        match = pattern.search(payload)
        if match is None:
            continue
        out[marker] = normalize_snippet(match.group(1))
    return out


def collect_director_snippets(
    *,
    single_call_payload: Optional[Dict[str, Any]],
    normalized_job: Dict[str, Any],
) -> Dict[str, str]:
    snippets: Dict[str, str] = {
        "MODEL_INIT": "",
        "EXTRACT_FEATURES": "",
        "LOGITS_HEAD": "",
        "PREPROCESSING_PIPELINE": "",
    }

    if isinstance(single_call_payload, dict):
        direct_snippets = single_call_payload.get("snippets")
        if isinstance(direct_snippets, dict):
            for marker in tuple(REQUIRED_MODEL_SNIPPETS) + tuple(REQUIRED_PREPROCESSING_SNIPPETS):
                snippets[marker] = normalize_snippet(direct_snippets.get(marker))

    code_instructions = normalized_job.get("code_instructions") if isinstance(normalized_job.get("code_instructions"), dict) else {}
    model_architecture_code = str(code_instructions.get("model_architecture_code", ""))
    parsed_model_snippets = extract_marked_model_snippets(model_architecture_code)
    for marker in REQUIRED_MODEL_SNIPPETS:
        if not snippets.get(marker):
            snippets[marker] = normalize_snippet(parsed_model_snippets.get(marker))

    if not snippets.get("PREPROCESSING_PIPELINE"):
        snippets["PREPROCESSING_PIPELINE"] = normalize_snippet(code_instructions.get("preprocessing_code"))

    missing = [name for name, value in snippets.items() if not value]
    if missing:
        raise RuntimeError(
            "Director could not resolve required code snippets for direct artifact generation. "
            f"Missing snippets: {missing}."
        )

    return snippets


def _normalize_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _normalize_string_list(value: Any, *, fallback: List[str], max_items: int = 12) -> List[str]:
    if not isinstance(value, list):
        return list(fallback)
    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        out.append(text[:240])
        if len(out) >= max_items:
            break
    return out or list(fallback)


def _normalize_named_parameters(value: Any, *, max_items: int = 16) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def _normalize_scalar(raw: Any) -> Any:
        if isinstance(raw, str):
            text = raw.strip()
            return text[:240] if text else ""
        try:
            serialized = json.dumps(raw, ensure_ascii=False)
        except Exception:
            serialized = str(raw)
        serialized = serialized.strip()
        return serialized[:240] if serialized else ""

    def _push(name_raw: Any, value_raw: Any) -> None:
        if len(rows) >= max_items:
            return
        name_text = str(name_raw).strip()
        if not name_text:
            return
        value = _normalize_scalar(value_raw)
        if isinstance(value, str) and not value:
            return
        rows.append({"name": name_text[:120], "value": value})

    if isinstance(value, dict):
        for key in sorted(value.keys(), key=lambda item: str(item)):
            _push(key, value.get(key))
        return rows

    if isinstance(value, list):
        for item in value:
            if len(rows) >= max_items:
                break
            if not isinstance(item, dict):
                continue
            _push(item.get("name"), item.get("value"))
        return rows

    return rows


def normalize_design_payload(
    design_payload: Any,
    *,
    target: Dict[str, Any],
) -> Dict[str, Any]:
    design_raw = design_payload if isinstance(design_payload, dict) else {}
    model_raw = design_raw.get("model_description") if isinstance(design_raw.get("model_description"), dict) else {}
    preprocessing_raw = (
        design_raw.get("preprocessing_description")
        if isinstance(design_raw.get("preprocessing_description"), dict)
        else {}
    )

    target_modality = ensure_non_empty_str(target.get("modality"), "unknown_modality", max_chars=80)
    target_class = ensure_non_empty_str(target.get("class_label"), "unknown_class", max_chars=80)

    model_description = {
        "family_name": ensure_non_empty_str(
            model_raw.get("family_name"),
            f"expert_{sanitize_token(target_modality)}_{sanitize_token(target_class)}",
            max_chars=120,
        ),
        "summary": ensure_non_empty_str(
            model_raw.get("summary"),
            f"Binary expert family for ({target_modality}, {target_class}).",
            max_chars=600,
        ),
        "input_expectation": ensure_non_empty_str(
            model_raw.get("input_expectation"),
            "Input tensor shape (batch, channels, timesteps) with finite float32 values.",
            max_chars=240,
        ),
        "feature_extractor_type": ensure_non_empty_str(
            model_raw.get("feature_extractor_type"),
            "temporal_conv",
            max_chars=120,
        ),
        "uses_multiscale_branches": _normalize_bool(
            model_raw.get("uses_multiscale_branches"),
            default=False,
        ),
        "uses_residual": _normalize_bool(model_raw.get("uses_residual"), default=False),
        "uses_batchnorm": _normalize_bool(model_raw.get("uses_batchnorm"), default=True),
        "uses_dropout": _normalize_bool(model_raw.get("uses_dropout"), default=False),
        "global_pooling": _normalize_string_list(
            model_raw.get("global_pooling"),
            fallback=["adaptive_avg_pool1d"],
            max_items=4,
        ),
        "embedding_dim_source": ensure_non_empty_str(
            model_raw.get("embedding_dim_source"),
            "final_feature_width",
            max_chars=120,
        ),
        "logits_head_type": ensure_non_empty_str(
            model_raw.get("logits_head_type"),
            "linear",
            max_chars=120,
        ),
        "key_hyperparameters": (
            _normalize_named_parameters(
                model_raw.get("key_hyperparameters"),
                max_items=16,
            )
        ),
        "shape_notes": _normalize_string_list(
            model_raw.get("shape_notes"),
            fallback=["Preserve sample axis and produce one logit vector per sample."],
            max_items=8,
        ),
    }

    preprocessing_description = {
        "family_name": ensure_non_empty_str(
            preprocessing_raw.get("family_name"),
            f"prep_{sanitize_token(target_modality)}_{sanitize_token(target_class)}",
            max_chars=120,
        ),
        "summary": ensure_non_empty_str(
            preprocessing_raw.get("summary"),
            "Numerically stable temporal preprocessing that preserves per-sample alignment.",
            max_chars=600,
        ),
        "preserves_sample_axis": _normalize_bool(
            preprocessing_raw.get("preserves_sample_axis"),
            default=True,
        ),
        "operates_along_time_axis_only": _normalize_bool(
            preprocessing_raw.get("operates_along_time_axis_only"),
            default=True,
        ),
        "operations": _normalize_string_list(
            preprocessing_raw.get("operations"),
            fallback=["cast_float32", "nan_to_num"],
            max_items=12,
        ),
        "key_parameters": (
            _normalize_named_parameters(
                preprocessing_raw.get("key_parameters"),
                max_items=16,
            )
        ),
        "shape_effect": ensure_non_empty_str(
            preprocessing_raw.get("shape_effect"),
            "preserve",
            max_chars=120,
        ),
        "finite_output_guarantee": _normalize_bool(
            preprocessing_raw.get("finite_output_guarantee"),
            default=True,
        ),
    }

    return {
        "model_description": model_description,
        "preprocessing_description": preprocessing_description,
    }


def compute_snippet_hashes(snippets: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in snippets.items():
        text = normalize_snippet(value)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        out[key] = digest
    return out


def update_design_history(
    *,
    project_root: Path,
    cycle_id: str,
    candidate_id: str,
    target: Dict[str, Any],
    design: Dict[str, Any],
    snippets: Dict[str, str],
    snippet_hashes: Dict[str, str],
) -> Path:
    history_path = project_root / "artifacts" / "design_history.json"
    history_payload = read_json(history_path) if history_path.exists() else None
    entries = history_payload.get("entries") if isinstance(history_payload, dict) else None
    if not isinstance(entries, list):
        entries = []

    entry = {
        "cycle_id": str(cycle_id),
        "candidate_id": ensure_non_empty_str(candidate_id, "unknown_candidate", max_chars=160),
        "target": {
            "modality": ensure_non_empty_str(target.get("modality"), "unknown_modality", max_chars=80),
            "class_label": ensure_non_empty_str(target.get("class_label"), "unknown_class", max_chars=80),
        },
        "design": design,
        "snippets": snippets,
        "snippet_hashes": snippet_hashes,
        "updated_at": utc_now_iso(),
    }

    deduped: List[Dict[str, Any]] = []
    for row in entries:
        if not isinstance(row, dict):
            continue
        same_cycle = str(row.get("cycle_id")) == str(entry["cycle_id"])
        same_candidate = str(row.get("candidate_id")) == str(entry["candidate_id"])
        if same_cycle and same_candidate:
            continue
        deduped.append(row)
    deduped.append(entry)

    def _entry_sort_key(item: Dict[str, Any]) -> Any:
        try:
            return (int(str(item.get("cycle_id"))), str(item.get("candidate_id")))
        except Exception:
            return (10**9, str(item.get("candidate_id")))

    deduped.sort(key=_entry_sort_key)
    try:
        max_entries_raw = int(str(os.environ.get("ARL_DESIGN_HISTORY_MAX_ENTRIES", "300")).strip())
    except Exception:
        max_entries_raw = 300
    max_entries = max(50, max_entries_raw)

    payload_out = {
        "schema_version": "1.0",
        "updated_at": utc_now_iso(),
        "entries": deduped[-max_entries:],
    }
    write_json(history_path, payload_out)
    return history_path


def infer_embedding_dim(model_code: str) -> Optional[int]:
    match = re.search(r"self\.embedding_dim\s*=\s*(?:int\()?([0-9]+)", model_code)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except Exception:
        return None
    return value if value > 0 else None


def build_director_repair_prompt(raw_output: str, *, project_id: str, cycle_id: str) -> str:
    clipped = trim_text(raw_output, 2500) or "<empty>"
    return f"""
The previous Director response was not valid JSON.

Reformat it into exactly one valid JSON object that follows the schema requested in the original prompt.

Rules:
- Return only JSON.
- Preserve intent from the original output.
- Keep `project_id` = "{project_id}" and `cycle_id` = "{cycle_id}" when those fields exist in the target schema.

ORIGINAL_OUTPUT:
```text
{clipped}
```
""".strip()


def save_llm_prompt(
    *,
    project_root: Path,
    role: str,
    stage: str,
    project_id: str,
    cycle_id: str,
    api_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
    text_verbosity: str,
    response_schema_name: Optional[str],
    system_prompt: str,
    user_prompt: str,
) -> Path:
    prompt_dir = project_root / "shared" / "prompts" / role / cycle_label(cycle_id)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stage_token = sanitize_token(stage)
    prompt_path = prompt_dir / f"{timestamp}_{stage_token}_prompt.json"
    counter = 1
    while prompt_path.exists():
        prompt_path = prompt_dir / f"{timestamp}_{stage_token}_prompt_{counter:02d}.json"
        counter += 1

    payload = {
        "saved_at": utc_now_iso(),
        "role": role,
        "stage": stage,
        "project_id": project_id,
        "cycle_id": cycle_id,
        "api_url": api_url,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "text_verbosity": text_verbosity,
        "response_schema_name": response_schema_name,
        "request_payload": {
            "model": model,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": max_tokens,
            "reasoning": {
                "effort": reasoning_effort,
            },
            "text": {
                "verbosity": text_verbosity,
                "format": {
                    "type": "json_schema",
                    "name": response_schema_name,
                    "strict": True,
                }
                if response_schema_name
                else None,
            },
        },
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    write_json(prompt_path, payload)
    return prompt_path


def find_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def build_context(project_root: Path) -> Dict[str, Any]:
    repo_root = find_repo_root()
    shared_outbound = project_root / "shared" / "outbound"
    shared_inbound = project_root / "shared" / "inbound"
    context_max_chars = int(
        os.environ.get(
            "ARL_DIRECTOR_MAX_CONTEXT_JSON_CHARS",
            os.environ.get("ARL_DIRECTOR_MAX_REFLECTION_CHARS", "12000"),
        )
    )

    cycle_context_json = read_cycle_context_json(shared_outbound)
    cycle_context_text = None
    if isinstance(cycle_context_json, dict):
        cycle_context_text = trim_text(json.dumps(cycle_context_json, indent=2, ensure_ascii=False), context_max_chars)

    run_cycle_chars = int(os.environ.get("ARL_DIRECTOR_MAX_RUN_CYCLE_CHARS", "12000"))
    model_template_chars = int(os.environ.get("ARL_DIRECTOR_MAX_MODEL_TEMPLATE_CHARS", "12000"))
    preprocessing_template_chars = int(os.environ.get("ARL_DIRECTOR_MAX_PREPROCESSING_TEMPLATE_CHARS", "12000"))

    run_cycle_path = Path(
        os.environ.get("ARL_RUN_CYCLE_PY_PATH", str(repo_root / "local" / "scripts" / "run_cycle.py"))
    )
    model_template_path = Path(
        os.environ.get(
            "ARL_BINARY_EXPERT_MODEL_TEMPLATE_PATH",
            str(repo_root / "remote" / "templates" / "binary_expert_model_template.py"),
        )
    )
    preprocessing_template_path = Path(
        os.environ.get(
            "ARL_PREPROCESSING_TEMPLATE_PATH",
            str(repo_root / "remote" / "templates" / "cycle_preprocessing_template.py"),
        )
    )

    return {
        "cycle_context_json": cycle_context_json,
        "cycle_context_text": cycle_context_text,
        "run_cycle_py": trim_text(read_text(run_cycle_path), run_cycle_chars),
        "model_template_py": trim_text(read_text(model_template_path), model_template_chars),
        "preprocessing_template_py": trim_text(read_text(preprocessing_template_path), preprocessing_template_chars),
        "user_inbox_json": read_json(shared_inbound / "user_inbox.json"),
    }


def build_user_prompt(
    role_card: str,
    project_id: str,
    cycle_id: str,
    context: Dict[str, Any],
) -> str:
        cycle_context_json = context.get("cycle_context_json")
        if not isinstance(cycle_context_json, dict):
                cycle_context_json = {}

        user_inbox = context.get("user_inbox_json")
        if user_inbox is None:
                user_inbox = []

        return f"""
{role_card.strip()}

## Provided metadata

- project_id: {project_id}
- cycle_id: {cycle_id}

## Provided cycle context JSON

```json
{json.dumps(cycle_context_json, indent=2, ensure_ascii=False)}
```

## Provided runtime behavior: run_cycle.py

```python
{context.get("run_cycle_py") or "<missing run_cycle.py>"}
```

## Provided model template

```python
{context.get("model_template_py") or "<missing binary_expert_model_template.py>"}
```

## Provided preprocessing template

```python
{context.get("preprocessing_template_py") or "<missing cycle_preprocessing_template.py>"}
```

## Optional user messages

```json
{json.dumps(user_inbox, indent=2, ensure_ascii=False)}
```
""".strip()


def is_single_call_director_payload(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {"target", "reasoning", "implementation", "snippets", "proposal_notes"}
    return required.issubset(set(payload.keys()))


def build_legacy_outputs_from_single_call(
    payload: Dict[str, Any],
    *,
    project_id: str,
    cycle_id: str,
) -> Dict[str, Any]:
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    reasoning = payload.get("reasoning") if isinstance(payload.get("reasoning"), dict) else {}
    implementation = payload.get("implementation") if isinstance(payload.get("implementation"), dict) else {}
    snippets = payload.get("snippets") if isinstance(payload.get("snippets"), dict) else {}
    notes = payload.get("proposal_notes") if isinstance(payload.get("proposal_notes"), dict) else {}

    modality = ensure_non_empty_str(target.get("modality"), "unknown_modality", max_chars=80)
    class_label = ensure_non_empty_str(target.get("class_label"), "unknown_class", max_chars=80)
    expected_upside = ensure_non_empty_str(target.get("expected_upside"), "moderate", max_chars=20).lower()
    if expected_upside not in {"low", "moderate", "high"}:
        expected_upside = "moderate"
    normalized_design = normalize_design_payload(payload.get("design"), target=target)

    candidate_id = ensure_non_empty_str(
        f"{cycle_label(cycle_id)}_{sanitize_token(modality)}_{sanitize_token(class_label)}_single_call",
        f"{cycle_label(cycle_id)}_candidate_001",
        max_chars=120,
    )

    model_init = ensure_non_empty_str(snippets.get("MODEL_INIT"), "pass", max_chars=12000)
    extract_features = ensure_non_empty_str(snippets.get("EXTRACT_FEATURES"), "return self.feature_proj(torch.zeros((x.size(0), self.embedding_dim), device=x.device, dtype=x.dtype))", max_chars=12000)
    logits_head = ensure_non_empty_str(snippets.get("LOGITS_HEAD"), "return self.classifier(features)", max_chars=12000)
    preprocessing_pipeline = ensure_non_empty_str(snippets.get("PREPROCESSING_PIPELINE"), "return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)", max_chars=12000)

    model_architecture_code = "\n\n".join(
        [
            "MODEL_INIT:\n" + model_init.strip(),
            "EXTRACT_FEATURES:\n" + extract_features.strip(),
            "LOGITS_HEAD:\n" + logits_head.strip(),
        ]
    ).strip()

    why_this_class = ensure_non_empty_str(
        reasoning.get("why_this_class"),
        "Target class selected from cycle context error profile.",
        max_chars=600,
    )
    why_this_modality = ensure_non_empty_str(
        reasoning.get("why_this_modality"),
        "Selected modality offers the strongest expected uplift based on cycle context evidence.",
        max_chars=600,
    )
    difficulty = ensure_non_empty_str(
        reasoning.get("difficulty"),
        "Class boundary remains challenging under current expert family capacity.",
        max_chars=600,
    )
    success_criterion = ensure_non_empty_str(
        reasoning.get("success_criterion"),
        "Increase validate F1 for the target expert family while preserving recall.",
        max_chars=600,
    )

    compatibility_checks = notes.get("compatibility_checks") if isinstance(notes.get("compatibility_checks"), list) else []
    compatibility_checks = [ensure_non_empty_str(x, "", max_chars=300) for x in compatibility_checks if str(x).strip()]

    manifest_lines = [
        "# Cycle Manifest",
        "",
        f"- Project: `{project_id}`",
        f"- Cycle: `{cycle_id}`",
        f"- Target binary expert family: `({modality}, {class_label})`",
        f"- Expected upside: `{expected_upside}`",
        "",
        "## Why This Target",
        f"- {why_this_class}",
        f"- {why_this_modality}",
        "",
        "## Difficulty",
        f"- {difficulty}",
        "",
        "## Success Criterion",
        f"- {success_criterion}",
    ]
    if compatibility_checks:
        manifest_lines.extend(["", "## Compatibility Checks"])
        for row in compatibility_checks[:5]:
            manifest_lines.append(f"- {row}")

    manifest_md = "\n".join(manifest_lines).rstrip() + "\n"

    rationale = [
        why_this_class,
        why_this_modality,
        difficulty,
        success_criterion,
    ]

    directive: Dict[str, Any] = {
        "schema_version": "1.0",
        "directive_id": ensure_non_empty_str(
            payload.get("directive_id"),
            f"{cycle_label(cycle_id)}_directive",
            max_chars=120,
        ),
        "cycle_id": str(cycle_id),
        "project_id": project_id,
        "created_at": utc_now_iso(),
        "created_by": "director",
        "ensemble_architecture_policy": "fixed",
        "objective": ensure_non_empty_str(
            f"Improve binary expert family ({modality}, {class_label}) with shared preprocessing/model updates.",
            "Improve one binary expert based on deterministic cycle context.",
            max_chars=280,
        ),
        "decision_rationale": [x for x in rationale if x],
        "proposal_mode": "explore",
        "train_fraction": normalize_fraction(payload.get("train_fraction"), default=0.2),
        "ensemble_validation_subset_fraction": normalize_fraction(
            payload.get("ensemble_validation_subset_fraction"),
            default=0.2,
        ),
        "jobs": [
            {
                "job_id": ensure_non_empty_str(
                    payload.get("job_id"),
                    f"{cycle_label(cycle_id)}_train_expert_001",
                    max_chars=120,
                ),
                "job_type": "train_expert",
                "target": {
                    "modality": modality,
                    "class_label": class_label,
                },
                "candidate": {
                    "candidate_id": candidate_id,
                    "model_py_ref": "shared/models/model.py",
                    "model_meta_ref": "shared/models/model.meta.json",
                    "preprocessing_py_ref": "shared/models/preprocessing.py",
                    "origin": "director",
                },
                "preprocessing": {
                    "preset": "snippet_defined",
                    "params": {
                        "expected_upside": expected_upside,
                        "shared_across_modality": bool(implementation.get("shared_across_modality", True)),
                    },
                },
                "code_instructions": {
                    "preprocessing_code": preprocessing_pipeline,
                    "model_architecture_code": model_architecture_code,
                },
                "requested_outputs": {
                    "run_end_of_cycle_ensemble_eval": True,
                },
            }
        ],
        "notes": ensure_non_empty_str(
            notes.get("why_it_might_help"),
            "Derived from single-call planner output.",
            max_chars=800,
        ),
    }

    return {
        "manifest_md": manifest_md,
        "directive": directive,
        "design": normalized_design,
    }


def extract_response_text(data: dict) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]

    output = data.get("output", [])
    parts = []
    for item in output:
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in ("output_text", "text"):
                    text = content.get("text")
                    if isinstance(text, str):
                        parts.append(text)

    if parts:
        return "\n".join(parts)

    raise RuntimeError(f"Could not extract text from Responses API payload: {json.dumps(data, indent=2)[:4000]}")


def call_openai_compatible_chat(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
    text_verbosity: str,
    response_schema: Optional[Dict[str, Any]] = None,
    response_schema_name: Optional[str] = None,
    call_label: str = "director",
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    client_request_id = f"eld-nas-{sanitize_token(call_label)}-{uuid.uuid4().hex[:10]}"

    payload = {
        "model": model,
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": max_tokens,
        "reasoning": {
            "effort": reasoning_effort
        },
        "text": {
            "verbosity": text_verbosity,
        },
    }
    if response_schema is not None:
        payload["text"]["format"] = {
            "type": "json_schema",
            "name": response_schema_name or "arl_director_response",
            "strict": True,
            "schema": response_schema,
        }

    req = request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Request-ID": client_request_id,
            "X-ELD-NAS-Request-ID": client_request_id,
            **(extra_headers or {}),
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=300) as resp:
            status_code = int(getattr(resp, "status", 200))
            response_headers = {k.lower(): v for (k, v) in resp.headers.items()}
            body = resp.read().decode("utf-8")
    except error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        response_headers = {k.lower(): v for (k, v) in e.headers.items()} if e.headers else {}
        raise RuntimeError(
            "HTTP error from LLM API: "
            f"{e.code} {e.reason}; "
            f"client_request_id={client_request_id}; "
            f"server_request_id={response_headers.get('x-request-id') or response_headers.get('openai-request-id')}\n"
            f"{err_body}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to call LLM API: {e}") from e

    data = json.loads(body)
    server_request_id = response_headers.get("x-request-id") or response_headers.get("openai-request-id")
    rate_limits = {
        "limit_requests": response_headers.get("x-ratelimit-limit-requests"),
        "remaining_requests": response_headers.get("x-ratelimit-remaining-requests"),
        "reset_requests": response_headers.get("x-ratelimit-reset-requests"),
        "limit_tokens": response_headers.get("x-ratelimit-limit-tokens"),
        "remaining_tokens": response_headers.get("x-ratelimit-remaining-tokens"),
        "reset_tokens": response_headers.get("x-ratelimit-reset-tokens"),
    }
    return {
        "text": extract_response_text(data),
        "client_request_id": client_request_id,
        "server_request_id": server_request_id,
        "http_status": status_code,
        "rate_limits": rate_limits,
    }


def main() -> int:
    if load_dotenv is not None:
        load_dotenv()

    repo_root = find_repo_root()

    project_root = resolve_file_from_env("ARL_PROJECT_ROOT")
    shared_outbound = project_root / "shared" / "outbound"
    shared_outbound.mkdir(parents=True, exist_ok=True)
    project_id = os.environ["ARL_PROJECT_ID"]
    cycle_id = os.environ["ARL_CYCLE_ID"]

    manifest_path = resolve_file_from_env("ARL_MANIFEST_PATH")
    directive_path = resolve_file_from_env("ARL_DIRECTIVE_PATH")
    model_py_path = resolve_file_from_env("ARL_MODEL_PY_PATH")
    model_meta_path = resolve_file_from_env("ARL_MODEL_META_PATH")
    proposal_summary_path = resolve_file_from_env("ARL_PROPOSAL_SUMMARY_PATH")
    preprocessing_py_path = Path(
        os.environ.get(
            "ARL_PREPROCESSING_PY_PATH",
            str(project_root / "shared" / "models" / "preprocessing.py"),
        )
    )

    prompt_path = Path(os.environ.get("ARL_DIRECTOR_PROMPT_PATH", str(repo_root / "prompts" / "director_system.md")))
    schema_path = Path(os.environ.get("ARL_DIRECTIVE_SCHEMA_PATH", str(repo_root / "schemas" / "directive.schema.json")))
    model_template_path = Path(
        os.environ.get(
            "ARL_BINARY_EXPERT_TEMPLATE_PATH",
            str(repo_root / "remote" / "templates" / "binary_expert_model_template.py"),
        )
    )
    preprocessing_template_path = Path(
        os.environ.get(
            "ARL_PREPROCESSING_TEMPLATE_PATH",
            str(repo_root / "remote" / "templates" / "cycle_preprocessing_template.py"),
        )
    )
    model_meta_schema_path = Path(
        os.environ.get(
            "ARL_MODEL_META_SCHEMA_PATH",
            str(repo_root / "schemas" / "model_meta.schema.json"),
        )
    )

    api_url = os.environ.get("ARL_LLM_API_URL", "https://api.openai.com/v1/responses")
    api_key = os.environ.get("ARL_LLM_API_KEY")
    model = os.environ.get("ARL_LLM_MODEL")
    temperature = float(os.environ.get("ARL_LLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.environ.get("ARL_DIRECTOR_MAX_TOKENS", os.environ.get("ARL_LLM_MAX_TOKENS", "3500")))


    if not api_key:
        raise RuntimeError("Missing ARL_LLM_API_KEY")
    if not model:
        raise RuntimeError("Missing ARL_LLM_MODEL")
    enforce_model_pin_policy(model, role="director")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Director prompt not found: {prompt_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Directive schema not found: {schema_path}")
    if not model_template_path.exists():
        raise FileNotFoundError(f"Model template not found: {model_template_path}")
    if not preprocessing_template_path.exists():
        raise FileNotFoundError(f"Preprocessing template not found: {preprocessing_template_path}")
    if not model_meta_schema_path.exists():
        raise FileNotFoundError(f"Model meta schema not found: {model_meta_schema_path}")

    reasoning_effort = normalize_reasoning_effort(os.environ.get("ARL_REASONING_EFFORT"), default="medium")
    text_verbosity = normalize_text_verbosity(os.environ.get("ARL_TEXT_VERBOSITY"), default="low")
    response_schema_name = os.environ.get("ARL_DIRECTOR_RESPONSE_SCHEMA_NAME", "arl_director_single_call_plan_v1")
    response_schema_path = os.environ.get("ARL_DIRECTOR_RESPONSE_SCHEMA_PATH")
    if response_schema_path:
        response_schema = json.loads(Path(response_schema_path).read_text(encoding="utf-8"))
    else:
        response_schema = DIRECTOR_SINGLE_CALL_JSON_SCHEMA

    llm_calls: List[Dict[str, Any]] = []

    def record_llm_call(stage: str, result: Dict[str, Any]) -> None:
        entry = {
            "stage": stage,
            "timestamp": utc_now_iso(),
            "client_request_id": result.get("client_request_id"),
            "server_request_id": result.get("server_request_id"),
            "http_status": result.get("http_status"),
            "rate_limits": result.get("rate_limits") or {},
        }
        llm_calls.append(entry)
        print(
            "LLM call "
            f"{stage}: client_request_id={entry['client_request_id']} "
            f"server_request_id={entry['server_request_id']}"
        )

    role_card = prompt_path.read_text(encoding="utf-8")
    context = build_context(project_root)

    if not isinstance(context.get("cycle_context_json"), dict):
        raise RuntimeError(
            "Director requires deterministic cycle context JSON, but none was found or parsed "
            "(expected ARL_CYCLE_CONTEXT_JSON_PATH or shared/outbound/cycle_context.json)."
        )

    system_prompt = (
        "You are the ELD-NAS experiment planner and code generator. "
        "Return exactly one valid JSON object matching the requested schema and no extra commentary."
    )
    user_prompt = build_user_prompt(
        role_card=role_card,
        project_id=project_id,
        cycle_id=cycle_id,
        context=context,
    )

    prompt_path_out = save_llm_prompt(
        project_root=project_root,
        role="director",
        stage="directive_generation",
        project_id=project_id,
        cycle_id=cycle_id,
        api_url=api_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
        response_schema_name=response_schema_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    primary_call = call_openai_compatible_chat(
        api_url=api_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
        response_schema=response_schema,
        response_schema_name=response_schema_name,
        call_label=f"director_primary_{project_id}_{cycle_id}",
    )
    record_llm_call("primary", primary_call)
    raw = str(primary_call.get("text") or "")

    raw_response_path = shared_outbound / f"director_cycle{cycle_id}_raw_response.txt"
    write_text(raw_response_path, (raw or "").rstrip() + "\n")

    parse_mode = "json"
    repair_prompt_path: Optional[Path] = None
    repair_raw_response_path: Optional[Path] = None
    retry_prompt_path: Optional[Path] = None
    retry_raw_response_path: Optional[Path] = None
    parse_error_primary: Optional[Exception] = None
    parse_error_retry: Optional[Exception] = None

    try:
        parsed = extract_json_object(raw)
    except Exception as parse_err:
        parse_error_primary = parse_err
        latest_raw_for_repair = raw
        retry_max_tokens_floor = int(os.environ.get("ARL_DIRECTOR_JSON_RETRY_MAX_TOKENS", "5000"))
        retry_max_tokens = max(max_tokens, retry_max_tokens_floor)

        if is_likely_truncated_json_output(raw) and retry_max_tokens > max_tokens:
            retry_system_prompt = (
                "You are the ELD-NAS experiment planner and code generator. "
                "Return one complete JSON object only. "
                "Keep output compact and strictly schema-aligned."
            )
            retry_user_prompt = user_prompt

            retry_prompt_path = save_llm_prompt(
                project_root=project_root,
                role="director",
                stage="directive_generation_retry_truncated_json",
                project_id=project_id,
                cycle_id=cycle_id,
                api_url=api_url,
                model=model,
                temperature=temperature,
                max_tokens=retry_max_tokens,
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity,
                response_schema_name=response_schema_name,
                system_prompt=retry_system_prompt,
                user_prompt=retry_user_prompt,
            )

            try:
                retry_call = call_openai_compatible_chat(
                    api_url=api_url,
                    api_key=api_key,
                    model=model,
                    system_prompt=retry_system_prompt,
                    user_prompt=retry_user_prompt,
                    temperature=temperature,
                    max_tokens=retry_max_tokens,
                    reasoning_effort=reasoning_effort,
                    text_verbosity=text_verbosity,
                    response_schema=response_schema,
                    response_schema_name=response_schema_name,
                    call_label=f"director_retry_{project_id}_{cycle_id}",
                )
                record_llm_call("retry_truncated_json", retry_call)
                retry_raw = str(retry_call.get("text") or "")
                retry_raw_response_path = shared_outbound / f"director_cycle{cycle_id}_raw_response_retry_truncated_json.txt"
                write_text(retry_raw_response_path, (retry_raw or "").rstrip() + "\n")
                latest_raw_for_repair = retry_raw
                try:
                    parsed = extract_json_object(retry_raw)
                    parse_mode = "json_retry_truncated"
                except Exception as retry_parse_err:
                    parse_error_retry = retry_parse_err
            except Exception as retry_call_err:
                parse_error_retry = retry_call_err

        if parse_mode != "json_retry_truncated":
            parse_mode = "json_parse_retry"
            repair_system_prompt = (
                "You are a strict JSON formatting assistant. "
                "Return only valid JSON with no additional commentary."
            )
            repair_user_prompt = build_director_repair_prompt(
                latest_raw_for_repair,
                project_id=project_id,
                cycle_id=cycle_id,
            )

            repair_prompt_path = save_llm_prompt(
                project_root=project_root,
                role="director",
                stage="directive_generation_repair_non_json",
                project_id=project_id,
                cycle_id=cycle_id,
                api_url=api_url,
                model=model,
                temperature=temperature,
                max_tokens=retry_max_tokens,
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity,
                response_schema_name=response_schema_name,
                system_prompt=repair_system_prompt,
                user_prompt=repair_user_prompt,
            )

            try:
                repair_call = call_openai_compatible_chat(
                    api_url=api_url,
                    api_key=api_key,
                    model=model,
                    system_prompt=repair_system_prompt,
                    user_prompt=repair_user_prompt,
                    temperature=temperature,
                    max_tokens=retry_max_tokens,
                    reasoning_effort=reasoning_effort,
                    text_verbosity=text_verbosity,
                    response_schema=response_schema,
                    response_schema_name=response_schema_name,
                    call_label=f"director_repair_{project_id}_{cycle_id}",
                )
                record_llm_call("repair_non_json", repair_call)
                repair_raw = str(repair_call.get("text") or "")
                repair_raw_response_path = shared_outbound / f"director_cycle{cycle_id}_raw_response_repair.txt"
                write_text(repair_raw_response_path, (repair_raw or "").rstrip() + "\n")
                parsed = extract_json_object(repair_raw)
                parse_mode = "json_repair"
            except Exception as repair_err:
                parse_mode = "error_non_json"
                parts = [
                    "Director output was not valid JSON after primary and repair attempts.",
                    f"Primary raw output: {raw_response_path}",
                ]
                if retry_prompt_path is not None:
                    parts.append(f"Retry prompt (higher max tokens): {retry_prompt_path}")
                if retry_raw_response_path is not None:
                    parts.append(f"Retry raw output: {retry_raw_response_path}")
                if repair_prompt_path is not None:
                    parts.append(f"Repair prompt: {repair_prompt_path}")
                if repair_raw_response_path is not None:
                    parts.append(f"Repair raw output: {repair_raw_response_path}")
                parts.append(f"Primary parse error: {parse_error_primary}")
                if parse_error_retry is not None:
                    parts.append(f"Retry parse error: {parse_error_retry}")
                parts.append(f"Repair error: {repair_err}")
                raise ValueError("\n".join(parts)) from repair_err

    single_call_payload_path: Optional[Path] = None
    single_call_payload: Optional[Dict[str, Any]] = None
    single_call_design_payload: Optional[Dict[str, Any]] = None

    if is_single_call_director_payload(parsed):
        single_call_payload = parsed
        converted = build_legacy_outputs_from_single_call(
            parsed,
            project_id=project_id,
            cycle_id=cycle_id,
        )
        manifest_md = converted.get("manifest_md")
        directive = converted.get("directive")
        single_call_design_payload = converted.get("design") if isinstance(converted.get("design"), dict) else None
        single_call_payload_path = shared_outbound / f"director_cycle{cycle_id}_single_call_plan.json"
        write_json(single_call_payload_path, parsed)
        parse_mode = f"{parse_mode}_single_call"
    elif isinstance(parsed, dict) and "manifest_md" in parsed and "directive" in parsed:
        manifest_md = parsed.get("manifest_md")
        directive = parsed.get("directive")
    else:
        parse_mode = "error_missing_required_keys"
        raise ValueError(
            "Director output JSON must contain either legacy keys (`manifest_md`, `directive`) "
            "or single-call keys (`target`, `reasoning`, `implementation`, `design`, `snippets`, `proposal_notes`). "
            f"See raw output at {raw_response_path}."
        )

    if not isinstance(manifest_md, str):
        parse_mode = "error_manifest_type"
        raise TypeError(
            "Director output field `manifest_md` must be a string. "
            f"See raw output at {raw_response_path}."
        )
    if not isinstance(directive, dict):
        parse_mode = "error_directive_type"
        raise TypeError(
            "Director output field `directive` must be an object. "
            f"See raw output at {raw_response_path}."
        )


    directive["schema_version"] = "1.0"
    directive["created_by"] = "director"
    directive["project_id"] = project_id
    directive["cycle_id"] = cycle_id
    directive["ensemble_architecture_policy"] = "fixed"
    directive.setdefault("created_at", utc_now_iso())
    directive["directive_id"] = ensure_non_empty_str(
        directive.get("directive_id"),
        f"{cycle_label(cycle_id)}_directive",
        max_chars=120,
    )
    directive["objective"] = ensure_non_empty_str(
        directive.get("objective"),
        "Improve one binary expert based on the deterministic cycle context.",
        max_chars=280,
    )

    reasons_raw = directive.get("decision_rationale")
    normalized_reasons: List[str] = []
    if isinstance(reasons_raw, list):
        for item in reasons_raw:
            text = ensure_non_empty_str(item, "", max_chars=220)
            if text:
                normalized_reasons.append(text)
    if len(normalized_reasons) > 4:
        normalized_reasons = normalized_reasons[:4]
    if not normalized_reasons:
        normalized_reasons = [
            "Selected from the deterministic cycle context as the highest-impact next binary expert target.",
        ]
    directive["decision_rationale"] = normalized_reasons

    valid_modes = {"explore", "exploit", "ablation", "retry_after_failure"}
    proposal_mode = str(directive.get("proposal_mode", "")).strip()
    if proposal_mode not in valid_modes:
        proposal_mode = "explore"
    directive["proposal_mode"] = proposal_mode

    directive["train_fraction"] = normalize_fraction(directive.get("train_fraction"), default=0.2)
    directive["ensemble_validation_subset_fraction"] = normalize_fraction(
        directive.get("ensemble_validation_subset_fraction"),
        default=0.2,
    )

    jobs = directive.get("jobs")
    if not isinstance(jobs, list):
        jobs = []

    first_job = jobs[0] if jobs and isinstance(jobs[0], dict) else {}
    target = first_job.get("target") if isinstance(first_job.get("target"), dict) else {}
    candidate = first_job.get("candidate") if isinstance(first_job.get("candidate"), dict) else {}
    preprocessing = first_job.get("preprocessing") if isinstance(first_job.get("preprocessing"), dict) else {}
    code_instructions = (
        first_job.get("code_instructions")
        if isinstance(first_job.get("code_instructions"), dict)
        else {}
    )
    model_py_ref = to_project_ref(model_py_path, project_root)
    model_meta_ref = to_project_ref(model_meta_path, project_root)
    preprocessing_py_ref = to_project_ref(preprocessing_py_path, project_root)

    normalized_job: Dict[str, Any] = {
        "job_id": ensure_non_empty_str(
            first_job.get("job_id"),
            f"{cycle_label(cycle_id)}_train_expert_001",
            max_chars=120,
        ),
        "job_type": "train_expert",
        "target": {
            "modality": ensure_non_empty_str(target.get("modality"), "unknown_modality", max_chars=80),
            "class_label": ensure_non_empty_str(target.get("class_label"), "unknown_class", max_chars=80),
        },
        "candidate": {
            "candidate_id": ensure_non_empty_str(
                candidate.get("candidate_id"),
                f"{cycle_label(cycle_id)}_candidate_001",
                max_chars=120,
            ),
            "model_py_ref": model_py_ref,
            "model_meta_ref": model_meta_ref,
            "preprocessing_py_ref": preprocessing_py_ref,
            "origin": "director",
        },
        "preprocessing": {
            "preset": ensure_non_empty_str(preprocessing.get("preset"), "no_preprocessing", max_chars=120),
            "params": preprocessing.get("params") if isinstance(preprocessing.get("params"), dict) else {},
        },
        "code_instructions": {
            "preprocessing_code": ensure_non_empty_str(
                code_instructions.get("preprocessing_code"),
                "Write a deterministic preprocessing function for this binary expert and document parameter choices.",
                max_chars=12000,
            ),
            "model_architecture_code": ensure_non_empty_str(
                code_instructions.get("model_architecture_code"),
                "Write a binary ExpertModel architecture update for this target while keeping the ensemble architecture unchanged.",
                max_chars=12000,
            ),
        },
        "requested_outputs": {
            "run_end_of_cycle_ensemble_eval": True,
        },
    }

    if isinstance(first_job.get("training_overrides"), dict):
        normalized_job["training_overrides"] = first_job["training_overrides"]

    normalized_directive: Dict[str, Any] = {
        "schema_version": directive["schema_version"],
        "directive_id": directive["directive_id"],
        "cycle_id": directive["cycle_id"],
        "project_id": directive["project_id"],
        "created_at": directive["created_at"],
        "created_by": directive["created_by"],
        "ensemble_architecture_policy": directive["ensemble_architecture_policy"],
        "objective": directive["objective"],
        "decision_rationale": directive["decision_rationale"],
        "proposal_mode": directive["proposal_mode"],
        "train_fraction": directive["train_fraction"],
        "ensemble_validation_subset_fraction": directive["ensemble_validation_subset_fraction"],
        "jobs": [normalized_job],
        "manifest_ref": to_project_ref(manifest_path, project_root),
    }

    if isinstance(directive.get("notes"), str) and directive.get("notes", "").strip():
        normalized_directive["notes"] = ensure_non_empty_str(
            directive.get("notes"),
            "",
            max_chars=3000,
        )

    directive = normalized_directive

    validate_json(directive, schema_path)

    snippets = collect_director_snippets(
        single_call_payload=single_call_payload,
        normalized_job=normalized_job,
    )
    target_payload_for_design = normalized_job.get("target") if isinstance(normalized_job.get("target"), dict) else {}
    design_payload = normalize_design_payload(
        single_call_design_payload or (single_call_payload.get("design") if isinstance(single_call_payload, dict) else None),
        target=target_payload_for_design,
    )
    snippet_hashes = compute_snippet_hashes(snippets)

    model_template_text = read_text(model_template_path)
    preprocessing_template_text = read_text(preprocessing_template_path)
    if not isinstance(model_template_text, str) or not model_template_text.strip():
        raise RuntimeError(f"Model template is missing or empty: {model_template_path}")
    if not isinstance(preprocessing_template_text, str) or not preprocessing_template_text.strip():
        raise RuntimeError(f"Preprocessing template is missing or empty: {preprocessing_template_path}")

    model_code = model_template_text
    for marker in REQUIRED_MODEL_SNIPPETS:
        model_code = replace_snippet_block(model_code, marker, snippets[marker])

    preprocessing_code = preprocessing_template_text
    for marker in REQUIRED_PREPROCESSING_SNIPPETS:
        preprocessing_code = replace_snippet_block(preprocessing_code, marker, snippets[marker])

    if "PROPOSER_SNIPPET_START:" in model_code or "PROPOSER_SNIPPET_START:" in preprocessing_code:
        raise RuntimeError("Snippet markers remain in Director-generated code; output is incomplete.")
    if "NotImplementedError(\"Fill" in model_code or "NotImplementedError(\"Fill" in preprocessing_code:
        raise RuntimeError("Template placeholders were not fully replaced by Director output.")

    write_text(model_py_path, model_code.rstrip() + "\n")
    write_text(preprocessing_py_path, preprocessing_code.rstrip() + "\n")

    candidate_payload = normalized_job.get("candidate") if isinstance(normalized_job.get("candidate"), dict) else {}
    target_payload = normalized_job.get("target") if isinstance(normalized_job.get("target"), dict) else {}
    candidate_id = ensure_non_empty_str(
        candidate_payload.get("candidate_id"),
        f"{cycle_label(cycle_id)}_candidate_001",
        max_chars=120,
    )

    rationale_payload = directive.get("decision_rationale") if isinstance(directive.get("decision_rationale"), list) else []
    rationale_text = " ".join(str(x).strip() for x in rationale_payload if str(x).strip())

    proposal_mode = str(directive.get("proposal_mode", "explore")).strip()
    if proposal_mode not in VALID_PROPOSAL_MODES:
        proposal_mode = "explore"

    model_meta_payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "project_id": project_id,
        "target": {
            "modality": ensure_non_empty_str(target_payload.get("modality"), "unknown_modality", max_chars=80),
            "class_label": ensure_non_empty_str(target_payload.get("class_label"), "unknown_class", max_chars=80),
        },
        "proposal_mode": proposal_mode,
        "model_family": "directive_generated_binary_expert",
        "input_contract": {
            "expected_tensor_shape": ["B", "C", "T"],
            "channels": None,
            "assumptions": "run_cycle.py normalizes preprocessed arrays to (B, C, T) before model forward.",
        },
        "output_contract": {
            "returns_logit": True,
            "returns_embedding": True,
            "embedding_dim": infer_embedding_dim(model_code),
        },
        "intended_hypothesis": ensure_non_empty_str(
            rationale_text or directive.get("objective"),
            "This candidate should improve target binary expert performance while preserving ensemble compatibility.",
            max_chars=500,
        ),
    }
    validate_json(model_meta_payload, model_meta_schema_path)
    write_json(model_meta_path, model_meta_payload)

    proposal_summary_payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "project_id": project_id,
        "cycle_id": str(cycle_id),
        "candidate_id": candidate_id,
        "proposal_mode": proposal_mode,
        "target": {
            "modality": ensure_non_empty_str(target_payload.get("modality"), "unknown_modality", max_chars=80),
            "class_label": ensure_non_empty_str(target_payload.get("class_label"), "unknown_class", max_chars=80),
        },
        "title": f"Cycle {cycle_id} binary expert proposal",
        "changed_factors": ["preprocessing", "model_architecture"],
        "rationale": rationale_payload,
        "generated_files": {
            "model_py_ref": to_project_ref(model_py_path, project_root),
            "preprocessing_py_ref": to_project_ref(preprocessing_py_path, project_root),
            "model_meta_ref": to_project_ref(model_meta_path, project_root),
        },
        "design": design_payload,
        "snippets": snippets,
        "snippet_hashes": snippet_hashes,
        "notes": ensure_non_empty_str(
            directive.get("notes"),
            "Generated directly by Director single-call output.",
            max_chars=600,
        ),
    }
    write_json(proposal_summary_path, proposal_summary_payload)

    design_history_path = update_design_history(
        project_root=project_root,
        cycle_id=str(cycle_id),
        candidate_id=candidate_id,
        target={
            "modality": ensure_non_empty_str(target_payload.get("modality"), "unknown_modality", max_chars=80),
            "class_label": ensure_non_empty_str(target_payload.get("class_label"), "unknown_class", max_chars=80),
        },
        design=design_payload,
        snippets=snippets,
        snippet_hashes=snippet_hashes,
    )

    write_text(manifest_path, manifest_md.rstrip() + "\n")
    write_json(directive_path, directive)

    llm_telemetry_path = shared_outbound / f"director_cycle{cycle_id}_llm_requests.json"
    write_json(
        llm_telemetry_path,
        {
            "project_id": project_id,
            "cycle_id": cycle_id,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "text_verbosity": text_verbosity,
            "response_schema_name": response_schema_name,
            "calls": llm_calls,
        },
    )

    print(f"Saved prompt:    {prompt_path_out}")
    if retry_prompt_path is not None:
        print(f"Saved retry prompt: {retry_prompt_path}")
    if retry_raw_response_path is not None:
        print(f"Saved retry output: {retry_raw_response_path}")
    if repair_prompt_path is not None:
        print(f"Saved repair prompt: {repair_prompt_path}")
    print(f"Saved raw response:  {raw_response_path}")
    if repair_raw_response_path is not None:
        print(f"Saved repair output: {repair_raw_response_path}")
    if single_call_payload_path is not None:
        print(f"Saved single-call plan: {single_call_payload_path}")
    print(f"Saved request log:  {llm_telemetry_path}")
    print(f"Parse mode:      {parse_mode}")
    print(f"Wrote manifest:  {manifest_path}")
    print(f"Wrote directive: {directive_path}")
    print(f"Wrote model code: {model_py_path}")
    print(f"Wrote preprocessing: {preprocessing_py_path}")
    print(f"Wrote model meta: {model_meta_path}")
    print(f"Wrote proposal summary: {proposal_summary_path}")
    print(f"Updated design history: {design_history_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)