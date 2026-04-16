#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request

try:
    import jsonschema  # type: ignore
except Exception:
    jsonschema = None

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # type: ignore[assignment,misc]


REQUIRED_MODEL_SNIPPETS = ("MODEL_INIT", "EXTRACT_FEATURES", "LOGITS_HEAD")
REQUIRED_PREPROCESSING_SNIPPETS = ("PREPROCESSING_PIPELINE",)
VALID_PROPOSAL_MODES = {"explore", "exploit", "ablation", "retry_after_failure"}
VALID_REASONING_EFFORTS = {"low", "medium", "high"}
VALID_TEXT_VERBOSITY = {"low", "medium", "high"}

PROPOSER_SNIPPET_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["model_snippets", "preprocessing_snippets", "proposal_notes"],
    "properties": {
        "model_snippets": {
            "type": "object",
            "additionalProperties": False,
            "required": ["MODEL_INIT", "EXTRACT_FEATURES", "LOGITS_HEAD"],
            "properties": {
                "MODEL_INIT": {"type": "string", "minLength": 1},
                "EXTRACT_FEATURES": {"type": "string", "minLength": 1},
                "LOGITS_HEAD": {"type": "string", "minLength": 1},
            },
        },
        "preprocessing_snippets": {
            "type": "object",
            "additionalProperties": False,
            "required": ["PREPROCESSING_PIPELINE"],
            "properties": {
                "PREPROCESSING_PIPELINE": {"type": "string", "minLength": 1},
            },
        },
        "proposal_notes": {"type": "string", "minLength": 1},
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def truncate_text(text: str, max_chars: int, *, label: str) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    removed = len(text) - max_chars
    trailer = f"\n...[{label}: removed {removed} chars]"
    keep = max(0, max_chars - len(trailer))
    return text[:keep] + trailer


def ensure_non_empty_str(value: Any, fallback: str, max_chars: Optional[int] = None) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        text = fallback
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def to_project_ref(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root)).replace(os.sep, "/")
    except ValueError:
        return str(path)


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
                        candidates.append(text[start : idx + 1])
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
        except Exception as exc:
            last_err = exc

    if start == -1:
        raise ValueError(
            "Could not parse JSON object from model output: no '{' found in response"
        )
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

    raise RuntimeError(
        f"Could not extract text from Responses API payload: {json.dumps(data, indent=2)[:4000]}"
    )


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
    call_label: str = "proposer",
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    client_request_id = f"eld-nas-{sanitize_token(call_label)}-{uuid.uuid4().hex[:10]}"

    payload = {
        "model": model,
        "instructions": system_prompt,
        "input": user_prompt,
        "max_output_tokens": max_tokens,
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "verbosity": text_verbosity,
        },
    }
    if response_schema is not None:
        payload["text"]["format"] = {
            "type": "json_schema",
            "name": response_schema_name or "arl_proposer_snippets",
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
    except error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        response_headers = {k.lower(): v for (k, v) in exc.headers.items()} if exc.headers else {}
        raise RuntimeError(
            "HTTP error from LLM API: "
            f"{exc.code} {exc.reason}; "
            f"client_request_id={client_request_id}; "
            f"server_request_id={response_headers.get('x-request-id') or response_headers.get('openai-request-id')}\n"
            f"{err_body}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to call LLM API: {exc}") from exc

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


def resolve_file_from_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return Path(value)


def find_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


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
            "reasoning": {"effort": reasoning_effort},
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


def validate_json(data: Dict[str, Any], schema_path: Path) -> None:
    if jsonschema is None:
        return
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=data, schema=schema)


def normalize_snippet(text: Any) -> str:
    if not isinstance(text, str):
        return ""

    raw = text.strip()
    if not raw:
        return ""

    fenced = re.findall(r"```(?:python)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = max((block.strip() for block in fenced if block.strip()), key=len, default="")

    return textwrap.dedent(raw).strip("\n")


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

    out = template_text[: match.start()] + replacement + template_text[match.end() :]
    return out


def extract_primary_job(directive: Dict[str, Any]) -> Dict[str, Any]:
    jobs = directive.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("directive.json must contain at least one job")
    first = jobs[0]
    if not isinstance(first, dict):
        raise TypeError("First directive job must be an object")
    return first


def build_generation_prompt(
    *,
    role_card: str,
    directive: Dict[str, Any],
    model_template: str,
    preprocessing_template: str,
    run_cycle_context: str,
) -> str:
    job = extract_primary_job(directive)
    target = job.get("target") if isinstance(job.get("target"), dict) else {}
    code_instructions = job.get("code_instructions") if isinstance(job.get("code_instructions"), dict) else {}

    return f"""
You are acting as the ELD-NAS Proposer. Fill template snippet blocks with valid Python code.

# ROLE CARD
{role_card}

# RETURN FORMAT (JSON ONLY)
Return exactly one JSON object with keys:
{{
  "model_snippets": {{
    "MODEL_INIT": "...",
    "EXTRACT_FEATURES": "...",
    "LOGITS_HEAD": "..."
  }},
  "preprocessing_snippets": {{
    "PREPROCESSING_PIPELINE": "..."
  }},
  "proposal_notes": "..."
}}

# STRICT RULES
- Return only JSON (no markdown fences).
- Each snippet must be a block body only (no enclosing class/def declarations).
- Do not include snippet marker comments in outputs.
- Generated model must keep class name `BinaryExpertModel` and methods `extract_features` and `forward`.
- `extract_features` must return (B, D).
- `forward` must return logits shape (B, n_classes).
- The model can be any PyTorch architecture.
- Preprocessing snippet must not include train/validation split loading.
- `proposal_notes` must include:
    - a short diagnosis of task difficulty,
    - expected improvement realism for this cycle (`low`, `moderate`, or `high`),
    - and, if relevant, a warning when targeting appears based only on weakest-link heuristics without clear modality promise.

# DIRECTIVE
```json
{json.dumps(directive, indent=2, ensure_ascii=False)}
```

# TARGET
```json
{json.dumps(target, indent=2, ensure_ascii=False)}
```

# CODE INSTRUCTIONS FROM DIRECTOR
preprocessing_code:
{ensure_non_empty_str(code_instructions.get("preprocessing_code"), "N/A", max_chars=2000)}

model_architecture_code:
{ensure_non_empty_str(code_instructions.get("model_architecture_code"), "N/A", max_chars=2000)}

# RUNNER CONTEXT (for compatibility expectations)
```python
{run_cycle_context}
```

# MODEL TEMPLATE
```python
{model_template}
```

# PREPROCESSING TEMPLATE
```python
{preprocessing_template}
```
""".strip()


def build_repair_prompt(raw_output: str) -> str:
    clipped = truncate_text(raw_output, 3500, label="TRUNCATED_RAW_OUTPUT")
    return f"""
Your previous response was not valid JSON for the required contract.

Return exactly one JSON object with this structure:
{{
  "model_snippets": {{
    "MODEL_INIT": "...",
    "EXTRACT_FEATURES": "...",
    "LOGITS_HEAD": "..."
  }},
  "preprocessing_snippets": {{
    "PREPROCESSING_PIPELINE": "..."
  }},
  "proposal_notes": "..."
}}

Rules:
- Output JSON only.
- Keep snippet values as plain code strings (no markdown fences).
- Include all required snippet keys.

PREVIOUS_OUTPUT:
```text
{clipped}
```
""".strip()


def parse_snippet_payload(parsed: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    model_snippets = parsed.get("model_snippets")
    preprocessing_snippets = parsed.get("preprocessing_snippets")
    if not isinstance(model_snippets, dict):
        raise TypeError("model_snippets must be an object")
    if not isinstance(preprocessing_snippets, dict):
        raise TypeError("preprocessing_snippets must be an object")

    out_model: Dict[str, str] = {}
    out_pre: Dict[str, str] = {}

    for key in REQUIRED_MODEL_SNIPPETS:
        snippet = normalize_snippet(model_snippets.get(key))
        if not snippet:
            raise ValueError(f"Missing/empty model snippet: {key}")
        out_model[key] = snippet

    for key in REQUIRED_PREPROCESSING_SNIPPETS:
        snippet = normalize_snippet(preprocessing_snippets.get(key))
        if not snippet:
            raise ValueError(f"Missing/empty preprocessing snippet: {key}")
        out_pre[key] = snippet

    return {
        "model": out_model,
        "preprocessing": out_pre,
    }


def infer_embedding_dim(model_code: str) -> Optional[int]:
    match = re.search(r"self\.embedding_dim\s*=\s*(?:int\()?(\d+)", model_code)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def build_model_meta(
    *,
    directive: Dict[str, Any],
    project_id: str,
    candidate_id: str,
    model_code: str,
) -> Dict[str, Any]:
    job = extract_primary_job(directive)
    target = job.get("target") if isinstance(job.get("target"), dict) else {}
    proposal_mode = str(directive.get("proposal_mode", "explore")).strip()
    if proposal_mode not in VALID_PROPOSAL_MODES:
        proposal_mode = "explore"

    objective = ensure_non_empty_str(directive.get("objective"), "Binary expert update", max_chars=240)
    rationale = directive.get("decision_rationale")
    rationale_text = ""
    if isinstance(rationale, list):
        rationale_text = " ".join(str(x).strip() for x in rationale if str(x).strip())

    intended_hypothesis = ensure_non_empty_str(
        rationale_text or objective,
        "This candidate should improve target binary expert performance while preserving ensemble-compatible embeddings.",
        max_chars=500,
    )

    embedding_dim = infer_embedding_dim(model_code)

    return {
        "schema_version": "1.0",
        "candidate_id": candidate_id,
        "project_id": project_id,
        "target": {
            "modality": ensure_non_empty_str(target.get("modality"), "unknown_modality", max_chars=80),
            "class_label": ensure_non_empty_str(target.get("class_label"), "unknown_class", max_chars=80),
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
            "embedding_dim": embedding_dim,
        },
        "intended_hypothesis": intended_hypothesis,
    }


def build_proposal_summary(
    *,
    directive: Dict[str, Any],
    project_id: str,
    cycle_id: str,
    candidate_id: str,
    model_py_ref: str,
    preprocessing_py_ref: str,
    model_meta_ref: str,
    proposal_notes: str,
) -> Dict[str, Any]:
    job = extract_primary_job(directive)
    target = job.get("target") if isinstance(job.get("target"), dict) else {}
    code_instructions = job.get("code_instructions") if isinstance(job.get("code_instructions"), dict) else {}

    mode = str(directive.get("proposal_mode", "explore")).strip()
    if mode not in VALID_PROPOSAL_MODES:
        mode = "explore"

    changed_factors = ["preprocessing", "model_architecture"]
    rationale = directive.get("decision_rationale") if isinstance(directive.get("decision_rationale"), list) else []

    return {
        "schema_version": "1.0",
        "project_id": project_id,
        "cycle_id": str(cycle_id),
        "candidate_id": candidate_id,
        "proposal_mode": mode,
        "target": {
            "modality": ensure_non_empty_str(target.get("modality"), "unknown_modality", max_chars=80),
            "class_label": ensure_non_empty_str(target.get("class_label"), "unknown_class", max_chars=80),
        },
        "title": f"Cycle {cycle_id} binary expert proposal",
        "changed_factors": changed_factors,
        "rationale": rationale,
        "instructions_digest": {
            "preprocessing_code": ensure_non_empty_str(code_instructions.get("preprocessing_code"), "N/A", max_chars=800),
            "model_architecture_code": ensure_non_empty_str(code_instructions.get("model_architecture_code"), "N/A", max_chars=800),
        },
        "generated_files": {
            "model_py_ref": model_py_ref,
            "preprocessing_py_ref": preprocessing_py_ref,
            "model_meta_ref": model_meta_ref,
        },
        "notes": ensure_non_empty_str(proposal_notes, "", max_chars=600),
    }


def build_candidate_snapshot_paths(project_root: Path, cycle_id: str, candidate_id: str) -> Dict[str, Path]:
    snapshot_dir = (
        project_root
        / "artifacts"
        / "candidate_snapshots"
        / cycle_label(cycle_id)
        / sanitize_token(candidate_id)
    )
    return {
        "snapshot_dir": snapshot_dir,
        "model_py_path": snapshot_dir / "model.py",
        "preprocessing_py_path": snapshot_dir / "preprocessing.py",
        "model_meta_path": snapshot_dir / "model.meta.json",
        "proposal_summary_path": snapshot_dir / "proposal_summary.json",
    }


def main() -> int:
    if load_dotenv is not None:
        load_dotenv()

    repo_root = find_repo_root()

    project_root = resolve_file_from_env("ARL_PROJECT_ROOT")
    project_id = os.environ["ARL_PROJECT_ID"]
    cycle_id = os.environ["ARL_CYCLE_ID"]

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

    prompt_path = Path(
        os.environ.get(
            "ARL_PROPOSER_PROMPT_PATH",
            str(repo_root / "prompts" / "proposer_system.md"),
        )
    )
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
    run_cycle_context_path = Path(
        os.environ.get(
            "ARL_RUN_CYCLE_CONTEXT_PATH",
            str(repo_root / "local" / "scripts" / "run_cycle.py"),
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
    max_tokens = int(os.environ.get("ARL_LLM_MAX_TOKENS", "4500"))

    max_runner_chars = int(os.environ.get("ARL_PROPOSER_MAX_RUNNER_CHARS", "28000"))
    max_template_chars = int(os.environ.get("ARL_PROPOSER_MAX_TEMPLATE_CHARS", "18000"))

    reasoning_effort = normalize_reasoning_effort(os.environ.get("ARL_REASONING_EFFORT"), default="medium")
    text_verbosity = normalize_text_verbosity(os.environ.get("ARL_TEXT_VERBOSITY"), default="low")
    response_schema_name = os.environ.get("ARL_PROPOSER_RESPONSE_SCHEMA_NAME", "arl_proposer_snippets_v1")
    response_schema_path = os.environ.get("ARL_PROPOSER_RESPONSE_SCHEMA_PATH")
    if response_schema_path:
        response_schema = json.loads(Path(response_schema_path).read_text(encoding="utf-8"))
    else:
        response_schema = PROPOSER_SNIPPET_JSON_SCHEMA

    llm_calls: list[Dict[str, Any]] = []

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

    if not api_key:
        raise RuntimeError("Missing ARL_LLM_API_KEY")
    if not model:
        raise RuntimeError("Missing ARL_LLM_MODEL")
    enforce_model_pin_policy(model, role="proposer")
    if not directive_path.exists():
        raise FileNotFoundError(f"Missing directive.json: {directive_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Proposer prompt not found: {prompt_path}")
    if not model_template_path.exists():
        raise FileNotFoundError(f"Model template not found: {model_template_path}")
    if not preprocessing_template_path.exists():
        raise FileNotFoundError(f"Preprocessing template not found: {preprocessing_template_path}")
    if not run_cycle_context_path.exists():
        raise FileNotFoundError(f"run_cycle context file not found: {run_cycle_context_path}")
    if not model_meta_schema_path.exists():
        raise FileNotFoundError(f"Model meta schema not found: {model_meta_schema_path}")

    role_card = prompt_path.read_text(encoding="utf-8")
    directive = read_json(directive_path)
    if not isinstance(directive, dict):
        raise RuntimeError(f"Invalid directive JSON at {directive_path}")

    model_template = model_template_path.read_text(encoding="utf-8")
    preprocessing_template = preprocessing_template_path.read_text(encoding="utf-8")
    run_cycle_context = run_cycle_context_path.read_text(encoding="utf-8")

    model_template = truncate_text(model_template, max_template_chars, label="TRUNCATED_MODEL_TEMPLATE")
    preprocessing_template = truncate_text(preprocessing_template, max_template_chars, label="TRUNCATED_PREPROCESSING_TEMPLATE")
    run_cycle_context = truncate_text(run_cycle_context, max_runner_chars, label="TRUNCATED_RUN_CYCLE")

    system_prompt = (
        "You are a careful code-generation assistant for template snippet filling. "
        "Return exactly the requested JSON object and no additional text."
    )
    user_prompt = build_generation_prompt(
        role_card=role_card,
        directive=directive,
        model_template=model_template,
        preprocessing_template=preprocessing_template,
        run_cycle_context=run_cycle_context,
    )

    prompt_path_out = save_llm_prompt(
        project_root=project_root,
        role="proposer",
        stage="snippet_generation",
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
        call_label=f"proposer_primary_{project_id}_{cycle_id}",
    )
    record_llm_call("primary", primary_call)
    raw = str(primary_call.get("text") or "")

    shared_proposals = project_root / "shared" / "proposals"
    shared_proposals.mkdir(parents=True, exist_ok=True)
    raw_response_path = shared_proposals / f"proposer_cycle{cycle_id}_raw_response.txt"
    write_text(raw_response_path, (raw or "").rstrip() + "\n")

    repair_prompt_path_out: Optional[Path] = None
    repair_raw_response_path: Optional[Path] = None
    retry_prompt_path_out: Optional[Path] = None
    retry_raw_response_path: Optional[Path] = None
    parse_mode = "json"
    parse_error_primary: Optional[Exception] = None
    parse_error_retry: Optional[Exception] = None
    retry_raw = ""

    try:
        parsed = extract_json_object(raw)
    except Exception as parse_err:
        parse_error_primary = parse_err

        latest_raw_for_repair = raw
        retry_max_tokens_floor = int(os.environ.get("ARL_PROPOSER_JSON_RETRY_MAX_TOKENS", "4500"))
        retry_max_tokens = max(max_tokens, retry_max_tokens_floor)

        if is_likely_truncated_json_output(raw) and retry_max_tokens > max_tokens:
            retry_system_prompt = (
                "You are a careful code-generation assistant for template snippet filling. "
                "Return exactly the requested JSON object and no additional text. "
                "Ensure the JSON object is complete and closed."
            )
            retry_user_prompt = user_prompt

            retry_prompt_path_out = save_llm_prompt(
                project_root=project_root,
                role="proposer",
                stage="snippet_generation_retry_truncated_json",
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
                    call_label=f"proposer_retry_{project_id}_{cycle_id}",
                )
                record_llm_call("retry_truncated_json", retry_call)
                retry_raw = str(retry_call.get("text") or "")
                retry_raw_response_path = shared_proposals / f"proposer_cycle{cycle_id}_raw_response_retry_truncated_json.txt"
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
            repair_system_prompt = "You are a strict JSON formatting assistant. Return only valid JSON."
            repair_user_prompt = build_repair_prompt(latest_raw_for_repair)

            repair_prompt_path_out = save_llm_prompt(
                project_root=project_root,
                role="proposer",
                stage="snippet_generation_repair_non_json",
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
                    call_label=f"proposer_repair_{project_id}_{cycle_id}",
                )
                record_llm_call("repair_non_json", repair_call)
                repair_raw = str(repair_call.get("text") or "")
                repair_raw_response_path = shared_proposals / f"proposer_cycle{cycle_id}_raw_response_repair.txt"
                write_text(repair_raw_response_path, (repair_raw or "").rstrip() + "\n")
                parsed = extract_json_object(repair_raw)
                parse_mode = "json_repair"
            except Exception as repair_err:
                parts = [
                    "Proposer output was not valid JSON after primary and repair attempts.",
                    f"Primary raw output: {raw_response_path}",
                ]
                if retry_prompt_path_out is not None:
                    parts.append(f"Retry prompt (higher max tokens): {retry_prompt_path_out}")
                if retry_raw_response_path is not None:
                    parts.append(f"Retry raw output: {retry_raw_response_path}")
                if repair_prompt_path_out is not None:
                    parts.append(f"Repair prompt: {repair_prompt_path_out}")
                if repair_raw_response_path is not None:
                    parts.append(f"Repair raw output: {repair_raw_response_path}")
                parts.append(f"Primary parse error: {parse_error_primary}")
                if parse_error_retry is not None:
                    parts.append(f"Retry parse error: {parse_error_retry}")
                parts.append(f"Repair error: {repair_err}")
                raise RuntimeError("\n".join(parts)) from repair_err

    snippets = parse_snippet_payload(parsed)

    model_code = model_template
    for marker in REQUIRED_MODEL_SNIPPETS:
        model_code = replace_snippet_block(model_code, marker, snippets["model"][marker])

    preprocessing_code = preprocessing_template
    for marker in REQUIRED_PREPROCESSING_SNIPPETS:
        preprocessing_code = replace_snippet_block(preprocessing_code, marker, snippets["preprocessing"][marker])

    if "PROPOSER_SNIPPET_START:" in model_code or "PROPOSER_SNIPPET_START:" in preprocessing_code:
        raise RuntimeError("Snippet markers remain in generated code; proposer output is incomplete.")
    if "NotImplementedError(\"Fill" in model_code or "NotImplementedError(\"Fill" in preprocessing_code:
        raise RuntimeError("Template placeholders were not fully replaced.")

    job = extract_primary_job(directive)
    candidate = job.get("candidate") if isinstance(job.get("candidate"), dict) else {}
    candidate_id = ensure_non_empty_str(candidate.get("candidate_id"), f"{cycle_id}_candidate", max_chars=120)

    snapshot_paths = build_candidate_snapshot_paths(project_root, cycle_id, candidate_id)
    snapshot_model_path = snapshot_paths["model_py_path"]
    snapshot_preprocessing_path = snapshot_paths["preprocessing_py_path"]
    snapshot_model_meta_path = snapshot_paths["model_meta_path"]
    snapshot_proposal_summary_path = snapshot_paths["proposal_summary_path"]

    snapshot_model_ref = to_project_ref(snapshot_model_path, project_root)
    snapshot_preprocessing_ref = to_project_ref(snapshot_preprocessing_path, project_root)
    snapshot_model_meta_ref = to_project_ref(snapshot_model_meta_path, project_root)
    snapshot_proposal_summary_ref = to_project_ref(snapshot_proposal_summary_path, project_root)


    write_text(model_py_path, model_code.rstrip() + "\n")
    write_text(preprocessing_py_path, preprocessing_code.rstrip() + "\n")
    write_text(snapshot_model_path, model_code.rstrip() + "\n")
    write_text(snapshot_preprocessing_path, preprocessing_code.rstrip() + "\n")

    model_meta = build_model_meta(
        directive=directive,
        project_id=project_id,
        candidate_id=candidate_id,
        model_code=model_code,
    )
    validate_json(model_meta, model_meta_schema_path)
    write_json(model_meta_path, model_meta)
    write_json(snapshot_model_meta_path, model_meta)

    jobs = directive.get("jobs") if isinstance(directive.get("jobs"), list) else []
    if jobs and isinstance(jobs[0], dict):
        candidate_payload = jobs[0].get("candidate")
        if not isinstance(candidate_payload, dict):
            candidate_payload = {}
            jobs[0]["candidate"] = candidate_payload
        candidate_payload["candidate_id"] = candidate_id
        candidate_payload["model_py_ref"] = snapshot_model_ref
        candidate_payload["model_meta_ref"] = snapshot_model_meta_ref
        candidate_payload["preprocessing_py_ref"] = snapshot_preprocessing_ref
        candidate_payload["origin"] = ensure_non_empty_str(candidate_payload.get("origin"), "proposer", max_chars=80)

        directive["jobs"] = jobs
        write_json(directive_path, directive)

    proposal_notes = ensure_non_empty_str(parsed.get("proposal_notes"), "", max_chars=600)
    proposal_summary = build_proposal_summary(
        directive=directive,
        project_id=project_id,
        cycle_id=cycle_id,
        candidate_id=candidate_id,
        model_py_ref=snapshot_model_ref,
        preprocessing_py_ref=snapshot_preprocessing_ref,
        model_meta_ref=snapshot_model_meta_ref,
        proposal_notes=proposal_notes,
    )
    write_json(proposal_summary_path, proposal_summary)
    write_json(snapshot_proposal_summary_path, proposal_summary)

    llm_telemetry_path = shared_proposals / f"proposer_cycle{cycle_id}_llm_requests.json"
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

    print(f"Saved prompt:            {prompt_path_out}")
    if retry_prompt_path_out is not None:
        print(f"Saved retry prompt:      {retry_prompt_path_out}")
    if retry_raw_response_path is not None:
        print(f"Saved retry output:      {retry_raw_response_path}")
    if repair_prompt_path_out is not None:
        print(f"Saved repair prompt:     {repair_prompt_path_out}")
    print(f"Saved raw response:      {raw_response_path}")
    if repair_raw_response_path is not None:
        print(f"Saved repair output:     {repair_raw_response_path}")
    print(f"Parse mode:              {parse_mode}")
    print(f"Wrote model code:        {model_py_path}")
    print(f"Wrote model snapshot:    {snapshot_model_path}")
    print(f"Wrote preprocessing:     {preprocessing_py_path}")
    print(f"Wrote preprocessing snap:{snapshot_preprocessing_path}")
    print(f"Wrote model metadata:    {model_meta_path}")
    print(f"Wrote model meta snap:   {snapshot_model_meta_path}")
    print(f"Wrote proposal summary:  {proposal_summary_path}")
    print(f"Wrote proposal snap:     {snapshot_proposal_summary_path}")
    print(f"Saved request log:       {llm_telemetry_path}")
    print(f"Updated directive refs:  {directive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())