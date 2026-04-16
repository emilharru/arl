#!/usr/bin/env python3
"""Export binary expert model-history plots to a single PDF figure.

This script mirrors the model-history logic used by the UI/API and generates
an MxN grid of subplots, where:
- M = number of modalities / dimensions / signals
- N = number of classes

Layout rules:
- rows correspond to modalities
- columns correspond to classes
- no overall title
- only the top row gets titles (class names)
- only the leftmost column gets y-axis labels (signal / modality names)
- no legend
- no gridlines
- no overall x-axis label
- x tick labels only on the bottom row
- y-limits extend slightly beyond [0, 1] so markers are fully visible
- x-limits are [-1, max_cycle + 1]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


UNKNOWN_LABEL_TOKENS = {
    "",
    "none",
    "null",
    "na",
    "n/a",
    "unknown",
    "unspecified",
}


def normalize_class_label(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in UNKNOWN_LABEL_TOKENS:
        return ""
    return text


def parse_cycle_number(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        return None

    m = re.match(r"^cycle_(\d+)$", text, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    if re.match(r"^\d+$", text):
        try:
            return int(text)
        except Exception:
            return None

    return None


def label_sort_key(value: Any):
    text = str(value or "").strip()
    if re.match(r"^-?\d+$", text):
        try:
            return (0, int(text))
        except Exception:
            pass
    return (1, text.lower())


def to_finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return float(number)


def read_json_file(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def extract_train_expert_f1(job_payload: Dict[str, Any]) -> Optional[float]:
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


def upsert_expert_history_point(
    expert_history_map: Dict[str, Dict[str, Any]],
    modality: str,
    class_label: str,
    cycle_value: int,
    f1_value: Optional[float],
    candidate_id: str,
) -> None:
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
    existing = cycle_history.get(cycle_value)
    existing_f1 = to_finite_float(existing.get("f1")) if isinstance(existing, dict) else None
    next_f1 = to_finite_float(f1_value)
    if next_f1 is None:
        return

    if existing_f1 is None or next_f1 > existing_f1:
        cycle_history[cycle_value] = {
            "f1": next_f1,
            "candidate_id": str(candidate_id or "").strip(),
        }


def load_class_description_map(project_root: Path) -> Dict[str, str]:
    context_path = project_root / "shared" / "context" / "data_context.md"
    if not context_path.exists():
        return {}

    try:
        content = context_path.read_text(encoding="utf-8")
    except Exception:
        return {}

    descriptions: Dict[str, str] = {}
    for line in content.splitlines():
        m = re.match(r"^- \*\*(?:Class\s+)?(.+?)\*\*:\s*(.*)$", line.strip())
        if not m:
            continue
        label_raw, desc_raw = m.groups()
        label = normalize_class_label(label_raw)
        desc = str(desc_raw).strip()
        if label:
            descriptions[label] = desc
    return descriptions


def clean_class_title(value: str) -> str:
    text = str(value or "").strip()
    m = re.match(r"^\s*[^()]+\((.+)\)\s*$", text)
    if m:
        return m.group(1).strip()
    return text


def format_modality_label(value: str) -> str:
    return str(value or "").strip().upper()


@dataclass
class ModelHistoryPayload:
    project: str
    cycles: List[int]
    experts: List[Dict[str, Any]]


def build_model_history_payload(project_root: Path) -> ModelHistoryPayload:
    project_name = project_root.name
    cycle_history_root = project_root / "artifacts" / "cycle_history"

    class_descriptions = load_class_description_map(project_root)

    if not cycle_history_root.exists() or not cycle_history_root.is_dir():
        return ModelHistoryPayload(project=project_name, cycles=[], experts=[])

    cycle_dirs: List[Tuple[int, Path]] = []
    for entry in cycle_history_root.iterdir():
        if not entry.is_dir():
            continue
        cycle_num = parse_cycle_number(entry.name)
        if cycle_num is None:
            continue
        cycle_dirs.append((cycle_num, entry))
    cycle_dirs.sort(key=lambda pair: pair[0])

    cycles_seen = set()
    expert_history_map: Dict[str, Dict[str, Any]] = {}

    for cycle_num, cycle_dir in cycle_dirs:
        if cycle_num == 0:
            cycles_seen.add(0)
            cycle0_matrix_path = cycle_dir / "expert_matrix.json"
            cycle0_matrix_payload = read_json_file(cycle0_matrix_path, {})
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
                        f1_value = to_finite_float(entry.get("f1"))
                        candidate_id = str(entry.get("candidate_id") or "").strip()
                        upsert_expert_history_point(
                            expert_history_map=expert_history_map,
                            modality=modality,
                            class_label=class_label,
                            cycle_value=0,
                            f1_value=f1_value,
                            candidate_id=candidate_id,
                        )

        results_path = cycle_dir / "results.json"
        if not results_path.exists():
            continue

        payload = read_json_file(results_path, {})
        if not isinstance(payload, dict):
            continue

        cycle_value = parse_cycle_number(payload.get("cycle_id"))
        if cycle_value is None:
            cycle_value = cycle_num
        cycles_seen.add(cycle_value)

        jobs = payload.get("jobs")
        if not isinstance(jobs, list):
            continue

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

    cycles_sorted = sorted(cycles_seen)

    matrix_path = project_root / "artifacts" / "expert_matrix.json"
    matrix_payload = read_json_file(matrix_path, {})
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

    experts_payload: List[Dict[str, Any]] = []
    for expert_id, expert_data in sorted(
        expert_history_map.items(),
        key=lambda pair: (
            str(pair[1].get("modality", "")).lower(),
            label_sort_key(pair[1].get("class_label", "")),
        ),
    ):
        class_label = str(expert_data.get("class_label") or "")
        class_desc = str(class_descriptions.get(class_label) or "").strip()
        class_name = f"{class_label} ({class_desc})" if class_desc else class_label

        history_by_cycle = expert_data.get("history_by_cycle")
        if not isinstance(history_by_cycle, dict):
            history_by_cycle = {}

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

    return ModelHistoryPayload(
        project=project_name,
        cycles=cycles_sorted,
        experts=experts_payload,
    )


def plot_expert_history(
    expert_payload: Optional[Dict[str, Any]],
    ax,
    show_trained_markers: bool,
    xlim: Optional[Tuple[float, float]],
) -> None:
    if not expert_payload:
        ax.set_ylim(-0.03, 1.03)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.grid(False)
        ax.tick_params(axis="both", width=0.6, length=2.5)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.7)
        return

    history = expert_payload.get("history") if isinstance(expert_payload.get("history"), list) else []

    cycles = []
    best_f1 = []
    trained_cycles = []
    trained_f1 = []
    changed_cycles = []
    changed_f1 = []

    for row in history:
        if not isinstance(row, dict):
            continue
        cycle = row.get("cycle")
        f1_value = to_finite_float(row.get("f1"))
        if cycle is None or f1_value is None:
            continue

        cycle_int = int(cycle)
        cycles.append(cycle_int)
        best_f1.append(f1_value)

        tr = to_finite_float(row.get("trained_f1"))
        if show_trained_markers and tr is not None:
            trained_cycles.append(cycle_int)
            trained_f1.append(tr)

        if bool(row.get("model_changed", False)):
            changed_cycles.append(cycle_int)
            changed_f1.append(f1_value)

    ax.plot(
        cycles,
        best_f1,
        color="black",
        linewidth=1.2,
        linestyle=":",
        zorder=2,
    )

    if show_trained_markers and trained_cycles:
        ax.scatter(
            trained_cycles,
            trained_f1,
            facecolors="white",
            edgecolors="black",
            linewidths=0.8,
            s=8,
            alpha=1.0,
            zorder=3,
        )


    ax.set_ylim(-0.03, 1.03)

    if xlim is not None:
        ax.set_xlim(*xlim)
    elif cycles:
        ax.set_xlim(-1, max(cycles) + 1)
    else:
        ax.set_xlim(-1, 1)

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.7)

    ax.tick_params(axis="both", colors="black", width=0.6, length=2.5)


def export_pdf(
    payload: ModelHistoryPayload,
    output_path: Path,
    fig_width: float,
    fig_height: float,
    dpi: int,
    show_trained_markers: bool,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rc_overrides = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }

    with plt.rc_context(rc_overrides):
        if not payload.experts:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"No binary expert history found for project '{payload.project}'.",
                ha="center",
                va="center",
                fontsize=12,
            )
            fig.savefig(output_path, dpi=max(150, int(dpi)), bbox_inches="tight", pad_inches=0.04)
            plt.close(fig)
            return {
                "experts_plotted": 0,
                "images_written": 1,
            }

        modalities = sorted(
            {
                str(expert.get("modality") or "").strip()
                for expert in payload.experts
                if str(expert.get("modality") or "").strip()
            },
            key=lambda x: x.lower(),
        )

        class_order = sorted(
            {
                str(expert.get("class_label") or "").strip()
                for expert in payload.experts
                if str(expert.get("class_label") or "").strip()
            },
            key=label_sort_key,
        )

        class_name_map: Dict[str, str] = {}
        expert_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for expert in payload.experts:
            modality = str(expert.get("modality") or "").strip()
            class_label = str(expert.get("class_label") or "").strip()
            if not modality or not class_label:
                continue
            expert_lookup[(modality, class_label)] = expert
            class_name_map[class_label] = clean_class_title(str(expert.get("class_name") or class_label))

        rows = max(1, len(modalities))
        cols = max(1, len(class_order))

        cycles_sorted = sorted(set(int(c) for c in payload.cycles))
        if cycles_sorted:
            xlim: Optional[Tuple[float, float]] = (-1, max(cycles_sorted) + 1)
        else:
            xlim = (-1, 1)

        panel_width = float(fig_width) / max(1, cols)
        panel_height = float(fig_height) / max(1, rows)
        actual_width = panel_width * cols
        actual_height = panel_height * rows

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(actual_width, actual_height),
            squeeze=False,
            sharex=True,
            sharey=True,
        )

        experts_plotted = 0

        for r, modality in enumerate(modalities):
            for c, class_label in enumerate(class_order):
                ax = axes[r][c]
                expert = expert_lookup.get((modality, class_label))

                plot_expert_history(
                    expert_payload=expert,
                    ax=ax,
                    show_trained_markers=show_trained_markers,
                    xlim=xlim,
                )

                if expert is not None:
                    experts_plotted += 1

                if r == 0:
                    ax.set_title(class_name_map.get(class_label, clean_class_title(class_label)), fontsize=8, pad=4)

                if c == 0:
                    ax.set_ylabel(format_modality_label(modality), fontsize=8)
                else:
                    ax.set_ylabel("")

                ax.set_xlabel("")


                ax.tick_params(axis="x", labelbottom=(r == rows - 1))

        fig.tight_layout(pad=0.5, w_pad=0.6, h_pad=0.8)
        fig.savefig(output_path, dpi=max(150, int(dpi)), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    return {
        "experts_plotted": experts_plotted,
        "images_written": 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export binary expert history plots as an MxN subplot grid PDF for a project, "
            "where rows are modalities and columns are classes."
        )
    )
    parser.add_argument("--project", required=True, help="Project name under projects/")
    parser.add_argument(
        "--projects-root",
        default="projects",
        help="Root directory containing project folders (default: projects)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PDF path (default: projects/<project>/artifacts/model_history_binary_experts.pdf)"
        ),
    )
    parser.add_argument("--fig-width", type=float, default=7.0, help="Figure width in inches")
    parser.add_argument("--fig-height", type=float, default=4.5, help="Figure height in inches")
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Output PDF DPI metadata (default: 400)",
    )
    parser.add_argument(
        "--no-trained-markers",
        action="store_true",
        help="Hide trained-in-cycle F1 markers",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    projects_root = Path(args.projects_root).resolve()
    project_root = (projects_root / args.project).resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"Project not found: {project_root}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else (project_root / "artifacts" / "model_history_binary_experts.pdf").resolve()
    )

    payload = build_model_history_payload(project_root)

    stats = export_pdf(
        payload=payload,
        output_path=output_path,
        fig_width=float(args.fig_width),
        fig_height=float(args.fig_height),
        dpi=int(args.dpi),
        show_trained_markers=not bool(args.no_trained_markers),
    )

    print(f"Project: {payload.project}")
    print(f"Output PDF: {output_path}")
    print(f"Experts plotted: {stats['experts_plotted']}")
    print(f"Images written: {stats['images_written']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())