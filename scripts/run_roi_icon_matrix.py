"""Run ROI icon VLM matrix experiments for small toolbar candidates.

The script answers a narrow question: when does a VLM start confusing small
icons, and is the failure caused by too many candidates, crop clarity, or marker
overlay? It does not write back to the main perception model.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vlm.roi_provider_worker import run_roi_vlm_provider_job


@dataclass(frozen=True)
class IconMatrixCase:
    case_id: str
    variant: str
    candidate_ids: list[str]
    bounds: list[int]
    draw_markers: bool


def parse_expected_labels(values: list[str]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected label must be candidate_id=label, got {value!r}")
        candidate_id, label = value.split("=", 1)
        candidate_id = candidate_id.strip()
        label = label.strip()
        if not candidate_id or not label:
            raise ValueError(f"Expected label must be candidate_id=label, got {value!r}")
        expected[candidate_id] = label
    return expected


def build_icon_matrix_cases(
    *,
    roi_bounds: list[int],
    candidate_ids: list[str],
    elements_by_id: dict[str, dict[str, Any]],
    group_sizes: list[int],
    variants: list[str],
    padding: int = 16,
) -> list[IconMatrixCase]:
    cases: list[IconMatrixCase] = []
    unique_ids = [candidate_id for candidate_id in dict.fromkeys(candidate_ids) if candidate_id in elements_by_id]
    for group_size in group_sizes:
        selected = unique_ids[: max(1, min(group_size, len(unique_ids)))]
        if not selected:
            continue
        for variant in variants:
            if variant == "roi_marked":
                bounds = list(roi_bounds)
                draw_markers = True
            elif variant == "roi_unmarked":
                bounds = list(roi_bounds)
                draw_markers = False
            elif variant == "tight_marked":
                bounds = union_bounds([elements_by_id[candidate_id]["bounds"] for candidate_id in selected], padding=padding)
                draw_markers = True
            elif variant == "tight_unmarked":
                bounds = union_bounds([elements_by_id[candidate_id]["bounds"] for candidate_id in selected], padding=padding)
                draw_markers = False
            else:
                raise ValueError(f"Unknown variant: {variant}")
            cases.append(
                IconMatrixCase(
                    case_id=f"{variant}_n{len(selected)}",
                    variant=variant,
                    candidate_ids=selected,
                    bounds=bounds,
                    draw_markers=draw_markers,
                )
            )
    return cases


def union_bounds(bounds_list: list[list[int]], *, padding: int) -> list[int]:
    if not bounds_list:
        return [0, 0, 1, 1]
    left = min(int(bounds[0]) for bounds in bounds_list) - padding
    top = min(int(bounds[1]) for bounds in bounds_list) - padding
    right = max(int(bounds[2]) for bounds in bounds_list) + padding
    bottom = max(int(bounds[3]) for bounds in bounds_list) + padding
    return [max(0, left), max(0, top), max(1, right), max(1, bottom)]


def score_annotations(
    annotations: list[dict[str, Any]],
    expected_labels: dict[str, str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    correct = 0
    wrong = 0
    missing = 0
    annotation_by_id = {
        str(item.get("candidate_id") or ""): item
        for item in annotations
        if item.get("candidate_id")
    }
    for candidate_id, expected in expected_labels.items():
        annotation = annotation_by_id.get(candidate_id)
        if annotation is None:
            missing += 1
            rows.append({"candidate_id": candidate_id, "expected": expected, "actual": "", "status": "missing"})
            continue
        actual_label = str(annotation.get("label") or "").strip()
        actual_role = str(annotation.get("role") or "").strip()
        actual = actual_label or actual_role
        ok = labels_match(actual_label, expected) or labels_match(actual_role, expected)
        if ok:
            correct += 1
        else:
            wrong += 1
        rows.append({
            "candidate_id": candidate_id,
            "expected": expected,
            "actual": actual,
            "status": "correct" if ok else "wrong",
        })
    total = max(1, len(expected_labels))
    return {
        "correct": correct,
        "wrong": wrong,
        "missing": missing,
        "accuracy": round(correct / total, 4),
        "rows": rows,
    }


def labels_match(actual: str, expected: str) -> bool:
    actual_norm = normalize_label(actual)
    expected_norm = normalize_label(expected)
    if not actual_norm or not expected_norm:
        return False
    if actual_norm == expected_norm:
        return True
    aliases = {
        "microphone": {"microphone", "mic", "voice"},
        "voice": {"microphone", "mic", "voice"},
        "emoji": {
            "emoji",
            "emojipicker",
            "smile",
            "smiley",
            "smileemoji",
            "smileyemoji",
            "smileyfaceemoji",
            "emoticon",
        },
        "attachment": {"attachment", "attachfile", "file", "folder"},
        "folder": {"attachment", "attachfile", "file", "folder"},
        "send": {"send", "sendbutton", "submit"},
        "screenshot": {"screenshot", "scissors", "cut"},
        "cut": {"screenshot", "scissors", "cut"},
    }
    expected_aliases = aliases.get(expected_norm, {expected_norm})
    return actual_norm in expected_aliases


def normalize_label(value: str) -> str:
    return "".join(ch for ch in value.strip().lower() if ch.isalnum())


def fetch_canvas(base_url: str, canvas_id: str) -> tuple[dict[str, Any], Image.Image]:
    detail_resp = requests.get(f"{base_url}/api/v1/canvases/{canvas_id}", timeout=20)
    detail_resp.raise_for_status()
    image_resp = requests.get(f"{base_url}/api/v1/canvases/{canvas_id}/screenshot", timeout=20)
    image_resp.raise_for_status()
    return detail_resp.json(), Image.open(io.BytesIO(image_resp.content)).convert("RGB")


def load_detail_and_screenshot(
    *,
    base_url: str,
    canvas_id: str,
    detail_json: Path | None = None,
    screenshot_file: Path | None = None,
) -> tuple[dict[str, Any], Image.Image]:
    if detail_json is not None or screenshot_file is not None:
        if detail_json is None or screenshot_file is None:
            raise ValueError("detail_json and screenshot_file must be provided together")
        detail = json.loads(detail_json.read_text(encoding="utf-8"))
        screenshot = Image.open(screenshot_file).convert("RGB")
        return detail, screenshot
    return fetch_canvas(base_url, canvas_id)


def run_matrix(
    *,
    base_url: str,
    canvas_id: str,
    roi_id: str,
    candidate_ids: list[str],
    expected_labels: dict[str, str],
    group_sizes: list[int],
    variants: list[str],
    output_dir: Path,
    timeout_seconds: float,
    image_max_edge: int,
    max_tokens: int,
    max_cases: int,
    detail_json: Path | None = None,
    screenshot_file: Path | None = None,
) -> dict[str, Any]:
    detail, screenshot = load_detail_and_screenshot(
        base_url=base_url,
        canvas_id=canvas_id,
        detail_json=detail_json,
        screenshot_file=screenshot_file,
    )
    canvas_id = str(detail.get("canvas_id") or canvas_id)
    roi = next((item for item in detail.get("roi_selection_plan", {}).get("rois", []) if item.get("roi_id") == roi_id), None)
    if roi is None:
        raise RuntimeError(f"ROI not found: {roi_id}")
    elements_by_id = {str(item.get("element_id")): item for item in detail.get("elements", []) if item.get("element_id")}
    selected_ids = candidate_ids or list(roi.get("candidate_ids") or [])
    cases = build_icon_matrix_cases(
        roi_bounds=list(roi.get("bounds") or []),
        candidate_ids=selected_ids,
        elements_by_id=elements_by_id,
        group_sizes=group_sizes,
        variants=variants,
    )
    if max_cases > 0:
        cases = cases[:max_cases]

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in cases:
        expected_subset = {
            candidate_id: expected_labels[candidate_id]
            for candidate_id in case.candidate_ids
            if candidate_id in expected_labels
        }
        job = {
            "canvas_id": canvas_id,
            "roi_id": f"{roi_id}_{case.case_id}",
            "bounds": case.bounds,
            "mode": str(roi.get("mode") or detail.get("visual_pattern", {}).get("mode") or ""),
            "purpose": str(roi.get("purpose") or ""),
            "experiment": "roi_icon_matrix",
            "candidate_ids": case.candidate_ids,
            "candidate_refs": [
                {
                    "marker": f"C{idx + 1}",
                    "candidate_id": candidate_id,
                    "bounds": list(elements_by_id[candidate_id].get("bounds") or []),
                    "control_type": str(elements_by_id[candidate_id].get("control_type") or ""),
                }
                for idx, candidate_id in enumerate(case.candidate_ids)
            ],
            "draw_markers": case.draw_markers,
        }
        started = time.perf_counter()
        error = ""
        result: dict[str, Any] = {}
        try:
            result = run_roi_vlm_provider_job(
                job,
                screenshot=screenshot,
                profile="fast",
                max_candidate_ids=len(case.candidate_ids),
                image_max_edge=image_max_edge,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - matrix rows preserve failures.
            error = str(exc)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        annotations = list(result.get("candidate_annotations") or [])
        score = score_annotations(annotations, expected_subset) if expected_subset else {
            "correct": 0,
            "wrong": 0,
            "missing": 0,
            "accuracy": 0,
            "rows": [],
        }
        rows.append({
            "case_id": case.case_id,
            "variant": case.variant,
            "candidate_count": len(case.candidate_ids),
            "draw_markers": case.draw_markers,
            "bounds": case.bounds,
            "candidate_ids": case.candidate_ids,
            "elapsed_ms": elapsed_ms,
            "provider_latency_ms": (result.get("_provider") or {}).get("latency_ms", 0),
            "debug_dir": (result.get("_debug_artifacts") or {}).get("dir", ""),
            "error": error,
            "annotations": annotations,
            "score": score,
        })

    summary = {
        "canvas_id": canvas_id,
        "roi_id": roi_id,
        "candidate_ids": selected_ids,
        "expected_labels": expected_labels,
        "case_count": len(rows),
        "rows": rows,
        "aggregate": aggregate_rows(rows),
    }
    write_outputs(summary, output_dir)
    return summary


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("variant") or "")
        item = by_variant.setdefault(key, {"case_count": 0, "correct": 0, "wrong": 0, "missing": 0, "elapsed_ms_total": 0})
        score = row.get("score") or {}
        item["case_count"] += 1
        item["correct"] += int(score.get("correct") or 0)
        item["wrong"] += int(score.get("wrong") or 0)
        item["missing"] += int(score.get("missing") or 0)
        item["elapsed_ms_total"] += int(row.get("elapsed_ms") or 0)
    for item in by_variant.values():
        total = max(1, item["correct"] + item["wrong"] + item["missing"])
        item["accuracy"] = round(item["correct"] / total, 4)
        item["avg_elapsed_ms"] = int(item["elapsed_ms_total"] / max(1, item["case_count"]))
    return {"by_variant": by_variant}


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "matrix_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "matrix_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "case_id",
                "variant",
                "candidate_count",
                "draw_markers",
                "elapsed_ms",
                "provider_latency_ms",
                "accuracy",
                "correct",
                "wrong",
                "missing",
                "error",
                "debug_dir",
                "candidate_ids",
            ],
        )
        writer.writeheader()
        for row in summary["rows"]:
            score = row.get("score") or {}
            writer.writerow({
                "case_id": row["case_id"],
                "variant": row["variant"],
                "candidate_count": row["candidate_count"],
                "draw_markers": row["draw_markers"],
                "elapsed_ms": row["elapsed_ms"],
                "provider_latency_ms": row["provider_latency_ms"],
                "accuracy": score.get("accuracy", 0),
                "correct": score.get("correct", 0),
                "wrong": score.get("wrong", 0),
                "missing": score.get("missing", 0),
                "error": row.get("error", ""),
                "debug_dir": row.get("debug_dir", ""),
                "candidate_ids": " ".join(row.get("candidate_ids") or []),
            })
    lines = [
        "# ROI Icon Matrix",
        "",
        f"- Canvas: `{summary['canvas_id']}`",
        f"- ROI: `{summary['roi_id']}`",
        f"- Cases: {summary['case_count']}",
        "",
        "## By Variant",
        "",
        "| Variant | Cases | Accuracy | Correct | Wrong | Missing | Avg ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, item in sorted(summary["aggregate"]["by_variant"].items()):
        lines.append(
            f"| {variant} | {item['case_count']} | {item['accuracy']} | {item['correct']} | "
            f"{item['wrong']} | {item['missing']} | {item['avg_elapsed_ms']} |"
        )
    lines.extend(["", "## Cases", "", "| Case | N | Accuracy | Error | Debug |", "|---|---:|---:|---|---|"])
    for row in summary["rows"]:
        score = row.get("score") or {}
        lines.append(
            f"| {row['case_id']} | {row['candidate_count']} | {score.get('accuracy', 0)} | "
            f"{row.get('error', '') or '-'} | `{row.get('debug_dir', '')}` |"
        )
    (output_dir / "matrix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--canvas-id", required=True)
    parser.add_argument("--roi-id", required=True)
    parser.add_argument("--candidate-ids", default="")
    parser.add_argument("--expected", action="append", default=[])
    parser.add_argument("--group-sizes", default="1,2,4,6")
    parser.add_argument("--variants", default="tight_marked,roi_marked,tight_unmarked,roi_unmarked")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/roi_icon_matrix"))
    parser.add_argument("--detail-json", type=Path)
    parser.add_argument("--screenshot-file", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    parser.add_argument("--image-max-edge", type=int, default=220)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    summary = run_matrix(
        base_url=args.base_url,
        canvas_id=args.canvas_id,
        roi_id=args.roi_id,
        candidate_ids=parse_csv_strings(args.candidate_ids),
        expected_labels=parse_expected_labels(args.expected),
        group_sizes=parse_csv_ints(args.group_sizes),
        variants=parse_csv_strings(args.variants),
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
        image_max_edge=args.image_max_edge,
        max_tokens=args.max_tokens,
        max_cases=args.max_cases,
        detail_json=args.detail_json,
        screenshot_file=args.screenshot_file,
    )
    print(
        f"cases={summary['case_count']} "
        f"variants={','.join(sorted(summary['aggregate']['by_variant'].keys()))} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
