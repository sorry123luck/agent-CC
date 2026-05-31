"""Audit local text evidence before sending ROI candidates to VLM.

The report answers which ROI candidates are already explained by UIA/OCR/Omni
text and which candidates are still pure visual/icon targets that need VLM help.
It does not mutate canvases.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def classify_candidate_evidence(
    element: dict[str, Any],
    *,
    ocr_blocks: list[dict[str, Any]],
    vision_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    bounds = _coerce_bounds(element.get("bounds"))
    candidate_kind = _candidate_kind(element)
    element_text = _first_text(
        element.get("text"),
        element.get("name"),
        element.get("value"),
    )
    attribute_ocr_text = (
        (element.get("attributes") or {}).get("ocr_text") if isinstance(element.get("attributes"), dict) else ""
    )
    local_text = element_text or ("" if candidate_kind in {"icon", "image"} else _first_text(attribute_ocr_text))
    text_sources: list[str] = []
    weak_ocr_text = _first_text(attribute_ocr_text) if candidate_kind in {"icon", "image"} else ""
    if element_text:
        text_sources.append("element")
    elif local_text:
        text_sources.append("ocr")

    ocr_text = ""
    if bounds is not None:
        ocr_matches = [
            block
            for block in ocr_blocks
            if _overlap_score(bounds, _coerce_bounds(block.get("bbox"))) >= 0.35
        ]
        ocr_matches.sort(key=lambda block: float(block.get("confidence") or 0.0), reverse=True)
        ocr_text = _first_text(*(block.get("text") for block in ocr_matches[:3]))
        if ocr_text and candidate_kind in {"icon", "image"}:
            weak_ocr_text = weak_ocr_text or ocr_text
            ocr_text = ""
        if ocr_text and "ocr" not in text_sources:
            text_sources.append("ocr")

    vision_text = ""
    candidate_id = str(element.get("element_id") or "")
    for candidate in vision_candidates:
        candidate_bounds = _coerce_bounds(candidate.get("bbox"))
        same_id = str(candidate.get("candidate_id") or "") == candidate_id
        overlaps = bounds is not None and _overlap_score(bounds, candidate_bounds) >= 0.45
        if same_id or overlaps:
            vision_text = _first_text(candidate.get("text"), candidate.get("content"), candidate.get("caption"))
            if vision_text:
                text_sources.append("vision")
                break

    merged_text = local_text or ocr_text or vision_text
    has_local_text = bool(merged_text)
    needs_vlm = not has_local_text and candidate_kind in {"icon", "button", "image", "unknown"}

    return {
        "candidate_id": candidate_id,
        "bounds": list(bounds) if bounds is not None else [],
        "control_type": str(element.get("control_type") or ""),
        "provider_sources": list(element.get("provider_sources") or []),
        "candidate_kind": candidate_kind,
        "has_local_text": has_local_text,
        "local_text": merged_text,
        "weak_ocr_text": weak_ocr_text,
        "text_sources": list(dict.fromkeys(text_sources)),
        "needs_vlm": needs_vlm,
    }


def analyze_canvas_detail(detail: dict[str, Any], *, source_file: str = "") -> dict[str, Any]:
    artifacts = detail.get("artifacts") or {}
    ocr_blocks = list(artifacts.get("ocr_blocks") or detail.get("ocr_blocks") or [])
    vision_candidates = list(artifacts.get("vision_candidates") or detail.get("vision_candidates") or [])
    elements_by_id = {
        str(element.get("element_id") or ""): element
        for element in list(detail.get("elements") or [])
        if element.get("element_id")
    }
    plan = detail.get("roi_selection_plan") or artifacts.get("roi_selection_plan") or {}
    rois = []
    totals = {
        "roi_count": 0,
        "candidate_count": 0,
        "local_text_count": 0,
        "needs_vlm_count": 0,
        "element_text_count": 0,
        "ocr_text_count": 0,
        "vision_text_count": 0,
    }

    for roi in list(plan.get("rois") or []):
        evidence_rows = [
            classify_candidate_evidence(elements_by_id[candidate_id], ocr_blocks=ocr_blocks, vision_candidates=vision_candidates)
            for candidate_id in [str(item) for item in list(roi.get("candidate_ids") or [])]
            if candidate_id in elements_by_id
        ]
        candidate_count = len(evidence_rows)
        local_text_count = sum(1 for row in evidence_rows if row["has_local_text"])
        needs_vlm_rows = [row for row in evidence_rows if row["needs_vlm"]]
        source_counts = {
            "element": sum(1 for row in evidence_rows if "element" in row["text_sources"]),
            "ocr": sum(1 for row in evidence_rows if "ocr" in row["text_sources"]),
            "vision": sum(1 for row in evidence_rows if "vision" in row["text_sources"]),
        }
        rois.append({
            "roi_id": str(roi.get("roi_id") or ""),
            "purpose": str(roi.get("purpose") or ""),
            "bounds": list(roi.get("bounds") or []),
            "candidate_count": candidate_count,
            "local_text_count": local_text_count,
            "needs_vlm_count": len(needs_vlm_rows),
            "local_text_ratio": round(local_text_count / max(1, candidate_count), 4),
            "needs_vlm_ratio": round(len(needs_vlm_rows) / max(1, candidate_count), 4),
            "source_counts": source_counts,
            "needs_vlm_ids": [row["candidate_id"] for row in needs_vlm_rows],
            "sample_text_candidates": [
                {"candidate_id": row["candidate_id"], "text": row["local_text"], "sources": row["text_sources"]}
                for row in evidence_rows
                if row["has_local_text"]
            ][:8],
        })
        totals["roi_count"] += 1
        totals["candidate_count"] += candidate_count
        totals["local_text_count"] += local_text_count
        totals["needs_vlm_count"] += len(needs_vlm_rows)
        totals["element_text_count"] += source_counts["element"]
        totals["ocr_text_count"] += source_counts["ocr"]
        totals["vision_text_count"] += source_counts["vision"]

    totals["local_text_ratio"] = round(totals["local_text_count"] / max(1, totals["candidate_count"]), 4)
    totals["needs_vlm_ratio"] = round(totals["needs_vlm_count"] / max(1, totals["candidate_count"]), 4)
    return {
        "source_file": source_file,
        "canvas_id": str(detail.get("canvas_id") or ""),
        "page_class": str(detail.get("page_class") or ""),
        "mode": str(plan.get("mode") or (detail.get("visual_pattern") or {}).get("mode") or ""),
        "totals": totals,
        "rois": rois,
    }


def audit_files(paths: list[Path]) -> dict[str, Any]:
    paths = select_detail_files(paths)
    samples = []
    totals = {
        "sample_count": 0,
        "roi_count": 0,
        "candidate_count": 0,
        "local_text_count": 0,
        "needs_vlm_count": 0,
        "element_text_count": 0,
        "ocr_text_count": 0,
        "vision_text_count": 0,
    }
    for path in paths:
        detail = json.loads(path.read_text(encoding="utf-8"))
        sample = analyze_canvas_detail(detail, source_file=str(path))
        samples.append(sample)
        totals["sample_count"] += 1
        for key in totals:
            if key == "sample_count":
                continue
            totals[key] += int(sample["totals"].get(key) or 0)
    totals["local_text_ratio"] = round(totals["local_text_count"] / max(1, totals["candidate_count"]), 4)
    totals["needs_vlm_ratio"] = round(totals["needs_vlm_count"] / max(1, totals["candidate_count"]), 4)
    return {"totals": totals, "samples": samples}


def select_detail_files(paths: list[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in sorted(paths):
        name = path.name
        key = name.replace(".post_vlm.detail.json", ".detail.json")
        existing = unique.get(key)
        if existing is None or ".post_vlm.detail.json" in existing.name:
            unique[key] = path
    return [unique[key] for key in sorted(unique)]


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "local_evidence_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "local_evidence_rois.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "canvas_id",
                "page_class",
                "mode",
                "roi_id",
                "purpose",
                "candidate_count",
                "local_text_count",
                "needs_vlm_count",
                "local_text_ratio",
                "needs_vlm_ratio",
                "element_text_count",
                "ocr_text_count",
                "vision_text_count",
                "source_file",
            ],
        )
        writer.writeheader()
        for sample in report["samples"]:
            for roi in sample["rois"]:
                source_counts = roi["source_counts"]
                writer.writerow({
                    "canvas_id": sample["canvas_id"],
                    "page_class": sample["page_class"],
                    "mode": sample["mode"],
                    "roi_id": roi["roi_id"],
                    "purpose": roi["purpose"],
                    "candidate_count": roi["candidate_count"],
                    "local_text_count": roi["local_text_count"],
                    "needs_vlm_count": roi["needs_vlm_count"],
                    "local_text_ratio": roi["local_text_ratio"],
                    "needs_vlm_ratio": roi["needs_vlm_ratio"],
                    "element_text_count": source_counts["element"],
                    "ocr_text_count": source_counts["ocr"],
                    "vision_text_count": source_counts["vision"],
                    "source_file": sample["source_file"],
                })
    _write_markdown(report, output_dir / "local_evidence_summary.md")


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    totals = report["totals"]
    lines = [
        "# Local Evidence Audit",
        "",
        f"- Samples: {totals['sample_count']}",
        f"- ROIs: {totals['roi_count']}",
        f"- Candidates: {totals['candidate_count']}",
        f"- Local text: {totals['local_text_count']} ({totals['local_text_ratio']})",
        f"- Needs VLM: {totals['needs_vlm_count']} ({totals['needs_vlm_ratio']})",
        f"- Element/UIA text hits: {totals['element_text_count']}",
        f"- OCR text hits: {totals['ocr_text_count']}",
        f"- Vision/Omni text hits: {totals['vision_text_count']}",
        "",
        "## ROI Summary",
        "",
        "| Canvas | Mode | ROI | Purpose | Candidates | Local Text | Needs VLM |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for sample in report["samples"]:
        for roi in sample["rois"]:
            lines.append(
                f"| `{sample['canvas_id']}` | `{sample['mode']}` | `{roi['roi_id']}` | "
                f"`{roi['purpose']}` | {roi['candidate_count']} | "
                f"{roi['local_text_count']} ({roi['local_text_ratio']}) | "
                f"{roi['needs_vlm_count']} ({roi['needs_vlm_ratio']}) |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _candidate_kind(element: dict[str, Any]) -> str:
    role = str(element.get("semantic_role") or "").strip().lower()
    control_type = str(element.get("control_type") or "").strip().lower()
    if "icon" in role or control_type == "icon":
        return "icon"
    if "button" in role or "button" in control_type:
        return "button"
    if "image" in role or "image" in control_type:
        return "image"
    if "edit" in control_type or "input" in role:
        return "input"
    if "text" in role or "text" in control_type:
        return "text"
    return "unknown"


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text[:80]
    return ""


def _coerce_bounds(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        left, top, right, bottom = [int(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _overlap_score(a: tuple[int, int, int, int], b: tuple[int, int, int, int] | None) -> float:
    if b is None:
        return 0.0
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / max(1, smaller)


def _expand_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.detail.json")))
        elif path.is_file():
            paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Detail JSON files or directories containing *.detail.json")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/local_evidence_audit"))
    args = parser.parse_args()

    paths = _expand_inputs(args.inputs)
    if not paths:
        raise SystemExit("no detail json files found")
    report = audit_files(paths)
    write_report(report, args.output_dir)
    print(
        f"samples={report['totals']['sample_count']} "
        f"rois={report['totals']['roi_count']} "
        f"candidates={report['totals']['candidate_count']} "
        f"needs_vlm={report['totals']['needs_vlm_count']} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
