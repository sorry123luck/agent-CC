"""Analyze whether ROI VLM evidence is ready for semantic supplementation.

The report deliberately keeps VLM as a supplement layer. It may explain unknown
regions or icon semantics, but it must not override coordinates, upgrade
safe_to_type, or directly project executable actions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


MIN_READY_SCORE = 80
BLOCKED_DEFAULTS = ["coordinate_override", "safe_to_type_upgrade", "direct_action_projection"]


def build_vlm_supplement_readiness_report(*, quality_summary: dict[str, Any]) -> dict[str, Any]:
    """Build a semantic-supplement readiness report from ROI VLM quality output."""
    if not quality_summary or int(quality_summary.get("sample_count") or 0) <= 0:
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "report_type": "vlm_supplement_readiness",
            "overall_status": "missing_evidence",
            "policy": _policy(),
            "global_blockers": ["missing_quality_summary"],
            "rows": [],
        }
    rows = [_row(item) for item in list(quality_summary.get("rows") or []) if isinstance(item, dict)]
    global_blockers = _global_blockers(quality_summary)
    row_blocked = any(row.get("status") != "ready" for row in rows)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_type": "vlm_supplement_readiness",
        "overall_status": "needs_review" if global_blockers or row_blocked else "semantic_supplement_ready",
        "policy": _policy(),
        "quality_summary": {
            "source": str(quality_summary.get("source") or ""),
            "sample_count": int(quality_summary.get("sample_count") or 0),
            "average_score": float(quality_summary.get("average_score") or 0.0),
            "low_score_count": int(quality_summary.get("low_score_count") or 0),
            "late_failure_total": int(quality_summary.get("late_failure_total") or 0),
            "invalid_region_total": int(quality_summary.get("invalid_region_total") or 0),
            "invalid_annotation_role_total": int(quality_summary.get("invalid_annotation_role_total") or 0),
        },
        "global_blockers": global_blockers,
        "rows": rows,
    }


def analyze_vlm_supplement_readiness_dirs(*, quality_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Load `quality_summary.json` and write VLM supplement readiness outputs."""
    report = build_vlm_supplement_readiness_report(
        quality_summary=_load_json(quality_dir / "quality_summary.json")
    )
    write_vlm_supplement_readiness_report(report, output_dir)
    return report


def write_vlm_supplement_readiness_report(report: dict[str, Any], output_dir: Path) -> None:
    """Persist JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vlm_supplement_readiness_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    policy = report.get("policy") if isinstance(report.get("policy"), dict) else {}
    lines = [
        "# VLM Supplement Readiness Report",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Allowed default: {policy.get('allowed_default', '')}",
        f"- Blocked default: {','.join(policy.get('blocked_default') or [])}",
        f"- Global blockers: {','.join(report.get('global_blockers') or []) or 'none'}",
        "",
        "| sample | status | score | projection_allowed | blockers | issues |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in list(report.get("rows") or []):
        lines.append(
            "| {sample} | {status} | {score} | {projection} | {blockers} | {issues} |".format(
                sample=_md(row.get("sample")),
                status=_md(row.get("status")),
                score=row.get("score", 0),
                projection="yes" if row.get("projection_allowed") else "no",
                blockers=",".join(row.get("blockers") or []).replace("|", "/"),
                issues=",".join(row.get("issues") or []).replace("|", "/"),
            )
        )
    (output_dir / "vlm_supplement_readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(item: dict[str, Any]) -> dict[str, Any]:
    score = int(item.get("score") or 0)
    issues = [str(issue) for issue in list(item.get("issues") or []) if str(issue)]
    blockers: list[str] = []
    if score < MIN_READY_SCORE:
        blockers.append("low_score")
    if any(issue.startswith("invalid_region_role:") or issue.startswith("invalid_annotation_role:") for issue in issues):
        blockers.append("invalid_region_or_role")
    if int(item.get("late_failure_count") or 0) > 0:
        blockers.append("late_failure")
    return {
        "sample": str(item.get("sample") or ""),
        "canvas_id": str(item.get("canvas_id") or ""),
        "mode": str(item.get("mode") or ""),
        "status": "ready" if not blockers else "needs_review",
        "score": score,
        "supplement_count": int(item.get("supplement_count") or 0),
        "annotation_count": int(item.get("annotation_count") or 0),
        "projection_allowed": False,
        "allowed_use": "semantic_supplement_only",
        "blockers": blockers,
        "issues": issues,
    }


def _global_blockers(summary: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if int(summary.get("low_score_count") or 0) > 0:
        blockers.append("low_score_samples")
    if int(summary.get("invalid_region_total") or 0) > 0 or int(summary.get("invalid_annotation_role_total") or 0) > 0:
        blockers.append("invalid_region_or_role")
    if int(summary.get("late_failure_total") or 0) > 0:
        blockers.append("late_failure")
    return blockers


def _policy() -> dict[str, Any]:
    return {
        "allowed_default": "semantic_supplement_only",
        "blocked_default": BLOCKED_DEFAULTS,
        "notes": (
            "VLM output can add region semantics, candidate annotations, and review-only hints. "
            "It cannot replace local bounds, enable typing, or create executable actions by itself."
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _md(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_vlm_supplement_readiness_dirs(quality_dir=args.quality_dir, output_dir=args.output_dir)
    print(f"overall={report.get('overall_status')} rows={len(report.get('rows') or [])}")
    return 0 if report.get("overall_status") == "semantic_supplement_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
