"""Build a read-only search action plan from a live sample matrix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def build_search_plan(summary: dict[str, Any]) -> dict[str, Any]:
    """Convert sample matrix rows into controlled search probe targets."""
    rows = [_plan_row(row) for row in list(summary.get("rows") or [])]
    rows = [row for row in rows if row["process_name"]]
    counts = {"ready": 0, "blocked": 0}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    overall = "blocked" if not rows or counts["blocked"] else "ready"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_generated_at": summary.get("generated_at"),
        "source_schema": summary.get("sample_matrix_schema_version"),
        "overall_status": overall,
        "counts": counts,
        "rows": rows,
    }


def write_search_plan(plan: dict[str, Any], output_dir: Path) -> None:
    """Persist search action plan JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "search_action_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Search Action Plan",
        "",
        f"- Generated: {plan.get('generated_at', '')}",
        f"- Source schema: {plan.get('source_schema', '')}",
        f"- Overall: {plan.get('overall_status', '')}",
        f"- Counts: {_format_counts(plan.get('counts') or {})}",
        "",
        "| sample | hwnd | process | mode | status | search_click | source | reason |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(plan.get("rows") or []):
        lines.append(
            "| {sample} | {hwnd} | {process} | {mode} | {status} | {click} | {source} | {reason} |".format(
                sample=str(row.get("sample") or "").replace("|", "/"),
                hwnd=int(row.get("hwnd") or 0),
                process=str(row.get("process_name") or "").replace("|", "/"),
                mode=str(row.get("mode") or "").replace("|", "/"),
                status=str(row.get("status") or "").replace("|", "/"),
                click=str(row.get("search_click") or "").replace("|", "/"),
                source=str(row.get("source") or "").replace("|", "/"),
                reason=",".join(row.get("reasons") or []),
            )
        )
    (output_dir / "search_action_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_matrix_dir(matrix_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    summary = json.loads((matrix_dir / "sample_matrix_summary.json").read_text(encoding="utf-8"))
    plan = build_search_plan(summary)
    write_search_plan(plan, output_dir or matrix_dir)
    return plan


def _plan_row(row: dict[str, Any]) -> dict[str, Any]:
    search_click = _point(row.get("search_primary_click_point"))
    reasons: list[str] = []
    if not search_click:
        reasons.append("missing_search_click")
    if int(row.get("search_candidate_count") or 0) < 1:
        reasons.append("missing_search_candidate")
    return {
        "sample": str(row.get("sample") or ""),
        "hwnd": int(row.get("hwnd") or 0),
        "title": str(row.get("title") or ""),
        "process_name": str(row.get("process_name") or ""),
        "mode": str(row.get("mode") or ""),
        "status": "ready" if not reasons else "blocked",
        "requires_controlled_probe": True,
        "search_click": search_click,
        "search_bounds": _rect(row.get("search_bounds")),
        "source": str(row.get("search_source") or ""),
        "evidence": str(row.get("search_evidence") or ""),
        "reasons": reasons,
    }


def _point(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return []
    try:
        x, y = [int(item) for item in value]
    except (TypeError, ValueError):
        return []
    if x < 0 or y < 0:
        return []
    return [x, y]


def _rect(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return []
    try:
        left, top, right, bottom = [int(item) for item in value]
    except (TypeError, ValueError):
        return []
    if right <= left or bottom <= top:
        return []
    return [left, top, right, bottom]


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) if counts else "none"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    plan = analyze_matrix_dir(args.matrix_dir, output_dir=args.output_dir)
    print(
        "overall={overall} ready={ready} blocked={blocked}".format(
            overall=plan["overall_status"],
            ready=plan["counts"].get("ready", 0),
            blocked=plan["counts"].get("blocked", 0),
        )
    )
    return 0 if plan["overall_status"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
