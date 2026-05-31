"""Build a read-only chat input/send action plan from a live sample matrix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


CHAT_MODES = {"chat_workspace", "chat_document", "collaboration_inbox"}


def build_action_plan(summary: dict[str, Any]) -> dict[str, Any]:
    """Convert sample matrix rows into controlled-probe action targets."""
    rows = [_plan_row(row) for row in list(summary.get("rows") or []) if _is_chat_row(row)]
    counts = {"ready": 0, "warn": 0, "blocked": 0}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    overall = "blocked" if counts["blocked"] else "warn" if counts["warn"] else "ready"
    if not rows:
        overall = "blocked"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_generated_at": summary.get("generated_at"),
        "source_schema": summary.get("sample_matrix_schema_version"),
        "overall_status": overall,
        "counts": counts,
        "rows": rows,
    }


def write_action_plan(plan: dict[str, Any], output_dir: Path) -> None:
    """Persist chat action plan JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chat_action_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Chat Action Plan",
        "",
        f"- Generated: {plan.get('generated_at', '')}",
        f"- Source schema: {plan.get('source_schema', '')}",
        f"- Overall: {plan.get('overall_status', '')}",
        f"- Counts: {_format_counts(plan.get('counts') or {})}",
        "",
        "| sample | process | mode | status | input_click | send_click | safe_to_type | reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(plan.get("rows") or []):
        lines.append(
            "| {sample} | {process} | {mode} | {status} | {input_click} | {send_click} | {safe} | {reason} |".format(
                sample=str(row.get("sample") or "").replace("|", "/"),
                process=str(row.get("process_name") or "").replace("|", "/"),
                mode=str(row.get("mode") or "").replace("|", "/"),
                status=str(row.get("status") or "").replace("|", "/"),
                input_click=str(row.get("input_click") or "").replace("|", "/"),
                send_click=str(row.get("send_click") or "").replace("|", "/"),
                safe="true" if row.get("safe_to_type") is True else "false",
                reason=",".join(row.get("reasons") or []),
            )
        )
    (output_dir / "chat_action_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_matrix_dir(matrix_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    summary = json.loads((matrix_dir / "sample_matrix_summary.json").read_text(encoding="utf-8"))
    plan = build_action_plan(summary)
    write_action_plan(plan, output_dir or matrix_dir)
    return plan


def _plan_row(row: dict[str, Any]) -> dict[str, Any]:
    input_click = _point(row.get("composer_input_primary_click_point"))
    send_bounds = _rect(row.get("composer_send_target_bounds"))
    send_click = _center(send_bounds)
    safe_count = int(row.get("composer_input_safe_count") or 0)
    review_count = int(row.get("composer_input_review_count") or 0)
    reasons: list[str] = []
    if safe_count > 0:
        reasons.append("input_marked_safe_unexpected")
    if review_count < 1:
        reasons.append("missing_review_input")
    if not input_click:
        reasons.append("missing_input_click")
    if not send_click:
        reasons.append("missing_send_click")
    if send_bounds and (send_bounds[2] - send_bounds[0]) < 32:
        reasons.append("send_target_too_narrow")
    if send_bounds and (send_bounds[3] - send_bounds[1]) < 20:
        reasons.append("send_target_too_short")
    status = "ready" if not reasons else "warn"
    if "missing_review_input" in reasons or "missing_input_click" in reasons or "missing_send_click" in reasons:
        status = "blocked"
    return {
        "sample": str(row.get("sample") or ""),
        "process_name": str(row.get("process_name") or ""),
        "mode": str(row.get("mode") or ""),
        "chat_variant": str(row.get("chat_variant") or ""),
        "status": status,
        "requires_controlled_probe": True,
        "safe_to_type": False,
        "input_click": input_click,
        "input_bounds": _rects(row.get("composer_input_review_bounds")),
        "send_click": send_click,
        "send_bounds": send_bounds,
        "reasons": reasons,
    }


def _is_chat_row(row: dict[str, Any]) -> bool:
    return str(row.get("mode") or "") in CHAT_MODES


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


def _rects(value: Any) -> list[list[int]]:
    if not isinstance(value, list):
        return []
    return [rect for rect in (_rect(item) for item in value) if rect]


def _center(bounds: list[int]) -> list[int]:
    if not bounds:
        return []
    return [int((bounds[0] + bounds[2]) / 2), int((bounds[1] + bounds[3]) / 2)]


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) if counts else "none"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    plan = analyze_matrix_dir(args.matrix_dir, output_dir=args.output_dir)
    print(
        "overall={overall} ready={ready} warn={warn} blocked={blocked}".format(
            overall=plan["overall_status"],
            ready=plan["counts"].get("ready", 0),
            warn=plan["counts"].get("warn", 0),
            blocked=plan["counts"].get("blocked", 0),
        )
    )
    return 0 if plan["overall_status"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
