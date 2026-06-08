"""Build an actionable collection plan from sample coverage gaps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def build_sample_collection_plan(*, coverage_report: dict[str, Any]) -> dict[str, Any]:
    """Convert missing/invalid sample coverage rows into concrete collection steps."""
    requirements = [row for row in list(coverage_report.get("requirements") or []) if isinstance(row, dict)]
    rows = [_plan_row(row) for row in requirements if str(row.get("status") or "") != "covered"]
    counts = {
        "ready_to_collect": sum(1 for row in rows if row["collection_status"] == "ready_to_collect"),
        "needs_human_setup": sum(1 for row in rows if row["collection_status"] == "needs_human_setup"),
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_type": "sample_collection_plan",
        "source_coverage_status": str(coverage_report.get("overall_status") or ""),
        "overall_status": "no_gaps" if not rows else "collection_needed",
        "counts": counts,
        "rows": rows,
    }


def build_sample_collection_plan_from_file(*, coverage_report_file: Path, output_dir: Path) -> dict[str, Any]:
    """Load a coverage report, build a collection plan, and write JSON/Markdown outputs."""
    coverage_report = json.loads(coverage_report_file.read_text(encoding="utf-8-sig"))
    plan = build_sample_collection_plan(coverage_report=coverage_report)
    write_sample_collection_plan(plan, output_dir)
    return plan


def write_sample_collection_plan(plan: dict[str, Any], output_dir: Path) -> None:
    """Persist collection plan JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sample_collection_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Sample Collection Plan",
        "",
        f"- Generated: {plan.get('generated_at', '')}",
        f"- Overall: {plan.get('overall_status', '')}",
        f"- Source coverage: {plan.get('source_coverage_status', '')}",
        f"- Counts: {_format_counts(plan.get('counts') or {})}",
        "",
        "| requirement | status | process | prerequisite | command | reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(plan.get("rows") or []):
        lines.append(
            "| {rid} | {status} | {process} | {prereq} | `{command}` | {reason} |".format(
                rid=_md(row.get("requirement_id")),
                status=_md(row.get("collection_status")),
                process=_md(",".join(row.get("include_processes") or [])),
                prereq=_md(row.get("prerequisite")),
                command=_md(row.get("suggested_command")),
                reason=_md(row.get("reason")),
            )
        )
    (output_dir / "sample_collection_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plan_row(requirement: dict[str, Any]) -> dict[str, Any]:
    requirement_id = str(requirement.get("requirement_id") or "")
    process = _include_process(requirement_id)
    prerequisite = _prerequisite(requirement_id)
    return {
        "requirement_id": requirement_id,
        "label": str(requirement.get("label") or ""),
        "coverage_status": str(requirement.get("status") or ""),
        "collection_status": "needs_human_setup",
        "include_processes": [process] if process else [],
        "prerequisite": prerequisite,
        "suggested_command": _suggested_command(process, requirement_id),
        "reason": str(requirement.get("reason") or ""),
        "collection_hint": str(requirement.get("collection_hint") or ""),
    }


def _include_process(requirement_id: str) -> str:
    return {
        "wechat_chat": "weixin.exe",
        "qq_private_chat": "qq.exe",
        "qq_group_chat": "qq.exe",
        "qq_valid_window": "qq.exe",
        "feishu_chat": "feishu.exe",
        "flclash_dashboard": "flclash.exe",
        "voicemeeter_control_matrix": "voicemeeter8x64.exe",
        "netease_music": "cloudmusic.exe",
    }.get(requirement_id, "")


def _prerequisite(requirement_id: str) -> str:
    return {
        "wechat_chat": "Restore WeChat and open a normal chat page.",
        "qq_private_chat": "Restore QQ and open a full private chat page.",
        "qq_group_chat": "Restore QQ and open a full group chat page.",
        "qq_valid_window": "Restore any full QQ main/chat window; ignore tiny utility windows.",
        "feishu_chat": "Restore Feishu and open a message conversation.",
        "flclash_dashboard": "Restore FlClash to dashboard/proxy page.",
        "voicemeeter_control_matrix": "Restore VoiceMeeter main mixer window.",
        "netease_music": "Launch or restore NetEase CloudMusic to a visible player/list page.",
    }.get(requirement_id, "Open the target app/page before collection.")


def _suggested_command(process: str, requirement_id: str) -> str:
    output_dir = f"artifacts/live_sample_matrix_<date>-{requirement_id}"
    if not process:
        return f"python scripts\\collect_live_sample_matrix.py --output-dir {output_dir} --max-windows 3 --wait-enhance-seconds 30 --deadline-ms 2000 --clean-output"
    return (
        "python scripts\\collect_live_sample_matrix.py "
        f"--output-dir {output_dir} "
        f"--include-process {process} "
        "--max-windows 3 --wait-enhance-seconds 30 --deadline-ms 2000 --clean-output"
    )


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"


def _md(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = build_sample_collection_plan_from_file(
        coverage_report_file=args.coverage_report,
        output_dir=args.output_dir,
    )
    print(f"overall={plan.get('overall_status')} counts={plan.get('counts')}")
    return 0 if plan.get("overall_status") == "no_gaps" else 1


if __name__ == "__main__":
    raise SystemExit(main())
