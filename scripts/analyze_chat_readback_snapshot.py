"""Build an acceptance report for read-only chat readback snapshots."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def build_snapshot_acceptance(snapshot_report: dict[str, Any]) -> dict[str, Any]:
    """Summarize snapshot rows into pass/warn/fail acceptance rows."""
    rows = [_evaluate_row(row) for row in list(snapshot_report.get("rows") or []) if isinstance(row, dict)]
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    overall = "fail" if counts["fail"] else "warn" if counts["warn"] else "pass"
    if not rows:
        overall = "fail"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_status": overall,
        "counts": counts,
        "rows": rows,
    }


def analyze_snapshot_dir(snapshot_dir: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    """Load snapshot report plus nested readback JSON files and write acceptance output."""
    report_path = snapshot_dir / "chat_readback_snapshot_report.json"
    snapshot_report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    for row in list(snapshot_report.get("rows") or []):
        if not isinstance(row, dict):
            continue
        row["readback_result"] = _load_readback_result(row, base_dir=snapshot_dir)
    acceptance = build_snapshot_acceptance(snapshot_report)
    write_acceptance_report(acceptance, output_dir or snapshot_dir)
    return acceptance


def write_acceptance_report(report: dict[str, Any], output_dir: Path) -> None:
    """Persist JSON and Markdown acceptance reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chat_readback_snapshot_acceptance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Chat Readback Snapshot Acceptance",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Counts: {_format_counts(report.get('counts') or {})}",
        "",
        "| sample | process | status | messages | warnings | texts |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in list(report.get("rows") or []):
        lines.append(
            "| {sample} | {process} | {status} | {messages} | {warnings} | {texts} |".format(
                sample=str(row.get("sample") or "").replace("|", "/"),
                process=str(row.get("process_name") or "").replace("|", "/"),
                status=str(row.get("status") or "").replace("|", "/"),
                messages=int(row.get("message_text_count") or 0),
                warnings=",".join(row.get("warnings") or []).replace("|", "/"),
                texts=" / ".join(row.get("message_texts") or []).replace("|", "/"),
            )
        )
    (output_dir / "chat_readback_snapshot_acceptance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
    readback = row.get("readback_result") if isinstance(row.get("readback_result"), dict) else {}
    events = [event for event in list(readback.get("events") or []) if isinstance(event, dict)]
    texts = [str(event.get("text") or "").strip() for event in events if str(event.get("text") or "").strip()]
    warnings = list(dict.fromkeys([str(item) for item in list(row.get("readback_warnings") or []) + list(readback.get("warnings") or []) if str(item)]))
    failures: list[str] = []
    if row.get("status") == "error":
        failures.append("snapshot_capture_error")
    if not texts and "message_stream_loading" not in warnings:
        warnings.append("no_message_events")
    status = "fail" if failures else "warn" if warnings else "pass"
    return {
        "sample": str(row.get("sample") or ""),
        "process_name": str(row.get("process_name") or ""),
        "snapshot_status": str(row.get("status") or ""),
        "status": status,
        "message_text_count": len(texts),
        "message_texts": texts,
        "readback_event_count": int(row.get("readback_event_count") or len(events)),
        "readback_result_path": str(row.get("readback_result_path") or ""),
        "warnings": warnings,
        "failures": failures,
    }


def _load_readback_result(row: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    path_text = str(row.get("readback_result_path") or "")
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.is_absolute():
        candidates = [
            base_dir / path,
            Path.cwd() / path,
            base_dir.parent / path,
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) if counts else "none"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = analyze_snapshot_dir(args.snapshot_dir, output_dir=args.output_dir)
    print(
        "overall={overall} pass={pass_count} warn={warn_count} fail={fail_count}".format(
            overall=report["overall_status"],
            pass_count=report["counts"].get("pass", 0),
            warn_count=report["counts"].get("warn", 0),
            fail_count=report["counts"].get("fail", 0),
        )
    )
    return 0 if report["overall_status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
