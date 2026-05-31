"""Read-only retention audit for artifacts directories.

The audit reports size, age, category, and a conservative recommendation.  It
does not move or delete any files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPORT_JSON = "artifacts_retention_audit.json"
REPORT_MD = "artifacts_retention_audit.md"

SUMMARY_REPORT_NAMES = {
    "agent_operability_regression_summary.json",
    "recognition_closure_report.json",
    "page_operability_report.json",
    "act_preflight_matrix_report.json",
    "input_safety_readiness_report.json",
    "recognition_acceptance.json",
    "chat_readiness_report.json",
}


def audit_artifacts_retention(
    *,
    artifacts_root: Path,
    now: datetime | None = None,
    archive_after_days: int = 7,
) -> dict[str, Any]:
    now = now or datetime.now()
    rows = [
        _audit_dir(path, now=now, archive_after_days=archive_after_days)
        for path in sorted(artifacts_root.iterdir(), key=lambda p: p.name.lower())
        if path.is_dir()
    ] if artifacts_root.exists() else []
    counts: dict[str, int] = {"total_dirs": len(rows)}
    for row in rows:
        action = str(row.get("recommended_action") or "review")
        counts[action] = counts.get(action, 0) + 1
    counts["total_size_bytes"] = sum(int(row.get("size_bytes") or 0) for row in rows)
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "report_type": "artifacts_retention_audit",
        "policy": (
            "Read-only audit. keep means the directory contains a current summary "
            "or baseline signal; archive_candidate means it is old and lacks a "
            "recognized summary report. No files are moved or deleted."
        ),
        "archive_after_days": archive_after_days,
        "counts": counts,
        "rows": rows,
    }


def write_artifacts_retention_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_JSON).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Artifacts Retention Audit",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Archive-after days: {report.get('archive_after_days', '')}",
        f"- Counts: {_format_counts(report.get('counts') or {})}",
        f"- Policy: {report.get('policy', '')}",
        "",
        "| name | type | action | age_days | size_bytes | signals |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(report.get("rows") or []):
        lines.append(
            "| {name} | {kind} | {action} | {age} | {size} | {signals} |".format(
                name=_md(row.get("name")),
                kind=_md(row.get("artifact_type")),
                action=_md(row.get("recommended_action")),
                age=_md(row.get("age_days")),
                size=_md(row.get("size_bytes")),
                signals=_md(",".join(row.get("signals") or [])),
            )
        )
    (output_dir / REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_artifacts_retention_dirs(
    *,
    artifacts_root: Path,
    output_dir: Path,
    archive_after_days: int = 7,
) -> dict[str, Any]:
    report = audit_artifacts_retention(
        artifacts_root=artifacts_root,
        archive_after_days=archive_after_days,
    )
    write_artifacts_retention_report(report, output_dir)
    return report


def _audit_dir(path: Path, *, now: datetime, archive_after_days: int) -> dict[str, Any]:
    latest_mtime = _latest_mtime(path)
    age_days = max(0, int((now.timestamp() - latest_mtime) // 86400)) if latest_mtime else 0
    signals = _signals(path)
    action = _recommended_action(age_days=age_days, signals=signals, archive_after_days=archive_after_days)
    return {
        "name": path.name,
        "path": str(path),
        "artifact_type": _artifact_type(path.name),
        "size_bytes": _dir_size(path),
        "file_count": _file_count(path),
        "age_days": age_days,
        "signals": signals,
        "recommended_action": action,
    }


def _signals(path: Path) -> list[str]:
    names = {item.name for item in path.rglob("*") if item.is_file()}
    signals: list[str] = []
    if names & SUMMARY_REPORT_NAMES:
        signals.append("summary_report")
    if "sample_matrix_summary.json" in names:
        signals.append("sample_matrix")
    if "chat_send_probe_report.json" in names:
        signals.append("send_probe_report")
    if "recognition_closure_report.json" in names:
        signals.append("closure_report")
    if path.name.startswith("_archive"):
        signals.append("already_archived")
    return signals


def _recommended_action(*, age_days: int, signals: list[str], archive_after_days: int) -> str:
    if "already_archived" in signals:
        return "keep"
    if "summary_report" in signals or "sample_matrix" in signals:
        return "keep"
    if age_days >= archive_after_days:
        return "archive_candidate"
    return "review"


def _artifact_type(name: str) -> str:
    if name.startswith("agent_operability_regression"):
        return "agent_operability_regression"
    if name.startswith("page_operability_matrix") or name.startswith("live_sample_matrix"):
        return "sample_matrix"
    if name.startswith("chat_"):
        return "chat_probe"
    if name.startswith("search_"):
        return "search_probe"
    if name.startswith("vlm") or "vlm" in name:
        return "vlm_probe"
    if name.startswith("_archive"):
        return "archive"
    return "diagnostic"


def _dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _file_count(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file())


def _latest_mtime(path: Path) -> float:
    mtimes = [path.stat().st_mtime]
    mtimes.extend(item.stat().st_mtime for item in path.rglob("*") if item.exists())
    return max(mtimes) if mtimes else 0.0


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) if counts else "none"


def _md(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/artifacts_retention_audit"))
    parser.add_argument("--archive-after-days", type=int, default=7)
    args = parser.parse_args()
    report = audit_artifacts_retention_dirs(
        artifacts_root=args.artifacts_root,
        output_dir=args.output_dir,
        archive_after_days=args.archive_after_days,
    )
    print(
        "dirs={dirs} keep={keep} archive_candidate={archive}".format(
            dirs=report["counts"].get("total_dirs", 0),
            keep=report["counts"].get("keep", 0),
            archive=report["counts"].get("archive_candidate", 0),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
