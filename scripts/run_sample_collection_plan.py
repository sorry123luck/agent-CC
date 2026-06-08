"""Run or dry-run an actionable sample collection plan."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import collect_live_sample_matrix
import analyze_sample_coverage
import build_sample_collection_plan


Collector = Callable[..., dict[str, Any]]
WindowProvider = Callable[[str], list[dict[str, Any]]]


def run_sample_collection_plan(
    *,
    plan_file: Path,
    output_dir: Path,
    base_url: str = "http://127.0.0.1:8000",
    execute: bool = False,
    max_windows: int = 3,
    wait_enhance_seconds: float = 30.0,
    deadline_ms: int = 2000,
    clean_output: bool = True,
    baseline_matrix_dirs: list[Path] | None = None,
    collector: Collector | None = None,
    window_provider: WindowProvider | None = None,
) -> dict[str, Any]:
    """Evaluate plan rows against live windows and optionally collect available samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(plan_file.read_text(encoding="utf-8-sig"))
    rows = [row for row in list(plan.get("rows") or []) if isinstance(row, dict)]
    collector = collector or collect_live_sample_matrix.collect_matrix
    window_provider = window_provider or _fetch_windows
    windows = window_provider(base_url.rstrip("/"))
    result_rows: list[dict[str, Any]] = []
    collected_dirs: list[str] = []
    for row in rows:
        result = _run_plan_row(
            row=row,
            windows=windows,
            output_dir=output_dir,
            base_url=base_url.rstrip("/"),
            execute=execute,
            max_windows=max_windows,
            wait_enhance_seconds=wait_enhance_seconds,
            deadline_ms=deadline_ms,
            clean_output=clean_output,
            collector=collector,
        )
        result_rows.append(result)
        if result.get("collection_output_dir"):
            collected_dirs.append(str(result["collection_output_dir"]))
    counts = _status_counts(result_rows)
    followup_coverage, followup_plan = _build_followup_reports(
        output_dir=output_dir,
        baseline_matrix_dirs=baseline_matrix_dirs or [],
        collection_rows=result_rows,
    )
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_type": "sample_collection_run",
        "execute": bool(execute),
        "source_plan": str(plan_file),
        "overall_status": _overall_status(counts),
        "counts": counts,
        "collected_matrix_dirs": collected_dirs,
        "followup_coverage": followup_coverage,
        "followup_collection_plan": followup_plan,
        "rows": result_rows,
    }
    write_sample_collection_run_report(report, output_dir)
    return report


def write_sample_collection_run_report(report: dict[str, Any], output_dir: Path) -> None:
    """Persist run report JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sample_collection_run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Sample Collection Run Report",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Execute: {report.get('execute')}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Counts: {_format_counts(report.get('counts') or {})}",
        "",
        "| requirement | status | process | window | output | reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(report.get("rows") or []):
        lines.append(
            "| {rid} | {status} | {process} | {window} | {output} | {reason} |".format(
                rid=_md(row.get("requirement_id")),
                status=_md(row.get("status")),
                process=_md(",".join(row.get("include_processes") or [])),
                window=_md(row.get("selected_window")),
                output=_md(row.get("collection_output_dir")),
                reason=_md(row.get("reason")),
            )
        )
    (output_dir / "sample_collection_run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_plan_row(
    *,
    row: dict[str, Any],
    windows: list[dict[str, Any]],
    output_dir: Path,
    base_url: str,
    execute: bool,
    max_windows: int,
    wait_enhance_seconds: float,
    deadline_ms: int,
    clean_output: bool,
    collector: Collector,
) -> dict[str, Any]:
    include_processes = [str(item).lower() for item in list(row.get("include_processes") or []) if str(item).strip()]
    selected = collect_live_sample_matrix.select_sample_windows(
        windows,
        include_processes=include_processes,
        max_windows=max_windows,
    )
    result = {
        "requirement_id": str(row.get("requirement_id") or ""),
        "include_processes": include_processes,
        "prerequisite": str(row.get("prerequisite") or ""),
        "status": "",
        "selected_window": _window_label(selected[0]) if selected else "",
        "collection_output_dir": "",
        "reason": "",
    }
    if not selected:
        result["status"] = "skipped_no_visible_window"
        result["reason"] = "No visible non-minimized target window matched the include_processes."
        return result
    row_dir = output_dir / "collections" / _safe_slug(str(row.get("requirement_id") or "sample"))
    result["collection_output_dir"] = str(row_dir)
    if not execute:
        result["status"] = "planned"
        result["reason"] = "Target window is visible; rerun with --execute to collect."
        return result
    try:
        summary = collector(
            base_url=base_url,
            output_dir=row_dir,
            include_processes=include_processes,
            max_windows=max_windows,
            run_vlm=False,
            max_rois_per_window=1,
            wait_late_seconds=0.0,
            wait_enhance_seconds=wait_enhance_seconds,
            deadline_ms=deadline_ms,
            clean_output=clean_output,
            async_enhance=False,
            roi_purposes=[],
        )
    except Exception as exc:  # noqa: BLE001 - report preserves collection failures.
        result["status"] = "failed"
        result["reason"] = str(exc)
        return result
    _ensure_summary_output(row_dir=row_dir, summary=summary)
    error_count = int(summary.get("error_count") or 0)
    sample_count = int(summary.get("sample_count") or 0)
    requirement_status = _requirement_status_after_collection(
        requirement_id=str(row.get("requirement_id") or ""),
        summary=summary,
    )
    if error_count:
        result["status"] = "failed"
        result["reason"] = f"collection errors={error_count}"
    elif requirement_status and requirement_status != "covered":
        result["status"] = "collected_not_matching_requirement"
        result["reason"] = f"samples={sample_count}; requirement_status={requirement_status}"
    elif sample_count:
        result["status"] = "collected"
        result["reason"] = f"samples={sample_count}"
    else:
        result["status"] = "collected_empty"
        result["reason"] = "collector returned no valid samples"
    return result


def _ensure_summary_output(*, row_dir: Path, summary: dict[str, Any]) -> None:
    row_dir.mkdir(parents=True, exist_ok=True)
    summary_file = row_dir / "sample_matrix_summary.json"
    if not summary_file.exists():
        summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_followup_reports(
    *,
    output_dir: Path,
    baseline_matrix_dirs: list[Path],
    collection_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix_dirs = [path for path in baseline_matrix_dirs if path]
    for row in collection_rows:
        collection_output = Path(str(row.get("collection_output_dir") or ""))
        if collection_output and (collection_output / "sample_matrix_summary.json").exists():
            matrix_dirs.append(collection_output)
    if not matrix_dirs:
        return {}, {}
    coverage_dir = output_dir / "followup_coverage"
    coverage_report = analyze_sample_coverage.analyze_sample_coverage_dirs(
        matrix_dirs=matrix_dirs,
        output_dir=coverage_dir,
    )
    plan_dir = output_dir / "followup_collection_plan"
    collection_plan = build_sample_collection_plan.build_sample_collection_plan(
        coverage_report=coverage_report,
    )
    build_sample_collection_plan.write_sample_collection_plan(collection_plan, plan_dir)
    return coverage_report, collection_plan


def _fetch_windows(base_url: str) -> list[dict[str, Any]]:
    response = requests.get(f"{base_url}/api/v1/windows", timeout=20)
    response.raise_for_status()
    data = response.json()
    return [row for row in list(data or []) if isinstance(row, dict)]


def _requirement_status_after_collection(*, requirement_id: str, summary: dict[str, Any]) -> str:
    if not requirement_id:
        return ""
    coverage = analyze_sample_coverage.build_sample_coverage_report(sample_matrix=summary)
    for item in list(coverage.get("requirements") or []):
        if str(item.get("requirement_id") or "") == requirement_id:
            return str(item.get("status") or "")
    return ""


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _overall_status(counts: dict[str, int]) -> str:
    if not counts:
        return "no_plan_rows"
    if counts.get("failed"):
        return "collection_failed"
    if counts.get("collected_not_matching_requirement"):
        return "collection_review"
    if counts.get("skipped_no_visible_window"):
        return "collection_waiting_for_targets"
    if counts.get("planned"):
        return "collection_planned"
    if counts.get("collected_empty"):
        return "collection_review"
    return "collection_done"


def _window_label(window: dict[str, Any]) -> str:
    return "{process} hwnd={hwnd} title={title}".format(
        process=window.get("process_name") or "",
        hwnd=window.get("hwnd") or "",
        title=window.get("title") or "",
    )


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug[:80] or "sample"


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"


def _md(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--wait-enhance-seconds", type=float, default=30.0)
    parser.add_argument("--deadline-ms", type=int, default=2000)
    parser.add_argument("--no-clean-output", action="store_true")
    parser.add_argument("--baseline-matrix-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    report = run_sample_collection_plan(
        plan_file=args.plan,
        output_dir=args.output_dir,
        base_url=args.base_url,
        execute=args.execute,
        max_windows=args.max_windows,
        wait_enhance_seconds=args.wait_enhance_seconds,
        deadline_ms=args.deadline_ms,
        clean_output=not args.no_clean_output,
        baseline_matrix_dirs=args.baseline_matrix_dir,
    )
    print(f"overall={report.get('overall_status')} counts={report.get('counts')}")
    return 0 if report.get("overall_status") in {"collection_done", "collection_planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
