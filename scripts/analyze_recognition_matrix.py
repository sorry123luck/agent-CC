"""Analyze recognition-chain sample matrix outputs against fixed acceptance rules."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


CHAT_MODES = {"chat_workspace", "chat_document", "collaboration_inbox"}
CHAT_PROCESSES = {"wechat.exe", "weixin.exe", "feishu.exe", "lark.exe"}
ACCOUNT_MODES = {"account_switcher"}
DENSE_MODES = {"control_matrix"}
ACCEPTED_SAMPLE_MATRIX_SCHEMA_VERSION = "2026-05-25.chat-variant.v3"


def classify_row(row: dict[str, Any]) -> str:
    """Classify one sample row into an acceptance class."""
    mode = str(row.get("mode") or "").strip()
    process = str(row.get("process_name") or "").strip().lower()
    if mode in ACCOUNT_MODES:
        return "account_switcher"
    if mode in CHAT_MODES or process in CHAT_PROCESSES:
        return mode or "chat_workspace"
    if mode in DENSE_MODES:
        return "control_matrix"
    if mode:
        return mode
    return "unknown"


def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return pass/warn/fail verdict for one sample matrix row."""
    sample_class = classify_row(row)
    failures: list[str] = []
    warnings: list[str] = []

    if str(row.get("error") or "").strip():
        failures.append("sample_collection_error")

    observe_ms = int(row.get("observe_elapsed_ms") or 0)
    if observe_ms > 2000:
        warnings.append("observe_over_2s")
    if int(row.get("roi_vlm_late_pending_count") or 0) > 0:
        warnings.append("roi_vlm_late_results_pending")
    if (
        str(row.get("process_name") or "").strip().lower() == "qq.exe"
        and str(row.get("mode") or "").strip() == "chat_workspace"
        and str(row.get("chat_variant") or "").strip() in {"", "unknown"}
    ):
        warnings.append("qq_chat_variant_unknown")
    quality_warnings = {
        str(warning).strip()
        for warning in list(row.get("quality_warnings") or [])
        if str(warning).strip()
    }
    if "screenshot_missing" in quality_warnings:
        warnings.append("screenshot_missing")
    if "root_only_detail" in quality_warnings:
        warnings.append("root_only_detail")

    roi_purposes = {
        str(purpose)
        for purpose in list(row.get("roi_purposes") or [])
        if str(purpose).strip()
    }

    if sample_class in {"chat_workspace", "chat_document"}:
        if "composer" not in roi_purposes:
            failures.append("missing_composer_roi")
        _evaluate_chat_input(row, failures)
        _evaluate_composer_vlm_budget(row, warnings)
    elif sample_class == "collaboration_inbox":
        expected = {"app_rail", "inbox_list", "message_thread", "composer"}
        if not expected.issubset(roi_purposes):
            missing = sorted(expected - roi_purposes)
            failures.append("missing_collaboration_rois:" + ",".join(missing))
        _evaluate_chat_input(row, failures)
        _evaluate_composer_vlm_budget(row, warnings)
    elif sample_class == "account_switcher":
        # Login/account pages can legitimately contain text inputs. The chat
        # input acceptance rule only applies to chat-like modes.
        pass
    elif sample_class == "control_matrix":
        max_candidates = int(row.get("roi_vlm_max_candidates_per_job") or 0)
        if max_candidates > 8:
            warnings.append("dense_roi_candidate_budget_high")

    status = "fail" if failures else "warn" if warnings else "pass"
    return {
        "sample": str(row.get("sample") or ""),
        "process_name": str(row.get("process_name") or ""),
        "mode": str(row.get("mode") or ""),
        "chat_variant": str(row.get("chat_variant") or ""),
        "sample_class": sample_class,
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "observe_elapsed_ms": observe_ms,
        "roi_purposes": sorted(roi_purposes),
    }


def _evaluate_chat_input(row: dict[str, Any], failures: list[str]) -> None:
    candidate_count = int(row.get("composer_input_candidate_count") or 0)
    safe_count = int(row.get("composer_input_safe_count") or 0)
    review_count = int(row.get("composer_input_review_count") or 0)
    review_in_composer = int(row.get("composer_input_review_in_composer_count") or 0)
    review_min_width = int(row.get("composer_input_review_min_width") or 0)
    review_min_height = int(row.get("composer_input_review_min_height") or 0)
    has_primary_click = "composer_input_primary_click_point" in row
    primary_click = row.get("composer_input_primary_click_point")
    has_review_quality = "composer_input_review_in_composer_count" in row
    has_review_width = "composer_input_review_min_width" in row
    has_review_height = "composer_input_review_min_height" in row
    composer_vlm_count = int(row.get("composer_vlm_candidate_count") or 0)
    send_hint_count = int(row.get("composer_vlm_send_hint_count") or 0)
    send_target_width = int(row.get("composer_send_target_width") or 0)
    send_target_height = int(row.get("composer_send_target_height") or 0)
    if candidate_count < 1:
        failures.append("missing_chat_input_candidate")
    if safe_count > 0:
        failures.append("chat_input_must_remain_review_only")
    if review_count < 1:
        failures.append("missing_review_chat_input_candidate")
    if review_count > 0 and has_review_quality and review_in_composer < 1:
        failures.append("review_chat_input_outside_composer_roi")
    if review_count > 0 and has_review_width and review_min_width < 120:
        failures.append("review_chat_input_too_narrow")
    if review_count > 0 and has_review_height and review_min_height < 24:
        failures.append("review_chat_input_too_short")
    if review_count > 0 and has_primary_click and not _valid_click_point(primary_click):
        failures.append("missing_review_chat_input_click_point")
    if composer_vlm_count > 0 and send_hint_count < 1:
        failures.append("missing_composer_send_hint")
    if send_hint_count > 0 and send_target_width > 0 and send_target_width < 32:
        failures.append("composer_send_target_too_narrow")
    if send_hint_count > 0 and send_target_height > 0 and send_target_height < 20:
        failures.append("composer_send_target_too_short")


def _evaluate_composer_vlm_budget(row: dict[str, Any], warnings: list[str]) -> None:
    composer_vlm_count = int(row.get("composer_vlm_candidate_count") or 0)
    if composer_vlm_count > 6:
        warnings.append("composer_vlm_candidate_budget_high")


def _valid_click_point(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    try:
        x, y = [int(item) for item in value]
    except (TypeError, ValueError):
        return False
    return x >= 0 and y >= 0


def build_acceptance_report(summary: dict[str, Any]) -> dict[str, Any]:
    """Build aggregate pass/warn/fail report from sample matrix summary."""
    verdicts = [evaluate_row(row) for row in list(summary.get("rows") or [])]
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for verdict in verdicts:
        status = str(verdict.get("status") or "fail")
        counts[status] = counts.get(status, 0) + 1
    matrix_warnings = _matrix_warnings(summary)
    overall = "fail" if counts["fail"] else "warn" if counts["warn"] else "pass"
    if overall == "pass" and matrix_warnings:
        overall = "warn"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_generated_at": summary.get("generated_at"),
        "sample_matrix_schema_version": str(summary.get("sample_matrix_schema_version") or ""),
        "accepted_sample_matrix_schema_version": ACCEPTED_SAMPLE_MATRIX_SCHEMA_VERSION,
        "matrix_warnings": matrix_warnings,
        "overall_status": overall,
        "counts": counts,
        "rows": verdicts,
    }


def write_acceptance_outputs(report: dict[str, Any], output_dir: Path) -> None:
    """Persist JSON and Markdown acceptance report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "recognition_acceptance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Recognition Acceptance",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Matrix schema: {report.get('sample_matrix_schema_version', '') or 'legacy'}",
        f"- Matrix warnings: {','.join(report.get('matrix_warnings') or []) or 'none'}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Counts: {_format_counts(report.get('counts') or {})}",
        "",
        "| sample | process | class | variant | status | failures | warnings |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(report.get("rows") or []):
        lines.append(
            "| {sample} | {process} | {klass} | {variant} | {status} | {failures} | {warnings} |".format(
                sample=str(row.get("sample") or "").replace("|", "/"),
                process=str(row.get("process_name") or "").replace("|", "/"),
                klass=str(row.get("sample_class") or "").replace("|", "/"),
                variant=str(row.get("chat_variant") or "").replace("|", "/"),
                status=str(row.get("status") or "").replace("|", "/"),
                failures=",".join(row.get("failures") or []),
                warnings=",".join(row.get("warnings") or []),
            )
        )
    (output_dir / "recognition_acceptance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) if counts else "none"


def _matrix_warnings(summary: dict[str, Any]) -> list[str]:
    schema_version = str(summary.get("sample_matrix_schema_version") or "")
    if schema_version != ACCEPTED_SAMPLE_MATRIX_SCHEMA_VERSION:
        return ["stale_or_legacy_sample_matrix"]
    return []


def analyze_matrix_dir(matrix_dir: Path) -> dict[str, Any]:
    """Load a matrix summary from a directory and write acceptance outputs."""
    summary_path = matrix_dir / "sample_matrix_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = build_acceptance_report(summary)
    write_acceptance_outputs(report, matrix_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_matrix_dir(args.matrix_dir)
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
