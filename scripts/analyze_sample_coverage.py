"""Analyze representative real-sample coverage for regression runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


RequirementPredicate = Callable[[dict[str, Any]], bool]


def build_sample_coverage_report(
    *,
    sample_matrix: dict[str, Any],
    coverage_manifest: dict[str, Any] | None = None,
    source_dirs: list[str] | None = None,
) -> dict[str, Any]:
    """Build a coverage report for required representative apps/pages."""
    coverage_manifest = coverage_manifest or {}
    rows = [row for row in list(sample_matrix.get("rows") or []) if isinstance(row, dict)]
    excluded = [row for row in list(sample_matrix.get("excluded_rows") or []) if isinstance(row, dict)]
    manifest_rows = [row for row in list(coverage_manifest.get("rows") or []) if isinstance(row, dict)]
    requirements = [
        _requirement("wechat_chat", "WeChat chat workspace", lambda row: _proc(row) in {"weixin.exe", "wechat.exe"} and _mode(row) == "chat_workspace"),
        _requirement("qq_private_chat", "QQ private chat", lambda row: _proc(row) == "qq.exe" and _variant(row) == "private_chat"),
        _requirement("qq_group_chat", "QQ group chat", lambda row: _proc(row) == "qq.exe" and _variant(row) == "group_chat"),
        _requirement("qq_valid_window", "QQ valid non-tiny window", lambda row: _proc(row) == "qq.exe" and _screenshot_ok(row)),
        _requirement("feishu_chat", "Feishu message workspace", lambda row: _proc(row) == "feishu.exe" and _mode(row) in {"collaboration_inbox", "chat_document"}),
        _requirement("flclash_dashboard", "FlClash dashboard/proxy page", lambda row: _proc(row) == "flclash.exe"),
        _requirement(
            "voicemeeter_control_matrix",
            "VoiceMeeter control matrix",
            lambda row: _proc(row) in {"voicemeeter.exe", "voicemeeter8x64.exe", "voicemeeterpro.exe"},
        ),
        _requirement(
            "netease_music",
            "NetEase CloudMusic player/list page",
            lambda row: _proc(row) in {"cloudmusic.exe", "neteasecloudmusic.exe"},
        ),
    ]
    evaluated = [
        _evaluate_requirement(req, rows=rows, excluded=excluded, manifest_rows=manifest_rows)
        for req in requirements
    ]
    counts = {
        "covered": sum(1 for item in evaluated if item["status"] == "covered"),
        "missing": sum(1 for item in evaluated if item["status"] == "missing"),
        "invalid": sum(1 for item in evaluated if item["status"] == "invalid"),
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_type": "sample_coverage",
        "overall_status": "coverage_ready" if counts["missing"] == 0 and counts["invalid"] == 0 else "coverage_incomplete",
        "source_dirs": source_dirs or [],
        "counts": counts,
        "requirements": evaluated,
    }


def analyze_sample_coverage_dirs(
    *,
    matrix_dir: Path | None = None,
    matrix_dirs: list[Path] | None = None,
    output_dir: Path,
) -> dict[str, Any]:
    """Load sample matrix and optional coverage manifest, then write outputs."""
    dirs = matrix_dirs or ([matrix_dir] if matrix_dir else [])
    sample_matrix = _merge_sample_matrices(dirs)
    coverage_manifest = _merge_coverage_manifests(dirs)
    report = build_sample_coverage_report(
        sample_matrix=sample_matrix,
        coverage_manifest=coverage_manifest,
        source_dirs=[str(path) for path in dirs],
    )
    write_sample_coverage_report(report, output_dir)
    return report


def _merge_sample_matrices(dirs: list[Path | None]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for directory in dirs:
        if not directory:
            continue
        data = _load_json(directory / "sample_matrix_summary.json")
        rows.extend(row for row in list(data.get("rows") or []) if isinstance(row, dict))
        excluded_rows.extend(row for row in list(data.get("excluded_rows") or []) if isinstance(row, dict))
    return {"rows": rows, "excluded_rows": excluded_rows, "sample_count": len(rows), "excluded_count": len(excluded_rows)}


def _merge_coverage_manifests(dirs: list[Path | None]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for directory in dirs:
        if not directory:
            continue
        data = _load_json(directory / "coverage_manifest.json")
        rows.extend(row for row in list(data.get("rows") or []) if isinstance(row, dict))
    return {"rows": rows}


def write_sample_coverage_report(report: dict[str, Any], output_dir: Path) -> None:
    """Persist JSON and Markdown coverage reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sample_coverage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Sample Coverage Report",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Counts: {_format_counts(report.get('counts') or {})}",
        "",
        "| requirement | status | sample | reason | collection_hint |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in list(report.get("requirements") or []):
        lines.append(
            "| {rid} | {status} | {sample} | {reason} | {hint} |".format(
                rid=_md(item.get("requirement_id")),
                status=_md(item.get("status")),
                sample=_md(item.get("sample")),
                reason=_md(item.get("reason")),
                hint=_md(item.get("collection_hint")),
            )
        )
    (output_dir / "sample_coverage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _requirement(requirement_id: str, label: str, predicate: RequirementPredicate) -> dict[str, Any]:
    return {"requirement_id": requirement_id, "label": label, "predicate": predicate}


def _evaluate_requirement(
    req: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    predicate = req["predicate"]
    for row in rows:
        if predicate(row) and _valid_sample(row):
            return {
                "requirement_id": req["requirement_id"],
                "label": req["label"],
                "status": "covered",
                "sample": str(row.get("sample") or ""),
                "reason": "valid sample in matrix",
                "collection_hint": "",
            }
    invalid_reason = _invalid_reason(req, excluded=excluded)
    if invalid_reason:
        return {
            "requirement_id": req["requirement_id"],
            "label": req["label"],
            "status": "invalid",
            "sample": "",
            "reason": invalid_reason,
            "collection_hint": _collection_hint(str(req["requirement_id"]), invalid_reason),
        }
    manifest_reason = _manifest_reason(req, manifest_rows=manifest_rows)
    return {
        "requirement_id": req["requirement_id"],
        "label": req["label"],
        "status": "missing",
        "sample": "",
        "reason": manifest_reason or "no matching valid sample",
        "collection_hint": _collection_hint(str(req["requirement_id"]), manifest_reason or "no matching valid sample"),
    }


def _valid_sample(row: dict[str, Any]) -> bool:
    if row.get("screenshot_success") is False:
        return False
    if str(row.get("excluded_reason") or ""):
        return False
    warnings = {str(item) for item in list(row.get("quality_warnings") or [])}
    return "screenshot_missing" not in warnings and "root_only_detail" not in warnings


def _invalid_reason(req: dict[str, Any], *, excluded: list[dict[str, Any]]) -> str:
    predicate = req["predicate"]
    reasons: list[str] = []
    for row in excluded:
        if predicate(row):
            reason = str(row.get("excluded_reason") or "excluded")
            if reason:
                reasons.append(reason)
    return ",".join(list(dict.fromkeys(reasons)))


def _manifest_reason(req: dict[str, Any], *, manifest_rows: list[dict[str, Any]]) -> str:
    target_processes = _requirement_processes(str(req["requirement_id"]))
    if not target_processes:
        return ""
    for row in manifest_rows:
        if str(row.get("process_name") or "").lower() in target_processes and row.get("covered") is False:
            return str(row.get("reason") or "not_covered")
    return ""


def _requirement_processes(requirement_id: str) -> set[str]:
    return {
        "wechat_chat": {"weixin.exe", "wechat.exe"},
        "qq_private_chat": {"qq.exe"},
        "qq_group_chat": {"qq.exe"},
        "qq_valid_window": {"qq.exe"},
        "feishu_chat": {"feishu.exe"},
        "flclash_dashboard": {"flclash.exe"},
        "voicemeeter_control_matrix": {"voicemeeter.exe", "voicemeeter8x64.exe", "voicemeeterpro.exe"},
        "netease_music": {"cloudmusic.exe", "neteasecloudmusic.exe"},
    }.get(requirement_id, set())


def _collection_hint(requirement_id: str, reason: str = "") -> str:
    hints = {
        "wechat_chat": "Open a normal WeChat chat window, then collect with --include-process weixin.exe.",
        "qq_private_chat": "Open a full QQ private-chat window, not the tiny tray/utility window, then collect with --include-process qq.exe.",
        "qq_group_chat": "Open a full QQ group-chat window with the right member list visible, then collect with --include-process qq.exe.",
        "qq_valid_window": "Restore a full QQ main/chat window before sampling; tiny windows are excluded.",
        "feishu_chat": "Open a Feishu message conversation, wait for the message pane to load, then collect with --include-process feishu.exe.",
        "flclash_dashboard": "Restore FlClash to the dashboard/proxy page, then collect with --include-process flclash.exe.",
        "voicemeeter_control_matrix": "Restore VoiceMeeter's main mixer window, then collect with --include-process voicemeeter8x64.exe.",
        "netease_music": "Open or restore NetEase CloudMusic with a visible player/list page, then collect with --include-process cloudmusic.exe.",
    }
    hint = hints.get(requirement_id, "Open the target application/page and rerun collect_live_sample_matrix.py.")
    if reason:
        return f"{hint} Current reason: {reason}."
    return hint


def _proc(row: dict[str, Any]) -> str:
    return str(row.get("process_name") or "").lower()


def _mode(row: dict[str, Any]) -> str:
    return str(row.get("mode") or "")


def _variant(row: dict[str, Any]) -> str:
    return str(row.get("chat_variant") or "")


def _screenshot_ok(row: dict[str, Any]) -> bool:
    return row.get("screenshot_success") is not False


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"


def _md(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_sample_coverage_dirs(matrix_dirs=args.matrix_dir, output_dir=args.output_dir)
    print(f"overall={report.get('overall_status')} counts={report.get('counts')}")
    return 0 if report.get("overall_status") == "coverage_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
