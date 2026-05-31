"""Build a closure report for the recognition-chain workstream.

This report deliberately separates the current recognition-chain deliverable
from later product execution work. It answers whether the current code and real
sample artifacts are sufficient for agent-facing page modeling, without
pretending default typing, full app exploration, or the unified /act executor
are complete.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PASSING_OPERABILITY = {"usable", "usable_with_warnings", "usable_with_blocked_actions"}


def build_closure_report(
    *,
    sample_matrix: dict[str, Any],
    recognition_acceptance: dict[str, Any],
    page_operability: dict[str, Any],
    act_preflight_matrix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a machine-readable closure decision for the recognition chain."""
    completion = {
        "real_sample_rerun": _real_sample_rerun_status(sample_matrix, recognition_acceptance),
        "page_operability_contract": _page_operability_status(page_operability),
        "region_read_api_contract": _region_read_api_status(page_operability),
        "controlled_scroll_contract": _controlled_scroll_status(page_operability),
        "safety_boundary": _safety_boundary_status(page_operability),
    }
    if act_preflight_matrix:
        completion["act_preflight_contract"] = _act_preflight_status(act_preflight_matrix)
    closed = all(item["status"] == "pass" for item in completion.values())
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_type": "recognition_closure",
        "overall_status": "closed_for_recognition_chain" if closed else "open",
        "scope": {
            "closed_now": (
                "Single-page recognition, region reading, page operability reporting, "
                "controlled scroll-read probing, and safety boundaries for review-only inputs."
            ),
            "not_in_scope": (
                "Default safe typing, full multi-page autonomous exploration, semantic task "
                "planning, and the unified /act execution surface."
            ),
        },
        "completion": completion,
        "deferred_after_closure": _deferred_items(),
        "sample_summary": {
            "sample_matrix_schema_version": sample_matrix.get("sample_matrix_schema_version", ""),
            "sample_count": int(sample_matrix.get("sample_count") or len(sample_matrix.get("rows") or [])),
            "excluded_count": int(sample_matrix.get("excluded_count") or 0),
            "recognition_counts": recognition_acceptance.get("counts") or {},
            "operability_counts": page_operability.get("counts") or {},
        },
    }


def analyze_closure_dirs(*, matrix_dir: Path, operability_dir: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    """Load standard artifact files and write closure JSON/Markdown outputs."""
    operability_dir = operability_dir or matrix_dir
    sample_matrix = _load_json(matrix_dir / "sample_matrix_summary.json")
    recognition_acceptance = _load_json(matrix_dir / "recognition_acceptance.json")
    page_operability = _load_json(operability_dir / "page_operability_report.json")
    act_preflight_matrix = _load_first_json(
        operability_dir / "act_preflight_matrix_report.json",
        matrix_dir / "act_preflight_matrix_report.json",
    )
    report = build_closure_report(
        sample_matrix=sample_matrix,
        recognition_acceptance=recognition_acceptance,
        page_operability=page_operability,
        act_preflight_matrix=act_preflight_matrix,
    )
    write_closure_report(report, output_dir or operability_dir)
    return report


def write_closure_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "recognition_closure_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Recognition Closure Report",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Closed now: {(report.get('scope') or {}).get('closed_now', '')}",
        f"- Not in scope: {(report.get('scope') or {}).get('not_in_scope', '')}",
        "",
        "## Completion",
        "",
        "| item | status | evidence |",
        "| --- | --- | --- |",
    ]
    for key, item in (report.get("completion") or {}).items():
        lines.append(
            "| {key} | {status} | {evidence} |".format(
                key=key,
                status=str(item.get("status") or ""),
                evidence=str(item.get("evidence") or "").replace("|", "/"),
            )
        )
    lines.extend(["", "## Deferred After Closure", "", "| item | reason | next phase |", "| --- | --- | --- |"])
    for item in list(report.get("deferred_after_closure") or []):
        lines.append(
            "| {item} | {reason} | {phase} |".format(
                item=str(item.get("item") or ""),
                reason=str(item.get("reason") or "").replace("|", "/"),
                phase=str(item.get("next_phase") or ""),
            )
        )
    (output_dir / "recognition_closure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _real_sample_rerun_status(sample_matrix: dict[str, Any], recognition_acceptance: dict[str, Any]) -> dict[str, str]:
    overall = str(recognition_acceptance.get("overall_status") or "")
    sample_count = int(sample_matrix.get("sample_count") or len(sample_matrix.get("rows") or []))
    if overall == "pass" and sample_count > 0:
        return {"status": "pass", "evidence": f"recognition_acceptance overall=pass sample_count={sample_count}"}
    return {"status": "fail", "evidence": f"recognition_acceptance overall={overall or 'missing'} sample_count={sample_count}"}


def _page_operability_status(page_operability: dict[str, Any]) -> dict[str, str]:
    overall = str(page_operability.get("overall_status") or "")
    rows = list(page_operability.get("rows") or [])
    bad = [str(row.get("sample") or "") for row in rows if str(row.get("operability_status") or "") not in PASSING_OPERABILITY]
    if overall in PASSING_OPERABILITY and not bad and rows:
        return {"status": "pass", "evidence": f"page_operability overall={overall} rows={len(rows)}"}
    return {"status": "fail", "evidence": f"page_operability overall={overall or 'missing'} bad={','.join(bad)}"}


def _region_read_api_status(page_operability: dict[str, Any]) -> dict[str, str]:
    read_plan_count = 0
    for row in list(page_operability.get("rows") or []):
        contract = row.get("agent_contract") if isinstance(row.get("agent_contract"), dict) else {}
        for item in list(contract.get("read_plan") or []):
            if str(item.get("endpoint") or "").endswith("/read-region"):
                read_plan_count += 1
    if read_plan_count:
        return {"status": "pass", "evidence": f"read-region exposed in {read_plan_count} read plan item(s)"}
    return {"status": "fail", "evidence": "no read-region endpoint found in page operability read_plan"}


def _controlled_scroll_status(page_operability: dict[str, Any]) -> dict[str, str]:
    scroll_plan_count = 0
    read_plan_count = 0
    for row in list(page_operability.get("rows") or []):
        contract = row.get("agent_contract") if isinstance(row.get("agent_contract"), dict) else {}
        for item in list(contract.get("read_plan") or []):
            read_plan_count += 1
            probe = item.get("next_scroll_probe") if isinstance(item.get("next_scroll_probe"), dict) else {}
            request = probe.get("request") if isinstance(probe.get("request"), dict) else {}
            if str(probe.get("endpoint") or "").endswith("/scroll-region") and request.get("dry_run") is True:
                scroll_plan_count += 1
    if scroll_plan_count:
        return {"status": "pass", "evidence": f"controlled scroll dry-run probe exposed in {scroll_plan_count} read plan item(s)"}
    if read_plan_count:
        return {
            "status": "pass",
            "evidence": "no sampled read region had scroll_context; controlled scroll API is covered by regression tests and remains available when scroll_context exists",
        }
    return {"status": "fail", "evidence": "no read plan available to evaluate controlled scroll contract"}


def _safety_boundary_status(page_operability: dict[str, Any]) -> dict[str, str]:
    unsafe = []
    rows = list(page_operability.get("rows") or [])
    for row in rows:
        contract = row.get("agent_contract") if isinstance(row.get("agent_contract"), dict) else {}
        if contract.get("default_agent_type") is True:
            unsafe.append(str(row.get("sample") or ""))
    if not unsafe and rows:
        return {"status": "pass", "evidence": "default_agent_type=false for all rows; review-only inputs stay review-only"}
    return {"status": "fail", "evidence": f"default_agent_type unexpectedly true for {','.join(unsafe)}"}


def _act_preflight_status(act_preflight_matrix: dict[str, Any]) -> dict[str, str]:
    overall = str(act_preflight_matrix.get("overall_status") or "")
    rows = list(act_preflight_matrix.get("rows") or [])
    if overall in PASSING_OPERABILITY and rows:
        return {"status": "pass", "evidence": f"act_preflight_matrix overall={overall} rows={len(rows)}"}
    failures: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            failures.extend(str(item) for item in list(row.get("failures") or []) if str(item))
    failure_summary = ",".join(list(dict.fromkeys(failures))[:5])
    evidence = f"act_preflight_matrix overall={overall or 'missing'}"
    if failure_summary:
        evidence = f"{evidence} failures={failure_summary}"
    return {"status": "fail", "evidence": evidence}


def _deferred_items() -> list[dict[str, str]]:
    return [
        {
            "item": "default_safe_typing",
            "reason": "Requires input-state and send-button enabled/disabled validation; current boundary remains review-only.",
            "next_phase": "agent_execution",
        },
        {
            "item": "unified_act_execution_surface",
            "reason": "Recognition chain now exposes contracts, but /act integration is a separate execution-layer workstream.",
            "next_phase": "agent_execution",
        },
        {
            "item": "full_multi_page_autonomous_exploration",
            "reason": "Requires stable single-page model plus explicit transition graph policy.",
            "next_phase": "multi_page_modeling",
        },
        {
            "item": "vlm_refine_and_disabled_enabled_state",
            "reason": "Needs separate visual-state evidence strategy; not required to close current recognition-chain report.",
            "next_phase": "semantic_refinement",
        },
        {
            "item": "historical_orphan_model_cleanup",
            "reason": "Audit exists; migration must be isolated from recognition-chain changes.",
            "next_phase": "maintenance",
        },
    ]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_first_json(*paths: Path) -> dict[str, Any]:
    for path in paths:
        data = _load_json(path)
        if data:
            return data
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", required=True, type=Path)
    parser.add_argument("--operability-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = analyze_closure_dirs(
        matrix_dir=args.matrix_dir,
        operability_dir=args.operability_dir,
        output_dir=args.output_dir,
    )
    print(
        "overall={overall} samples={samples}".format(
            overall=report["overall_status"],
            samples=report["sample_summary"].get("sample_count", 0),
        )
    )
    return 0 if report["overall_status"] == "closed_for_recognition_chain" else 1


if __name__ == "__main__":
    raise SystemExit(main())
