"""Aggregate recognition, evidence, and readiness into page operability.

The output is a product-boundary report for agents.  It summarizes what the
current page model can support without interpreting the user's task text and
without executing clicks, typing, or sending messages.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


CHAT_CLASSES = {"chat_workspace", "chat_document", "collaboration_inbox"}
AGENT_CONTRACT_SCHEMA_VERSION = "2026-05-28.page-operability.v1"
READ_REGION_PURPOSES = {
    "message_stream",
    "message_thread",
    "chat_area",
    "detail_pane",
    "content_area",
    "content_feed",
    "main_workspace",
    "table",
    "list",
    "dashboard_cards",
    "status_panel",
}
NAVIGATION_REGION_PURPOSES = {
    "app_rail",
    "navigation_list",
    "navigation_and_list",
    "workspace_navigation",
    "inbox_list",
    "item_list",
    "account_list",
}
ACTION_REGION_PURPOSES = {
    "action_panel",
    "toolbar",
    "login_actions",
    "pagination",
    "window_controls",
}
INPUT_REGION_PURPOSES = {
    "composer",
    "search_results_list",
    "search_overlay",
}


def build_page_operability_report(
    *,
    sample_matrix: dict[str, Any],
    recognition_acceptance: dict[str, Any] | None = None,
    chat_readiness: dict[str, Any] | None = None,
    page_evidence_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a page-level operability report from existing diagnostics."""
    acceptance_rows = list((recognition_acceptance or {}).get("rows") or [])
    readiness_rows = list((chat_readiness or {}).get("rows") or [])
    evidence_checks = list((page_evidence_gate or {}).get("checks") or [])
    rows = [
        _build_row(
            row,
            acceptance=_match_row(row, acceptance_rows),
            readiness=_match_row(row, readiness_rows),
            evidence=_match_row(row, evidence_checks),
        )
        for row in list(sample_matrix.get("rows") or [])
        if isinstance(row, dict)
    ]
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("operability_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_type": "page_operability",
        "policy": (
            "Summarizes page regions, read capability, review-only inputs, "
            "controlled probes, and risk flags. It does not execute actions or "
            "promote chat input to default-safe typing."
        ),
        "overall_status": _overall_status(rows),
        "counts": counts,
        "rows": rows,
    }


def analyze_page_operability_dirs(
    *,
    matrix_dir: Path,
    acceptance_dir: Path | None = None,
    readiness_dir: Path | None = None,
    evidence_gate_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Load standard artifact files and write page operability outputs."""
    sample_matrix = _load_json(matrix_dir / "sample_matrix_summary.json")
    recognition_acceptance = _load_json((acceptance_dir or matrix_dir) / "recognition_acceptance.json")
    chat_readiness = _load_json((readiness_dir or matrix_dir) / "chat_readiness_report.json")
    page_evidence_gate = _load_json((evidence_gate_dir or matrix_dir) / "page_evidence_gate.json")
    report = build_page_operability_report(
        sample_matrix=sample_matrix,
        recognition_acceptance=recognition_acceptance,
        chat_readiness=chat_readiness,
        page_evidence_gate=page_evidence_gate,
    )
    write_page_operability_report(report, output_dir or matrix_dir)
    return report


def write_page_operability_report(report: dict[str, Any], output_dir: Path) -> None:
    """Persist JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "page_operability_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Page Operability Report",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Counts: {_format_counts(report.get('counts') or {})}",
        f"- Policy: {report.get('policy', '')}",
        "",
        "| sample | process | class | status | read_regions | review_inputs | controlled_probe | default_type | risks |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(report.get("rows") or []):
        caps = row.get("capabilities") if isinstance(row.get("capabilities"), dict) else {}
        lines.append(
            "| {sample} | {process} | {klass} | {status} | {read_regions} | {review_inputs} | {probe} | {default_type} | {risks} |".format(
                sample=str(row.get("sample") or "").replace("|", "/"),
                process=str(row.get("process_name") or "").replace("|", "/"),
                klass=str(row.get("page_class") or "").replace("|", "/"),
                status=str(row.get("operability_status") or "").replace("|", "/"),
                read_regions=",".join(caps.get("read_regions") or []).replace("|", "/"),
                review_inputs=str(len(caps.get("review_input_candidates") or [])),
                probe="yes" if caps.get("controlled_send_probe") else "no",
                default_type="yes" if caps.get("default_agent_type") else "no",
                risks=",".join(row.get("risk_flags") or []).replace("|", "/"),
            )
        )
    (output_dir / "page_operability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_row(
    row: dict[str, Any],
    *,
    acceptance: dict[str, Any],
    readiness: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    sample = str(row.get("sample") or "")
    process = str(row.get("process_name") or "")
    page_class = _page_class(row)
    acceptance_status = str(acceptance.get("status") or "").strip()
    readiness_status = str(readiness.get("status") or "").strip()
    evidence_decision = str(evidence.get("decision") or "").strip()
    risk_flags = _risk_flags(row, acceptance=acceptance, readiness=readiness, evidence=evidence)
    capabilities = _capabilities(row, page_class=page_class, readiness=readiness)
    if acceptance_status == "fail":
        capabilities["controlled_send_probe"] = False
        status = "unreliable"
    elif readiness_status.startswith("blocked_") or evidence_decision == "stop":
        capabilities["controlled_send_probe"] = False
        status = "usable_with_blocked_actions"
    elif acceptance_status == "warn" or _warning_risks(risk_flags):
        status = "usable_with_warnings"
    else:
        status = "usable"
    return {
        "sample": sample,
        "process_name": process,
        "mode": str(row.get("mode") or ""),
        "page_class": page_class,
        "chat_variant": str(row.get("chat_variant") or ""),
        "operability_status": status,
        "observe_elapsed_ms": int(row.get("observe_elapsed_ms") or 0),
        "recognition_status": acceptance_status or "unknown",
        "readiness_status": readiness_status,
        "evidence_decision": evidence_decision,
        "regions": sorted(_purposes(row)),
        "capabilities": capabilities,
        "agent_contract": _agent_contract(row, page_class=page_class, capabilities=capabilities),
        "risk_flags": risk_flags,
    }


def _capabilities(row: dict[str, Any], *, page_class: str, readiness: dict[str, Any]) -> dict[str, Any]:
    read_regions = sorted(purpose for purpose in _purposes(row) if purpose in READ_REGION_PURPOSES)
    review_inputs = []
    if int(row.get("composer_input_review_count") or 0) > 0:
        review_inputs.append(
            {
                "role": "message_input",
                "actionability": "review",
                "click_point": row.get("composer_input_primary_click_point") or [],
                "safe_to_type": False,
            }
        )
    action_targets = []
    send_bounds = row.get("composer_send_target_bounds")
    if send_bounds and int(row.get("composer_vlm_send_hint_count") or 0) > 0:
        action_targets.append(
            {
                "role": "send_button",
                "bounds": send_bounds,
                "actionability": "controlled_probe",
            }
        )
    search_point = row.get("search_primary_click_point")
    if search_point:
        action_targets.append(
            {
                "role": "search_entry",
                "click_point": search_point,
                "actionability": "review",
            }
        )
    return {
        "read_regions": read_regions,
        "review_input_candidates": review_inputs,
        "action_targets": action_targets,
        "controlled_send_probe": bool(readiness.get("can_run_controlled_send_probe")),
        "default_agent_type": bool(readiness.get("can_default_agent_type")),
        "input_safety": _input_safety(row, readiness=readiness, review_inputs=review_inputs),
        "chat_like": page_class in CHAT_CLASSES,
    }


def _risk_flags(
    row: dict[str, Any],
    *,
    acceptance: dict[str, Any],
    readiness: dict[str, Any],
    evidence: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    flags.extend(str(item) for item in list(acceptance.get("failures") or []) if str(item))
    flags.extend(str(item) for item in list(acceptance.get("warnings") or []) if str(item))
    flags.extend(str(item) for item in list(readiness.get("warnings") or []) if str(item))
    if str(evidence.get("decision") or "") == "stop":
        flags.append("evidence_gate_stop")
    if int(row.get("observe_elapsed_ms") or 0) > 2000:
        flags.append("observe_over_2s")
    if int(row.get("composer_input_review_count") or 0) > 0:
        flags.append("chat_input_review_only")
    if bool(readiness.get("can_default_agent_type")):
        flags.append("unexpected_default_agent_type")
    return list(dict.fromkeys(flags))


def _agent_contract(row: dict[str, Any], *, page_class: str, capabilities: dict[str, Any]) -> dict[str, Any]:
    purposes = _purposes(row)
    read_regions = sorted(purpose for purpose in purposes if purpose in READ_REGION_PURPOSES)
    review_targets = []
    review_targets.extend(capabilities.get("review_input_candidates") or [])
    review_targets.extend(
        target
        for target in list(capabilities.get("action_targets") or [])
        if str(target.get("actionability") or "") != "safe"
    )
    safe_targets = [
        target
        for target in list(capabilities.get("action_targets") or [])
        if str(target.get("actionability") or "") == "safe"
    ]
    return {
        "schema_version": AGENT_CONTRACT_SCHEMA_VERSION,
        "page_class": page_class,
        "navigation_regions": sorted(purpose for purpose in purposes if purpose in NAVIGATION_REGION_PURPOSES),
        "read_regions": read_regions,
        "read_plan": [
            _read_plan_item(region, read_region_contexts=_read_region_contexts(row))
            for region in read_regions
        ],
        "action_regions": sorted(purpose for purpose in purposes if purpose in ACTION_REGION_PURPOSES),
        "input_regions": sorted(purpose for purpose in purposes if purpose in INPUT_REGION_PURPOSES),
        "safe_action_targets": safe_targets,
        "review_required_targets": review_targets,
        "input_safety": capabilities.get("input_safety") or {},
        "controlled_send_probe": bool(capabilities.get("controlled_send_probe")),
        "default_agent_type": bool(capabilities.get("default_agent_type")),
    }


def _input_safety(
    row: dict[str, Any],
    *,
    readiness: dict[str, Any],
    review_inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    safe_input_count = int(row.get("composer_input_safe_count") or 0)
    review_input_count = int(row.get("composer_input_review_count") or len(review_inputs) or 0)
    default_agent_type = bool(readiness.get("can_default_agent_type"))
    safe_to_type = bool(default_agent_type and safe_input_count > 0)
    if safe_to_type:
        return {
            "status": "safe",
            "safe_to_type": True,
            "default_agent_type": True,
            "blockers": [],
            "next_probe_plan": [],
        }
    blockers: list[str] = []
    if review_input_count > 0 and safe_input_count == 0:
        blockers.append("safe_to_type_false")
    if review_input_count > 0:
        blockers.extend(
            [
                "input_click_requires_probe",
                "send_state_requires_probe",
                "readback_required_before_safe_type",
            ]
        )
    status = "review_only" if review_input_count > 0 else "no_input_candidate"
    return {
        "status": status,
        "safe_to_type": False,
        "default_agent_type": False,
        "blockers": list(dict.fromkeys(blockers)),
        "next_probe_plan": _input_safety_next_probe_plan() if review_input_count > 0 else [],
    }


def _input_safety_next_probe_plan() -> list[dict[str, Any]]:
    return [
        {
            "probe": "type_text_preflight",
            "endpoint": "POST /api/v1/act",
            "request": {
                "action": "type_text",
                "candidate_role": "message_input",
                "dry_run": True,
                "execute_confirmed": False,
                "params": {"text": "<probe_text>"},
            },
            "required_evidence": [
                "state_probe_plan.current.input_state",
                "readback_plan.expected_text",
            ],
        },
        {
            "probe": "send_preflight",
            "endpoint": "POST /api/v1/act",
            "request": {
                "action": "send",
                "candidate_role": "send_button",
                "dry_run": True,
                "execute_confirmed": False,
                "params": {"expected_text": "<probe_text>"},
            },
            "required_evidence": [
                "state_probe_plan.current.send_enabled",
                "readback_plan.expected_text",
            ],
        },
    ]


def _read_plan_item(region: str, *, read_region_contexts: dict[str, Any] | None = None) -> dict[str, Any]:
    method = "chat_crop_ocr_readback" if region in {"message_stream", "message_thread", "chat_area"} else "region_text_harvest"
    allow_crop_ocr = method == "chat_crop_ocr_readback"
    scroll_context = _read_plan_scroll_context((read_region_contexts or {}).get(region))
    long_content_reason = "controlled_scroll_available" if scroll_context else "no_scroll_context"
    return {
        "region": region,
        "method": method,
        "endpoint": "POST /api/v1/canvases/{canvas_id}/read-region",
        "request": {
            "region_role": region,
            "include_elements": True,
            "include_ocr": True,
            "allow_crop_ocr": allow_crop_ocr,
        },
        "read_scope": "current_viewport",
        "scroll_context": scroll_context,
        "long_content_strategy": {
            "status": "current_viewport_only",
            "reason": long_content_reason,
        },
        "next_scroll_probe": _next_scroll_probe(region, scroll_context=scroll_context),
        "scroll_supported": bool(scroll_context),
    }


def _read_region_contexts(row: dict[str, Any]) -> dict[str, Any]:
    contexts = row.get("read_region_contexts")
    return contexts if isinstance(contexts, dict) else {}


def _read_plan_scroll_context(context: Any) -> dict[str, Any] | None:
    if not isinstance(context, dict):
        return None
    scroll_context_id = str(context.get("scroll_context_id") or "").strip()
    if not scroll_context_id:
        return None
    return {
        "scroll_context_id": scroll_context_id,
        "region_id": str(context.get("region_id") or "").strip(),
        "scroll_type": str(context.get("scroll_type") or "vertical"),
        "is_virtual": bool(context.get("is_virtual")),
    }


def _next_scroll_probe(region: str, *, scroll_context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not scroll_context:
        return None
    return {
        "endpoint": "POST /api/v1/canvases/{canvas_id}/scroll-region",
        "request": {
            "region_role": region,
            "direction": "down",
            "amount": "page",
            "dry_run": True,
            "execute_confirmed": False,
            "read_after": True,
        },
    }


def _warning_risks(risk_flags: list[str]) -> list[str]:
    """Return risk flags that should degrade the page status."""
    informational = {"chat_input_review_only", "review_only_input"}
    return [flag for flag in risk_flags if flag not in informational]


def _page_class(row: dict[str, Any]) -> str:
    mode = str(row.get("mode") or "").strip()
    process = str(row.get("process_name") or "").strip().lower()
    if mode:
        return mode
    if process in {"weixin.exe", "wechat.exe", "qq.exe"}:
        return "chat_workspace"
    if process in {"feishu.exe", "lark.exe"}:
        return "collaboration_inbox"
    return "unknown"


def _purposes(row: dict[str, Any]) -> set[str]:
    return {str(item).strip() for item in list(row.get("roi_purposes") or []) if str(item).strip()}


def _match_row(source: dict[str, Any], rows: list[Any]) -> dict[str, Any]:
    sample = str(source.get("sample") or "").lower()
    process = str(source.get("process_name") or "").lower()
    for item in rows:
        if isinstance(item, dict) and sample and str(item.get("sample") or "").lower() == sample:
            return item
    for item in rows:
        if isinstance(item, dict) and process and str(item.get("process_name") or item.get("app") or "").lower() == process:
            return item
    return {}


def _overall_status(rows: list[dict[str, Any]]) -> str:
    statuses = {str(row.get("operability_status") or "") for row in rows}
    if "unreliable" in statuses:
        return "unreliable"
    if "usable_with_blocked_actions" in statuses:
        return "usable_with_blocked_actions"
    if "usable_with_warnings" in statuses:
        return "usable_with_warnings"
    if "usable" in statuses:
        return "usable"
    return "empty"


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) if counts else "none"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--acceptance-dir", type=Path)
    parser.add_argument("--readiness-dir", type=Path)
    parser.add_argument("--evidence-gate-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = analyze_page_operability_dirs(
        matrix_dir=args.matrix_dir,
        acceptance_dir=args.acceptance_dir,
        readiness_dir=args.readiness_dir,
        evidence_gate_dir=args.evidence_gate_dir,
        output_dir=args.output_dir,
    )
    print(
        "overall={overall} rows={rows}".format(
            overall=report["overall_status"],
            rows=len(report["rows"]),
        )
    )
    return 1 if report["overall_status"] == "unreliable" else 0


if __name__ == "__main__":
    raise SystemExit(main())
