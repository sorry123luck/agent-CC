"""Analyze /act preflight responses as an agent operability matrix.

This report is deliberately about protocol completeness, not about whether the
desktop action actually ran.  Review-only typing and disabled sends can still be
usable for an agent when the API returns enough state/readback proof for the
caller to keep the action blocked or to run a later confirmed probe.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPORT_JSON = "act_preflight_matrix_report.json"
REPORT_MD = "act_preflight_matrix_report.md"


def build_act_preflight_matrix_report(matrix: dict[str, Any]) -> dict[str, Any]:
    """Build a normalized report from saved /act preflight responses."""
    rows = [_build_row(row) for row in list(matrix.get("rows") or []) if isinstance(row, dict)]
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unusable")
        counts[status] = counts.get(status, 0) + 1
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_type": "act_preflight_matrix",
        "policy": (
            "Evaluates /act preflight protocol completeness only. It does not "
            "execute desktop input, does not promote review-only inputs to "
            "safe_to_type, and treats safety blocks as usable only when the "
            "response includes state probe and readback plans."
        ),
        "overall_status": _overall(rows),
        "counts": counts,
        "rows": rows,
    }


def analyze_act_preflight_matrix_dirs(*, input_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Load act_preflight_matrix.json from a directory and write reports."""
    matrix = _load_json(input_dir / "act_preflight_matrix.json")
    report = build_act_preflight_matrix_report(matrix)
    write_act_preflight_matrix_report(report, output_dir or input_dir)
    return report


def write_act_preflight_matrix_report(report: dict[str, Any], output_dir: Path) -> None:
    """Persist JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_JSON).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Act Preflight Matrix Report",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Counts: {_format_counts(report.get('counts') or {})}",
        f"- Policy: {report.get('policy', '')}",
        "",
        "| sample | process | status | type_text | send | click | scroll | input_state | send_enabled | warnings | failures |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(report.get("rows") or []):
        type_text = row.get("type_text") if isinstance(row.get("type_text"), dict) else {}
        send = row.get("send") if isinstance(row.get("send"), dict) else {}
        click = row.get("click") if isinstance(row.get("click"), dict) else {}
        scroll = row.get("scroll") if isinstance(row.get("scroll"), dict) else {}
        state = type_text.get("input_state") or send.get("input_state") or ""
        send_enabled = type_text.get("send_enabled")
        if send_enabled is None:
            send_enabled = send.get("send_enabled")
        lines.append(
            "| {sample} | {process} | {status} | {type_text} | {send} | {click} | {scroll} | {state} | {send_enabled} | {warnings} | {failures} |".format(
                sample=_md(row.get("sample")),
                process=_md(row.get("process_name")),
                status=_md(row.get("status")),
                type_text=_action_cell(type_text),
                send=_action_cell(send),
                click=_action_cell(click),
                scroll=_action_cell(scroll),
                state=_md(state),
                send_enabled=_md("" if send_enabled is None else str(send_enabled).lower()),
                warnings=_md(",".join(row.get("warnings") or [])),
                failures=_md(",".join(row.get("failures") or [])),
            )
        )
    (output_dir / REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_row(row: dict[str, Any]) -> dict[str, Any]:
    actions = row.get("actions") if isinstance(row.get("actions"), dict) else row
    type_text = _evaluate_action("type_text", actions.get("type_text"))
    send = _evaluate_action("send", actions.get("send"))
    click = _evaluate_action("click", actions.get("click"))
    scroll = _evaluate_action("scroll", actions.get("scroll"))
    action_reports = [item for item in (type_text, send, click, scroll) if item.get("present")]
    failures = _dedupe([failure for item in action_reports for failure in list(item.get("failures") or [])])
    if not action_reports:
        failures.append("missing_action_preflight_responses")
    warnings = _dedupe([warning for item in action_reports for warning in list(item.get("warnings") or [])])
    status = "unusable" if failures or not action_reports else _row_usable_status(action_reports)
    return {
        "sample": str(row.get("sample") or ""),
        "process_name": str(row.get("process_name") or row.get("app") or ""),
        "status": status,
        "default_agent_type": False,
        "type_text": type_text,
        "send": send,
        "click": click,
        "scroll": scroll,
        "warnings": warnings,
        "failures": failures,
    }


def _evaluate_action(action: str, response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {
            "present": False,
            "protocol_complete": False,
            "safe_boundary": "missing",
            "warnings": [],
            "failures": [],
        }
    plan = response.get("action_plan") if isinstance(response.get("action_plan"), dict) else {}
    execution_policy = plan.get("execution_policy") if isinstance(plan.get("execution_policy"), dict) else {}
    verification = plan.get("verification_plan") if isinstance(plan.get("verification_plan"), dict) else {}
    readback_plan = verification.get("readback_plan") if isinstance(verification.get("readback_plan"), dict) else {}
    state_probe = verification.get("state_probe_plan") if isinstance(verification.get("state_probe_plan"), dict) else {}
    current = state_probe.get("current") if isinstance(state_probe.get("current"), dict) else {}
    warnings = [str(item) for item in list(response.get("warnings") or []) if str(item)]
    failures: list[str] = []
    if not execution_policy:
        failures.append(f"{action}_missing_execution_policy")
    else:
        failures.extend(_execution_policy_failures(action, execution_policy))
    if action in {"type_text", "send"} and not state_probe:
        failures.append(f"{action}_missing_state_probe_plan")
    if action in {"type_text", "send", "click", "double_click", "scroll"} and not readback_plan:
        failures.append(f"{action}_missing_readback_plan")
    if action in {"click", "double_click", "scroll"}:
        if verification.get("observe_after") is not True:
            failures.append(f"{action}_missing_observe_after")
        if verification.get("diff_after") is not True:
            failures.append(f"{action}_missing_diff_after")
        if verification.get("readback_after") is not True:
            failures.append(f"{action}_missing_readback_after")
        if readback_plan and readback_plan.get("after_observe") is not True:
            failures.append(f"{action}_readback_not_after_observe")
    expected_text = str(readback_plan.get("expected_text") or "")
    if action == "send" and not expected_text.strip():
        failures.append("send_missing_expected_text")
    if action == "send" and "send_requires_send_candidate" in warnings:
        failures.append("send_candidate_not_accepted_by_act")
    protocol_complete = not failures
    return {
        "present": True,
        "execution_result": str(response.get("execution_result") or ""),
        "action_level": str(plan.get("action_level") or ""),
        "protocol_complete": protocol_complete,
        "safe_boundary": _safe_boundary(action, warnings=warnings, current=current),
        "execution_policy": execution_policy,
        "input_state": str(current.get("input_state") or ""),
        "send_enabled": current.get("send_enabled"),
        "has_readback_plan": bool(readback_plan),
        "has_state_probe_plan": bool(state_probe),
        "expected_text": expected_text,
        "before_requirements": list(state_probe.get("before_requirements") or []) if state_probe else [],
        "after_expectations": list(state_probe.get("after_expectations") or []) if state_probe else [],
        "warnings": warnings,
        "failures": failures,
    }


def _execution_policy_failures(action: str, execution_policy: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if action in {"click", "double_click", "scroll"}:
        if execution_policy.get("confirmed_execution_would_touch_desktop") is not True:
            failures.append(f"{action}_execution_policy_missing_confirmed_desktop_touch")
        if execution_policy.get("requires_execute_confirmed") is not True:
            failures.append(f"{action}_execution_policy_missing_confirmation_requirement")
        if execution_policy.get("enabled_for_execution") is not True:
            failures.append(f"{action}_execution_policy_not_enabled_for_controlled_execution")
        if execution_policy.get("executes_desktop_input") is not False:
            failures.append(f"{action}_preflight_policy_must_not_execute_desktop_input")
    if action in {"type_text", "send"}:
        if execution_policy.get("enabled_for_execution") is not False:
            failures.append(f"{action}_execution_policy_must_not_enable_execution")
        if execution_policy.get("executes_desktop_input") is not False:
            failures.append(f"{action}_execution_policy_must_not_execute_desktop_input")
    return failures


def _safe_boundary(action: str, *, warnings: list[str], current: dict[str, Any]) -> str:
    warning_set = set(warnings)
    if action in {"click", "double_click", "scroll"} and "act_execution_adapter_pending" in warning_set:
        return "controlled_requires_confirmation"
    if action == "type_text" and "type_text_requires_safe_to_type" in warning_set:
        return "review_only"
    if action == "send" and "send_button_disabled" in warning_set:
        return "disabled_send_blocked"
    if action == "send" and current.get("send_enabled") is False:
        return "disabled_send_blocked"
    if warning_set:
        return "blocked_or_review"
    return "planned"


def _row_usable_status(action_reports: list[dict[str, Any]]) -> str:
    if any(str(item.get("execution_result") or "") == "blocked" for item in action_reports):
        return "usable_with_blocked_actions"
    if any(item.get("warnings") for item in action_reports):
        return "usable_with_warnings"
    return "usable"


def _overall(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "empty"
    statuses = {str(row.get("status") or "") for row in rows}
    if "unusable" in statuses:
        return "unusable"
    if "usable_with_blocked_actions" in statuses:
        return "usable_with_blocked_actions"
    if "usable_with_warnings" in statuses:
        return "usable_with_warnings"
    return "usable"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _action_cell(action_report: dict[str, Any]) -> str:
    if not action_report or not action_report.get("present"):
        return "missing"
    complete = "complete" if action_report.get("protocol_complete") else "incomplete"
    boundary = str(action_report.get("safe_boundary") or "")
    return _md(f"{complete}:{boundary}")


def _md(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = analyze_act_preflight_matrix_dirs(input_dir=args.input_dir, output_dir=args.output_dir)
    print(f"overall={report.get('overall_status')} rows={len(report.get('rows') or [])}")
    return 1 if report.get("overall_status") == "unusable" else 0


if __name__ == "__main__":
    raise SystemExit(main())
