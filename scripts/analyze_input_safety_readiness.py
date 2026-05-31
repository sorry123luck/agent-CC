"""Analyze whether review-only input candidates have complete safety probes.

This report does not promote any candidate to safe typing.  It translates the
page operability input_safety contract plus /act preflight evidence into a
machine-readable readiness state for the input-safety workstream.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPORT_JSON = "input_safety_readiness_report.json"
REPORT_MD = "input_safety_readiness_report.md"


def build_input_safety_readiness_report(
    *,
    page_operability: dict[str, Any],
    act_preflight_matrix: dict[str, Any],
    send_probe_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    send_probe_rows = list((send_probe_report or {}).get("rows") or [])
    rows = [
        _build_row(
            row,
            act_row=_match_row(row, list(act_preflight_matrix.get("rows") or [])),
            send_probe_row=_match_send_probe_row(row, send_probe_rows),
        )
        for row in list(page_operability.get("rows") or [])
        if isinstance(row, dict)
    ]
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "incomplete")
        counts[status] = counts.get(status, 0) + 1
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_type": "input_safety_readiness",
        "policy": (
            "Evaluates input safety probe evidence only. It never upgrades "
            "safe_to_type and never executes typing or sending."
        ),
        "overall_status": _overall(rows),
        "counts": counts,
        "rows": rows,
    }


def analyze_input_safety_readiness_dirs(*, input_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    page_operability = _load_json(input_dir / "page_operability_report.json")
    act_preflight = _load_json(input_dir / "act_preflight_matrix_report.json")
    if not act_preflight:
        act_preflight = _load_json(input_dir / "act_preflight" / "analysis" / "act_preflight_matrix_report.json")
    send_probe_report = _load_json(input_dir / "chat_send_probe_report.json")
    report = build_input_safety_readiness_report(
        page_operability=page_operability,
        act_preflight_matrix=act_preflight,
        send_probe_report=send_probe_report,
    )
    write_input_safety_readiness_report(report, output_dir or input_dir)
    return report


def write_input_safety_readiness_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_JSON).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Input Safety Readiness Report",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Counts: {_format_counts(report.get('counts') or {})}",
        f"- Policy: {report.get('policy', '')}",
        "",
        "| sample | process | status | type_text | send | controlled_probe | safe_to_type | blockers |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(report.get("rows") or []):
        readiness = row.get("probe_readiness") if isinstance(row.get("probe_readiness"), dict) else {}
        lines.append(
            "| {sample} | {process} | {status} | {type_text} | {send} | {probe} | {safe} | {blockers} |".format(
                sample=_md(row.get("sample")),
                process=_md(row.get("process_name")),
                status=_md(row.get("status")),
                type_text=_md(readiness.get("type_text_preflight")),
                send=_md(readiness.get("send_preflight")),
                probe=_md(row.get("controlled_probe_status")),
                safe=_md(str(bool(row.get("safe_to_type"))).lower()),
                blockers=_md(",".join(row.get("remaining_blockers") or [])),
            )
        )
    (output_dir / REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_row(row: dict[str, Any], *, act_row: dict[str, Any], send_probe_row: dict[str, Any]) -> dict[str, Any]:
    contract = row.get("agent_contract") if isinstance(row.get("agent_contract"), dict) else {}
    input_safety = contract.get("input_safety") if isinstance(contract.get("input_safety"), dict) else {}
    probe_readiness = {
        "type_text_preflight": _probe_status(act_row.get("type_text") if isinstance(act_row, dict) else {}),
        "send_preflight": _probe_status(act_row.get("send") if isinstance(act_row, dict) else {}),
    }
    controlled_probe_status = _controlled_probe_status(send_probe_row)
    remaining = _remaining_blockers(
        input_safety=input_safety,
        act_row=act_row,
        probe_readiness=probe_readiness,
        controlled_probe_status=controlled_probe_status,
    )
    if remaining in ([], ["safe_to_type_false"]):
        status = "probe_ready_review_only"
    elif remaining == ["safe_to_type_false", "safe_type_upgrade_policy_pending"]:
        status = "controlled_probe_verified_review_only"
    else:
        status = "incomplete"
    safe_to_type = input_safety.get("safe_to_type") is True
    return {
        "sample": str(row.get("sample") or ""),
        "process_name": str(row.get("process_name") or ""),
        "page_class": str(row.get("page_class") or ""),
        "status": "safe" if safe_to_type else status,
        "safe_to_type": safe_to_type,
        "can_upgrade_safe_to_type": False,
        "input_safety_status": str(input_safety.get("status") or ""),
        "probe_readiness": probe_readiness,
        "controlled_probe_status": controlled_probe_status,
        "safe_type_upgrade_gate": _safe_type_upgrade_gate(
            safe_to_type=safe_to_type,
            probe_readiness=probe_readiness,
            controlled_probe_status=controlled_probe_status,
        ),
        "remaining_blockers": [] if safe_to_type else remaining,
    }


def _probe_status(action: Any) -> str:
    if not isinstance(action, dict) or not action.get("present"):
        return "missing"
    if action.get("has_state_probe_plan") is not True:
        return "missing_state_probe"
    if action.get("has_readback_plan") is not True:
        return "missing_readback"
    if action.get("protocol_complete") is not True:
        return "incomplete"
    return "complete"


def _remaining_blockers(
    *,
    input_safety: dict[str, Any],
    act_row: dict[str, Any],
    probe_readiness: dict[str, str],
    controlled_probe_status: str,
) -> list[str]:
    blockers: list[str] = []
    if input_safety.get("safe_to_type") is not True:
        blockers.append("safe_to_type_false")
    if probe_readiness.get("type_text_preflight") != "complete":
        blockers.append(f"type_text_preflight_{probe_readiness.get('type_text_preflight') or 'missing'}")
    if probe_readiness.get("send_preflight") != "complete":
        blockers.append(f"send_preflight_{probe_readiness.get('send_preflight') or 'missing'}")
    if isinstance(act_row, dict):
        blockers.extend(str(item) for item in list(act_row.get("failures") or []) if str(item))
    if controlled_probe_status == "pass" and blockers == ["safe_to_type_false"]:
        blockers.append("safe_type_upgrade_policy_pending")
    return list(dict.fromkeys(blockers))


def _overall(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "empty"
    statuses = {str(row.get("status") or "") for row in rows}
    if "incomplete" in statuses:
        return "incomplete"
    if "controlled_probe_verified_review_only" in statuses:
        if statuses <= {"controlled_probe_verified_review_only", "safe"}:
            return "controlled_probe_verified_review_only"
        return "probe_ready_review_only"
    if "probe_ready_review_only" in statuses:
        return "probe_ready_review_only"
    if statuses == {"safe"}:
        return "safe"
    return "empty"


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


def _match_send_probe_row(source: dict[str, Any], rows: list[Any]) -> dict[str, Any]:
    sample = str(source.get("sample") or "").lower()
    process = str(source.get("process_name") or "").lower()
    process_key = process.removesuffix(".exe")
    for item in rows:
        if not isinstance(item, dict):
            continue
        app = str(item.get("app") or "").lower()
        if app and (app in sample or app == process or app == process_key):
            return item
    return {}


def _controlled_probe_status(row: dict[str, Any]) -> str:
    if not row:
        return "not_run"
    if row.get("status") == "pass" and row.get("text_observed_after") is True:
        return "pass"
    if row.get("status") == "warn":
        return "warn"
    if row.get("status") == "fail":
        return "fail"
    return "unknown"


def _safe_type_upgrade_gate(
    *,
    safe_to_type: bool,
    probe_readiness: dict[str, str],
    controlled_probe_status: str,
) -> dict[str, Any]:
    satisfied: list[str] = []
    missing: list[str] = []
    if probe_readiness.get("type_text_preflight") == "complete":
        satisfied.append("type_text_preflight_complete")
    else:
        missing.append("type_text_preflight_complete")
    if probe_readiness.get("send_preflight") == "complete":
        satisfied.append("send_preflight_complete")
    else:
        missing.append("send_preflight_complete")
    if controlled_probe_status == "pass":
        satisfied.append("controlled_probe_passed")
    else:
        missing.append("controlled_probe_passed")
    if safe_to_type:
        satisfied.extend(
            [
                "explicit_safe_type_policy",
                "state_template_input_transition_verified",
                "multi_sample_stability_verified",
            ]
        )
        decision = "allow"
    else:
        missing.extend(
            [
                "explicit_safe_type_policy",
                "state_template_input_transition_verified",
                "multi_sample_stability_verified",
            ]
        )
        decision = "blocked"
    return {
        "decision": decision,
        "satisfied": list(dict.fromkeys(satisfied)),
        "missing": list(dict.fromkeys(missing)),
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) if counts else "none"


def _md(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = analyze_input_safety_readiness_dirs(input_dir=args.input_dir, output_dir=args.output_dir)
    print(f"overall={report.get('overall_status')} rows={len(report.get('rows') or [])}")
    return 1 if report.get("overall_status") == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
