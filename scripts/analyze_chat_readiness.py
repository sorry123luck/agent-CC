"""Aggregate chat action, send, and readback evidence into readiness status.

This report is intentionally conservative. It never upgrades chat input
candidates to safe default typing; it only decides whether further controlled
probes are justified.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


APP_ALIASES = {
    "qq": ("qq", "qq.exe"),
    "qq.exe": ("qq", "qq.exe"),
    "wechat": ("wechat", "weixin", "weixin.exe", "wechat.exe"),
    "weixin": ("wechat", "weixin", "weixin.exe", "wechat.exe"),
    "weixin.exe": ("wechat", "weixin", "weixin.exe", "wechat.exe"),
    "feishu": ("feishu", "feishu.exe", "lark", "lark.exe"),
    "feishu.exe": ("feishu", "feishu.exe", "lark", "lark.exe"),
    "lark": ("feishu", "feishu.exe", "lark", "lark.exe"),
}


def build_chat_readiness(
    *,
    action_plan: dict[str, Any],
    send_report: dict[str, Any] | None = None,
    snapshot_acceptance: dict[str, Any] | None = None,
    evidence_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build readiness rows from chat action plan and evidence reports."""
    send_rows = list((send_report or {}).get("rows") or [])
    snapshot_rows = list((snapshot_acceptance or {}).get("rows") or [])
    evidence_checks = list((evidence_gate or {}).get("checks") or [])
    gate_type = str((evidence_gate or {}).get("gate_type") or "").strip().lower()
    rows = [
        _evaluate_action_row(
            row,
            send_rows=send_rows,
            snapshot_rows=snapshot_rows,
            evidence_checks=evidence_checks,
            gate_type=gate_type,
        )
        for row in list(action_plan.get("rows") or [])
        if isinstance(row, dict)
    ]
    counts = {"controlled_ready": 0, "readonly_ready": 0, "blocked": 0}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if str(row["status"]).startswith("blocked_"):
            counts["blocked"] = counts.get("blocked", 0) + 1
    overall = _overall(rows)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_status": overall,
        "counts": counts,
        "rows": rows,
    }


def analyze_chat_readiness_dirs(
    *,
    action_plan_dir: Path,
    send_report_dir: Path | None = None,
    snapshot_acceptance_dir: Path | None = None,
    evidence_gate_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Load report files from directories and write readiness output."""
    action_plan = _load_json(action_plan_dir / "chat_action_plan.json")
    send_report = _load_json((send_report_dir or action_plan_dir) / "chat_send_probe_report.json")
    snapshot_acceptance = _load_json(
        (snapshot_acceptance_dir or action_plan_dir) / "chat_readback_snapshot_acceptance.json"
    )
    evidence_gate = _load_json((evidence_gate_dir or action_plan_dir) / "page_evidence_gate.json")
    report = build_chat_readiness(
        action_plan=action_plan,
        send_report=send_report,
        snapshot_acceptance=snapshot_acceptance,
        evidence_gate=evidence_gate,
    )
    write_readiness_report(report, output_dir or action_plan_dir)
    return report


def write_readiness_report(report: dict[str, Any], output_dir: Path) -> None:
    """Persist JSON and Markdown readiness reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chat_readiness_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Chat Readiness Report",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Counts: {_format_counts(report.get('counts') or {})}",
        "",
        "| sample | process | status | readonly | controlled_send | default_type | warnings |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(report.get("rows") or []):
        lines.append(
            "| {sample} | {process} | {status} | {readonly} | {controlled} | {default_type} | {warnings} |".format(
                sample=str(row.get("sample") or "").replace("|", "/"),
                process=str(row.get("process_name") or "").replace("|", "/"),
                status=str(row.get("status") or "").replace("|", "/"),
                readonly="yes" if row.get("can_run_readonly_snapshot") else "no",
                controlled="yes" if row.get("can_run_controlled_send_probe") else "no",
                default_type="yes" if row.get("can_default_agent_type") else "no",
                warnings=",".join(row.get("warnings") or []).replace("|", "/"),
            )
        )
    (output_dir / "chat_readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_action_row(
    row: dict[str, Any],
    *,
    send_rows: list[Any],
    snapshot_rows: list[Any],
    evidence_checks: list[Any] | None = None,
    gate_type: str = "",
) -> dict[str, Any]:
    app_key = _app_key(str(row.get("process_name") or row.get("app") or ""))
    action_ready = str(row.get("status") or "") == "ready"
    action_status = str(row.get("status") or "")
    snapshot = _matching_snapshot_row(row, snapshot_rows, app_key=app_key)
    send = _matching_send_row(row, send_rows, app_key=app_key)
    snapshot_pass = bool(snapshot) and str(snapshot.get("status") or "") == "pass"
    send_pass = bool(send) and str(send.get("status") or "") == "pass"
    warnings: list[str] = []
    if row.get("safe_to_type") is not True:
        warnings.append("review_only_input")
    evidence_check = _matching_evidence_check(row, evidence_checks or [], app_key=app_key)
    if evidence_check and str(evidence_check.get("decision") or "").lower() != "proceed":
        action_status = "blocked_evidence_mismatch" if gate_type == "page_evidence" else "blocked_target_mismatch"
        warning = "evidence_gate_stop" if gate_type == "page_evidence" else "target_identity_check_stop"
        warnings.append(warning)
        reason = str(evidence_check.get("reason") or "")
        if reason:
            warnings.append(reason)
        return {
            "sample": str(row.get("sample") or ""),
            "process_name": str(row.get("process_name") or ""),
            "status": action_status,
            "safe_to_type": False,
            "can_run_readonly_snapshot": False,
            "can_run_controlled_send_probe": False,
            "can_default_agent_type": False,
            "action_plan_status": str(row.get("status") or ""),
            "snapshot_status": str(snapshot.get("status") or "") if snapshot else "",
            "send_probe_status": str(send.get("status") or "") if send else "",
            "evidence_gate_decision": str(evidence_check.get("decision") or ""),
            "input_click": row.get("input_click") or [],
            "send_click": row.get("send_click") or [],
            "warnings": list(dict.fromkeys(warnings)),
        }
    if action_status in {"blocked_target_mismatch", "blocked_evidence_mismatch"}:
        warnings.extend(str(item) for item in list(row.get("reasons") or []) if str(item))
        default_warning = "evidence_gate_stop" if action_status == "blocked_evidence_mismatch" else "target_identity_check_stop"
        if not any(item == default_warning for item in warnings):
            warnings.append(default_warning)
        if action_status == "blocked_target_mismatch" and not any(item == "target_identity_check_stop" for item in warnings):
            warnings.append("target_identity_check_stop")
        return {
            "sample": str(row.get("sample") or ""),
            "process_name": str(row.get("process_name") or ""),
            "status": action_status,
            "safe_to_type": False,
            "can_run_readonly_snapshot": False,
            "can_run_controlled_send_probe": False,
            "can_default_agent_type": False,
            "action_plan_status": action_status,
            "snapshot_status": str(snapshot.get("status") or "") if snapshot else "",
            "send_probe_status": "",
            "evidence_gate_decision": "",
            "input_click": row.get("input_click") or [],
            "send_click": row.get("send_click") or [],
            "warnings": list(dict.fromkeys(warnings)),
        }
    if not action_ready:
        warnings.append("action_plan_not_ready")
    if not snapshot_pass:
        warnings.append("missing_snapshot_pass")
    if not send_pass:
        warnings.append("missing_send_probe_pass")
    can_readonly = action_ready and snapshot_pass
    can_controlled = action_ready and snapshot_pass and send_pass
    status = "controlled_ready" if can_controlled else "readonly_ready" if can_readonly else "blocked"
    return {
        "sample": str(row.get("sample") or ""),
        "process_name": str(row.get("process_name") or ""),
        "status": status,
        "safe_to_type": False,
        "can_run_readonly_snapshot": can_readonly,
        "can_run_controlled_send_probe": can_controlled,
        "can_default_agent_type": False,
        "action_plan_status": action_status,
        "snapshot_status": str(snapshot.get("status") or "") if snapshot else "",
        "send_probe_status": str(send.get("status") or "") if send else "",
        "evidence_gate_decision": str(evidence_check.get("decision") or "") if evidence_check else "",
        "input_click": row.get("input_click") or [],
        "send_click": row.get("send_click") or [],
        "warnings": list(dict.fromkeys(warnings)),
    }


def _matching_snapshot_row(action_row: dict[str, Any], snapshot_rows: list[Any], *, app_key: str) -> dict[str, Any]:
    action_sample = str(action_row.get("sample") or "").lower()
    for item in snapshot_rows:
        if not isinstance(item, dict):
            continue
        if action_sample and str(item.get("sample") or "").lower() == action_sample:
            return item
    for item in snapshot_rows:
        if isinstance(item, dict) and _app_key(str(item.get("process_name") or "")) == app_key:
            return item
    return {}


def _matching_send_row(action_row: dict[str, Any], send_rows: list[Any], *, app_key: str) -> dict[str, Any]:
    action_sample = str(action_row.get("sample") or "").lower()
    for item in send_rows:
        if not isinstance(item, dict):
            continue
        if action_sample and action_sample == str(item.get("sample") or "").lower():
            return item
    for item in send_rows:
        if isinstance(item, dict) and _app_key(str(item.get("app") or item.get("process_name") or "")) == app_key:
            return item
    return {}


def _matching_evidence_check(action_row: dict[str, Any], evidence_checks: list[Any], *, app_key: str) -> dict[str, Any]:
    action_sample = str(action_row.get("sample") or "").lower()
    for item in evidence_checks:
        if not isinstance(item, dict):
            continue
        if action_sample and action_sample == str(item.get("sample") or "").lower():
            return item
    for item in evidence_checks:
        if not isinstance(item, dict):
            continue
        item_process = str(item.get("process_name") or item.get("app") or "")
        if item_process and _app_key(item_process) == app_key:
            return item
    return {}


def _app_key(value: str) -> str:
    normalized = value.strip().lower()
    aliases = APP_ALIASES.get(normalized)
    if aliases:
        return aliases[0]
    return normalized.removesuffix(".exe")


def _overall(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "blocked"
    if any(row["status"] == "controlled_ready" for row in rows):
        return "controlled_ready"
    if any(row["status"] == "readonly_ready" for row in rows):
        return "readonly_ready"
    return "blocked"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-plan-dir", type=Path, required=True)
    parser.add_argument("--send-report-dir", type=Path)
    parser.add_argument("--snapshot-acceptance-dir", type=Path)
    parser.add_argument("--evidence-gate-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = analyze_chat_readiness_dirs(
        action_plan_dir=args.action_plan_dir,
        send_report_dir=args.send_report_dir,
        snapshot_acceptance_dir=args.snapshot_acceptance_dir,
        evidence_gate_dir=args.evidence_gate_dir,
        output_dir=args.output_dir,
    )
    print(
        "overall={overall} controlled_ready={controlled} readonly_ready={readonly} blocked={blocked}".format(
            overall=report["overall_status"],
            controlled=report["counts"].get("controlled_ready", 0),
            readonly=report["counts"].get("readonly_ready", 0),
            blocked=report["counts"].get("blocked", 0),
        )
    )
    return 0 if report["overall_status"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
