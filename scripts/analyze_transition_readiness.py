"""Analyze observed page transitions for agent-facing multi-page readiness.

This script is intentionally read-only. It summarizes existing search probe
artifacts and transition-memory API snapshots into an agent contract. It does
not execute clicks, typing, sends, or any other desktop action.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


CONTROLLED_ACTIONS = {"click", "double_click", "scroll"}
REQUIRED_VERIFICATION = ["observe_after", "diff_after", "readback_after"]


def build_transition_readiness_report(
    *,
    search_probe: dict[str, Any] | None = None,
    transition_graph: dict[str, Any] | None = None,
    control_transition_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only transition readiness report."""
    rows: list[dict[str, Any]] = []
    rows.extend(_search_probe_rows(search_probe or {}))
    rows.extend(_transition_graph_rows(transition_graph or {}))
    rows.extend(_control_transition_rows(control_transition_graph or {}))
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_type": "transition_readiness",
        "policy": (
            "Read existing transition evidence only. Agents may inspect this "
            "memory by default, but autonomous click/type/send remains blocked "
            "unless a later execution policy explicitly confirms it."
        ),
        "overall_status": _overall_status(rows),
        "counts": counts,
        "rows": rows,
    }


def analyze_transition_readiness_dirs(
    *,
    search_probe_dir: Path | None = None,
    transition_graph_file: Path | None = None,
    control_transition_graph_file: Path | None = None,
    output_dir: Path,
) -> dict[str, Any]:
    """Load known artifact files and write transition readiness outputs."""
    search_probe = _load_json((search_probe_dir / "search_probe_report.json") if search_probe_dir else None)
    transition_graph = _load_json(transition_graph_file)
    control_transition_graph = _load_json(control_transition_graph_file)
    report = build_transition_readiness_report(
        search_probe=search_probe,
        transition_graph=transition_graph,
        control_transition_graph=control_transition_graph,
    )
    write_transition_readiness_report(report, output_dir)
    return report


def write_transition_readiness_report(report: dict[str, Any], output_dir: Path) -> None:
    """Write JSON and Markdown transition readiness reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transition_readiness_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Transition Readiness Report",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Counts: {_format_counts(report.get('counts') or {})}",
        f"- Policy: {report.get('policy', '')}",
        "",
        "| sample | process | source | status | path | boundary | autonomous | blockers |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in list(report.get("rows") or []):
        lines.append(
            "| {sample} | {process} | {source} | {status} | {path} | {boundary} | {autonomous} | {blockers} |".format(
                sample=_md(row.get("sample")),
                process=_md(row.get("process_name")),
                source=_md(row.get("source")),
                status=_md(row.get("status")),
                path=_md(row.get("transition_path")),
                boundary=_md(row.get("execution_boundary")),
                autonomous="yes" if row.get("autonomous_execution") else "no",
                blockers=",".join(row.get("blockers") or []).replace("|", "/"),
            )
        )
    (output_dir / "transition_readiness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _search_probe_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in list(report.get("rows") or []):
        if not isinstance(row, dict):
            continue
        transition = row.get("state_transition") if isinstance(row.get("state_transition"), dict) else {}
        path = str(transition.get("path") or "")
        blockers = _search_transition_blockers(row, transition=transition, path=path)
        status = "observed_roundtrip" if not blockers else "incomplete_transition"
        rows.append(
            {
                "sample": str(row.get("sample") or ""),
                "process_name": str(row.get("process_name") or ""),
                "source": "search_probe",
                "status": status,
                "transition_path": path,
                "query": str(row.get("query") or ""),
                "before_canvas_id": str(row.get("before_canvas_id") or ""),
                "after_canvas_id": str(row.get("after_canvas_id") or ""),
                "selected_canvas_id": str(row.get("selected_canvas_id") or ""),
                "execution_boundary": "observed_only",
                "autonomous_execution": False,
                "required_verification": REQUIRED_VERIFICATION,
                "blockers": blockers,
                "agent_contract": _agent_contract(),
            }
        )
    return rows


def _transition_graph_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(report.get("transitions") or []):
        if not isinstance(item, dict):
            continue
        from_class = str(item.get("from_page_class") or "")
        to_class = str(item.get("to_page_class") or "")
        action = str(item.get("trigger_action") or "")
        blockers = _identity_blockers(from_class=from_class, to_class=to_class)
        if int(item.get("success_count") or 0) <= 0:
            blockers.append("no_successful_observation")
        rows.append(
            {
                "sample": str(item.get("trigger_candidate_key") or ""),
                "process_name": "",
                "source": "transition_graph",
                "status": "transition_observed" if not blockers else "incomplete_transition",
                "transition_path": _path(from_class, to_class),
                "candidate_key": str(item.get("trigger_candidate_key") or ""),
                "action_type": action,
                "execution_boundary": "observed_only",
                "autonomous_execution": False,
                "required_verification": REQUIRED_VERIFICATION,
                "blockers": blockers,
                "agent_contract": _agent_contract(),
            }
        )
    return rows


def _control_transition_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(report.get("transitions") or []):
        if not isinstance(item, dict):
            continue
        from_class = str(item.get("from_page_class") or "")
        to_class = str(item.get("to_page_class") or "")
        action = str(item.get("action_type") or "")
        blockers = _identity_blockers(from_class=from_class, to_class=to_class)
        if action not in CONTROLLED_ACTIONS:
            blockers.append("unsupported_action_type")
        if int(item.get("success_count") or 0) <= 0:
            blockers.append("no_successful_observation")
        if not str(item.get("canvas_id_before") or ""):
            blockers.append("missing_canvas_id_before")
        if not str(item.get("canvas_id_after") or ""):
            blockers.append("missing_canvas_id_after")
        rows.append(
            {
                "sample": str(item.get("candidate_key") or ""),
                "process_name": "",
                "source": "control_transition_graph",
                "status": "controlled_transition_observed" if not blockers else "incomplete_transition",
                "transition_path": _path(from_class, to_class),
                "candidate_key": str(item.get("candidate_key") or ""),
                "action_type": action,
                "success_rate": float(item.get("success_rate") or 0.0),
                "execution_boundary": "requires_confirmed_execution",
                "autonomous_execution": False,
                "required_verification": REQUIRED_VERIFICATION,
                "blockers": blockers,
                "agent_contract": _agent_contract(),
            }
        )
    return rows


def _search_transition_blockers(row: dict[str, Any], *, transition: dict[str, Any], path: str) -> list[str]:
    blockers: list[str] = []
    if str(row.get("status") or "") != "captured":
        blockers.append("probe_not_captured")
    if " -> " not in path:
        blockers.append("missing_transition_path")
    if not bool(transition.get("selected_returned_to_before_state")):
        blockers.append("selected_not_returned_to_before_state")
    return blockers


def _identity_blockers(*, from_class: str, to_class: str) -> list[str]:
    blockers: list[str] = []
    if not from_class:
        blockers.append("missing_from_page_class")
    if not to_class:
        blockers.append("missing_to_page_class")
    if from_class and to_class and from_class == to_class:
        blockers.append("no_page_class_change")
    return blockers


def _overall_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "empty"
    if any(str(row.get("status") or "") == "incomplete_transition" for row in rows):
        return "incomplete"
    statuses = {str(row.get("status") or "") for row in rows}
    if "controlled_transition_observed" in statuses:
        return "controlled_transition_memory_ready"
    if statuses <= {"observed_roundtrip", "transition_observed"}:
        return "observed_transition_memory_ready"
    return "review"


def _agent_contract() -> dict[str, Any]:
    return {
        "allowed_default": "read_transition_memory",
        "blocked_default": ["click", "type_text", "send"],
        "execution_requires": ["explicit_confirmation", *REQUIRED_VERIFICATION],
    }


def _load_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _path(from_class: str, to_class: str) -> str:
    if from_class and to_class:
        return f"{from_class} -> {to_class}"
    return from_class or to_class


def _format_counts(counts: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"


def _md(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-probe-dir", type=Path)
    parser.add_argument("--transition-graph-file", type=Path)
    parser.add_argument("--control-transition-graph-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_transition_readiness_dirs(
        search_probe_dir=args.search_probe_dir,
        transition_graph_file=args.transition_graph_file,
        control_transition_graph_file=args.control_transition_graph_file,
        output_dir=args.output_dir,
    )
    print(f"overall={report.get('overall_status')} rows={len(report.get('rows') or [])}")
    return 0 if report.get("overall_status") in {"observed_transition_memory_ready", "controlled_transition_memory_ready"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
