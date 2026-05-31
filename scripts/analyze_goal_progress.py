"""Summarize the six active product goals from existing regression reports.

This report is a progress map, not a completion claim. It converts the existing
agent-operability, input-safety, transition, VLM, sample, and maintenance
reports into one machine-readable view of what is proven and what remains.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


GOAL_IDS = [
    "goal_1_agent_execution",
    "goal_2_input_safety",
    "goal_3_multi_page_exploration",
    "goal_4_vlm_semantic_supplement",
    "goal_5_real_sample_regression",
    "goal_6_model_data_maintenance",
]


def build_goal_progress_report(
    *,
    agent_operability: dict[str, Any],
    input_safety: dict[str, Any] | None = None,
    sample_coverage: dict[str, Any] | None = None,
    artifacts_retention: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build objective-level progress from current reports."""
    input_safety = input_safety or {}
    sample_coverage = sample_coverage or {}
    artifacts_retention = artifacts_retention or {}
    goals = [
        _goal_agent_execution(agent_operability),
        _goal_input_safety(agent_operability, input_safety),
        _goal_multi_page(agent_operability),
        _goal_vlm(agent_operability),
        _goal_real_samples(agent_operability, sample_coverage),
        _goal_maintenance(artifacts_retention),
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_type": "goal_progress",
        "overall_status": "complete" if all(goal["status"] == "complete" for goal in goals) else "in_progress",
        "policy": "Progress evidence only; do not mark the thread goal complete from this report alone.",
        "goals": goals,
    }


def analyze_goal_progress_dirs(
    *,
    input_dir: Path,
    output_dir: Path,
    artifacts_retention_dir: Path | None = None,
) -> dict[str, Any]:
    """Load standard reports and write goal progress outputs."""
    report = build_goal_progress_report(
        agent_operability=_load_json(input_dir / "agent_operability_regression_summary.json"),
        input_safety=_load_json(input_dir / "input_safety_readiness_report.json"),
        sample_coverage=_load_json(input_dir / "sample_coverage_report.json"),
        artifacts_retention=_load_json((artifacts_retention_dir / "artifacts_retention_report.json") if artifacts_retention_dir else None),
    )
    write_goal_progress_report(report, output_dir)
    return report


def write_goal_progress_report(report: dict[str, Any], output_dir: Path) -> None:
    """Persist goal progress JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "goal_progress_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Goal Progress Report",
        "",
        f"- Generated: {report.get('generated_at', '')}",
        f"- Overall: {report.get('overall_status', '')}",
        f"- Policy: {report.get('policy', '')}",
        "",
        "| goal | status | evidence | remaining | next_step |",
        "| --- | --- | --- | --- | --- |",
    ]
    for goal in list(report.get("goals") or []):
        lines.append(
            "| {goal} | {status} | {evidence} | {remaining} | {next_step} |".format(
                goal=_md(goal.get("goal_id")),
                status=_md(goal.get("status")),
                evidence=_md("; ".join(goal.get("evidence") or [])),
                remaining=_md("; ".join(goal.get("remaining_work") or [])),
                next_step=_md(goal.get("next_step")),
            )
        )
    (output_dir / "goal_progress_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _goal_agent_execution(agent: dict[str, Any]) -> dict[str, Any]:
    status = str(agent.get("act_preflight_status") or "")
    if not status:
        return _goal(
            "goal_1_agent_execution",
            "missing_evidence",
            [],
            ["missing act_preflight_status"],
            "run_agent_operability_regression with current backend",
            boundary="read/inspect and controlled click/scroll must stay policy-gated",
        )
    remaining = []
    if status == "usable_with_blocked_actions":
        remaining.append("type_text/send execution still blocked by design")
    if status != "usable":
        remaining.append("confirm controlled click/scroll execution on broader non-chat samples")
    return _goal(
        "goal_1_agent_execution",
        status,
        [f"act_preflight_status={status}"],
        remaining,
        "expand controlled click/scroll real-sample execution checks before any input execution",
        boundary="read-only actions allowed; click/scroll controlled; type_text/send blocked",
    )


def _goal_input_safety(agent: dict[str, Any], input_safety: dict[str, Any]) -> dict[str, Any]:
    status = str(agent.get("input_safety_status") or input_safety.get("overall_status") or "")
    if not status:
        return _goal(
            "goal_2_input_safety",
            "missing_evidence",
            [],
            ["missing input_safety_readiness_report"],
            "run input safety readiness from current page and act preflight reports",
            boundary="safe_to_type remains false until gate is satisfied",
        )
    missing = _input_gate_missing(input_safety)
    remaining = missing or ["safe_to_type remains false until policy explicitly upgrades it"]
    return _goal(
        "goal_2_input_safety",
        status,
        [f"input_safety_status={status}"],
        remaining,
        "collect controlled probe, StateTemplate input transition, and multi-sample stability evidence",
        boundary="review-only input; no default typing",
    )


def _goal_multi_page(agent: dict[str, Any]) -> dict[str, Any]:
    status = str(agent.get("transition_readiness_status") or "")
    if not status:
        return _goal(
            "goal_3_multi_page_exploration",
            "missing_evidence",
            [],
            ["missing transition_readiness_report"],
            "run transition readiness with search probe or transition graph inputs",
            boundary="read transition memory first; no autonomous clicking",
        )
    remaining = []
    if status == "observed_transition_memory_ready":
        remaining.extend(["state graph is read-only", "autonomous exploration policy not implemented"])
    return _goal(
        "goal_3_multi_page_exploration",
        status,
        [f"transition_readiness_status={status}"],
        remaining,
        "build read-only exploration graph from observed search/dialog transitions before enabling controlled exploration",
        boundary="read_transition_memory only by default",
    )


def _goal_vlm(agent: dict[str, Any]) -> dict[str, Any]:
    status = str(agent.get("vlm_supplement_status") or "")
    if not status:
        return _goal(
            "goal_4_vlm_semantic_supplement",
            "missing_evidence",
            [],
            ["missing vlm_supplement_readiness_report"],
            "run VLM supplement readiness from ROI/VLM quality summary",
            boundary="VLM must not own coordinates or executable action projection",
        )
    remaining = []
    if status == "semantic_supplement_ready":
        remaining.append("continue expanding scenario-specific ROI prompt coverage")
    return _goal(
        "goal_4_vlm_semantic_supplement",
        status,
        [f"vlm_supplement_status={status}"],
        remaining,
        "add mismatch diagnostics for image messages and unknown attachments",
        boundary="semantic supplement only; no coordinate/action projection",
    )


def _goal_real_samples(agent: dict[str, Any], sample_coverage: dict[str, Any]) -> dict[str, Any]:
    coverage_status = str(sample_coverage.get("overall_status") or "")
    if coverage_status:
        remaining = []
        for item in list(sample_coverage.get("requirements") or []):
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "")
            if status != "covered":
                remaining.append(f"{status}:{item.get('requirement_id')}")
        counts = sample_coverage.get("counts") if isinstance(sample_coverage.get("counts"), dict) else {}
        return _goal(
            "goal_5_real_sample_regression",
            coverage_status,
            [f"sample_coverage={coverage_status}", f"covered={counts.get('covered', 0)}"],
            remaining,
            "refresh missing representative apps and rerun fixed sample coverage gate",
            boundary="artifacts are evidence; source control only tracks scripts/docs/baselines",
        )
    sample_summary = agent.get("sample_summary") if isinstance(agent.get("sample_summary"), dict) else {}
    sample_count = int(sample_summary.get("sample_count") or 0)
    overall = str(agent.get("overall_status") or "")
    if not overall or sample_count <= 0:
        return _goal(
            "goal_5_real_sample_regression",
            "missing_evidence",
            [],
            ["missing real sample regression summary"],
            "run_agent_operability_regression against the current representative sample matrix",
            boundary="fixed reports, no ad-hoc sample conclusions",
        )
    remaining = []
    if sample_count < 7:
        remaining.append("matrix still needs broader apps: QQ private/group, Feishu, WeChat, FlClash, VoiceMeeter, NetEase")
    return _goal(
        "goal_5_real_sample_regression",
        "partial" if remaining else "ready",
        [f"overall_status={overall}", f"sample_count={sample_count}"],
        remaining,
        "refresh representative matrix and keep one fixed summary directory per run",
        boundary="artifacts are evidence; source control only tracks scripts/docs/baselines",
    )


def _goal_maintenance(artifacts_retention: dict[str, Any]) -> dict[str, Any]:
    summary = artifacts_retention.get("summary") if isinstance(artifacts_retention.get("summary"), dict) else {}
    if not summary:
        return _goal(
            "goal_6_model_data_maintenance",
            "audit_ready",
            ["artifacts retention audit script exists"],
            ["maintenance cleanup remains read-only; no DB/cache cleanup applied"],
            "run artifacts retention audit and model integrity audit before any cleanup",
            boundary="audit first; no destructive cleanup without explicit policy",
        )
    archive_candidate = int(summary.get("archive_candidate") or 0)
    return _goal(
        "goal_6_model_data_maintenance",
        "audit_ready",
        [f"archive_candidate={archive_candidate}"],
        ["orphan model migration and cache cleanup still pending"],
        "turn audits into non-destructive cleanup plans, then require explicit cleanup gate",
        boundary="audit first; cleanup policy separate",
    )


def _goal(goal_id: str, status: str, evidence: list[str], remaining: list[str], next_step: str, *, boundary: str) -> dict[str, Any]:
    return {
        "goal_id": goal_id,
        "status": status,
        "evidence": evidence,
        "remaining_work": list(dict.fromkeys(item for item in remaining if item)),
        "next_step": next_step,
        "boundary": boundary,
    }


def _input_gate_missing(input_safety: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for row in list(input_safety.get("rows") or []):
        if not isinstance(row, dict):
            continue
        gate = row.get("safe_type_upgrade_gate") if isinstance(row.get("safe_type_upgrade_gate"), dict) else {}
        missing.extend(str(item) for item in list(gate.get("missing") or []) if str(item))
    return list(dict.fromkeys(missing))


def _load_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _md(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifacts-retention-dir", type=Path)
    args = parser.parse_args()
    report = analyze_goal_progress_dirs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        artifacts_retention_dir=args.artifacts_retention_dir,
    )
    print(f"overall={report.get('overall_status')} goals={len(report.get('goals') or [])}")
    return 0 if report.get("overall_status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
