"""Run the fixed agent operability regression chain.

This runner glues together the existing product-level reports:

1. page operability
2. /act preflight matrix
3. recognition closure with optional act preflight evidence

It does not execute typing or sending.  The only backend calls are /act dry-run
preflight requests made by the act preflight regression step.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_page_operability
import analyze_input_safety_readiness
import analyze_goal_progress
import analyze_recognition_closure
import analyze_sample_coverage
import analyze_transition_readiness
import analyze_vlm_supplement_readiness
import run_act_preflight_regression


def run_agent_operability_regression(
    *,
    matrix_dir: Path,
    detail_dir: Path,
    output_dir: Path,
    acceptance_dir: Path | None = None,
    readiness_dir: Path | None = None,
    evidence_gate_dir: Path | None = None,
    search_probe_dir: Path | None = None,
    transition_graph_file: Path | None = None,
    control_transition_graph_file: Path | None = None,
    vlm_quality_dir: Path | None = None,
    api_base: str = "http://127.0.0.1:8000",
    probe_text: str = "DeskCanvas matrix probe",
    act_client: run_act_preflight_regression.ActClient | None = None,
) -> dict[str, Any]:
    """Run page, act, and closure reports into one output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    page_report = analyze_page_operability.analyze_page_operability_dirs(
        matrix_dir=matrix_dir,
        acceptance_dir=acceptance_dir,
        readiness_dir=readiness_dir,
        evidence_gate_dir=evidence_gate_dir,
        output_dir=output_dir,
    )

    act_output_dir = output_dir / "act_preflight"
    act_report = run_act_preflight_regression.run_act_preflight_regression(
        detail_dir=detail_dir,
        output_dir=act_output_dir,
        sample_matrix=matrix_dir / "sample_matrix_summary.json",
        act_client=act_client,
        api_base=api_base,
        probe_text=probe_text,
    )
    _copy_if_exists(
        act_output_dir / "analysis" / "act_preflight_matrix_report.json",
        output_dir / "act_preflight_matrix_report.json",
    )
    _copy_if_exists(
        act_output_dir / "analysis" / "act_preflight_matrix_report.md",
        output_dir / "act_preflight_matrix_report.md",
    )
    input_safety_report = analyze_input_safety_readiness.analyze_input_safety_readiness_dirs(
        input_dir=output_dir,
        output_dir=output_dir,
    )
    sample_coverage_report = analyze_sample_coverage.analyze_sample_coverage_dirs(
        matrix_dir=matrix_dir,
        output_dir=output_dir,
    )
    transition_report: dict[str, Any] | None = None
    if search_probe_dir or transition_graph_file or control_transition_graph_file:
        transition_report = analyze_transition_readiness.analyze_transition_readiness_dirs(
            search_probe_dir=search_probe_dir,
            transition_graph_file=transition_graph_file,
            control_transition_graph_file=control_transition_graph_file,
            output_dir=output_dir,
        )
    vlm_supplement_report: dict[str, Any] | None = None
    if vlm_quality_dir:
        vlm_supplement_report = analyze_vlm_supplement_readiness.analyze_vlm_supplement_readiness_dirs(
            quality_dir=vlm_quality_dir,
            output_dir=output_dir,
        )

    closure_report = analyze_recognition_closure.analyze_closure_dirs(
        matrix_dir=matrix_dir,
        operability_dir=output_dir,
        output_dir=output_dir,
    )
    summary = _summary(
        page_report=page_report,
        act_report=act_report,
        input_safety_report=input_safety_report,
        transition_report=transition_report,
        vlm_supplement_report=vlm_supplement_report,
        closure_report=closure_report,
    )
    _write_summary(summary, output_dir)
    goal_progress_report = analyze_goal_progress.build_goal_progress_report(
        agent_operability=summary,
        input_safety=input_safety_report,
        sample_coverage=sample_coverage_report,
    )
    analyze_goal_progress.write_goal_progress_report(goal_progress_report, output_dir)
    return summary


def _summary(
    *,
    page_report: dict[str, Any],
    act_report: dict[str, Any],
    input_safety_report: dict[str, Any],
    transition_report: dict[str, Any] | None,
    vlm_supplement_report: dict[str, Any] | None,
    closure_report: dict[str, Any],
) -> dict[str, Any]:
    outputs = {
        "page_operability": "page_operability_report.json",
        "act_preflight": "act_preflight/analysis/act_preflight_matrix_report.json",
        "input_safety": "input_safety_readiness_report.json",
        "sample_coverage": "sample_coverage_report.json",
        "recognition_closure": "recognition_closure_report.json",
        "goal_progress": "goal_progress_report.json",
    }
    if transition_report is not None:
        outputs["transition_readiness"] = "transition_readiness_report.json"
    if vlm_supplement_report is not None:
        outputs["vlm_supplement_readiness"] = "vlm_supplement_readiness_report.json"
    return {
        "overall_status": str(closure_report.get("overall_status") or ""),
        "page_operability_status": str(page_report.get("overall_status") or ""),
        "act_preflight_status": str(act_report.get("overall_status") or ""),
        "input_safety_status": str(input_safety_report.get("overall_status") or ""),
        "transition_readiness_status": str((transition_report or {}).get("overall_status") or ""),
        "vlm_supplement_status": str((vlm_supplement_report or {}).get("overall_status") or ""),
        "closure_completion": closure_report.get("completion") or {},
        "sample_summary": closure_report.get("sample_summary") or {},
        "outputs": outputs,
    }


def _write_summary(summary: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "agent_operability_regression_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Agent Operability Regression Summary",
        "",
        f"- Overall: {summary.get('overall_status', '')}",
        f"- Page operability: {summary.get('page_operability_status', '')}",
        f"- Act preflight: {summary.get('act_preflight_status', '')}",
        f"- Input safety: {summary.get('input_safety_status', '')}",
        f"- Transition readiness: {summary.get('transition_readiness_status', '') or 'not_run'}",
        f"- VLM supplement: {summary.get('vlm_supplement_status', '') or 'not_run'}",
        "",
        "| completion | status | evidence |",
        "| --- | --- | --- |",
    ]
    for key, item in (summary.get("closure_completion") or {}).items():
        lines.append(
            "| {key} | {status} | {evidence} |".format(
                key=_md(key),
                status=_md(item.get("status") if isinstance(item, dict) else ""),
                evidence=_md(item.get("evidence") if isinstance(item, dict) else ""),
            )
        )
    (output_dir / "agent_operability_regression_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copyfile(src, dst)


def _md(value: Any) -> str:
    return str(value or "").replace("|", "/").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--detail-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acceptance-dir", type=Path)
    parser.add_argument("--readiness-dir", type=Path)
    parser.add_argument("--evidence-gate-dir", type=Path)
    parser.add_argument("--search-probe-dir", type=Path)
    parser.add_argument("--transition-graph-file", type=Path)
    parser.add_argument("--control-transition-graph-file", type=Path)
    parser.add_argument("--vlm-quality-dir", type=Path)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--probe-text", default="DeskCanvas matrix probe")
    args = parser.parse_args()
    summary = run_agent_operability_regression(
        matrix_dir=args.matrix_dir,
        detail_dir=args.detail_dir,
        output_dir=args.output_dir,
        acceptance_dir=args.acceptance_dir,
        readiness_dir=args.readiness_dir,
        evidence_gate_dir=args.evidence_gate_dir,
        search_probe_dir=args.search_probe_dir,
        transition_graph_file=args.transition_graph_file,
        control_transition_graph_file=args.control_transition_graph_file,
        vlm_quality_dir=args.vlm_quality_dir,
        api_base=args.api_base,
        probe_text=args.probe_text,
    )
    print(
        "overall={overall} page={page} act={act}".format(
            overall=summary.get("overall_status"),
            page=summary.get("page_operability_status"),
            act=summary.get("act_preflight_status"),
        )
    )
    return 0 if summary.get("overall_status") == "closed_for_recognition_chain" else 1


if __name__ == "__main__":
    raise SystemExit(main())
