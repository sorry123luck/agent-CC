"""Run the offline ROI planning and ROI VLM quality regression checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_roi_plan_samples
import evaluate_roi_vlm_quality


def build_combined_summary(
    plan_summary: dict[str, Any],
    quality_summary: dict[str, Any] | None,
    *,
    min_quality_score: float,
    max_mode_mismatches: int,
    max_plan_errors: int,
    sample_matrix_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact gate summary from plan and VLM quality reports."""
    plan_error_count = len(list(plan_summary.get("errors") or []))
    failures: list[str] = []
    mode_mismatch_count = int(plan_summary.get("mode_mismatch_count") or 0)
    if mode_mismatch_count > max_mode_mismatches:
        failures.append(f"mode_mismatch_count>{max_mode_mismatches}")
    if plan_error_count > max_plan_errors:
        failures.append(f"plan_error_count>{max_plan_errors}")

    quality_block = _quality_block(quality_summary)
    if quality_summary is not None:
        average_score = float(quality_summary.get("average_score") or 0)
        if average_score < min_quality_score:
            failures.append(f"average_quality_score<{_format_number(min_quality_score)}")
        if int(quality_summary.get("late_failure_total") or 0) > 0:
            failures.append("late_failure_total>0")

    return {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "plan": {
            "sample_count": int(plan_summary.get("sample_count") or 0),
            "expected_count": int(plan_summary.get("expected_count") or 0),
            "mode_match_count": int(plan_summary.get("mode_match_count") or 0),
            "mode_mismatch_count": mode_mismatch_count,
            "error_count": plan_error_count,
        },
        "quality": quality_block,
        "sample_matrix": _sample_matrix_block(sample_matrix_summary),
        "thresholds": {
            "min_quality_score": min_quality_score,
            "max_mode_mismatches": max_mode_mismatches,
            "max_plan_errors": max_plan_errors,
        },
    }


def run_regression(
    *,
    samples_root: Path,
    output_dir: Path,
    quality_summary_path: Path | None,
    min_quality_score: float,
    max_mode_mismatches: int,
    max_plan_errors: int,
) -> dict[str, Any]:
    """Evaluate local ROI plans and optionally score saved ROI VLM comparison output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_output_dir = output_dir / "roi_plan_eval"
    plan_paths = evaluate_roi_plan_samples.collect_sample_paths(samples_root)
    plan_summary = evaluate_roi_plan_samples.evaluate_paths(plan_paths)
    evaluate_roi_plan_samples.write_outputs(plan_summary, plan_output_dir)

    quality_summary = None
    if quality_summary_path is not None:
        quality_output_dir = output_dir / "roi_vlm_quality"
        quality_summary = evaluate_roi_vlm_quality.evaluate_summary(quality_summary_path)
        evaluate_roi_vlm_quality.write_outputs(quality_summary, quality_output_dir)

    summary = build_combined_summary(
        plan_summary,
        quality_summary,
        min_quality_score=min_quality_score,
        max_mode_mismatches=max_mode_mismatches,
        max_plan_errors=max_plan_errors,
        sample_matrix_summary=load_sample_matrix_summary(samples_root),
    )
    write_outputs(summary, output_dir)
    return summary


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "regression_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# ROI/VLM Regression Summary",
        "",
        f"- Status: {summary['status']}",
        f"- Failures: {', '.join(summary['failures']) if summary['failures'] else 'none'}",
        "",
        "## ROI Plan",
        "",
        f"- Samples: {summary['plan']['sample_count']}",
        f"- Expected labels: {summary['plan']['expected_count']}",
        f"- Mode matches: {summary['plan']['mode_match_count']}",
        f"- Mode mismatches: {summary['plan']['mode_mismatch_count']}",
        f"- Errors: {summary['plan']['error_count']}",
        "",
        "## ROI VLM Quality",
        "",
    ]
    quality = summary.get("quality") or {}
    if quality.get("available"):
        lines.extend(
            [
                f"- Source: {quality['source']}",
                f"- Samples: {quality['sample_count']}",
                f"- Average score: {quality['average_score']}",
                f"- Low score count: {quality['low_score_count']}",
                f"- Generic region total: {quality['generic_region_total']}",
                f"- Invalid region total: {quality['invalid_region_total']}",
                f"- Invalid annotation role total: {quality['invalid_annotation_role_total']}",
                f"- Late failure total: {quality['late_failure_total']}",
            ]
        )
    else:
        lines.append("- Not run")
    matrix = summary.get("sample_matrix") or {}
    lines.extend(
        [
            "",
            "## Live Sample Matrix",
            "",
        ]
    )
    if matrix.get("available"):
        lines.extend(
            [
                f"- Samples: {matrix['sample_count']}",
                f"- Errors: {matrix['error_count']}",
                f"- Semantic status counts: {_format_counts(matrix['semantic_status_counts'])}",
                f"- Mode counts: {_format_counts(matrix['mode_counts'])}",
            ]
        )
    else:
        lines.append("- Not available")
    (output_dir / "regression_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _quality_block(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {"available": False}
    return {
        "available": True,
        "source": str(summary.get("source") or ""),
        "sample_count": int(summary.get("sample_count") or 0),
        "average_score": float(summary.get("average_score") or 0),
        "low_score_count": int(summary.get("low_score_count") or 0),
        "generic_region_total": int(summary.get("generic_region_total") or 0),
        "invalid_region_total": int(summary.get("invalid_region_total") or 0),
        "invalid_annotation_role_total": int(summary.get("invalid_annotation_role_total") or 0),
        "late_failure_total": int(summary.get("late_failure_total") or 0),
    }


def load_sample_matrix_summary(samples_root: Path) -> dict[str, Any] | None:
    """Load optional live sample matrix summary from the sample root."""
    path = samples_root / "sample_matrix_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_matrix_block(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {"available": False}
    return {
        "available": True,
        "sample_count": int(summary.get("sample_count") or 0),
        "error_count": int(summary.get("error_count") or 0),
        "semantic_status_counts": dict(summary.get("semantic_status_counts") or {}),
        "mode_counts": dict(summary.get("mode_counts") or {}),
    }


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _format_counts(counts: dict[str, Any]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, default=Path("artifacts/live_roi_vlm_samples_2026-05-21"))
    parser.add_argument("--quality-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/roi_vlm_regression_2026-05-21"))
    parser.add_argument("--min-quality-score", type=float, default=85.0)
    parser.add_argument("--max-mode-mismatches", type=int, default=0)
    parser.add_argument("--max-plan-errors", type=int, default=0)
    args = parser.parse_args()

    summary = run_regression(
        samples_root=args.samples_root,
        output_dir=args.output_dir,
        quality_summary_path=args.quality_summary,
        min_quality_score=args.min_quality_score,
        max_mode_mismatches=args.max_mode_mismatches,
        max_plan_errors=args.max_plan_errors,
    )
    print(
        f"status={summary['status']} plan_samples={summary['plan']['sample_count']} "
        f"mismatches={summary['plan']['mode_mismatch_count']} errors={summary['plan']['error_count']} "
        f"quality_score={summary['quality'].get('average_score', 'n/a')}"
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
