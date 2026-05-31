"""Compatibility wrapper for chat target identity gate reports.

Prefer ``analyze_page_evidence_gate.py`` for new generic callers.  This module
keeps the older target-identity filenames and function names for existing tests
and artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_page_evidence_gate import (
    analyze_page_evidence_dir,
    build_page_evidence_gate,
    write_page_evidence_gate,
)


def build_target_identity_check(
    *,
    snapshot_report: dict[str, Any],
    targets: dict[str, list[str]],
) -> dict[str, Any]:
    """Return proceed/stop decisions by matching expected texts in readback events."""
    report = build_page_evidence_gate(snapshot_report=snapshot_report, requirements=targets)
    report["policy"] = "Rows must match all expected_texts before controlled send; missing or unmatched targets stop."
    return report


def analyze_target_identity_dir(
    *,
    snapshot_dir: Path,
    targets_file: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Load snapshot/readback artifacts and write target identity gate reports."""
    report = analyze_page_evidence_dir(
        snapshot_dir=snapshot_dir,
        requirements_file=targets_file,
        output_dir=output_dir or snapshot_dir,
    )
    report["policy"] = "Rows must match all expected_texts before controlled send; missing or unmatched targets stop."
    write_target_identity_check(report, output_dir or snapshot_dir)
    return report


def write_target_identity_check(report: dict[str, Any], output_dir: Path) -> None:
    """Persist JSON and Markdown target identity gate reports."""
    write_page_evidence_gate(report, output_dir)
    page_json = output_dir / "page_evidence_gate.json"
    page_md = output_dir / "page_evidence_gate.md"
    if page_json.exists():
        (output_dir / "target_identity_check.json").write_text(page_json.read_text(encoding="utf-8"), encoding="utf-8")
    if page_md.exists():
        text = page_md.read_text(encoding="utf-8").replace("# Page Evidence Gate", "# Target Identity Check", 1)
        (output_dir / "target_identity_check.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True, help="JSON file with target expected_texts")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = analyze_target_identity_dir(
        snapshot_dir=args.snapshot_dir,
        targets_file=args.targets,
        output_dir=args.output_dir,
    )
    print(
        "overall={overall} proceed={proceed} stop={stop}".format(
            overall=report["overall_decision"],
            proceed=sum(1 for row in report["checks"] if row.get("decision") == "proceed"),
            stop=sum(1 for row in report["checks"] if row.get("decision") != "proceed"),
        )
    )
    return 0 if any(row.get("decision") == "proceed" for row in report["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
