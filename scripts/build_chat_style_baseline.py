"""Build chat sender bubble style baselines from readback artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat.readback_style import build_sender_style_baseline, extract_style_samples


def find_readback_results(paths: list[Path]) -> list[dict[str, Any]]:
    """Load nested readback result JSON payloads from artifact directories."""
    results: list[dict[str, Any]] = []
    for root in paths:
        if root.is_file():
            candidates = [root]
        else:
            candidates = sorted(root.rglob("*readback*.json"))
        for path in candidates:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and isinstance(data.get("events"), list) and data.get("app_process"):
                data["_source_path"] = str(path)
                results.append(data)
            elif isinstance(data, dict) and isinstance(data.get("results"), list):
                for item in data["results"]:
                    if isinstance(item, dict) and isinstance(item.get("events"), list) and item.get("app_process"):
                        item = dict(item)
                        item["_source_path"] = str(path)
                        results.append(item)
    return results


def write_baseline_report(output_dir: Path, baseline: dict[str, Any]) -> None:
    """Persist JSON and Markdown baseline reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(baseline)
    payload.setdefault("generated_at", datetime.now().isoformat(timespec="seconds"))
    (output_dir / "chat_sender_style_baseline.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Chat Sender Style Baseline",
        "",
        f"- Generated: {payload.get('generated_at', '')}",
        f"- Sample count: {payload.get('sample_count', 0)}",
        "",
        "| app | usable | counts | hints |",
        "| --- | --- | --- | --- |",
    ]
    for app, data in sorted((payload.get("apps") or {}).items()):
        counts = ", ".join(f"{sender}={count}" for sender, count in sorted((data.get("sample_counts") or {}).items()))
        hints = ", ".join(
            f"{sender}={rgb}"
            for sender, rgb in sorted((data.get("sender_style_hints") or {}).items())
        )
        lines.append(f"| {app} | {str(bool(data.get('usable'))).lower()} | {counts} | {hints} |")
    (output_dir / "chat_sender_style_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="Artifact directory or readback JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--min-sender-distance", type=float, default=40)
    args = parser.parse_args()
    results = find_readback_results(args.input)
    samples = extract_style_samples(results)
    baseline = build_sender_style_baseline(
        samples,
        min_samples=args.min_samples,
        min_sender_distance=args.min_sender_distance,
    )
    baseline["generated_at"] = datetime.now().isoformat(timespec="seconds")
    baseline["readback_result_count"] = len(results)
    write_baseline_report(args.output_dir, baseline)
    print(f"apps={len(baseline.get('apps') or {})} samples={baseline.get('sample_count', 0)}")
    return 0 if baseline.get("apps") else 1


if __name__ == "__main__":
    raise SystemExit(main())
