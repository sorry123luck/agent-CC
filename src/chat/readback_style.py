"""Build sender style hints from chat readback results."""

from __future__ import annotations

from statistics import median
from typing import Any


def extract_style_samples(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return valid text bubble style samples from readback JSON payloads."""
    samples: list[dict[str, Any]] = []
    for result in results:
        app_process = str(result.get("app_process") or "").lower()
        if not app_process:
            continue
        for event in list(result.get("events") or []):
            if not isinstance(event, dict):
                continue
            if str(event.get("message_type") or "") != "text":
                continue
            sender = str(event.get("sender") or "")
            if sender not in {"me", "peer"}:
                continue
            rgb = _rgb_triplet((event.get("style") or {}).get("bubble_rgb"))
            if rgb is None:
                continue
            samples.append(
                {
                    "app_process": app_process,
                    "sender": sender,
                    "bubble_rgb": list(rgb),
                    "text": str(event.get("text") or ""),
                }
            )
    return samples


def build_sender_style_baseline(
    samples: list[dict[str, Any]],
    *,
    min_samples: int = 2,
    min_sender_distance: float = 40,
) -> dict[str, Any]:
    """Aggregate bubble RGB samples into app-level sender style hints."""
    grouped: dict[str, dict[str, list[tuple[int, int, int]]]] = {}
    for sample in samples:
        app_process = str(sample.get("app_process") or "").lower()
        sender = str(sample.get("sender") or "")
        rgb = _rgb_triplet(sample.get("bubble_rgb"))
        if not app_process or sender not in {"me", "peer"} or rgb is None:
            continue
        grouped.setdefault(app_process, {}).setdefault(sender, []).append(rgb)

    apps: dict[str, Any] = {}
    for app_process, by_sender in sorted(grouped.items()):
        hints = {
            sender: _median_rgb(values)
            for sender, values in sorted(by_sender.items())
            if len(values) >= min_samples
        }
        counts = {sender: len(values) for sender, values in sorted(by_sender.items())}
        warnings: list[str] = []
        if "me" in hints and "peer" in hints and _rgb_distance(hints["me"], hints["peer"]) < min_sender_distance:
            warnings.append("sender_style_ambiguous")
            hints = {}
        apps[app_process] = {
            "sender_style_hints": hints,
            "sample_counts": counts,
            "usable": bool(hints) and not warnings,
            "warnings": warnings,
        }
    return {"apps": apps, "sample_count": sum(sum(len(values) for values in senders.values()) for senders in grouped.values())}


def _median_rgb(values: list[tuple[int, int, int]]) -> list[int]:
    return [round(median([rgb[index] for rgb in values])) for index in range(3)]


def _rgb_distance(left: list[int], right: list[int]) -> float:
    return sum((int(left[index]) - int(right[index])) ** 2 for index in range(3)) ** 0.5


def _rgb_triplet(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        rgb = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None
    if any(item < 0 or item > 255 for item in rgb):
        return None
    return rgb  # type: ignore[return-value]
