from __future__ import annotations

from typing import Any


def attach_observe_timing(canvas: Any, timing: dict[str, float]) -> None:
    """Attach compact observe timing diagnostics to a canvas artifact."""
    artifacts = getattr(canvas, "artifacts", None)
    if artifacts is None:
        artifacts = {}
        setattr(canvas, "artifacts", artifacts)

    stages = {
        str(name): round(float(seconds), 3)
        for name, seconds in timing.items()
        if name != "TOTAL"
    }
    slowest_name = ""
    slowest_seconds = 0.0
    if stages:
        slowest_name, slowest_seconds = max(stages.items(), key=lambda item: item[1])

    artifacts["observe_timing"] = {
        "stages": stages,
        "total_seconds": round(float(timing.get("TOTAL", 0.0)), 3),
        "slowest_stage": {
            "name": slowest_name,
            "seconds": slowest_seconds,
        },
    }
