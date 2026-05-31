"""P0 inspector exports.

Keep package import light so execution/runtime modules can reuse inspector-adjacent
helpers without pulling the full inspector service stack during import time.
"""

__all__ = ["InspectorService"]


def __getattr__(name: str):
    if name == "InspectorService":
        from src.inspector.service import InspectorService

        return InspectorService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
