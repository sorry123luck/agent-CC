"""Canvas engine: query, diff, and scroll control for InteractionCanvas."""

from src.canvas.query_engine import CanvasQueryEngine, QueryTarget, QueryResult
from src.canvas.diff_engine import CanvasDiffEngine, CanvasChanges

__all__ = [
    "CanvasQueryEngine",
    "QueryTarget",
    "QueryResult",
    "CanvasDiffEngine",
    "CanvasChanges",
]
