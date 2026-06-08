"""Provider metadata — unified descriptor for every Vision Runtime provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ProviderMetadata:
    """Unified provider descriptor.

    Instances are registered into :class:`ProviderRegistry` at startup and
    queried by :class:`HealthMonitor` / future ``VisionRuntimeManager``.

    Fields are intentionally read-only (frozen) — updates go through
    ``ProviderRegistry.update_endpoint()`` which replaces the whole object.
    """

    provider_id: str
    """Unique identifier, e.g. ``"uia"``, ``"paddleocr"``, ``"omniparser"``."""

    provider_type: str
    """Category: ``"perception"``, ``"ocr"``, ``"vision"``, ``"vlm"``."""

    mode: str
    """Execution mode: ``"in_process"``, ``"subprocess"``, ``"remote_api"``."""

    capabilities: frozenset[str] = field(default_factory=frozenset)
    """Capability tags, e.g. ``{"text_extraction", "icon_detection"}``."""

    default_timeout_seconds: float = 30.0
    """Suggested timeout for a single call to this provider."""

    endpoint: str | None = None
    """Network endpoint, if applicable (e.g. ``"http://127.0.0.1:8001/parse/"``)."""

    process_info: str | None = None
    """Process descriptor, e.g. ``"persistent_worker"``, ``"com_thread"``."""

    model_dependency: str | None = None
    """Required model, e.g. ``"PP-OCRv5"``, ``"YOLO+Florence-2"``."""

    health_check: Callable[[], dict[str, Any]] | None = field(
        default=None, repr=False, compare=False
    )
    """Optional zero-arg callable that returns a health dict.

    The callable MUST be lightweight — no heavy I/O, no model loading.
    It is invoked by :class:`HealthMonitor.check_one()`.
    """
