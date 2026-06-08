"""Health monitor — non-intrusive health probes for all registered providers."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from src.runtime.provider_metadata import ProviderMetadata
from src.runtime.provider_registry import ProviderRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class HealthResult:
    """Health probe result for a single provider."""

    provider_id: str
    ok: bool
    latency_ms: float | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeHealthSnapshot:
    """Aggregated health snapshot across all registered providers."""

    timestamp: str
    providers: list[HealthResult]
    summary: dict[str, int]  # {"total": 6, "healthy": 5, "unhealthy": 1}


# ---------------------------------------------------------------------------
# Built-in lightweight health checks
#
# Each function MUST be non-intrusive:
#   - No heavy I/O (no OCR, no OmniParser parse, no VLM call)
#   - No model loading
#   - No side effects beyond reading process/module state
# ---------------------------------------------------------------------------


def _check_uia() -> dict[str, Any]:
    """Check if UIA COM is importable (does NOT enumerate any window)."""
    try:
        import uiautomation  # noqa: F401

        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_screenshot() -> dict[str, Any]:
    """Check if Win32 GDI screenshot primitives are importable."""
    try:
        import win32gui  # noqa: F401
        import win32ui  # noqa: F401

        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_paddleocr() -> dict[str, Any]:
    """Check persistent OCR worker process liveness (no actual OCR call)."""
    try:
        from src.perception.ocr_service import get_ocr_service

        svc = get_ocr_service()
        worker = getattr(svc, "_persistent_worker", None)
        proc = getattr(worker, "_process", None) if worker else None
        alive = proc is not None and proc.poll() is None
        return {
            "ok": True,
            "worker_alive": alive,
            "worker_mode": getattr(svc._config, "worker_mode", "unknown"),
            "bridge_enabled": getattr(svc._config, "bridge_enabled", True),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_omniparser() -> dict[str, Any]:
    """HTTP GET probe to OmniParser (lightweight, no image sent).

    Uses a short 3-second timeout for health probes — the full 90-second
    timeout is only for actual parse requests.
    """
    try:
        import requests as _requests

        from src.perception.providers.remote_vision_provider import (
            OmniParserRemoteVisionProvider,
        )

        provider = OmniParserRemoteVisionProvider()
        cfg = provider._config
        if not cfg.enabled:
            return {"ok": False, "endpoint": cfg.endpoint, "error": "vision_provider_disabled"}
        try:
            resp = _requests.get(cfg.probe_endpoint, timeout=3.0)
            return {
                "ok": resp.ok,
                "endpoint": cfg.endpoint,
                "probe": {"ok": resp.ok, "http_status": resp.status_code},
            }
        except Exception as probe_exc:
            return {
                "ok": False,
                "endpoint": cfg.endpoint,
                "probe": {"ok": False, "error": str(probe_exc)},
            }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_vlm_generic(provider_module: str, provider_class: str) -> dict[str, Any]:
    """Generic VLM provider ``is_available()`` check."""
    try:
        import importlib

        mod = importlib.import_module(provider_module)
        cls = getattr(mod, provider_class)
        instance = cls()
        available = instance.is_available()
        return {
            "ok": True,
            "available": available,
            "model_id": getattr(instance, "model_id", "unknown"),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _check_vlm_openai() -> dict[str, Any]:
    return _check_vlm_generic("src.vlm.providers.openai_provider", "OpenAICompatibleProvider")


def _check_vlm_anthropic() -> dict[str, Any]:
    return _check_vlm_generic("src.vlm.providers.anthropic_provider", "AnthropicProvider")


def _check_vlm_minimax() -> dict[str, Any]:
    return _check_vlm_generic("src.vlm.providers.minimax_provider", "MiniMaxProvider")


# ---------------------------------------------------------------------------
# Config reader
# ---------------------------------------------------------------------------


def _read_config() -> dict[str, Any]:
    """Read merged config from config_manager (non-failing)."""
    try:
        from src.common.config_manager import load_config

        cfg = load_config()
        return {
            "ocr": {
                "provider": cfg.ocr.provider,
                "worker_mode": cfg.ocr.worker_mode,
                "bridge_enabled": cfg.ocr.bridge_enabled,
                "worker_timeout_seconds": cfg.ocr.worker_timeout_seconds,
            },
            "vision": {
                "provider": cfg.vision.provider,
                "endpoint": cfg.vision.endpoint,
                "timeout": cfg.vision.timeout,
            },
            "vlm": {
                "provider": cfg.vlm.provider,
                "model": cfg.vlm.model,
                "endpoint": cfg.vlm.endpoint,
                "timeout_seconds": cfg.vlm.timeout_seconds,
            },
            "semantic_modeler": {
                "enabled": cfg.semantic_modeler.enabled,
                "provider": cfg.semantic_modeler.provider,
                "model": cfg.semantic_modeler.model,
            },
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Default provider metadata (registered at startup)
# ---------------------------------------------------------------------------


def build_default_provider_metadata() -> list[ProviderMetadata]:
    """Return the standard set of provider metadata entries.

    Reads actual values from config (OCR worker mode, OmniParser endpoint,
    VLM provider/model) so the registry reflects the real deployment.

    Called once during ``api_server.lifespan`` to populate the registry.
    """
    cfg = _read_config()
    ocr_cfg = cfg.get("ocr", {})
    vision_cfg = cfg.get("vision", {})
    vlm_cfg = cfg.get("vlm", {})
    sm_cfg = cfg.get("semantic_modeler", {})

    # Determine VLM endpoint/model from semantic_modeler if vlm is disabled
    vlm_provider = (sm_cfg.get("provider") or vlm_cfg.get("provider") or "disabled").strip()
    vlm_model = (sm_cfg.get("model") or vlm_cfg.get("model") or "").strip()
    vlm_timeout = int(sm_cfg.get("timeout_seconds") or vlm_cfg.get("timeout_seconds") or 60)

    providers: list[ProviderMetadata] = [
        ProviderMetadata(
            provider_id="uia",
            provider_type="perception",
            mode="in_process",
            capabilities=frozenset({"element_tree", "bounding_rect", "control_type"}),
            default_timeout_seconds=8.0,
            process_info="com_thread",
            health_check=_check_uia,
        ),
        ProviderMetadata(
            provider_id="screenshot",
            provider_type="perception",
            mode="in_process",
            capabilities=frozenset({"window_capture", "region_capture", "fullscreen_capture"}),
            default_timeout_seconds=5.0,
            process_info="win32_gdi",
            health_check=_check_screenshot,
        ),
        ProviderMetadata(
            provider_id="paddleocr",
            provider_type="ocr",
            mode="subprocess",
            capabilities=frozenset({"text_extraction", "bbox_detection", "confidence"}),
            default_timeout_seconds=float(ocr_cfg.get("worker_timeout_seconds", 120)),
            process_info=ocr_cfg.get("worker_mode", "persistent"),
            model_dependency="PP-OCRv5",
            health_check=_check_paddleocr,
        ),
        ProviderMetadata(
            provider_id="omniparser",
            provider_type="vision",
            mode="remote_api",
            capabilities=frozenset({"icon_detection", "interactive_element", "caption"}),
            default_timeout_seconds=float(vision_cfg.get("timeout", 90)),
            endpoint=vision_cfg.get("endpoint") or "http://127.0.0.1:8001/parse/",
            process_info="subprocess_uvicorn",
            model_dependency="YOLO+Florence-2",
            health_check=_check_omniparser,
        ),
    ]

    # VLM providers: only register real (non-mock, non-disabled) providers.
    #
    # vlm_mock is excluded — it is a test-only provider that returns canned
    # responses; registering it would pollute health snapshots with a fake
    # "always available" entry and mislead monitoring.
    _VLM_PROVIDER_MAP: dict[str, tuple[str, str, Callable[[], dict[str, Any]]]] = {
        "openai": ("vlm_openai", "gpt-4o", _check_vlm_openai),
        "openrouter": ("vlm_openai", "openrouter", _check_vlm_openai),
        "anthropic": ("vlm_anthropic", "claude-sonnet", _check_vlm_anthropic),
        "minimax": ("vlm_minimax", "MiniMax-VL-01", _check_vlm_minimax),
        "mimo": ("vlm_minimax", "mimo", _check_vlm_minimax),
    }

    vlm_key = vlm_provider.lower().split("/")[0]  # normalize "openai/gpt-4o" -> "openai"
    if vlm_key in _VLM_PROVIDER_MAP:
        pid, default_model, check_fn = _VLM_PROVIDER_MAP[vlm_key]
        providers.append(ProviderMetadata(
            provider_id=pid,
            provider_type="vlm",
            mode="remote_api",
            capabilities=frozenset({"semantic_understanding", "element_detection", "json_output"}),
            default_timeout_seconds=float(vlm_timeout),
            model_dependency=vlm_model or default_model,
            health_check=check_fn,
        ))

    return providers


# ---------------------------------------------------------------------------
# Health monitor
# ---------------------------------------------------------------------------


class HealthMonitor:
    """Non-intrusive health probe for all registered providers.

    The monitor queries each provider's ``health_check`` callable (if any).
    Health checks are **lightweight** — they must NOT trigger OCR, OmniParser
    parse, or VLM calls.  They only read process/module state.
    """

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def check_one(self, provider_id: str) -> HealthResult:
        """Run health check for a single provider."""
        meta = self._registry.get(provider_id)
        if meta is None:
            return HealthResult(
                provider_id=provider_id,
                ok=False,
                error="provider_not_registered",
            )
        if meta.health_check is None:
            return HealthResult(
                provider_id=provider_id,
                ok=True,
                details={"note": "no_health_check_defined"},
            )

        start = time.perf_counter()
        try:
            details = meta.health_check()
            elapsed_ms = (time.perf_counter() - start) * 1000
            ok = bool(details.get("ok", False))
            error = details.pop("error", None) if not ok else None
            return HealthResult(
                provider_id=provider_id,
                ok=ok,
                latency_ms=round(elapsed_ms, 1),
                error=error,
                details=details,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("Health check failed for %s: %s", provider_id, exc)
            return HealthResult(
                provider_id=provider_id,
                ok=False,
                latency_ms=round(elapsed_ms, 1),
                error=str(exc),
            )

    def check_all(self) -> RuntimeHealthSnapshot:
        """Run health checks for all registered providers."""
        results: list[HealthResult] = []
        for meta in self._registry.list_all():
            results.append(self.check_one(meta.provider_id))

        healthy = sum(1 for r in results if r.ok)
        return RuntimeHealthSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            providers=results,
            summary={
                "total": len(results),
                "healthy": healthy,
                "unhealthy": len(results) - healthy,
            },
        )
