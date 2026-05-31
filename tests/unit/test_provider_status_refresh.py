"""Provider status derivation tests."""

from src.integration.api_server import _refresh_provider_status_from_trace
from src.perception.page_compiler_models import (
    AppInfo,
    InteractionCanvas,
    PageInfo,
    ProviderTrace,
    SurfaceInfo,
    SurfaceType,
    WindowInfoSnapshot,
)


def test_refresh_provider_status_removes_stale_failures_when_trace_succeeds():
    """providers_failed must match provider_trace after background enhancement."""
    canvas = InteractionCanvas(
        canvas_id="c1",
        app=AppInfo(app_id="app", process_name="app.exe"),
        window=WindowInfoSnapshot(hwnd=1, title="App"),
        surface=SurfaceInfo(surface_type=SurfaceType.NATIVE_UIA, confidence=0.9),
        page=PageInfo(page_class="app/main/default", class_confidence=0.9),
        provider_trace=ProviderTrace(
            uia_used=True,
            ocr_used=True,
            vision_used=True,
            provider_details={
                "ocr_bridge": {"success": True},
                "vision_provider": {"success": True},
            },
        ),
        providers_used=["uia"],
        providers_failed=["ocr_bridge", "vision_provider"],
    )

    _refresh_provider_status_from_trace(canvas)

    assert canvas.providers_used == ["uia", "ocr", "vision"]
    assert canvas.providers_failed == []


def test_refresh_provider_status_keeps_failed_provider_when_trace_fails():
    canvas = InteractionCanvas(
        canvas_id="c1",
        app=AppInfo(app_id="app", process_name="app.exe"),
        window=WindowInfoSnapshot(hwnd=1, title="App"),
        surface=SurfaceInfo(surface_type=SurfaceType.NATIVE_UIA, confidence=0.9),
        page=PageInfo(page_class="app/main/default", class_confidence=0.9),
        provider_trace=ProviderTrace(
            uia_used=True,
            vision_used=False,
            provider_details={"vision_provider": {"success": False, "error": "timeout"}},
        ),
        providers_used=[],
        providers_failed=[],
    )

    _refresh_provider_status_from_trace(canvas)

    assert canvas.providers_used == ["uia"]
    assert canvas.providers_failed == ["vision_provider"]
