"""
E2E test fixtures.

Uses FastAPI TestClient with:
- In-memory SQLite database (fresh per test)
- Mocked PerceptionService (no real window dependency)
- Canvas cache cleared between tests
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.perception.page_compiler_models import (
    AppInfo,
    Candidate,
    ConfidenceLevel,
    ElementState,
    InteractionCanvas,
    PageInfo,
    ProviderTrace,
    Region,
    RiskLevel,
    SemanticRole,
    SurfaceInfo,
    SurfaceType,
    WindowInfoSnapshot,
)


def _make_test_canvas(
    hwnd: int = 12345,
    app_id: str = "test_app",
    page_class: str = "test_app/main/default",
    element_count: int = 3,
) -> InteractionCanvas:
    """Create a minimal but complete InteractionCanvas for testing."""
    elements = []
    for i in range(element_count):
        elements.append(
            Candidate(
                element_id=f"elem_{i}",
                region_id="content",
                semantic_role=SemanticRole.BUTTON if i == 0 else SemanticRole.TEXT,
                control_type="Button" if i == 0 else "Text",
                bounds=(100 + i * 50, 100, 200 + i * 50, 140),
                text=f"Element {i}",
                name=f"element_{i}",
                interactable=(i == 0),
                state=ElementState(enabled=True, visible=True),
                provider_sources=["uia"],
                confidence_level=ConfidenceLevel.HIGH if i == 0 else ConfidenceLevel.MEDIUM,
                risk_level=RiskLevel.L0,
            )
        )

    regions = [
        Region(
            region_id="content",
            role="content_area",
            bounds=(0, 0, 800, 600),
            element_ids=[e.element_id for e in elements],
        ),
    ]

    canvas = InteractionCanvas(
        canvas_id=str(uuid.uuid4()),
        app=AppInfo(app_id=app_id, process_name=f"{app_id}.exe"),
        window=WindowInfoSnapshot(hwnd=hwnd, title=f"{app_id} Window"),
        surface=SurfaceInfo(surface_type=SurfaceType.NATIVE_UIA, confidence=0.9),
        page=PageInfo(page_class=page_class, class_confidence=0.85),
        regions=regions,
        elements=elements,
        provider_trace=ProviderTrace(uia_used=True),
        providers_used=["uia"],
        canvas_schema_version="1.0",
    )
    return canvas


@pytest.fixture(autouse=True)
def _clear_canvas_cache():
    """Clear the global canvas cache before each test."""
    from src.canvas.canvas_cache import get_canvas_cache

    cache = get_canvas_cache()
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def client():
    """FastAPI TestClient with fresh in-memory DB per test."""
    import sqlalchemy
    from sqlalchemy.orm import sessionmaker

    from sqlalchemy.pool import StaticPool
    engine = sqlalchemy.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    from src.storage.schema import Base

    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    with patch("src.integration.api_server.init_db") as mock_init_db:
        class FakeDB:
            def create_all(self):
                Base.metadata.create_all(engine)

        mock_init_db.return_value = FakeDB()

        with patch("src.storage.db.Session", TestSession):
            from src.integration.api_server import app as fastapi_app

            with TestClient(fastapi_app) as c:
                yield c


@pytest.fixture
def seed_canvas():
    """Seed the canvas cache directly with a test canvas (no observe call needed).

    Returns (canvas_id, canvas) so tests can reference the cached canvas.
    """
    from src.canvas.canvas_cache import get_canvas_cache

    canvas = _make_test_canvas()
    cache = get_canvas_cache()
    cache.put(canvas)
    return canvas.canvas_id, canvas


@pytest.fixture
def seed_canvas_varying():
    """Seed the canvas cache with two canvases (for diff tests).

    Returns (first_canvas_id, first_canvas, second_canvas_id, second_canvas).
    """
    from src.canvas.canvas_cache import get_canvas_cache

    cache = get_canvas_cache()
    first = _make_test_canvas(element_count=3)
    second = _make_test_canvas(element_count=4)
    cache.put(first)
    cache.put(second)
    return first.canvas_id, first, second.canvas_id, second


@pytest.fixture
def mock_observe():
    """Patch _do_observe to return a test canvas instead of calling real perception."""
    canvases = []

    def _fake_observe(
        hwnd: int,
        allow_vlm: bool = False,
        force_vlm: bool = False,
        run_enhancement_phases: bool = True,
        fast_perception: bool | None = None,
        lightweight_uia: bool = False,
    ) -> tuple[InteractionCanvas, str, dict]:
        canvas = _make_test_canvas(hwnd=hwnd)
        canvases.append(canvas)
        from src.canvas.canvas_cache import get_canvas_cache
        get_canvas_cache().put(canvas)
        return canvas, "new_page", {"used": False, "status": "skipped", "provider": ""}

    with patch("src.integration.api_server._do_observe", side_effect=_fake_observe) as m, \
        patch("src.windows.screenshot_service.ScreenshotService.normalize_hwnd", side_effect=lambda hwnd: hwnd), \
        patch("src.windows.screenshot_service.ScreenshotService.is_minimized", return_value=False):
        # Return a getter so tests can access the latest canvas
        yield m, lambda: canvases[-1] if canvases else None


@pytest.fixture
def mock_observe_varying():
    """Patch _do_observe to return different canvases on each call (for diff tests)."""
    call_count = [0]

    def _fake_observe(
        hwnd: int,
        allow_vlm: bool = False,
        force_vlm: bool = False,
        run_enhancement_phases: bool = True,
        fast_perception: bool | None = None,
        lightweight_uia: bool = False,
    ) -> tuple[InteractionCanvas, str, dict]:
        call_count[0] += 1
        if call_count[0] == 1:
            canvas = _make_test_canvas(hwnd=hwnd, element_count=3)
        else:
            canvas = _make_test_canvas(hwnd=hwnd, element_count=4)
        # Also cache it like the real _do_observe does
        from src.canvas.canvas_cache import get_canvas_cache
        get_canvas_cache().put(canvas)
        return canvas, "new_page", {"used": False, "status": "skipped", "provider": ""}

    with patch("src.integration.api_server._do_observe", side_effect=_fake_observe) as m, \
        patch("src.windows.screenshot_service.ScreenshotService.normalize_hwnd", side_effect=lambda hwnd: hwnd), \
        patch("src.windows.screenshot_service.ScreenshotService.is_minimized", return_value=False):
        yield m


@pytest.fixture
def seed_canvas_with_screenshot():
    """Seed the canvas cache with a test canvas AND a screenshot image.

    Returns (canvas_id, canvas, screenshot).
    """
    from PIL import Image

    from src.canvas.canvas_cache import get_canvas_cache

    canvas = _make_test_canvas()
    # Create a small solid-color test image (never written to disk)
    screenshot = Image.new("RGB", (800, 600), color=(100, 149, 237))
    cache = get_canvas_cache()
    cache.put(canvas, screenshot=screenshot)
    return canvas.canvas_id, canvas, screenshot


@pytest.fixture
def seed_canvas_with_vlm_element():
    """Seed the canvas cache with a canvas that includes a VLM-only element.

    The VLM element has provider_sources=["vlm"], confidence=0.85,
    and is not matched by any UIA element.
    Returns (canvas_id, canvas, vlm_element_id).
    """
    from src.canvas.canvas_cache import get_canvas_cache

    vlm_element_id = "vlm_only_send_btn"
    elements = [
        Candidate(
            element_id="elem_0",
            region_id="content",
            semantic_role=SemanticRole.BUTTON,
            control_type="Button",
            bounds=(100, 100, 200, 140),
            text="Element 0",
            name="element_0",
            interactable=True,
            state=ElementState(enabled=True, visible=True),
            provider_sources=["uia"],
            confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
            risk_level=RiskLevel.L0,
        ),
        Candidate(
            element_id=vlm_element_id,
            region_id="content",
            semantic_role=SemanticRole.UNKNOWN,
            control_type="button",
            bounds=(300, 200, 400, 240),
            text="VLM Send",
            name=None,
            interactable=True,
            state=ElementState(enabled=True, visible=True),
            provider_sources=["vlm"],
            confidence=0.85,
            confidence_level=ConfidenceLevel.MEDIUM,
            risk_level=RiskLevel.L0,
            attributes={"vision_only": True, "source": "vlm"},
        ),
    ]

    regions = [
        Region(
            region_id="content",
            role="content_area",
            bounds=(0, 0, 800, 600),
            element_ids=[e.element_id for e in elements],
        ),
    ]

    canvas = InteractionCanvas(
        canvas_id=str(uuid.uuid4()),
        app=AppInfo(app_id="test_app", process_name="test_app.exe"),
        window=WindowInfoSnapshot(hwnd=12345, title="test_app Window"),
        surface=SurfaceInfo(surface_type=SurfaceType.NATIVE_UIA, confidence=0.9),
        page=PageInfo(page_class="test_app/main/default", class_confidence=0.85),
        regions=regions,
        elements=elements,
        provider_trace=ProviderTrace(uia_used=True, vlm_used=True),
        providers_used=["uia", "vlm"],
        canvas_schema_version="1.0",
    )
    cache = get_canvas_cache()
    cache.put(canvas)
    return canvas.canvas_id, canvas, vlm_element_id
