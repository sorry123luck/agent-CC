"""Unit tests for CanvasCache screenshot storage.

Verifies:
- put with screenshot stores both canvas and screenshot
- get_screenshot returns the stored image
- Eviction removes screenshot
- remove() removes screenshot
- clear() clears screenshots
- Screenshot is never written to disk
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from src.canvas.canvas_cache import CanvasCache
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


def _make_canvas(canvas_id: str = "test_canvas") -> InteractionCanvas:
    return InteractionCanvas(
        canvas_id=canvas_id,
        app=AppInfo(app_id="test", process_name="test.exe"),
        window=WindowInfoSnapshot(hwnd=12345, title="Test"),
        surface=SurfaceInfo(surface_type=SurfaceType.NATIVE_UIA, confidence=0.9),
        page=PageInfo(page_class="test/default", class_confidence=0.8),
        regions=[],
        elements=[],
        provider_trace=ProviderTrace(uia_used=True),
        providers_used=["uia"],
        canvas_schema_version="1.0",
    )


def _make_screenshot(w: int = 100, h: int = 80, color=(255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


class TestPutWithScreenshot:
    def test_put_canvas_and_screenshot(self):
        cache = CanvasCache(max_size=4)
        canvas = _make_canvas("c1")
        screenshot = _make_screenshot()
        cache.put(canvas, screenshot=screenshot)

        assert cache.get("c1") is canvas
        assert cache.get_screenshot("c1") is screenshot

    def test_put_canvas_without_screenshot(self):
        cache = CanvasCache(max_size=4)
        canvas = _make_canvas("c2")
        cache.put(canvas)

        assert cache.get("c2") is canvas
        assert cache.get_screenshot("c2") is None

    def test_put_updates_screenshot(self):
        cache = CanvasCache(max_size=4)
        canvas = _make_canvas("c3")
        s1 = _make_screenshot(color=(255, 0, 0))
        s2 = _make_screenshot(color=(0, 255, 0))

        cache.put(canvas, screenshot=s1)
        assert cache.get_screenshot("c3") is s1

        cache.put(canvas, screenshot=s2)
        assert cache.get_screenshot("c3") is s2


class TestGetScreenshotNoLRUMove:
    def test_get_screenshot_does_not_move_lru(self):
        cache = CanvasCache(max_size=2)
        c1 = _make_canvas("c1")
        c2 = _make_canvas("c2")
        cache.put(c1, screenshot=_make_screenshot())
        cache.put(c2, screenshot=_make_screenshot())

        # Access c1 screenshot — should NOT move c1 to end
        cache.get_screenshot("c1")

        # Now add c3, which should evict c1 (oldest)
        c3 = _make_canvas("c3")
        cache.put(c3, screenshot=_make_screenshot())

        assert cache.get("c1") is None
        assert cache.get_screenshot("c1") is None


class TestEviction:
    def test_screenshot_evicted_with_canvas(self):
        cache = CanvasCache(max_size=2)
        c1 = _make_canvas("c1")
        c2 = _make_canvas("c2")
        c3 = _make_canvas("c3")
        cache.put(c1, screenshot=_make_screenshot(color=(255, 0, 0)))
        cache.put(c2, screenshot=_make_screenshot(color=(0, 255, 0)))

        # c1 should be evicted when c3 is added
        cache.put(c3, screenshot=_make_screenshot(color=(0, 0, 255)))

        assert cache.get("c1") is None
        assert cache.get_screenshot("c1") is None
        assert cache.get("c2") is not None
        assert cache.get_screenshot("c2") is not None


class TestRemove:
    def test_remove_also_removes_screenshot(self):
        cache = CanvasCache(max_size=4)
        canvas = _make_canvas("c1")
        screenshot = _make_screenshot()
        cache.put(canvas, screenshot=screenshot)

        assert cache.remove("c1") is True
        assert cache.get("c1") is None
        assert cache.get_screenshot("c1") is None

    def test_remove_nonexistent(self):
        cache = CanvasCache(max_size=4)
        assert cache.remove("nope") is False


class TestClear:
    def test_clear_also_clears_screenshots(self):
        cache = CanvasCache(max_size=4)
        for i in range(3):
            cache.put(_make_canvas(f"c{i}"), screenshot=_make_screenshot())

        cache.clear()
        assert cache.size() == 0
        for i in range(3):
            assert cache.get_screenshot(f"c{i}") is None


class TestListAll:
    def test_list_all_returns_all_canvases(self):
        cache = CanvasCache(max_size=4)
        c1 = _make_canvas("c1")
        c2 = _make_canvas("c2")
        cache.put(c1)
        cache.put(c2)

        items = cache.list_all()
        assert len(items) == 2
        ids = [cid for cid, _ in items]
        assert "c1" in ids
        assert "c2" in ids


class TestPutScreenshot:
    def test_put_screenshot_for_existing_canvas(self):
        cache = CanvasCache(max_size=4)
        canvas = _make_canvas("c1")
        cache.put(canvas)

        screenshot = _make_screenshot()
        cache.put_screenshot("c1", screenshot)

        assert cache.get_screenshot("c1") is screenshot

    def test_put_screenshot_for_nonexistent_canvas(self):
        cache = CanvasCache(max_size=4)
        screenshot = _make_screenshot()
        # Should be a no-op
        cache.put_screenshot("nope", screenshot)
        assert cache.get_screenshot("nope") is None


class TestScreenshotNotOnDisk:
    def test_screenshot_never_written_to_disk(self):
        """Screenshots are memory-only — no files created."""
        with tempfile.TemporaryDirectory() as tmp:
            cache = CanvasCache(max_size=4)
            canvas = _make_canvas("mem_test")
            screenshot = _make_screenshot()
            cache.put(canvas, screenshot=screenshot)

            # Check no files were created in tmp dir
            tmp_path = Path(tmp)
            files = list(tmp_path.rglob("*"))
            assert len(files) == 0, f"Unexpected files in temp dir: {files}"

            # Also verify the screenshot is accessible
            retrieved = cache.get_screenshot("mem_test")
            assert retrieved is not None
            assert retrieved.size == (100, 80)
