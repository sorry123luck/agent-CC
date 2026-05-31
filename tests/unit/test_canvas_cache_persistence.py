"""Unit tests for CanvasCache L2 disk persistence (Phase 4).

Verifies:
- CacheConfig loads from YAML with clamping
- L2 warm write creates JPEG file on disk
- L2 FIFO eviction removes oldest files
- L2 write is non-blocking (daemon thread)
- L2 write failure does not raise
- get_screenshot_from_disk reads warm cache
- put() without persist=True does NOT write to disk
- Singleton initialization reads config
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.exc import IntegrityError
from PIL import Image

from src.canvas.canvas_cache import CanvasCache, _WARM_DIR
from src.perception.page_compiler_models import (
    AppInfo,
    Candidate,
    InteractionCanvas,
    PageInfo,
    ProviderTrace,
    SurfaceInfo,
    SurfaceType,
    WindowInfoSnapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def warm_dir(tmp_path, monkeypatch):
    """Patch _WARM_DIR to a temporary directory."""
    warm = tmp_path / "warm"
    warm.mkdir(parents=True)
    monkeypatch.setattr("src.canvas.canvas_cache._WARM_DIR", warm)
    return warm


@pytest.fixture
def cache(warm_dir, monkeypatch):
    """Create a CanvasCache with known config, patched warm dir."""
    monkeypatch.setattr(
        "src.canvas.canvas_cache.CanvasCache._write_canvas_json_to_db",
        staticmethod(lambda *_args, **_kwargs: None),
    )
    return CanvasCache(max_size=4, l2_max_files=3, l2_jpeg_quality=80)


# ---------------------------------------------------------------------------
# CacheConfig tests
# ---------------------------------------------------------------------------

class TestCacheConfig:
    def test_defaults(self):
        from src.common.config_manager import CacheConfig
        cfg = CacheConfig()
        assert cfg.l1_max_size == 8
        assert cfg.l2_max_files == 300
        assert cfg.l2_jpeg_quality == 80
        assert cfg.l3_keyframe_quality == 85

    def test_from_yaml(self):
        from src.common.config_manager import load_config
        mock_yaml = {
            "app.yaml": {"cache": {"l1_max_size": 16, "l2_max_files": 500, "l2_jpeg_quality": 75}},
            "models.yaml": {},
        }
        with patch("src.common.config_manager._load_yaml", side_effect=lambda f: mock_yaml.get(f, {})):
            cfg = load_config()
        assert cfg.cache.l1_max_size == 16
        assert cfg.cache.l2_max_files == 500
        assert cfg.cache.l2_jpeg_quality == 75

    def test_clamps_l1_max_size(self):
        from src.common.config_manager import load_config
        mock_yaml = {
            "app.yaml": {"cache": {"l1_max_size": 200}},
            "models.yaml": {},
        }
        with patch("src.common.config_manager._load_yaml", side_effect=lambda f: mock_yaml.get(f, {})):
            cfg = load_config()
        assert cfg.cache.l1_max_size == 64  # clamped to max

    def test_clamps_l1_max_size_min(self):
        from src.common.config_manager import load_config
        mock_yaml = {
            "app.yaml": {"cache": {"l1_max_size": 0}},
            "models.yaml": {},
        }
        with patch("src.common.config_manager._load_yaml", side_effect=lambda f: mock_yaml.get(f, {})):
            cfg = load_config()
        assert cfg.cache.l1_max_size == 1  # clamped to min

    def test_clamps_jpeg_quality(self):
        from src.common.config_manager import load_config
        mock_yaml = {
            "app.yaml": {"cache": {"l2_jpeg_quality": 100}},
            "models.yaml": {},
        }
        with patch("src.common.config_manager._load_yaml", side_effect=lambda f: mock_yaml.get(f, {})):
            cfg = load_config()
        assert cfg.cache.l2_jpeg_quality == 85  # clamped to max

    def test_empty_yaml_returns_defaults(self):
        from src.common.config_manager import load_config
        with patch("src.common.config_manager._load_yaml", return_value={}):
            cfg = load_config()
        assert cfg.cache.l1_max_size == 8
        assert cfg.cache.l2_max_files == 300


# ---------------------------------------------------------------------------
# L2 warm write tests
# ---------------------------------------------------------------------------

class TestL2WarmWrite:
    def test_persist_creates_jpeg(self, cache, warm_dir):
        canvas = _make_canvas("c_persist")
        screenshot = _make_screenshot()
        cache.put(canvas, screenshot=screenshot, persist=True)

        # Wait for daemon thread
        time.sleep(0.5)

        path = warm_dir / "c_persist.jpg"
        assert path.exists()
        assert path.stat().st_size > 0

    def test_no_persist_no_disk_write(self, cache, warm_dir):
        canvas = _make_canvas("c_no_persist")
        screenshot = _make_screenshot()
        cache.put(canvas, screenshot=screenshot)  # persist=False (default)

        time.sleep(0.3)

        path = warm_dir / "c_no_persist.jpg"
        assert not path.exists()

    def test_rgba_converted(self, cache, warm_dir):
        canvas = _make_canvas("c_rgba")
        screenshot = Image.new("RGBA", (100, 80), color=(255, 0, 0, 128))
        cache.put(canvas, screenshot=screenshot, persist=True)

        time.sleep(0.5)

        path = warm_dir / "c_rgba.jpg"
        assert path.exists()

    def test_write_failure_no_raise(self, cache, warm_dir, monkeypatch):
        """Image.save raising should be caught, not propagated."""
        canvas = _make_canvas("c_fail")
        screenshot = _make_screenshot()

        def bad_save(*args, **kwargs):
            raise IOError("disk full")

        monkeypatch.setattr(screenshot, "save", bad_save)
        # Should not raise
        cache.put(canvas, screenshot=screenshot, persist=True)
        time.sleep(0.5)

    def test_no_screenshot_no_write(self, cache, warm_dir):
        canvas = _make_canvas("c_no_ss")
        cache.put(canvas, screenshot=None, persist=True)

        time.sleep(0.3)

        assert not (warm_dir / "c_no_ss.jpg").exists()

    def test_deleted_canvas_skips_background_persist(self, cache, warm_dir, monkeypatch):
        """Deleting while the background writer is queued must prevent L2/L3 resurrection."""
        canvas = _make_canvas("c_deleted")
        screenshot = _make_screenshot()
        writes: list[str] = []
        release = False

        def slow_write(canvas_id, img):
            nonlocal release
            while not release:
                time.sleep(0.01)
            writes.append(canvas_id)
            (warm_dir / f"{canvas_id}.jpg").write_bytes(b"late-write")

        monkeypatch.setattr(cache, "_write_l2_warm", slow_write)
        monkeypatch.setattr(cache, "_write_canvas_json_to_db", lambda c, *_args: writes.append(f"db:{c.canvas_id}"))

        cache.put(canvas, screenshot=screenshot, persist=True)
        cache.remove_persistent(canvas.canvas_id)
        release = True
        time.sleep(0.2)

        assert "db:c_deleted" not in writes
        assert not (warm_dir / "c_deleted.jpg").exists()

    def test_canvas_json_persist_refreshes_snapshot_metadata(self, tmp_path):
        """Persisting an enhanced canvas must refresh the existing snapshot row counts."""
        from src.canvas.canvas_cache import CanvasCache
        from src.storage.db import Session, init_db
        from src.storage.schema import CanvasSnapshotRecord

        init_db(str(tmp_path / "cache_meta.db")).create_all()
        with Session() as session:
            session.add(CanvasSnapshotRecord(
                snapshot_id="snap-row",
                canvas_id="c_meta",
                captured_at="2026-01-01T00:00:00",
                element_count=1,
                has_screenshot=0,
                page_model_id="pm_old",
                state_template_id="st_old",
            ))

        canvas = _make_canvas("c_meta")
        canvas.page_model_id = "pm_new"
        canvas.state_template_id = "st_new"
        canvas.elements = [Candidate(element_id=f"e{i}") for i in range(4)]

        CanvasCache._write_canvas_json_to_db(canvas)

        with Session() as session:
            row = session.query(CanvasSnapshotRecord).filter_by(canvas_id="c_meta").one()
            assert row.element_count == 4
            assert row.has_screenshot == 1
            assert row.page_model_id == "pm_new"
            assert row.state_template_id == "st_new"

    def test_canvas_json_insert_uses_conflict_safe_upsert(self, monkeypatch):
        """A late duplicate insert race should be handled by the SQL upsert itself."""
        from src.canvas.canvas_cache import CanvasCache

        class Query:
            def filter(self, *_args, **_kwargs):
                return self

            def first(self):
                return None

        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def query(self, *_args, **_kwargs):
                return Query()

            def execute(self, statement, *_args, **_kwargs):
                if "ON CONFLICT" not in str(statement):
                    raise IntegrityError("insert", {}, Exception("duplicate canvas_id"))

            def commit(self):
                return None

        monkeypatch.setattr("src.storage.db.Session", Session)

        canvas = _make_canvas("c_race")

        CanvasCache._write_canvas_json_to_db(canvas)


# ---------------------------------------------------------------------------
# L2 FIFO eviction tests
# ---------------------------------------------------------------------------

class TestL2FIFOEviction:
    def test_removes_oldest_files(self, cache, warm_dir):
        """5 files with l2_max_files=3 → 2 oldest removed."""
        # Create 5 files with staggered mtimes
        for i in range(5):
            path = warm_dir / f"evict_{i}.jpg"
            _make_screenshot().save(str(path), format="JPEG", quality=80)
            # Ensure different mtimes
            time.sleep(0.05)

        cache._evict_l2_fifo()

        remaining = sorted(p.name for p in warm_dir.glob("*.jpg"))
        assert len(remaining) == 3
        # Oldest (evict_0, evict_1) should be removed
        assert "evict_0.jpg" not in remaining
        assert "evict_1.jpg" not in remaining
        assert "evict_2.jpg" in remaining

    def test_no_op_under_capacity(self, cache, warm_dir):
        """2 files with l2_max_files=3 → all survive."""
        for i in range(2):
            path = warm_dir / f"keep_{i}.jpg"
            _make_screenshot().save(str(path), format="JPEG", quality=80)

        cache._evict_l2_fifo()

        remaining = list(warm_dir.glob("*.jpg"))
        assert len(remaining) == 2


# ---------------------------------------------------------------------------
# get_screenshot_from_disk tests
# ---------------------------------------------------------------------------

class TestGetScreenshotFromDisk:
    def test_read_existing(self, warm_dir):
        path = warm_dir / "c_read.jpg"
        _make_screenshot(color=(0, 128, 255)).save(str(path), format="JPEG", quality=80)

        img = CanvasCache.get_screenshot_from_disk("c_read")
        assert img is not None
        assert img.size == (100, 80)

    def test_read_nonexistent(self, warm_dir):
        img = CanvasCache.get_screenshot_from_disk("no_such_canvas")
        assert img is None

    def test_read_corrupted(self, warm_dir):
        path = warm_dir / "c_corrupt.jpg"
        path.write_bytes(b"not a real jpeg")

        img = CanvasCache.get_screenshot_from_disk("c_corrupt")
        assert img is None


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------

class TestSingletonInit:
    def test_reads_config(self, monkeypatch):
        import src.canvas.canvas_cache as mod
        from src.common.config_manager import CacheConfig

        # Reset singleton
        monkeypatch.setattr(mod, "_singleton", None)

        mock_cache = CacheConfig(l1_max_size=12, l2_max_files=200, l2_jpeg_quality=75)
        with patch("src.common.config_manager.load_config") as mock_load:
            mock_load.return_value.cache = mock_cache
            cache = mod.get_canvas_cache()

        assert cache._max_size == 12
        assert cache._l2_max_files == 200
        assert cache._l2_jpeg_quality == 75

        # Clean up singleton for other tests
        monkeypatch.setattr(mod, "_singleton", None)

    def test_idempotent(self, monkeypatch):
        import src.canvas.canvas_cache as mod
        from src.common.config_manager import CacheConfig

        monkeypatch.setattr(mod, "_singleton", None)

        mock_cache = CacheConfig()
        with patch("src.common.config_manager.load_config") as mock_load:
            mock_load.return_value.cache = mock_cache
            c1 = mod.get_canvas_cache()
            c2 = mod.get_canvas_cache()

        assert c1 is c2

        monkeypatch.setattr(mod, "_singleton", None)
