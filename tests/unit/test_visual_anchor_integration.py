"""Tests for VisualAnchorEngine — 视觉锚点对比集成测试。"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.memory.visual_anchor import VisualObservationResult
from src.memory.visual_anchor_engine import (
    VisualAnchorEngine,
    _bounds_valid,
    _drift_to_confidence,
    _safe_filename,
)
from src.memory.visual_asset_store import VisualAssetStore
from src.memory.visual_observation_store import VisualObservationStore
from src.storage.db import Database
from src.storage.migration_evidence import migrate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_all()
    migrate(db)
    return db


@pytest.fixture
def asset_store():
    return VisualAssetStore()


@pytest.fixture
def obs_store():
    return VisualObservationStore()


@pytest.fixture
def engine(asset_store, obs_store):
    return VisualAnchorEngine(asset_store=asset_store, obs_store=obs_store)


class _FakeElement:
    def __init__(
        self,
        element_id="e1",
        stable_key_id=None,
        bounds=None,
        attributes=None,
    ):
        self.element_id = element_id
        self.stable_key_id = stable_key_id
        self.bounds = bounds
        self.attributes = attributes or {}


class _FakeCanvas:
    def __init__(self, canvas_id="c1", elements=None, page_model_id="pm1", state_template_id="st1"):
        self.canvas_id = canvas_id
        self.elements = elements or []
        self.page_model_id = page_model_id
        self.state_template_id = state_template_id


def _make_screenshot(w: int = 200, h: int = 150, color=(100, 150, 200)) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


# ---------------------------------------------------------------------------
# Test: _drift_to_confidence
# ---------------------------------------------------------------------------

class TestDriftToConfidence:
    def test_exact_match_1_0(self):
        result = VisualObservationResult(
            dhash_distance=0, ssim_score=1.0, template_score=None,
            coordinate_drift=None, match_status="exact_match", method="dhash_ssim",
        )
        assert _drift_to_confidence(result) == 1.0

    def test_strong_match_0_85(self):
        result = VisualObservationResult(
            dhash_distance=2, ssim_score=0.90, template_score=0.95,
            coordinate_drift=1.0, match_status="strong_match", method="dhash_ssim_template",
        )
        assert _drift_to_confidence(result) == 0.85

    def test_weak_match_0_50(self):
        result = VisualObservationResult(
            dhash_distance=8, ssim_score=0.75, template_score=0.60,
            coordinate_drift=3.0, match_status="weak_match", method="dhash_ssim_template",
        )
        assert _drift_to_confidence(result) == 0.50

    def test_no_match_0_0(self):
        result = VisualObservationResult(
            dhash_distance=20, ssim_score=None, template_score=None,
            coordinate_drift=None, match_status="no_match", method="dhash_only",
        )
        assert _drift_to_confidence(result) == 0.0

    def test_drift_detected_scales_with_drift(self):
        result_low = VisualObservationResult(
            dhash_distance=4, ssim_score=0.85, template_score=0.90,
            coordinate_drift=2.0, match_status="drift_detected", method="dhash_ssim_template",
        )
        result_high = VisualObservationResult(
            dhash_distance=4, ssim_score=0.85, template_score=0.90,
            coordinate_drift=10.0, match_status="drift_detected", method="dhash_ssim_template",
        )
        assert _drift_to_confidence(result_low) > _drift_to_confidence(result_high)
        assert _drift_to_confidence(result_low) == pytest.approx(0.50, abs=0.01)
        assert _drift_to_confidence(result_high) == pytest.approx(0.10, abs=0.01)


# ---------------------------------------------------------------------------
# Test: _bounds_valid
# ---------------------------------------------------------------------------

class TestBoundsValid:
    def test_valid(self):
        assert _bounds_valid((10, 20, 100, 80)) is True

    def test_zero_area(self):
        assert _bounds_valid((10, 20, 10, 80)) is False

    def test_too_small(self):
        assert _bounds_valid((0, 0, 1, 1)) is False

    def test_none(self):
        assert _bounds_valid(None) is False

    def test_wrong_length(self):
        assert _bounds_valid((10, 20, 100)) is False


# ---------------------------------------------------------------------------
# Test: _safe_filename
# ---------------------------------------------------------------------------

class TestSafeFilename:
    def test_alphanumeric(self):
        assert _safe_filename("abc123") == "abc123"

    def test_special_chars(self):
        assert _safe_filename("key/with:special*chars") == "key_with_special_chars"

    def test_dots_and_dashes(self):
        assert _safe_filename("my-key.v1") == "my-key.v1"


# ---------------------------------------------------------------------------
# Test: First observation stores crop
# ---------------------------------------------------------------------------

class TestVisualAnchorFirstObservation:
    def test_first_obs_stores_crop(self, db, engine, tmp_path):
        """First observation should store a control_crop file."""
        from unittest.mock import patch
        controls_dir = tmp_path / "controls"
        controls_dir.mkdir(parents=True)
        with patch("src.memory.visual_anchor_engine._CONTROLS_DIR", controls_dir):
            element = _FakeElement(
                element_id="e1",
                stable_key_id="sk1",
                bounds=(10, 20, 60, 70),
            )
            canvas = _FakeCanvas(elements=[element])
            screenshot = _make_screenshot()

            with db.session() as session:
                count = engine.apply_to_canvas(session, canvas, screenshot)

            assert count == 1
            crop_files = list(controls_dir.glob("*.png"))
            assert len(crop_files) == 1
            assert "sk1" in crop_files[0].name

    def test_first_obs_creates_asset(self, db, engine, asset_store, tmp_path):
        """First observation should create a visual_assets record."""
        from unittest.mock import patch
        controls_dir = tmp_path / "controls"
        controls_dir.mkdir(parents=True)
        with patch("src.memory.visual_anchor_engine._CONTROLS_DIR", controls_dir):
            element = _FakeElement(
                element_id="e1",
                stable_key_id="sk1",
                bounds=(10, 20, 60, 70),
            )
            canvas = _FakeCanvas(elements=[element])
            screenshot = _make_screenshot()

            with db.session() as session:
                engine.apply_to_canvas(session, canvas, screenshot)
                assets = asset_store.list_by_stable_key(session, "sk1", asset_type="control_crop")

            assert len(assets) == 1
            assert assets[0]["asset_type"] == "control_crop"
            assert assets[0]["stable_key_id"] == "sk1"
            assert assets[0]["format"] == "png"

    def test_first_obs_no_comparison(self, db, engine, obs_store, tmp_path):
        """First observation should NOT write visual_observations."""
        from unittest.mock import patch
        controls_dir = tmp_path / "controls"
        controls_dir.mkdir(parents=True)
        with patch("src.memory.visual_anchor_engine._CONTROLS_DIR", controls_dir):
            element = _FakeElement(
                element_id="e1",
                stable_key_id="sk1",
                bounds=(10, 20, 60, 70),
            )
            canvas = _FakeCanvas(elements=[element])
            screenshot = _make_screenshot()

            with db.session() as session:
                engine.apply_to_canvas(session, canvas, screenshot)
                obs = obs_store.list_by_stable_key(session, "sk1")

            assert len(obs) == 0


# ---------------------------------------------------------------------------
# Test: Subsequent observation compares
# ---------------------------------------------------------------------------

class TestVisualAnchorComparison:
    def _seed_baseline(self, db, engine, tmp_path, controls_dir):
        """Store baseline crop for sk1."""
        from unittest.mock import patch
        controls_dir.mkdir(parents=True)
        with patch("src.memory.visual_anchor_engine._CONTROLS_DIR", controls_dir):
            element = _FakeElement(
                element_id="e1",
                stable_key_id="sk1",
                bounds=(10, 20, 60, 70),
            )
            canvas = _FakeCanvas(elements=[element])
            screenshot = _make_screenshot()
            with db.session() as session:
                engine.apply_to_canvas(session, canvas, screenshot)

    def test_same_element_compares(self, db, engine, asset_store, obs_store, tmp_path):
        """Same stable_key_id second observation should do comparison."""
        controls_dir = tmp_path / "controls"
        self._seed_baseline(db, engine, tmp_path, controls_dir)

        element2 = _FakeElement(
            element_id="e2",
            stable_key_id="sk1",
            bounds=(10, 20, 60, 70),
        )
        canvas2 = _FakeCanvas(canvas_id="c2", elements=[element2])
        screenshot2 = _make_screenshot()

        with db.session() as session:
            count = engine.apply_to_canvas(session, canvas2, screenshot2)

        assert count == 1

    def test_match_status_exact(self, db, engine, asset_store, obs_store, tmp_path):
        """Same screenshot → exact_match."""
        controls_dir = tmp_path / "controls"
        self._seed_baseline(db, engine, tmp_path, controls_dir)

        element2 = _FakeElement(
            element_id="e2",
            stable_key_id="sk1",
            bounds=(10, 20, 60, 70),
        )
        canvas2 = _FakeCanvas(canvas_id="c2", elements=[element2])
        screenshot2 = _make_screenshot()

        with db.session() as session:
            engine.apply_to_canvas(session, canvas2, screenshot2)
            obs = obs_store.list_by_stable_key(session, "sk1")

        assert len(obs) == 1
        assert obs[0]["match_status"] == "exact_match"

    def test_observation_written(self, db, engine, obs_store, tmp_path):
        """visual_observations table should have a record."""
        controls_dir = tmp_path / "controls"
        self._seed_baseline(db, engine, tmp_path, controls_dir)

        element2 = _FakeElement(
            element_id="e2",
            stable_key_id="sk1",
            bounds=(10, 20, 60, 70),
        )
        canvas2 = _FakeCanvas(canvas_id="c2", elements=[element2])
        screenshot2 = _make_screenshot()

        with db.session() as session:
            engine.apply_to_canvas(session, canvas2, screenshot2)
            obs = obs_store.list_by_stable_key(session, "sk1")

        assert len(obs) == 1
        assert obs[0]["asset_id_a"] is not None
        assert obs[0]["asset_id_b"] is None
        assert obs[0]["dhash_distance"] is not None
        assert obs[0]["match_status"] is not None

    def test_confidence_written_to_attributes(self, db, engine, tmp_path):
        """visual_anchor_confidence should be in element.attributes."""
        controls_dir = tmp_path / "controls"
        self._seed_baseline(db, engine, tmp_path, controls_dir)

        element2 = _FakeElement(
            element_id="e2",
            stable_key_id="sk1",
            bounds=(10, 20, 60, 70),
        )
        canvas2 = _FakeCanvas(canvas_id="c2", elements=[element2])
        screenshot2 = _make_screenshot()

        with db.session() as session:
            engine.apply_to_canvas(session, canvas2, screenshot2)

        assert "visual_anchor_confidence" in element2.attributes
        assert element2.attributes["visual_anchor_confidence"] == pytest.approx(1.0)
        assert element2.attributes["visual_anchor_status"] == "exact_match"


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestVisualAnchorEdgeCases:
    def test_no_screenshot_returns_0(self, db, engine):
        element = _FakeElement(stable_key_id="sk1", bounds=(10, 20, 60, 70))
        canvas = _FakeCanvas(elements=[element])
        with db.session() as session:
            count = engine.apply_to_canvas(session, canvas, None)
        assert count == 0

    def test_no_stable_key_skipped(self, db, engine):
        element = _FakeElement(element_id="e1", bounds=(10, 20, 60, 70))
        canvas = _FakeCanvas(elements=[element])
        screenshot = _make_screenshot()
        with db.session() as session:
            count = engine.apply_to_canvas(session, canvas, screenshot)
        assert count == 0

    def test_invalid_bounds_skipped(self, db, engine):
        element = _FakeElement(stable_key_id="sk1", bounds=(0, 0, 0, 0))
        canvas = _FakeCanvas(elements=[element])
        screenshot = _make_screenshot()
        with db.session() as session:
            count = engine.apply_to_canvas(session, canvas, screenshot)
        assert count == 0

    def test_none_bounds_skipped(self, db, engine):
        element = _FakeElement(stable_key_id="sk1", bounds=None)
        canvas = _FakeCanvas(elements=[element])
        screenshot = _make_screenshot()
        with db.session() as session:
            count = engine.apply_to_canvas(session, canvas, screenshot)
        assert count == 0

    def test_element_exception_no_propagate(self, db, engine, tmp_path):
        """Exception in one element should not propagate to caller."""
        from unittest.mock import patch
        controls_dir = tmp_path / "controls"
        controls_dir.mkdir(parents=True)

        bad_element = _FakeElement(
            element_id="bad",
            stable_key_id="sk_bad",
            bounds=(10, 20, 60, 70),
        )
        # Make crop_element fail by passing a mock that raises
        bad_screenshot = MagicMock(spec=Image.Image)
        bad_screenshot.size = (200, 150)
        bad_screenshot.crop.side_effect = RuntimeError("crop failed")

        canvas = _FakeCanvas(elements=[bad_element])
        with patch("src.memory.visual_anchor_engine._CONTROLS_DIR", controls_dir):
            with db.session() as session:
                count = engine.apply_to_canvas(session, canvas, bad_screenshot)
        assert count == 0

    def test_canvas_format_unchanged(self, db, engine, tmp_path):
        """Canvas structure should not be modified by apply_to_canvas."""
        from unittest.mock import patch
        controls_dir = tmp_path / "controls"
        controls_dir.mkdir(parents=True)

        element = _FakeElement(
            element_id="e1",
            stable_key_id="sk1",
            bounds=(10, 20, 60, 70),
        )
        canvas = _FakeCanvas(elements=[element])
        screenshot = _make_screenshot()

        original_element_count = len(canvas.elements)
        original_canvas_id = canvas.canvas_id

        with patch("src.memory.visual_anchor_engine._CONTROLS_DIR", controls_dir):
            with db.session() as session:
                engine.apply_to_canvas(session, canvas, screenshot)

        assert len(canvas.elements) == original_element_count
        assert canvas.canvas_id == original_canvas_id

    def test_multiple_elements_partial_stable_key(self, db, engine, tmp_path):
        """Only elements with stable_key_id should be processed."""
        from unittest.mock import patch
        controls_dir = tmp_path / "controls"
        controls_dir.mkdir(parents=True)

        e1 = _FakeElement(element_id="e1", stable_key_id="sk1", bounds=(10, 20, 60, 70))
        e2 = _FakeElement(element_id="e2", bounds=(70, 20, 120, 70))  # no stable_key_id
        e3 = _FakeElement(element_id="e3", stable_key_id="sk3", bounds=(10, 80, 60, 130))

        canvas = _FakeCanvas(elements=[e1, e2, e3])
        screenshot = _make_screenshot()

        with patch("src.memory.visual_anchor_engine._CONTROLS_DIR", controls_dir):
            with db.session() as session:
                count = engine.apply_to_canvas(session, canvas, screenshot)

        assert count == 2  # e1 and e3 processed, e2 skipped
