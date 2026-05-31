"""VisualAnchorEngine — 跨截图视觉锚点对比引擎。

在 observe 管线中调用，对有 stable_key_id 的元素做视觉对比。
首次观察：存储 crop 作为 baseline。
后续观察：对比当前 crop 与 baseline，记录 VisualObservation。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from PIL import Image

from src.memory.visual_anchor import VisualObservationResult, compare_assets
from src.memory.visual_asset_store import VisualAssetStore
from src.memory.visual_observation_store import VisualObservationStore

_logger = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
_CONTROLS_DIR = _ROOT_DIR / "data" / "visual_assets" / "controls"


def _safe_filename(s: str) -> str:
    """Sanitize string for use as a filename (defense-in-depth)."""
    return re.sub(r"[^\w\-.]", "_", s)


def _drift_to_confidence(result: VisualObservationResult) -> float:
    """Convert VisualObservationResult to 0.0-1.0 visual_anchor_confidence."""
    if result.match_status == "exact_match":
        return 1.0
    if result.match_status == "strong_match":
        return 0.85
    if result.match_status == "weak_match":
        return 0.50
    if result.match_status == "drift_detected":
        drift = result.coordinate_drift or 0.0
        return max(0.0, 0.60 - drift * 0.05)
    # no_match
    return 0.0


def _bounds_valid(bounds) -> bool:
    """Check if bounds are valid (4-tuple, positive area, min 2px)."""
    if not bounds or len(bounds) != 4:
        return False
    l, t, r, b = bounds
    return r > l and b > t and (r - l) >= 2 and (b - t) >= 2


class VisualAnchorEngine:
    """跨截图视觉锚点对比引擎。"""

    def __init__(
        self,
        asset_store: VisualAssetStore | None = None,
        obs_store: VisualObservationStore | None = None,
    ) -> None:
        self._asset_store = asset_store or VisualAssetStore()
        self._obs_store = obs_store or VisualObservationStore()

    def apply_to_canvas(
        self,
        session,
        canvas,
        screenshot: Image.Image | None,
    ) -> int:
        """对所有有 stable_key_id 的元素执行视觉对比。返回处理数。"""
        if screenshot is None:
            return 0
        count = 0
        for element in canvas.elements:
            try:
                stable_key_id = getattr(element, "stable_key_id", None)
                bounds = getattr(element, "bounds", None)
                if not stable_key_id or not bounds:
                    continue
                if not _bounds_valid(bounds):
                    continue

                baseline_assets = self._asset_store.list_by_stable_key(
                    session,
                    stable_key_id,
                    asset_type="control_crop",
                    status="active",
                )

                if not baseline_assets:
                    self._store_control_crop(session, element, canvas, screenshot)
                    count += 1
                    continue

                current_crop = self._crop_element(screenshot, bounds)
                if current_crop is None:
                    continue

                baseline_path = baseline_assets[0]["path"]
                result = compare_assets(
                    baseline_path=baseline_path,
                    current_image=_pil_to_ndarray(current_crop),
                    expected_bounds=tuple(int(v) for v in bounds),
                )

                self._obs_store.create_observation(
                    session,
                    asset_id_a=baseline_assets[0]["asset_id"],
                    asset_id_b=None,
                    stable_key_id=stable_key_id,
                    observed_canvas_id=canvas.canvas_id,
                    observed_bounds_json=json.dumps(list(bounds)),
                    match_status=result.match_status,
                    dhash_distance=result.dhash_distance,
                    ssim_score=result.ssim_score,
                    template_score=result.template_score,
                    coordinate_drift=result.coordinate_drift,
                    method=result.method,
                )

                if not hasattr(element, "attributes") or element.attributes is None:
                    element.attributes = {}
                element.attributes["visual_anchor_confidence"] = _drift_to_confidence(result)
                element.attributes["visual_anchor_status"] = result.match_status
                count += 1
            except Exception as exc:
                _logger.warning(
                    "visual anchor failed for element=%s: %s",
                    getattr(element, "element_id", "?"),
                    exc,
                )
        return count

    def _store_control_crop(
        self,
        session,
        element,
        canvas,
        screenshot: Image.Image,
    ) -> str | None:
        """Crop element area and store as control_crop asset. Returns asset_id."""
        bounds = getattr(element, "bounds", None)
        crop = self._crop_element(screenshot, bounds)
        if crop is None:
            return None

        stable_key_id = getattr(element, "stable_key_id", None)
        _CONTROLS_DIR.mkdir(parents=True, exist_ok=True)

        version = self._asset_store.next_version(session, stable_key_id, "control_crop")
        filename = f"{_safe_filename(stable_key_id)}_v{version}.png"
        path = _CONTROLS_DIR / filename
        crop.save(str(path), format="PNG")

        asset_id = self._asset_store.create_asset(
            session,
            asset_type="control_crop",
            path=str(path.resolve()),
            format="png",
            stable_key_id=stable_key_id,
            canvas_id=canvas.canvas_id,
            width=crop.width,
            height=crop.height,
            bounds_json=json.dumps(list(bounds)),
            source_provider="visual_anchor_engine",
        )
        return asset_id

    @staticmethod
    def _crop_element(screenshot: Image.Image, bounds) -> Image.Image | None:
        """Crop element area from screenshot. bounds = (left, top, right, bottom)."""
        if not bounds or len(bounds) != 4:
            return None
        l, t, r, b = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
        if r <= l or b <= t:
            return None
        w, h = screenshot.size
        l, t = max(0, l), max(0, t)
        r, b = min(w, r), min(h, b)
        if r <= l or b <= t:
            return None
        return screenshot.crop((l, t, r, b))


def _pil_to_ndarray(img: Image.Image):
    """Convert PIL Image to BGR numpy array for visual_anchor.compare_assets().

    visual_anchor._load_image() returns BGR when OpenCV is available (the
    common case). To ensure consistent channel order between baseline and
    current image, we flip PIL's RGB to BGR.
    """
    import numpy as np
    rgb = img.convert("RGB")
    return np.array(rgb)[:, :, ::-1]  # RGB -> BGR
