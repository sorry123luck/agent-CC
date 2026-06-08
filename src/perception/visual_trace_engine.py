"""Visual Trace Layer (VTL) Engine.

Shadow-only engine that detects page containers and local objects
from screenshot pixels. Does NOT participate in the main pipeline.

VTL-0a: page_container detection (bg_color_block, low_texture, shell)
VTL-0b: local_object detection (connected_component, bordered_rect, color_blob)

Gate: OPENCLAW_VTL_0_SHADOW=1 to enable (default off).
Output: artifacts["visual_trace_shadow"]
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.perception.visual_trace_model import (
    DETECT_BG_COLOR_BLOCK,
    DETECT_BORDERED_RECT,
    DETECT_BOTTOM_BAR,
    DETECT_COLOR_BLOB,
    DETECT_CONNECTED_COMPONENT,
    DETECT_LOW_TEXTURE,
    DETECT_SHELL_SIDE_RAIL,
    DETECT_SHELL_TOP_BAND,
    OBJECT_CLASS_ACTIONABLE,
    OBJECT_CLASS_CONTENT,
    OBJECT_CLASS_PAGE_CONTAINER,
    OBJECT_CLASS_TEXT_SUB,
    OBJECT_CLASS_VISUAL_CANDIDATE,
    OBJECT_LEVEL_LOCAL_OBJECT,
    OBJECT_LEVEL_PAGE_CONTAINER,
    VisualTraceObject,
    make_trace_object,
)

logger = logging.getLogger(__name__)


class VisualTraceEngine:
    """Detect page containers and local objects from screenshot pixels.

    VTL-0: Shadow-only, no main pipeline integration.
    """

    # Page container thresholds
    MIN_CONTAINER_AREA_RATIO = 0.02   # 2% of window
    MAX_CONTAINER_AREA_RATIO = 0.95   # 95% of window
    LOW_TEXTURE_STD_THRESHOLD = 15.0  # pixel std below this = low texture
    COLOR_SIMILARITY_THRESHOLD = 20.0 # mean color diff below this = same block

    # Local object thresholds
    MIN_OBJECT_AREA_RATIO = 0.0005    # 0.05% of window
    MAX_OBJECT_AREA_RATIO = 0.25      # 25% of window
    MIN_OBJECT_SIZE = 8               # min 8px in any dimension
    MAX_OBJECT_SIZE_RATIO = 15.0      # max 15:1 aspect ratio

    # object_class classification
    ACTIONABLE_MAX_AREA_RATIO = 0.01  # 1% of window
    ACTIONABLE_MIN_EDGE_DENSITY = 0.08
    TEXT_SUB_MAX_EDGE_DENSITY = 0.04
    TEXT_SUB_MIN_ASPECT_RATIO = 3.0
    TEXT_SUB_MIN_OCR_COUNT = 2

    def build(
        self,
        screenshot: Image.Image | None,
        ocr_blocks: list[dict[str, Any]] | None = None,
        vision_candidates: list[dict[str, Any]] | None = None,
        raw_elements: list[dict[str, Any]] | None = None,
        window_width: int = 0,
        window_height: int = 0,
    ) -> dict[str, Any]:
        """Build visual_trace_shadow artifact."""
        if screenshot is None:
            return {"page_containers": [], "local_objects": [], "text_substructures": [],
                    "total_containers": 0, "total_objects": 0, "total_text_subs": 0,
                    "excluded_reason": "no_screenshot"}

        if window_width <= 0 or window_height <= 0:
            return {"page_containers": [], "local_objects": [], "text_substructures": [],
                    "total_containers": 0, "total_objects": 0, "total_text_subs": 0,
                    "excluded_reason": "invalid_dimensions"}

        rgb = np.asarray(screenshot.convert("RGB"))
        gray = np.asarray(screenshot.convert("L"))
        height, width = gray.shape
        window_area = width * height

        # Skip if image is too uniform (no meaningful visual structure)
        if float(gray.std()) < 5.0:
            return {"page_containers": [], "local_objects": [], "text_substructures": [],
                    "total_containers": 0, "total_objects": 0, "total_text_subs": 0,
                    "excluded_reason": "uniform_image"}

        # VTL-0a: page container candidates (all), then filter to high-confidence
        container_candidates = self._detect_page_containers(rgb, gray, width, height, window_area)
        page_containers = self._filter_page_containers(container_candidates, width, height, window_area)

        # VTL-0b: local object candidates (all), then classify and filter
        raw_objects = self._detect_local_objects(gray, rgb, width, height, window_area)
        local_objects, text_subs, rejected_text = self._classify_and_filter_objects(
            raw_objects, gray, ocr_blocks, vision_candidates, raw_elements, window_area,
        )

        # Set parent/child relationships (containment ratio, not IoU)
        self._set_hierarchy(page_containers, local_objects)

        # Get audit metadata
        audit = getattr(self, "_last_audit", {})

        # Actionable audit
        actionable_objects = [o for o in local_objects if o.object_class == OBJECT_CLASS_ACTIONABLE]
        actionable_reasons: dict[str, int] = {}
        for obj in actionable_objects:
            reason = getattr(obj, "_actionable_reason", "unknown")
            actionable_reasons[reason] = actionable_reasons.get(reason, 0) + 1

        # Get classify audit
        classify_audit = getattr(self, "_classify_audit", {})

        return {
            "page_containers": [c.to_dict() for c in page_containers],
            "container_candidates": [c.to_dict() for c in container_candidates],
            "local_objects": [o.to_dict() for o in local_objects],
            "text_substructures": [t.to_dict() for t in text_subs],
            "rejected_text_like": [r.to_dict() for r in rejected_text],
            "total_containers": len(page_containers),
            "total_container_candidates": len(container_candidates),
            "total_objects": len(local_objects),
            "total_text_subs": len(text_subs),
            "total_rejected_text": len(rejected_text),
            "total_actionable": len(actionable_objects),
            "accepted_actionable_by_reason": actionable_reasons,
            "shape_only_actionable_count": 0,  # Always 0 (removed)
            "negative_label_demoted_count": classify_audit.get("negative_label_demoted", 0),
            "icon_only_demoted_count": classify_audit.get("icon_only_demoted", 0),
            "uia_actionable_match_details": classify_audit.get("uia_actionable_match_details", []),
            "audit": audit,
            "window_width": width,
            "window_height": height,
        }

    # ── VTL-0a: Page Container Detection ────────────────────────

    def _detect_page_containers(
        self,
        rgb: np.ndarray,
        gray: np.ndarray,
        width: int,
        height: int,
        window_area: int,
    ) -> list[VisualTraceObject]:
        """Detect page-level containers from screenshot."""
        containers: list[VisualTraceObject] = []
        trace_idx = 0

        # 1. Background color blocks
        bg_blocks = self._detect_bg_color_blocks(rgb, gray, width, height, window_area)
        for bounds, conf in bg_blocks:
            containers.append(make_trace_object(
                trace_id=f"VT{trace_idx}",
                bounds=bounds,
                shape_type="container",
                object_level=OBJECT_LEVEL_PAGE_CONTAINER,
                object_class=OBJECT_CLASS_PAGE_CONTAINER,
                detection_method=DETECT_BG_COLOR_BLOCK,
                confidence=conf,
                window_area=window_area,
            ))
            trace_idx += 1

        # 2. Low texture blocks
        lt_blocks = self._detect_low_texture_blocks(gray, width, height, window_area)
        for bounds, conf in lt_blocks:
            if not self._overlaps_existing(bounds, containers, 0.5):
                containers.append(make_trace_object(
                    trace_id=f"VT{trace_idx}",
                    bounds=bounds,
                    shape_type="container",
                    object_level=OBJECT_LEVEL_PAGE_CONTAINER,
                    object_class=OBJECT_CLASS_PAGE_CONTAINER,
                    detection_method=DETECT_LOW_TEXTURE,
                    confidence=conf,
                    window_area=window_area,
                ))
                trace_idx += 1

        # 3. Top band
        top = self._detect_top_band(gray, rgb, width, height, window_area)
        if top and not self._overlaps_existing(top[0], containers, 0.5):
            containers.append(make_trace_object(
                trace_id=f"VT{trace_idx}",
                bounds=top[0],
                shape_type="rectangle",
                object_level=OBJECT_LEVEL_PAGE_CONTAINER,
                object_class=OBJECT_CLASS_PAGE_CONTAINER,
                detection_method=DETECT_SHELL_TOP_BAND,
                confidence=top[1],
                window_area=window_area,
            ))
            trace_idx += 1

        # 4. Bottom bar
        bottom = self._detect_bottom_bar(gray, rgb, width, height, window_area)
        if bottom and not self._overlaps_existing(bottom[0], containers, 0.5):
            containers.append(make_trace_object(
                trace_id=f"VT{trace_idx}",
                bounds=bottom[0],
                shape_type="rectangle",
                object_level=OBJECT_LEVEL_PAGE_CONTAINER,
                object_class=OBJECT_CLASS_PAGE_CONTAINER,
                detection_method=DETECT_BOTTOM_BAR,
                confidence=bottom[1],
                window_area=window_area,
            ))
            trace_idx += 1

        # 5. Side rail
        side = self._detect_side_rail(gray, rgb, width, height, window_area)
        if side and not self._overlaps_existing(side[0], containers, 0.5):
            containers.append(make_trace_object(
                trace_id=f"VT{trace_idx}",
                bounds=side[0],
                shape_type="rectangle",
                object_level=OBJECT_LEVEL_PAGE_CONTAINER,
                object_class=OBJECT_CLASS_PAGE_CONTAINER,
                detection_method=DETECT_SHELL_SIDE_RAIL,
                confidence=side[1],
                window_area=window_area,
            ))
            trace_idx += 1

        return containers

    def _detect_bg_color_blocks(
        self, rgb: np.ndarray, gray: np.ndarray, width: int, height: int, window_area: int,
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        """Detect large background color blocks."""
        # Downsample for speed
        scale = max(1, min(width, height) // 200)
        small = rgb[::scale, ::scale]
        h_s, w_s = small.shape[:2]

        # K-means clustering
        pixels = small.reshape(-1, 3).astype(np.float32)
        n_clusters = min(6, max(2, len(np.unique(pixels, axis=0)) // 10))
        if n_clusters < 2:
            return []

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        except cv2.error:
            return []

        label_image = labels.reshape(h_s, w_s)
        results = []

        for cluster_id in range(n_clusters):
            mask = (label_image == cluster_id).astype(np.uint8) * 255
            # Scale back to original size
            mask_full = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                area_ratio = area / window_area
                if area_ratio < self.MIN_CONTAINER_AREA_RATIO or area_ratio > self.MAX_CONTAINER_AREA_RATIO:
                    continue
                # Check uniformity
                region = gray[y:y+h, x:x+w]
                if region.size == 0:
                    continue
                std = float(region.std())
                conf = max(0.3, min(0.9, 1.0 - std / 100.0))
                results.append(((x, y, x + w, y + h), conf))

        return results

    def _detect_low_texture_blocks(
        self, gray: np.ndarray, width: int, height: int, window_area: int,
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        """Detect large low-texture regions."""
        # Compute local std in blocks
        block_size = max(16, min(width, height) // 20)
        results = []

        for y in range(0, height - block_size, block_size // 2):
            for x in range(0, width - block_size, block_size // 2):
                block = gray[y:y+block_size, x:x+block_size]
                if block.size == 0:
                    continue
                std = float(block.std())
                if std < self.LOW_TEXTURE_STD_THRESHOLD:
                    # Try to expand the block
                    ex, ey, ew, eh = self._expand_low_texture(gray, x, y, block_size, block_size, width, height)
                    area = ew * eh
                    area_ratio = area / window_area
                    if area_ratio >= self.MIN_CONTAINER_AREA_RATIO:
                        conf = max(0.3, min(0.8, 1.0 - std / 50.0))
                        results.append(((ex, ey, ex + ew, ey + eh), conf))

        # Deduplicate
        return self._dedupe_rects(results, 0.5)

    def _expand_low_texture(
        self, gray: np.ndarray, x: int, y: int, w: int, h: int,
        max_w: int, max_h: int, step: int = 16,
    ) -> tuple[int, int, int, int]:
        """Expand a low-texture block in all directions."""
        l, t, r, b = x, y, x + w, y + h
        mean_val = float(gray[t:b, l:r].mean())

        for _ in range(20):
            expanded = False
            # Expand left
            if l > 0:
                new_l = max(0, l - step)
                strip = gray[t:b, new_l:l]
                if strip.size > 0 and abs(float(strip.mean()) - mean_val) < self.COLOR_SIMILARITY_THRESHOLD:
                    l = new_l
                    expanded = True
            # Expand right
            if r < max_w:
                new_r = min(max_w, r + step)
                strip = gray[t:b, r:new_r]
                if strip.size > 0 and abs(float(strip.mean()) - mean_val) < self.COLOR_SIMILARITY_THRESHOLD:
                    r = new_r
                    expanded = True
            # Expand top
            if t > 0:
                new_t = max(0, t - step)
                strip = gray[new_t:t, l:r]
                if strip.size > 0 and abs(float(strip.mean()) - mean_val) < self.COLOR_SIMILARITY_THRESHOLD:
                    t = new_t
                    expanded = True
            # Expand bottom
            if b < max_h:
                new_b = min(max_h, b + step)
                strip = gray[b:new_b, l:r]
                if strip.size > 0 and abs(float(strip.mean()) - mean_val) < self.COLOR_SIMILARITY_THRESHOLD:
                    b = new_b
                    expanded = True
            if not expanded:
                break

        return l, t, r - l, b - t

    def _detect_top_band(
        self, gray: np.ndarray, rgb: np.ndarray, width: int, height: int, window_area: int,
    ) -> tuple[tuple[int, int, int, int], float] | None:
        """Detect top band (title/menu bar)."""
        band_h = min(int(height * 0.08), 60)
        if band_h < 10:
            return None
        band = gray[:band_h, :]
        std = float(band.std())
        if std < 40:  # Top bands are usually relatively uniform
            return ((0, 0, width, band_h), max(0.4, min(0.8, 1.0 - std / 100.0)))
        return None

    def _detect_bottom_bar(
        self, gray: np.ndarray, rgb: np.ndarray, width: int, height: int, window_area: int,
    ) -> tuple[tuple[int, int, int, int], float] | None:
        """Detect bottom bar (status/player/input)."""
        band_h = min(int(height * 0.06), 50)
        if band_h < 10:
            return None
        band = gray[height - band_h:, :]
        std = float(band.std())
        if std < 50:
            return ((0, height - band_h, width, height), max(0.3, min(0.7, 1.0 - std / 100.0)))
        return None

    def _detect_side_rail(
        self, gray: np.ndarray, rgb: np.ndarray, width: int, height: int, window_area: int,
    ) -> tuple[tuple[int, int, int, int], float] | None:
        """Detect left side rail."""
        rail_w = min(int(width * 0.15), 200)
        if rail_w < 20:
            return None
        rail = gray[:, :rail_w]
        # Check if the right edge of the rail has a strong vertical transition
        edge_col = gray[:, rail_w:rail_w + 5]
        if edge_col.size == 0:
            return None
        edge_strength = float(np.abs(np.diff(edge_col.mean(axis=1))).mean())
        if edge_strength > 10:
            return ((0, 0, rail_w, height), max(0.4, min(0.8, edge_strength / 50.0)))
        return None

    def _filter_page_containers(
        self,
        candidates: list[VisualTraceObject],
        width: int,
        height: int,
        window_area: int,
    ) -> list[VisualTraceObject]:
        """Filter container candidates to keep only high-confidence, deduplicated containers.

        Target: 3-12 page_containers per app.
        """
        if not candidates:
            return []

        # Sort by area (largest first)
        sorted_cands = sorted(candidates, key=lambda c: -c.area_ratio)

        # Deduplicate: keep only containers that don't overlap too much
        kept: list[VisualTraceObject] = []
        for cand in sorted_cands:
            # Skip if too small
            if cand.area_ratio < self.MIN_CONTAINER_AREA_RATIO:
                continue
            # Skip if too large (almost full window)
            if cand.area_ratio > 0.90:
                continue
            # Check overlap with already kept containers
            overlaps = False
            for existing in kept:
                containment = self._containment_ratio(cand.bounds, existing.bounds)
                if containment > 0.7:
                    overlaps = True
                    break
            if not overlaps:
                kept.append(cand)
            # Stop if we have enough
            if len(kept) >= 12:
                break

        # If too few, relax constraints
        if len(kept) < 3 and len(sorted_cands) >= 3:
            kept = sorted_cands[:min(6, len(sorted_cands))]

        return kept

    # ── VTL-0b: Local Object Detection ──────────────────────────

    def _detect_local_objects(
        self,
        gray: np.ndarray,
        rgb: np.ndarray,
        width: int,
        height: int,
        window_area: int,
    ) -> list[tuple[tuple[int, int, int, int], float, str]]:
        """Detect local objects (buttons, icons, cards, etc.)."""
        raw_objects: list[tuple[tuple[int, int, int, int], float, str]] = []

        # 1. Connected components
        cc_objects = self._detect_connected_components(gray, width, height, window_area)
        raw_objects.extend(cc_objects)

        # 2. Bordered rectangles
        br_objects = self._detect_bordered_rects(gray, width, height, window_area)
        raw_objects.extend(br_objects)

        # 3. Color blobs
        cb_objects = self._detect_color_blobs(rgb, gray, width, height, window_area)
        raw_objects.extend(cb_objects)

        # Deduplicate
        return self._dedupe_raw_objects(raw_objects, 0.5)

    def _detect_connected_components(
        self, gray: np.ndarray, width: int, height: int, window_area: int,
    ) -> list[tuple[tuple[int, int, int, int], float, str]]:
        """Detect objects via connected components."""
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5,
        )
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        results = []

        for i in range(1, num_labels):
            x, y, w, h, area = (int(v) for v in stats[i])
            if not self._valid_object_geometry(x, y, w, h, area, width, height, window_area):
                continue
            # Compute edge density
            region = gray[y:y+h, x:x+w]
            if region.size == 0:
                continue
            edges = cv2.Canny(region, 50, 150)
            edge_density = float(edges.sum()) / (255 * max(1, region.size))
            conf = max(0.3, min(0.8, edge_density * 5))
            results.append(((x, y, x + w, y + h), conf, DETECT_CONNECTED_COMPONENT))

        return results

    def _detect_bordered_rects(
        self, gray: np.ndarray, width: int, height: int, window_area: int,
    ) -> list[tuple[tuple[int, int, int, int], float, str]]:
        """Detect bordered rectangles via edge detection."""
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if not self._valid_object_geometry(x, y, w, h, area, width, height, window_area):
                continue
            # Check rectangularity
            cnt_area = cv2.contourArea(cnt)
            if cnt_area > 0 and area / cnt_area < 0.6:
                continue  # Not rectangular enough
            conf = max(0.4, min(0.8, cnt_area / max(1, area)))
            results.append(((x, y, x + w, y + h), conf, DETECT_BORDERED_RECT))

        return results

    def _detect_color_blobs(
        self, rgb: np.ndarray, gray: np.ndarray, width: int, height: int, window_area: int,
    ) -> list[tuple[tuple[int, int, int, int], float, str]]:
        """Detect color blobs via clustering."""
        # Downsample
        scale = max(1, min(width, height) // 200)
        small = rgb[::scale, ::scale]
        h_s, w_s = small.shape[:2]

        pixels = small.reshape(-1, 3).astype(np.float32)
        n_clusters = min(4, max(2, len(np.unique(pixels, axis=0)) // 20))
        if n_clusters < 2:
            return []

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        except cv2.error:
            return []

        label_image = labels.reshape(h_s, w_s)
        results = []

        for cluster_id in range(n_clusters):
            mask = (label_image == cluster_id).astype(np.uint8) * 255
            mask_full = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if not self._valid_object_geometry(x, y, w, h, area, width, height, window_area):
                    continue
                # Skip if too similar to background (very large)
                if area / window_area > 0.15:
                    continue
                conf = max(0.3, min(0.7, area / window_area * 10))
                results.append(((x, y, x + w, y + h), conf, DETECT_COLOR_BLOB))

        return results

    # ── Object Classification ───────────────────────────────────

    def _classify_and_filter_objects(
        self,
        raw_objects: list[tuple[tuple[int, int, int, int], float, str]],
        gray: np.ndarray,
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
        raw_elements: list[dict[str, Any]] | None,
        window_area: int,
    ) -> tuple[list[VisualTraceObject], list[VisualTraceObject], list[VisualTraceObject]]:
        """Classify raw objects into local_objects, text_substructures, rejected_text_like.

        Returns:
            (local_objects, text_substructures, rejected_text_like)
        """
        local_objects: list[VisualTraceObject] = []
        text_subs: list[VisualTraceObject] = []
        rejected_text: list[VisualTraceObject] = []
        trace_idx = 100

        # Audit counters
        negative_label_demoted = 0
        icon_only_demoted = 0
        uia_actionable_details: list[dict[str, Any]] = []

        for bounds, conf, method in raw_objects:
            l, t, r, b = (int(x) for x in bounds)
            w = max(1, r - l)
            h = max(1, b - t)
            area_ratio = (w * h) / window_area
            aspect = w / max(h, 1)

            # Compute REAL edge density from pixel data
            region = gray[t:b, l:r]
            if region.size == 0:
                continue
            edges = cv2.Canny(region, 50, 150)
            edge_density = float(edges.sum()) / (255 * max(1, region.size))

            # Count OCR items and total text length (expanded region)
            margin = 10
            ocr_items = self._get_items_in_region(
                ocr_blocks, l - margin, t - margin, r + margin, b + margin, "bbox",
            )
            ocr_count = len(ocr_items)
            ocr_text_length = sum(len(item.get("text", "")) for item in ocr_items)

            # Count vision items and check for control labels
            vis_items = self._get_items_in_region(vision_candidates, l, t, r, b, "bounding_rect")
            vis_count = len(vis_items)

            # Positive control keywords (Omni labels that suggest actionable)
            # Only specific interactive controls, NOT generic "icon"
            control_keywords = {"button", "input", "checkbox", "switch",
                                "toggle", "menu", "tab", "toolbar", "slider", "dropdown",
                                "submit", "send", "play", "pause", "close", "search"}
            # Negative keywords (Omni labels that suggest content/media, NOT actionable)
            negative_keywords = {
                "image", "photo", "avatar", "cover", "screenshot", "thumbnail",
                "content", "card", "list", "media", "album", "artwork", "picture",
                "poster", "banner", "preview", "feed", "story", "reel",
            }

            has_control_label = False
            has_negative_label = False
            has_icon_only = False  # Only "icon" kind, no specific control label
            for vi in vis_items:
                label = (vi.get("label", "") or vi.get("kind", "") or "").lower()
                icon_type = (vi.get("icon_type", "") or "").lower()
                text = (vi.get("text", "") or "").lower()
                kind = (vi.get("kind", "") or "").lower()
                combined = f"{label} {icon_type} {text}"

                if any(kw in combined for kw in control_keywords):
                    has_control_label = True
                if any(kw in combined for kw in negative_keywords):
                    has_negative_label = True
                # Check if only "icon" kind with no specific control label
                if kind == "icon" and not any(kw in combined for kw in control_keywords):
                    has_icon_only = True

            # ── Strict actionable classification ──
            # Actionable REQUIRES strong evidence:
            #   Option A: UIA actionable control (strong evidence)
            #   Option B: Omni control label + NO negative label + no OCR + strict shape
            # Negative labels (image/cover/screenshot/content/card) disqualify actionable.
            # Shape-only is NEVER sufficient for actionable.

            # Option A: UIA actionable control (strong evidence, no negative guard needed)
            uia_actionable = self._has_uia_actionable(raw_elements, l, t, r, b)
            if uia_actionable:
                uia_actionable_details.append({
                    "trace_id": f"VT{trace_idx}",
                    "bounds": [l, t, r, b],
                    "evidence": "uia_actionable",
                })

            # Option B: Omni control + negative guard + strict conditions
            omni_actionable = (
                has_control_label
                and not has_negative_label  # Negative label guard
                and not has_icon_only       # Icon-only guard
                and ocr_text_length == 0
                and edge_density >= 0.10
                and 0.3 < aspect < 3.0
                and area_ratio < 0.01
            )

            is_actionable = uia_actionable or omni_actionable

            # Track demotion reasons
            if has_negative_label and has_control_label:
                negative_label_demoted += 1
            if has_icon_only and not uia_actionable:
                icon_only_demoted += 1

            # ── Text/content guards ──
            # Text-label guard: OCR text with significant length + not too high edge
            has_significant_text = ocr_text_length > 3
            is_text_label = has_significant_text and edge_density < 0.15
            is_avatar = self._is_avatar_like(area_ratio, aspect, edge_density, ocr_count)
            is_cover = self._is_cover_like(area_ratio, aspect, edge_density)
            is_strong_text = self._is_strong_text_like(aspect, edge_density, ocr_count, ocr_text_length)

            # ── Classify ──
            # Priority: text exclusion > content guards > actionable
            if is_strong_text:
                obj_class = OBJECT_CLASS_TEXT_SUB
            elif is_text_label:
                # OCR text present → label/text, NOT actionable
                obj_class = "label_evidence"
            elif has_significant_text and not uia_actionable:
                # Significant OCR text but no UIA actionable → demote
                obj_class = "label_evidence"
            elif is_avatar:
                obj_class = OBJECT_CLASS_CONTENT
            elif is_cover:
                obj_class = OBJECT_CLASS_CONTENT
            elif is_actionable and ocr_count == 0:
                # Actionable only if NO OCR overlap and strong evidence
                obj_class = OBJECT_CLASS_ACTIONABLE
                # Track reason for audit
                if uia_actionable:
                    actionable_reason = "uia_actionable"
                elif omni_actionable:
                    actionable_reason = "omni_control_confirmed"
                else:
                    actionable_reason = "unknown"
            else:
                obj_class = OBJECT_CLASS_VISUAL_CANDIDATE
                actionable_reason = None

            # Build evidence sources list
            evidence_sources = ["pixel"]
            if ocr_count > 0:
                evidence_sources.append("ocr")
            if vis_count > 0:
                evidence_sources.append("vision")
            if uia_actionable:
                evidence_sources.append("uia")

            # Collect contained IDs
            contained_ocr_ids = [
                item.get("text", "")[:20] for item in ocr_items[:5]
            ]
            contained_omni_ids = [
                item.get("candidate_id", "") or item.get("label", "")
                for item in vis_items[:5]
            ]

            obj = VisualTraceObject(
                trace_id=f"VT{trace_idx}",
                bounds=(l, t, r, b),
                shape_type="rectangle",
                object_level=OBJECT_LEVEL_LOCAL_OBJECT,
                object_class=obj_class,
                detection_method=method,
                evidence_sources=evidence_sources,
                contained_omni_ids=contained_omni_ids,
                contained_ocr_ids=contained_ocr_ids,
                confidence=conf,
                area_ratio=round(area_ratio, 4),
                aspect_ratio=round(aspect, 3),
                edge_density=round(edge_density, 4),
            )
            # Store actionable reason for audit
            if obj_class == OBJECT_CLASS_ACTIONABLE:
                obj._actionable_reason = actionable_reason
            else:
                obj._actionable_reason = None

            # Route to appropriate list
            if obj_class == OBJECT_CLASS_TEXT_SUB:
                text_subs.append(obj)
            elif obj_class == "label_evidence":
                rejected_text.append(obj)
            elif obj_class == OBJECT_CLASS_VISUAL_CANDIDATE:
                # Visual candidate: not actionable, not content
                local_objects.append(obj)
            elif obj_class == OBJECT_CLASS_ACTIONABLE:
                local_objects.append(obj)
            elif self._is_text_like_region(obj, ocr_count, ocr_text_length):
                obj.object_class = "rejected_text_like"
                rejected_text.append(obj)
            else:
                local_objects.append(obj)

            trace_idx += 1

        # ── Post-processing: actionable cap with audit ──
        actionable = [o for o in local_objects if o.object_class == OBJECT_CLASS_ACTIONABLE]
        non_actionable = [o for o in local_objects if o.object_class != OBJECT_CLASS_ACTIONABLE]
        pre_cap_actionable = len(actionable)

        demoted_by_cap = []
        if len(actionable) > 15:
            actionable.sort(key=lambda o: -o.confidence)
            kept = actionable[:15]
            demoted = actionable[15:]
            for obj in demoted:
                obj.object_class = OBJECT_CLASS_VISUAL_CANDIDATE
                demoted_by_cap.append(obj.trace_id)
            local_objects = kept + non_actionable + demoted
        else:
            local_objects = actionable + non_actionable

        # Store audit metadata
        self._last_audit = {
            "raw_visual_candidates": trace_idx - 100,
            "pre_cap_actionable": pre_cap_actionable,
            "accepted_actionable": min(pre_cap_actionable, 15),
            "demoted_by_cap": demoted_by_cap,
        }

        # Store audit counters for build() to access
        self._classify_audit = {
            "negative_label_demoted": negative_label_demoted,
            "icon_only_demoted": icon_only_demoted,
            "uia_actionable_match_details": uia_actionable_details,
        }

        return local_objects, text_subs, rejected_text

    def _is_avatar_like(
        self, area_ratio: float, aspect: float, edge_density: float, ocr_count: int,
    ) -> bool:
        """Check if object looks like a contact avatar/photo.

        Avatars are: small square, moderate edge, no OCR text.
        """
        return (
            area_ratio < 0.005
            and 0.5 < aspect < 2.0
            and edge_density >= 0.08
            and ocr_count == 0
        )

    def _has_uia_actionable(
        self,
        raw_elements: list[dict[str, Any]] | None,
        l: int, t: int, r: int, b: int,
    ) -> bool:
        """Check if region contains UIA actionable controls."""
        if not raw_elements:
            return False
        actionable_types = {
            "button", "hyperlink", "menuitem", "tabitem",
            "checkbox", "radiobutton", "togglebutton",
            "combobox", "spinner", "slider", "switch",
        }
        margin = 10
        for elem in raw_elements:
            bbox = elem.get("bounding_rect") or elem.get("bbox", [])
            if not bbox or len(bbox) < 4:
                continue
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            if l - margin <= cx < r + margin and t - margin <= cy < b + margin:
                ct = (elem.get("control_type", "") or "").lower()
                if any(ct.startswith(at) for at in actionable_types):
                    return True
        return False

    def _is_cover_like(
        self, area_ratio: float, aspect: float, edge_density: float,
    ) -> bool:
        """Check if object looks like an album cover/image card.

        Covers are: medium square, moderate edge, no text.
        """
        return (
            0.003 < area_ratio < 0.03
            and 0.5 < aspect < 2.0
            and edge_density < 0.15
        )

    def _is_strong_text_like(
        self,
        aspect_ratio: float,
        edge_density: float,
        ocr_count: int,
        ocr_text_length: int,
    ) -> bool:
        """Strong text-like indicators → text_substructure."""
        # Elongated + low edge
        if aspect_ratio > 3.5 and edge_density < 0.06:
            return True
        # OCR text + low edge
        if ocr_text_length > 15 and edge_density < 0.08:
            return True
        # Multiple OCR blocks + elongated
        if ocr_count >= 2 and aspect_ratio > 2.5:
            return True
        # Very elongated thin strip
        if aspect_ratio > 6.0:
            return True
        return False

    def _get_items_in_region(
        self,
        items: list[dict[str, Any]] | None,
        l: int, t: int, r: int, b: int,
        bbox_key: str,
    ) -> list[dict[str, Any]]:
        """Get items whose bbox overlaps with region.

        Uses bbox overlap instead of center-in-region to catch small text
        labels whose center may be just outside the region boundary.
        """
        if not items:
            return []
        result = []
        for item in items:
            bbox = item.get(bbox_key) or item.get("bbox", [])
            if not bbox or len(bbox) < 4:
                continue
            # Check bbox overlap (not center-in-region)
            il = max(l, bbox[0])
            it = max(t, bbox[1])
            ir = min(r, bbox[2])
            ib = min(b, bbox[3])
            if il < ir and it < ib:
                # There is overlap
                overlap_area = (ir - il) * (ib - it)
                item_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if item_area > 0 and overlap_area / item_area > 0.2:
                    result.append(item)
        return result

    def _is_text_like_region(
        self,
        obj: VisualTraceObject,
        ocr_count: int,
        ocr_text_length: int,
    ) -> bool:
        """Check if a region looks like text content that should not be a local_object.

        Text-like characteristics:
        - Contains OCR text (any length > 0 is a strong signal)
        - Elongated (high aspect ratio)
        - Low/moderate edge density
        - Multiple OCR blocks present
        """
        aspect = obj.aspect_ratio
        edge = obj.edge_density
        area = obj.area_ratio

        # Any OCR text present → likely text content
        # (buttons/icons rarely have OCR text unless they're labeled)
        if ocr_text_length > 0:
            return True

        # Multiple OCR blocks → text content
        if ocr_count >= 2:
            return True

        # High aspect ratio + low edge = likely text line
        if aspect > 3.5 and edge < 0.06:
            return True

        # Very elongated thin strip
        if aspect > 6.0 and area < 0.02:
            return True

        # Moderate aspect + moderate area + moderate edge = likely text block
        if aspect > 2.0 and area > 0.003 and edge < 0.07:
            return True

        return False

    def _determine_object_class(
        self,
        area_ratio: float,
        aspect_ratio: float,
        edge_density: float,
        ocr_count: int,
        ocr_text_length: int,
        has_omni_overlap: bool,
        has_uia_control: bool,
    ) -> str:
        """Determine object_class for a detected object.

        Strict rules:
        - text_substructure: elongated, low edge, OCR text
        - actionable: ONLY with strong evidence (Omni/UIA overlap, or
          small+square+low-OCR+in-toolbar-area)
        - content: everything else
        """
        # ── Text exclusion (strong) ──
        # Elongated + low edge = text line
        if aspect_ratio > 3.5 and edge_density < 0.06:
            return OBJECT_CLASS_TEXT_SUB
        # OCR text present + low edge
        if ocr_text_length > 10 and edge_density < 0.08:
            return OBJECT_CLASS_TEXT_SUB
        # OCR confirmed + elongated
        if ocr_count >= 2 and aspect_ratio > 2.5:
            return OBJECT_CLASS_TEXT_SUB
        # Very elongated thin strip
        if aspect_ratio > 6.0 and area_ratio < 0.02:
            return OBJECT_CLASS_TEXT_SUB

        # ── Actionable (strict: REQUIRES external evidence) ──
        # Shape alone is NEVER sufficient for actionable.
        # Must have Omni/vision control overlap or UIA control evidence.
        if has_omni_overlap or has_uia_control:
            if area_ratio < 0.02 and 0.2 < aspect_ratio < 5.0:
                return OBJECT_CLASS_ACTIONABLE

        # No external evidence → NEVER actionable
        # (downstream will downgrade to local_visual_candidate)

        # ── Content (default for non-text, non-actionable) ──
        return OBJECT_CLASS_CONTENT

    # ── Hierarchy ───────────────────────────────────────────────

    def _set_hierarchy(
        self,
        containers: list[VisualTraceObject],
        local_objects: list[VisualTraceObject],
    ) -> None:
        """Set parent/child relationships using containment ratio.

        containment_ratio = intersection_area / object_area
        Only set parent if object is mostly inside the container (>70%).
        """
        for obj in local_objects:
            best_container = None
            best_ratio = 0.0

            for cont in containers:
                ratio = self._containment_ratio(obj.bounds, cont.bounds)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_container = cont

            if best_container and best_ratio > 0.7:
                obj.parent_trace_id = best_container.trace_id
                best_container.child_trace_ids.append(obj.trace_id)

    # ── Helpers ──────────────────────────────────────────────────

    def _valid_object_geometry(
        self, x: int, y: int, w: int, h: int, area: int,
        width: int, height: int, window_area: int,
    ) -> bool:
        """Check if object geometry is valid."""
        if w < self.MIN_OBJECT_SIZE or h < self.MIN_OBJECT_SIZE:
            return False
        area_ratio = area / window_area
        if area_ratio < self.MIN_OBJECT_AREA_RATIO or area_ratio > self.MAX_OBJECT_AREA_RATIO:
            return False
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > self.MAX_OBJECT_SIZE_RATIO:
            return False
        return True

    def _count_items_in_region(
        self,
        items: list[dict[str, Any]] | None,
        l: int, t: int, r: int, b: int,
        bbox_key: str,
    ) -> int:
        """Count items whose center falls within region."""
        if not items:
            return 0
        count = 0
        for item in items:
            bbox = item.get(bbox_key) or item.get("bbox", [])
            if not bbox or len(bbox) < 4:
                continue
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            if l <= cx < r and t <= cy < b:
                count += 1
        return count

    def _overlaps_existing(
        self,
        bounds: tuple[int, int, int, int],
        existing: list[VisualTraceObject],
        threshold: float,
    ) -> bool:
        """Check if bounds overlaps significantly with existing objects."""
        for obj in existing:
            if self._compute_iou(bounds, obj.bounds) > threshold:
                return True
        return False

    def _compute_iou(
        self,
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection over Union."""
        il = max(a[0], b[0])
        it = max(a[1], b[1])
        ir = min(a[2], b[2])
        ib = min(a[3], b[3])
        if il >= ir or it >= ib:
            return 0.0
        intersection = (ir - il) * (ib - it)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / max(union, 1)

    def _containment_ratio(
        self,
        obj: tuple[int, int, int, int],
        container: tuple[int, int, int, int],
    ) -> float:
        """Compute what fraction of obj is inside container.

        Returns: intersection_area / obj_area (0-1).
        1.0 = obj is entirely inside container.
        """
        il = max(obj[0], container[0])
        it = max(obj[1], container[1])
        ir = min(obj[2], container[2])
        ib = min(obj[3], container[3])
        if il >= ir or it >= ib:
            return 0.0
        intersection = (ir - il) * (ib - it)
        obj_area = (obj[2] - obj[0]) * (obj[3] - obj[1])
        return intersection / max(obj_area, 1)

    def _dedupe_rects(
        self,
        rects: list[tuple[tuple[int, int, int, int], float]],
        threshold: float,
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        """Deduplicate overlapping rectangles."""
        if not rects:
            return []
        sorted_rects = sorted(rects, key=lambda r: -(r[0][2] - r[0][0]) * (r[0][3] - r[0][1]))
        kept = [sorted_rects[0]]
        for rect in sorted_rects[1:]:
            if not any(self._compute_iou(rect[0], k[0]) > threshold for k in kept):
                kept.append(rect)
        return kept

    def _dedupe_raw_objects(
        self,
        objects: list[tuple[tuple[int, int, int, int], float, str]],
        threshold: float,
    ) -> list[tuple[tuple[int, int, int, int], float, str]]:
        """Deduplicate overlapping raw objects."""
        if not objects:
            return []
        sorted_objs = sorted(objects, key=lambda o: -(o[0][2] - o[0][0]) * (o[0][3] - o[0][1]))
        kept = [sorted_objs[0]]
        for obj in sorted_objs[1:]:
            if not any(self._compute_iou(obj[0], k[0]) > threshold for k in kept):
                kept.append(obj)
        return kept
