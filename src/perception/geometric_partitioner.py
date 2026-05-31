"""Pure geometric region partitioning from screenshot pixels.

Layer 1 of the two-layer architecture: detects visual boundaries (lines,
density changes, border enclosures, floating blobs, card grids) and outputs
neutral GeometricRegion objects with no semantic labels.

Output region IDs are neutral (R0, R1, R2...) and carry only
boundary_evidence (e.g. "vertical_separator_L78%") — never semantic roles
like "side_panel" or "composer_area".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


# =============================================================================
# Data model
# =============================================================================


@dataclass(frozen=True)
class GeometricRegion:
    """A neutral geometric partition of the screenshot.

    Carries only boundary evidence, no semantic label. Semantic labeling
    is done by SemanticFusion in Layer 2.
    """

    region_id: str  # "R0", "R1", "R2" ...
    bounds: tuple[int, int, int, int]  # left, top, right, bottom
    boundary_evidence: list[str] = field(default_factory=list)
    density_profile: tuple[float, float] = (0.0, 0.0)  # (mean_std, edge_density)
    geometry_confidence: float = 0.0
    parent_region_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "bounds": list(self.bounds),
            "boundary_evidence": list(self.boundary_evidence),
            "density_profile": list(self.density_profile),
            "geometry_confidence": self.geometry_confidence,
            "parent_region_id": self.parent_region_id,
        }


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class PartitionerConfig:
    """Centralized thresholds for GeometricPartitioner.

    All values are initial defaults — tune based on real screenshot diagnostics.
    """

    separator_threshold: float = 10.0
    min_span_ratio: float = 0.55
    density_contrast_threshold: float = 0.12
    min_region_area_ratio: float = 0.006
    border_edge_threshold: int = 12
    floating_min_confidence: float = 0.65
    grid_min_card_count: int = 4
    content_std_min: float = 6.0
    content_std_max: float = 42.0
    low_texture_std_max: float = 4.0


# =============================================================================
# Partition diagnostics
# =============================================================================


@dataclass
class PartitionDiagnostics:
    """Record why each boundary was accepted or filtered out."""

    total_separator_candidates: int = 0
    accepted_separators: int = 0
    rejected_separators: list[dict[str, Any]] = field(default_factory=list)
    accepted_separator_details: list[dict[str, Any]] = field(default_factory=list)
    enclosed_rect_count: int = 0
    density_region_count: int = 0
    floating_blob_count: int = 0
    card_grid_count: int = 0
    final_region_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = {
            "total_separator_candidates": self.total_separator_candidates,
            "accepted_separators": self.accepted_separators,
            "rejected_separators": self.rejected_separators,
            "accepted_separator_details": self.accepted_separator_details,
            "enclosed_rect_count": self.enclosed_rect_count,
            "density_region_count": self.density_region_count,
            "floating_blob_count": self.floating_blob_count,
            "card_grid_count": self.card_grid_count,
            "final_region_count": self.final_region_count,
        }
        if hasattr(self, "fast_mode"):
            data["fast_mode"] = bool(getattr(self, "fast_mode"))
        if hasattr(self, "bounded_full"):
            data["bounded_full"] = bool(getattr(self, "bounded_full"))
        if hasattr(self, "scale"):
            data["scale"] = float(getattr(self, "scale"))
        return data


# =============================================================================
# Engine
# =============================================================================


class GeometricPartitioner:
    """Detect neutral geometric regions from screenshot pixels.

    Pure geometry — no semantic labels, no app-shell templates, no
    software-specific rules. Output region IDs are R0, R1, R2...
    """

    def __init__(self, config: PartitionerConfig | None = None) -> None:
        self.config = config or PartitionerConfig()
        self.diagnostics = PartitionDiagnostics()

    def partition(self, image: Image.Image | None) -> list[GeometricRegion]:
        """Partition a screenshot into neutral geometric regions."""
        self.diagnostics = PartitionDiagnostics()
        if image is None:
            return []

        rgb = image.convert("RGB")
        arr = np.asarray(rgb).astype(np.int16)
        height, width = arr.shape[:2]
        if width <= 1 or height <= 1:
            return []

        regions: list[GeometricRegion] = []

        # 1. Detect separators (vertical and horizontal boundary lines)
        v_seps = self._detect_vertical_separators(arr)
        h_seps = self._detect_horizontal_separators(arr)
        regions.extend(v_seps)
        regions.extend(h_seps)
        regions.extend(self._detect_edge_coverage_separators(arr))

        # 2. Detect bordered rectangles (enclosed by edges on all four sides)
        regions.extend(self._detect_enclosed_rectangles(arr))

        # 3. Detect density-based regions (texture variance differs from surroundings)
        regions.extend(self._detect_density_regions(arr))

        # 4. Detect floating blobs (high-contrast isolated blocks, typically bottom-right)
        floating = self._detect_floating_blobs(arr)
        if floating is not None:
            regions.append(floating)

        # 5. Detect repeated card grids (similar-sized bordered boxes in a cluster)
        card_grid = self._detect_repeated_card_grid(arr)
        if card_grid is not None:
            regions.append(card_grid)

        # 6. Partition by strong separators (recursive split into region tree)
        tree_regions = self._partition_by_separators(arr, regions)
        if tree_regions:
            regions = tree_regions

        # Deduplicate and clip
        regions = self._dedupe_and_clip(regions, width, height)
        if not regions:
            regions = [
                GeometricRegion(
                    region_id="R0",
                    bounds=(0, 0, width, height),
                    boundary_evidence=["full_image_fallback"],
                    density_profile=self._compute_density(arr),
                    geometry_confidence=0.20,
                )
            ]
        self.diagnostics.final_region_count = len(regions)
        return regions

    # ------------------------------------------------------------------
    # Separator detection
    # ------------------------------------------------------------------

    def _detect_vertical_separators(self, arr: np.ndarray) -> list[GeometricRegion]:
        height, width = arr.shape[:2]
        min_span = max(12, int(height * self.config.min_span_ratio))
        col_mean = arr.mean(axis=0)
        diff = np.abs(col_mean[1:] - col_mean[:-1]).mean(axis=1)
        candidates = self._runs(diff >= self.config.separator_threshold)

        regions: list[GeometricRegion] = []
        self.diagnostics.total_separator_candidates += len(candidates)

        for idx, (start, end) in enumerate(candidates):
            x1 = max(0, start - 1)
            x2 = min(width, end + 2)
            line_width = x2 - x1
            if line_width > max(8, width * 0.05):
                self.diagnostics.rejected_separators.append({
                    "direction": "vertical", "position": x1,
                    "reason": "too_wide", "width": line_width,
                })
                continue

            # Span check: measure actual vertical extent of the boundary
            left_data = arr[:, max(0, x1 - 6):x1, :].astype(np.float32) if x1 > 0 else arr[:, :1, :].astype(np.float32)
            right_data = arr[:, x2:min(width, x2 + 6), :].astype(np.float32) if x2 < width else arr[:, -1:, :].astype(np.float32)
            left_brightness = left_data.mean(axis=(1, 2))
            right_brightness = right_data.mean(axis=(1, 2))
            row_diff = np.abs(left_brightness - right_brightness)
            span = int((row_diff > self.config.separator_threshold * 0.5).sum())
            if span < min_span:
                self.diagnostics.rejected_separators.append({
                    "direction": "vertical", "position": x1,
                    "reason": "span_too_short",
                    "span": span, "min_span": min_span,
                })
                continue

            # Density contrast check: are the two sides different enough?
            left_band = arr[:, max(0, x1 - 8):x1, :] if x1 > 0 else None
            right_band = arr[:, x2:min(width, x2 + 8), :] if x2 < width else None
            if left_band is not None and left_band.size > 0 and right_band is not None and right_band.size > 0:
                left_std = float(left_band.std())
                right_std = float(right_band.std())
                density_contrast = abs(left_std - right_std) / max(max(left_std, right_std), 0.1)
                if density_contrast < self.config.density_contrast_threshold:
                    self.diagnostics.rejected_separators.append({
                        "direction": "vertical", "position": x1,
                        "reason": "density_contrast_insufficient",
                        "contrast": round(density_contrast, 3),
                        "threshold": self.config.density_contrast_threshold,
                    })
                    continue

            span_pct = round(100 * span / max(height, 1))
            score = min(0.95, float(diff[start:end].mean() if end > start else diff[start]) / 35.0)
            regions.append(GeometricRegion(
                region_id=f"R_sep_v_{idx}",
                bounds=(x1, 0, x2, height),
                boundary_evidence=[f"vertical_separator_L{span_pct}%"],
                density_profile=(0.0, 0.0),
                geometry_confidence=max(0.45, score),
            ))
            self.diagnostics.accepted_separators += 1
            self.diagnostics.accepted_separator_details.append({
                "direction": "vertical", "position": x1,
                "span_pct": span_pct, "confidence": max(0.45, score),
            })

        return regions

    def _detect_horizontal_separators(self, arr: np.ndarray) -> list[GeometricRegion]:
        height, width = arr.shape[:2]
        min_span = max(12, int(width * self.config.min_span_ratio))
        row_mean = arr.mean(axis=1)
        diff = np.abs(row_mean[1:] - row_mean[:-1]).mean(axis=1)
        candidates = self._runs(diff >= self.config.separator_threshold)

        regions: list[GeometricRegion] = []
        self.diagnostics.total_separator_candidates += len(candidates)

        for idx, (start, end) in enumerate(candidates):
            y1 = max(0, start - 1)
            y2 = min(height, end + 2)
            line_height = y2 - y1
            if line_height > max(8, height * 0.05):
                self.diagnostics.rejected_separators.append({
                    "direction": "horizontal", "position": y1,
                    "reason": "too_tall", "height": line_height,
                })
                continue

            # Span check: measure actual horizontal extent of the boundary
            top_data = arr[max(0, y1 - 6):y1, :, :].astype(np.float32) if y1 > 0 else arr[:1, :, :].astype(np.float32)
            bottom_data = arr[y2:min(height, y2 + 6), :, :].astype(np.float32) if y2 < height else arr[-1:, :, :].astype(np.float32)
            top_brightness = top_data.mean(axis=(0, 2))
            bottom_brightness = bottom_data.mean(axis=(0, 2))
            col_diff = np.abs(top_brightness - bottom_brightness)
            span = int((col_diff > self.config.separator_threshold * 0.5).sum())
            if span < min_span:
                self.diagnostics.rejected_separators.append({
                    "direction": "horizontal", "position": y1,
                    "reason": "span_too_short",
                    "span": span, "min_span": min_span,
                })
                continue

            # Density contrast check
            top_band = arr[max(0, y1 - 8):y1, :, :] if y1 > 0 else None
            bottom_band = arr[y2:min(height, y2 + 8), :, :] if y2 < height else None
            if top_band is not None and top_band.size > 0 and bottom_band is not None and bottom_band.size > 0:
                top_std = float(top_band.std())
                bottom_std = float(bottom_band.std())
                density_contrast = abs(top_std - bottom_std) / max(max(top_std, bottom_std), 0.1)
                if density_contrast < self.config.density_contrast_threshold:
                    self.diagnostics.rejected_separators.append({
                        "direction": "horizontal", "position": y1,
                        "reason": "density_contrast_insufficient",
                        "contrast": round(density_contrast, 3),
                    })
                    continue

            span_pct = round(100 * span / max(width, 1))
            score = min(0.95, float(diff[start:end].mean() if end > start else diff[start]) / 35.0)
            regions.append(GeometricRegion(
                region_id=f"R_sep_h_{idx}",
                bounds=(0, y1, width, y2),
                boundary_evidence=[f"horizontal_separator_L{span_pct}%"],
                density_profile=(0.0, 0.0),
                geometry_confidence=max(0.45, score),
            ))
            self.diagnostics.accepted_separators += 1
            self.diagnostics.accepted_separator_details.append({
                "direction": "horizontal", "position": y1,
                "span_pct": span_pct, "confidence": max(0.45, score),
            })

        return regions

    def _detect_edge_coverage_separators(self, arr: np.ndarray) -> list[GeometricRegion]:
        """Detect separators by edge coverage instead of whole-row/column mean.

        Mean-based separators miss UI boundaries that are visually strong but local
        to part of a window, such as a chat input top border or a pane divider.
        This detector counts how much of each row/column contains a brightness
        edge and keeps long, high-coverage runs as neutral separator evidence.
        """
        height, width = arr.shape[:2]
        gray = arr.mean(axis=2).astype(np.float32)
        edge_threshold = max(6.0, float(self.config.separator_threshold) * 0.75)

        vertical_edges = np.zeros((height, width), dtype=bool)
        vertical_edges[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1]) >= edge_threshold
        horizontal_edges = np.zeros((height, width), dtype=bool)
        horizontal_edges[1:, :] = np.abs(gray[1:, :] - gray[:-1, :]) >= edge_threshold

        min_coverage = max(0.35, float(self.config.min_span_ratio) * 0.72)
        v_coverage = vertical_edges.mean(axis=0)
        h_coverage = horizontal_edges.mean(axis=1)

        regions: list[GeometricRegion] = []
        v_runs = self._runs(v_coverage >= min_coverage)
        h_runs = self._runs(h_coverage >= min_coverage)
        self.diagnostics.total_separator_candidates += len(v_runs) + len(h_runs)

        for idx, (start, end) in enumerate(v_runs):
            x1 = max(0, start - 1)
            x2 = min(width, end + 1)
            line_width = x2 - x1
            if line_width > max(10, int(width * 0.04)):
                self.diagnostics.rejected_separators.append({
                    "direction": "vertical_edge",
                    "position": x1,
                    "reason": "too_wide",
                    "width": line_width,
                })
                continue
            coverage = float(v_coverage[start:end].max() if end > start else v_coverage[start])
            span_pct = round(coverage * 100)
            confidence = max(0.55, min(0.95, coverage))
            regions.append(GeometricRegion(
                region_id=f"R_edge_v_{idx}",
                bounds=(x1, 0, x2, height),
                boundary_evidence=[f"vertical_edge_coverage_L{span_pct}%"],
                density_profile=(0.0, 0.0),
                geometry_confidence=confidence,
            ))
            self.diagnostics.accepted_separators += 1
            self.diagnostics.accepted_separator_details.append({
                "direction": "vertical_edge",
                "position": x1,
                "span_pct": span_pct,
                "confidence": confidence,
            })

        for idx, (start, end) in enumerate(h_runs):
            y1 = max(0, start - 1)
            y2 = min(height, end + 1)
            line_height = y2 - y1
            if line_height > max(10, int(height * 0.04)):
                self.diagnostics.rejected_separators.append({
                    "direction": "horizontal_edge",
                    "position": y1,
                    "reason": "too_tall",
                    "height": line_height,
                })
                continue
            coverage = float(h_coverage[start:end].max() if end > start else h_coverage[start])
            span_pct = round(coverage * 100)
            confidence = max(0.55, min(0.95, coverage))
            regions.append(GeometricRegion(
                region_id=f"R_edge_h_{idx}",
                bounds=(0, y1, width, y2),
                boundary_evidence=[f"horizontal_edge_coverage_L{span_pct}%"],
                density_profile=(0.0, 0.0),
                geometry_confidence=confidence,
            ))
            self.diagnostics.accepted_separators += 1
            self.diagnostics.accepted_separator_details.append({
                "direction": "horizontal_edge",
                "position": y1,
                "span_pct": span_pct,
                "confidence": confidence,
            })

        return regions

    # ------------------------------------------------------------------
    # Enclosed rectangles (borders, no semantic "input" label)
    # ------------------------------------------------------------------

    def _detect_enclosed_rectangles(self, arr: np.ndarray) -> list[GeometricRegion]:
        height, width = arr.shape[:2]
        gray = arr.mean(axis=2)
        background = float(np.percentile(gray, 92))
        edge_mask = (background - gray) > self.config.border_edge_threshold
        components = self._connected_components(edge_mask)

        regions: list[GeometricRegion] = []
        min_area = width * height * self.config.min_region_area_ratio

        for idx, (x1, y1, x2, y2, count) in enumerate(components):
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w <= 8 or box_h <= 8:
                continue
            if box_w * box_h < min_area:
                continue
            if count / max(box_w * box_h, 1) > 0.65:
                continue
            regions.append(GeometricRegion(
                region_id=f"R_border_{idx}",
                bounds=(x1, y1, x2, y2),
                boundary_evidence=["border_enclosure"],
                density_profile=(float(gray[y1:y2, x1:x2].std()), 0.0),
                geometry_confidence=0.55,
            ))

        self.diagnostics.enclosed_rect_count = len(regions)
        return regions

    # ------------------------------------------------------------------
    # Density-based regions
    # ------------------------------------------------------------------

    def _detect_density_regions(self, arr: np.ndarray) -> list[GeometricRegion]:
        height, width = arr.shape[:2]
        gray = arr.mean(axis=2)
        min_area = width * height * self.config.min_region_area_ratio

        block_h = max(40, height // 8)
        block_w = max(60, width // 6)
        regions: list[GeometricRegion] = []
        idx = 0

        for y in range(0, height - block_h, block_h // 2):
            for x in range(0, width - block_w, block_w // 2):
                x2_val = min(width, x + block_w)
                y2_val = min(height, y + block_h)
                block = gray[y:y2_val, x:x2_val]
                if block.size == 0:
                    continue
                area = (x2_val - x) * (y2_val - y)
                if area < min_area:
                    continue
                std_val = float(block.std())

                # Low texture (empty panels, solid backgrounds)
                if std_val <= self.config.low_texture_std_max:
                    regions.append(GeometricRegion(
                        region_id=f"R_lowtex_{idx}",
                        bounds=(x, y, x2_val, y2_val),
                        boundary_evidence=["low_texture"],
                        density_profile=(std_val, 0.0),
                        geometry_confidence=0.35,
                    ))
                    idx += 1
                # Moderate texture (content-like: text, lists, images)
                elif self.config.content_std_min <= std_val <= self.config.content_std_max:
                    right_half = x >= width * 0.45
                    evidence = ["moderate_texture"]
                    if right_half:
                        evidence.append("right_placement")
                    regions.append(GeometricRegion(
                        region_id=f"R_content_{idx}",
                        bounds=(x, y, x2_val, y2_val),
                        boundary_evidence=evidence,
                        density_profile=(std_val, 0.0),
                        geometry_confidence=0.38,
                    ))
                    idx += 1

        self.diagnostics.density_region_count = len(regions)
        # Limit to avoid explosion
        return regions[:24]

    # ------------------------------------------------------------------
    # Floating blobs
    # ------------------------------------------------------------------

    def _detect_floating_blobs(self, arr: np.ndarray) -> GeometricRegion | None:
        height, width = arr.shape[:2]
        gray = arr.mean(axis=2)
        x_start = int(width * 0.55)
        y_start = int(height * 0.55)
        crop = gray[y_start:, x_start:]
        if crop.size == 0:
            return None

        threshold = max(float(np.percentile(crop, 88)), float(np.median(crop) + 35))
        mask = crop >= threshold
        components = self._connected_components(mask)
        best: tuple[int, int, int, int, int] | None = None

        for x1, y1, x2, y2, count in components:
            box_w = x2 - x1
            box_h = y2 - y1
            if not (int(width * 0.06) <= box_w <= int(width * 0.30)):
                continue
            if not (int(height * 0.035) <= box_h <= int(height * 0.16)):
                continue
            if count < box_w * box_h * 0.30:
                continue
            if best is None or count > best[4]:
                best = (x1, y1, x2, y2, count)

        if best is None:
            self.diagnostics.floating_blob_count = 0
            return None

        x1, y1, x2, y2, _count = best
        confidence = 0.70
        if confidence < self.config.floating_min_confidence:
            self.diagnostics.floating_blob_count = 0
            return None

        self.diagnostics.floating_blob_count = 1
        return GeometricRegion(
            region_id="R_floating_0",
            bounds=(x_start + x1, y_start + y1, x_start + x2, y_start + y2),
            boundary_evidence=["floating_blob", "bottom_right_quadrant"],
            density_profile=(0.0, 0.0),
            geometry_confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Card grid detection
    # ------------------------------------------------------------------

    def _detect_repeated_card_grid(self, arr: np.ndarray) -> GeometricRegion | None:
        height, width = arr.shape[:2]
        gray = arr.mean(axis=2)
        gx = np.zeros_like(gray, dtype=bool)
        gy = np.zeros_like(gray, dtype=bool)
        gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1]) > 14
        gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :]) > 14
        edge_mask = gx | gy
        components = self._connected_components(edge_mask)

        boxes: list[tuple[int, int, int, int]] = []
        for x1, y1, x2, y2, count in components:
            box_w = x2 - x1
            box_h = y2 - y1
            if not (int(width * 0.12) <= box_w <= int(width * 0.45)):
                continue
            if not (45 <= box_h <= int(height * 0.22)):
                continue
            density = count / max(box_w * box_h, 1)
            if density > 0.45:
                continue
            boxes.append((x1, y1, x2, y2))

        if len(boxes) < self.config.grid_min_card_count:
            return None

        boxes.sort(key=lambda b: (b[1], b[0]))
        left = min(b[0] for b in boxes)
        top = min(b[1] for b in boxes)
        right = max(b[2] for b in boxes)
        bottom = max(b[3] for b in boxes)
        if (right - left) * (bottom - top) < width * height * 0.12:
            return None

        self.diagnostics.card_grid_count = 1
        return GeometricRegion(
            region_id="R_card_grid_0",
            bounds=(left, top, right, bottom),
            boundary_evidence=["repeated_cards", f"card_count:{len(boxes)}"],
            density_profile=(0.0, 0.0),
            geometry_confidence=min(0.86, 0.52 + len(boxes) * 0.035),
        )

    # ------------------------------------------------------------------
    # Partition by separators (recursive)
    # ------------------------------------------------------------------

    def _partition_by_separators(
        self, arr: np.ndarray, existing_regions: list[GeometricRegion]
    ) -> list[GeometricRegion]:
        """Use strong separators to recursively partition the image into a region tree.

        Only separators with confidence >= 0.55 and span >= min_span_ratio are used.
        """
        height, width = arr.shape[:2]
        separators = [
            r for r in existing_regions
            if any(
                ev.startswith("vertical_separator")
                or ev.startswith("horizontal_separator")
                or ev.startswith("vertical_edge_coverage")
                or ev.startswith("horizontal_edge_coverage")
                for ev in r.boundary_evidence
            )
            and r.geometry_confidence >= 0.55
        ]

        if len(separators) < 1:
            return []

        margin_x = max(12, int(width * 0.018))
        margin_y = max(12, int(height * 0.018))
        v_positions = self._cluster_separator_positions(
            [
                ((s.bounds[0] + s.bounds[2]) // 2, s.geometry_confidence)
                for s in separators
                if any(ev.startswith("vertical") for ev in s.boundary_evidence)
                and margin_x <= ((s.bounds[0] + s.bounds[2]) // 2) <= width - margin_x
            ],
            min_gap=max(24, int(width * 0.025)),
        )[:6]
        h_positions = self._cluster_separator_positions(
            [
                ((s.bounds[1] + s.bounds[3]) // 2, s.geometry_confidence)
                for s in separators
                if any(ev.startswith("horizontal") for ev in s.boundary_evidence)
                and margin_y <= ((s.bounds[1] + s.bounds[3]) // 2) <= height - margin_y
            ],
            min_gap=max(24, int(height * 0.03)),
        )[:6]

        if not v_positions and not h_positions:
            return []

        x_splits = [0] + sorted(v_positions) + [width]
        y_splits = [0] + sorted(h_positions) + [height]
        min_area = width * height * self.config.min_region_area_ratio
        regions: list[GeometricRegion] = []
        region_idx = 0
        for top, bottom in zip(y_splits, y_splits[1:]):
            if bottom - top < 24:
                continue
            for left, right in zip(x_splits, x_splits[1:]):
                if right - left < 24:
                    continue
                area = (right - left) * (bottom - top)
                if area < min_area:
                    continue
                regions.append(GeometricRegion(
                    region_id=f"R{region_idx}",
                    bounds=(left, top, right, bottom),
                    boundary_evidence=["separator_partition"],
                    density_profile=self._compute_density(arr[top:bottom, left:right]),
                    geometry_confidence=0.55,
                ))
                region_idx += 1

        if regions:
            return self._build_region_tree(regions)

        # Fallback for degenerate splits.
        v_seps = sorted(
            [s for s in separators if any(ev.startswith("vertical") for ev in s.boundary_evidence)],
            key=lambda s: -s.geometry_confidence,
        )
        h_seps = sorted(
            [s for s in separators if any(ev.startswith("horizontal") for ev in s.boundary_evidence)],
            key=lambda s: -s.geometry_confidence,
        )

        regions: list[GeometricRegion] = []
        region_idx = 0

        if v_seps and h_seps:
            # Cross partition: use strongest V and H separator
            v_x = (v_seps[0].bounds[0] + v_seps[0].bounds[2]) // 2
            h_y = (h_seps[0].bounds[1] + h_seps[0].bounds[3]) // 2
            quads = [
                (0, 0, v_x, h_y),
                (v_x, 0, width, h_y),
                (0, h_y, v_x, height),
                (v_x, h_y, width, height),
            ]
            for left, top, right, bottom in quads:
                if right - left >= 24 and bottom - top >= 24:
                    regions.append(GeometricRegion(
                        region_id=f"R{region_idx}",
                        bounds=(left, top, right, bottom),
                        boundary_evidence=["separator_partition"],
                        density_profile=self._compute_density(arr[top:bottom, left:right]),
                        geometry_confidence=0.55,
                    ))
                    region_idx += 1
        elif v_seps:
            # Vertical splits only
            splits = [0] + sorted({(s.bounds[0] + s.bounds[2]) // 2 for s in v_seps[:3]}) + [width]
            for i in range(len(splits) - 1):
                left, right = splits[i], splits[i + 1]
                if right - left >= 24:
                    regions.append(GeometricRegion(
                        region_id=f"R{region_idx}",
                        bounds=(left, 0, right, height),
                        boundary_evidence=["separator_partition"],
                        density_profile=self._compute_density(arr[:, left:right]),
                        geometry_confidence=0.50,
                    ))
                    region_idx += 1
        elif h_seps:
            # Horizontal splits only
            splits = [0] + sorted({(s.bounds[1] + s.bounds[3]) // 2 for s in h_seps[:3]}) + [height]
            for i in range(len(splits) - 1):
                top, bottom = splits[i], splits[i + 1]
                if bottom - top >= 24:
                    regions.append(GeometricRegion(
                        region_id=f"R{region_idx}",
                        bounds=(0, top, width, bottom),
                        boundary_evidence=["separator_partition"],
                        density_profile=self._compute_density(arr[top:bottom, :]),
                        geometry_confidence=0.50,
                    ))
                    region_idx += 1

        if not regions:
            return []

        # Assign parent-child relationships
        return self._build_region_tree(regions)

    @staticmethod
    def _cluster_separator_positions(
        positions: list[tuple[int, float]],
        min_gap: int,
    ) -> list[int]:
        """Merge nearby separator positions and prefer higher-confidence lines."""
        if not positions:
            return []

        ordered = sorted(positions, key=lambda item: item[0])
        groups: list[list[tuple[int, float]]] = []
        current: list[tuple[int, float]] = [ordered[0]]
        for position, confidence in ordered[1:]:
            last_position = current[-1][0]
            if position - last_position <= min_gap:
                current.append((position, confidence))
            else:
                groups.append(current)
                current = [(position, confidence)]
        groups.append(current)

        representatives: list[tuple[int, float]] = []
        for group in groups:
            weighted_total = sum(max(conf, 0.01) for _pos, conf in group)
            weighted_position = round(
                sum(pos * max(conf, 0.01) for pos, conf in group) / weighted_total
            )
            best_confidence = max(conf for _pos, conf in group)
            representatives.append((int(weighted_position), best_confidence))

        representatives.sort(key=lambda item: item[1], reverse=True)
        return [position for position, _confidence in representatives]

    def _build_region_tree(self, regions: list[GeometricRegion]) -> list[GeometricRegion]:
        """Build parent-child relationships for nested regions."""
        sorted_regions = sorted(regions, key=lambda r: (
            (r.bounds[2] - r.bounds[0]) * (r.bounds[3] - r.bounds[1])
        ), reverse=True)

        result: list[GeometricRegion] = []
        assigned: set[int] = set()

        for i, region in enumerate(sorted_regions):
            for j, other in enumerate(sorted_regions):
                if i == j or j in assigned:
                    continue
                containment = self._compute_containment(other.bounds, region.bounds)
                if containment > 0.80:
                    sorted_regions[j] = GeometricRegion(
                        region_id=other.region_id,
                        bounds=other.bounds,
                        boundary_evidence=list(other.boundary_evidence),
                        density_profile=other.density_profile,
                        geometry_confidence=other.geometry_confidence,
                        parent_region_id=region.region_id,
                    )
                    assigned.add(j)

        for i, region in enumerate(sorted_regions):
            if i not in assigned:
                result.append(region)
        for j in assigned:
            result.append(sorted_regions[j])
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_density(block: np.ndarray) -> tuple[float, float]:
        if block.size == 0:
            return (0.0, 0.0)
        std_val = float(block.std())
        if block.ndim >= 2 and block.shape[0] > 1 and block.shape[1] > 1:
            edge_h = np.abs(block[:, 1:] - block[:, :-1]).mean() if block.shape[1] > 1 else 0.0
            edge_v = np.abs(block[1:, :] - block[:-1, :]).mean() if block.shape[0] > 1 else 0.0
            edge_density = float((edge_h + edge_v) / 2)
        else:
            edge_density = 0.0
        return (std_val, edge_density)

    @staticmethod
    def _compute_containment(
        inner: tuple[int, int, int, int],
        outer: tuple[int, int, int, int],
    ) -> float:
        left = max(inner[0], outer[0])
        top = max(inner[1], outer[1])
        right = min(inner[2], outer[2])
        bottom = min(inner[3], outer[3])
        if right <= left or bottom <= top:
            return 0.0
        inter = (right - left) * (bottom - top)
        inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
        return inter / max(inner_area, 1)

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        left = max(a[0], b[0])
        top = max(a[1], b[1])
        right = min(a[2], b[2])
        bottom = min(a[3], b[3])
        if right <= left or bottom <= top:
            return 0.0
        inter = (right - left) * (bottom - top)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / max(area_a + area_b - inter, 1)

    @staticmethod
    def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for i, active in enumerate(mask.tolist()):
            if active and start is None:
                start = i
            elif not active and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(mask)))
        return runs

    @staticmethod
    def _connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
        height, width = mask.shape
        visited = np.zeros(mask.shape, dtype=bool)
        components: list[tuple[int, int, int, int, int]] = []
        for y in range(height):
            xs = np.where(mask[y] & ~visited[y])[0]
            for x_start in xs.tolist():
                if visited[y, x_start] or not mask[y, x_start]:
                    continue
                stack = [(x_start, y)]
                visited[y, x_start] = True
                min_x = max_x = x_start
                min_y = max_y = y
                count = 0
                while stack:
                    x, cy = stack.pop()
                    count += 1
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)
                    for nx, ny in ((x + 1, cy), (x - 1, cy), (x, cy + 1), (x, cy - 1)):
                        if nx < 0 or nx >= width or ny < 0 or ny >= height:
                            continue
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((nx, ny))
                components.append((min_x, min_y, max_x + 1, max_y + 1, count))
        return components

    def _dedupe_and_clip(
        self,
        regions: list[GeometricRegion],
        width: int,
        height: int,
    ) -> list[GeometricRegion]:
        kept: list[GeometricRegion] = []
        for region in regions:
            left, top, right, bottom = region.bounds
            clipped = (
                max(0, min(width, int(left))),
                max(0, min(height, int(top))),
                max(0, min(width, int(right))),
                max(0, min(height, int(bottom))),
            )
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            normalized = GeometricRegion(
                region_id=region.region_id,
                bounds=clipped,
                boundary_evidence=list(region.boundary_evidence),
                density_profile=region.density_profile,
                geometry_confidence=region.geometry_confidence,
                parent_region_id=region.parent_region_id,
            )
            if any(self._iou(normalized.bounds, existing.bounds) > 0.9 for existing in kept):
                continue
            kept.append(normalized)
        return kept
