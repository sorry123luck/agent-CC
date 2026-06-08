"""U2-shadow: UIA Boundary Shadow Pass.

Read-only diagnostic that evaluates whether UIA element boundaries
can improve geometric partitioning. Does NOT modify GeometricPartitioner,
snapshot.regions, element.region_id, ROI, query, or backfill.

Only runs when invoked explicitly (QA script or OPENCLAW_U2_UIA_SHADOW=1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from src.perception.geometric_partitioner import GeometricRegion, PartitionerConfig


# Rejection reasons
ROOT_ONLY_UIA = "root_only_uia"
NON_UIA_SOURCE = "non_uia_source"
SCREENSHOT_MISSING = "screenshot_missing"
INSUFFICIENT_SPAN = "insufficient_span"
INSUFFICIENT_ELEMENT_SUPPORT = "insufficient_element_support"
DENSITY_CONTRAST_LOW = "density_contrast_low"
TOO_CLOSE_TO_EXISTING = "too_close_to_existing_boundary"
TOO_MANY_MICRO = "too_many_micro_boundaries"


@dataclass
class UiBoundaryDiagnostic:
    """Diagnostic output for one UIA boundary candidate."""

    position: int  # y for horizontal, x for vertical
    direction: str  # "horizontal" or "vertical"
    accepted: bool
    reason: str = ""
    span_ratio: float = 0.0
    element_count: int = 0
    density_contrast: float = 0.0


@dataclass
class UiaBoundaryShadowResult:
    """Result of U2-shadow pass."""

    raw_elements_count: int = 0
    uia_element_count: int = 0
    usable_uia_boundary_count: int = 0
    accepted_boundary_count: int = 0
    rejected_boundary_count: int = 0
    rejected_reasons: dict[str, int] = field(default_factory=dict)
    no_uia_signal: bool = False
    over_fragmentation_score: float = 0.0
    boundary_diagnostics: list[UiBoundaryDiagnostic] = field(default_factory=list)
    uia_aware_regions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_elements_count": self.raw_elements_count,
            "uia_element_count": self.uia_element_count,
            "usable_uia_boundary_count": self.usable_uia_boundary_count,
            "accepted_boundary_count": self.accepted_boundary_count,
            "rejected_boundary_count": self.rejected_boundary_count,
            "rejected_reasons": dict(self.rejected_reasons),
            "no_uia_signal": self.no_uia_signal,
            "over_fragmentation_score": round(self.over_fragmentation_score, 3),
            "uia_aware_regions": list(self.uia_aware_regions),
        }


class UiaBoundaryShadowPass:
    """Detect UIA element boundaries and evaluate them as geometric separators.

    This is a read-only diagnostic pass. It does NOT modify any shared state.
    """

    # Thresholds (matching GeometricPartitioner defaults)
    MIN_SPAN_RATIO = 0.55
    DENSITY_CONTRAST_THRESHOLD = 0.12
    MIN_ELEMENT_SUPPORT = 3
    MIN_BOUNDARY_GAP = 8  # pixels
    MIN_BOUNDARY_WIDTH = 12  # pixels
    MAX_REGIONS = 20
    MIN_REGION_AREA_RATIO = 0.006

    def run(
        self,
        screenshot: Image.Image | None,
        raw_elements: list[dict[str, Any]] | None,
        existing_regions: list[GeometricRegion] | None = None,
        window_width: int = 0,
        window_height: int = 0,
    ) -> UiaBoundaryShadowResult:
        """Run U2-shadow pass.

        Args:
            screenshot: PIL Image (for density contrast check)
            raw_elements: UIA elements with bounding_rect
            existing_regions: existing GeometricRegions (to avoid overlap)
            window_width: window width
            window_height: window height

        Returns:
            UiaBoundaryShadowResult with diagnostics
        """
        result = UiaBoundaryShadowResult()
        result.raw_elements_count = len(raw_elements or [])

        # Filter to UIA-only elements
        uia_elements = self._filter_uia_elements(raw_elements or [])
        result.uia_element_count = len(uia_elements)

        # Check for root-only UIA
        if len(uia_elements) <= 1:
            result.no_uia_signal = True
            result.rejected_reasons[ROOT_ONLY_UIA] = 1
            return result

        if screenshot is None:
            result.rejected_reasons[SCREENSHOT_MISSING] = 1
            return result

        if window_width <= 0 or window_height <= 0:
            result.rejected_reasons[SCREENSHOT_MISSING] = 1
            return result

        arr = np.asarray(screenshot.convert("RGB")).astype(np.int16)

        # Detect horizontal UIA boundaries
        h_boundaries = self._detect_boundaries(
            uia_elements, "horizontal", arr, window_width, window_height,
            existing_regions,
        )

        # Detect vertical UIA boundaries
        v_boundaries = self._detect_boundaries(
            uia_elements, "vertical", arr, window_width, window_height,
            existing_regions,
        )

        all_boundaries = h_boundaries + v_boundaries
        result.boundary_diagnostics = all_boundaries

        # Count accepted/rejected
        accepted = [b for b in all_boundaries if b.accepted]
        rejected = [b for b in all_boundaries if not b.accepted]
        result.accepted_boundary_count = len(accepted)
        result.rejected_boundary_count = len(rejected)
        result.usable_uia_boundary_count = len(accepted)

        for b in rejected:
            result.rejected_reasons[b.reason] = result.rejected_reasons.get(b.reason, 0) + 1

        # Build uia-aware regions from accepted boundaries
        if accepted:
            regions = self._build_regions_from_boundaries(
                accepted, window_width, window_height, arr,
            )
            result.uia_aware_regions = [r.to_dict() for r in regions]
            result.over_fragmentation_score = self._compute_fragmentation(regions, window_width, window_height)

        return result

    def _filter_uia_elements(self, raw_elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter to UIA-only elements (exclude OCR synth, boundary_candidate, etc.)."""
        uia = []
        for elem in raw_elements:
            source = elem.get("synthetic_source", "")
            if source in ("ocr", "ocr_actionable", "boundary_candidate", "app_layout"):
                continue
            # Must have bounding_rect
            bounds = elem.get("bounding_rect")
            if not bounds or len(bounds) < 4:
                continue
            uia.append(elem)
        return uia

    def _detect_boundaries(
        self,
        uia_elements: list[dict[str, Any]],
        direction: str,
        arr: np.ndarray,
        window_width: int,
        window_height: int,
        existing_regions: list[GeometricRegion] | None,
    ) -> list[UiBoundaryDiagnostic]:
        """Detect UIA element boundaries in one direction."""
        diagnostics: list[UiBoundaryDiagnostic] = []
        height, width = arr.shape[:2]

        if direction == "horizontal":
            dim_size = window_height
            min_span = max(12, int(window_width * self.MIN_SPAN_RATIO))
        else:
            dim_size = window_width
            min_span = max(12, int(window_height * self.MIN_SPAN_RATIO))

        # Collect all boundary positions from UIA elements
        boundary_positions: dict[int, list[dict]] = {}  # position → [elements]
        for elem in uia_elements:
            bounds = elem.get("bounding_rect", [])
            if len(bounds) < 4:
                continue
            l, t, r, b = bounds
            if direction == "horizontal":
                for pos in [t, b]:
                    if 0 < pos < dim_size:
                        boundary_positions.setdefault(pos, []).append(elem)
            else:
                for pos in [l, r]:
                    if 0 < pos < dim_size:
                        boundary_positions.setdefault(pos, []).append(elem)

        # Cluster nearby positions
        sorted_positions = sorted(boundary_positions.keys())
        clusters: list[tuple[int, list[dict]]] = []
        i = 0
        while i < len(sorted_positions):
            pos = sorted_positions[i]
            elems = list(boundary_positions[pos])
            j = i + 1
            while j < len(sorted_positions) and sorted_positions[j] - pos <= self.MIN_BOUNDARY_GAP:
                elems.extend(boundary_positions[sorted_positions[j]])
                j += 1
            cluster_pos = pos  # take first position
            clusters.append((cluster_pos, elems))
            i = j

        # Evaluate each cluster
        for pos, elems in clusters:
            diag = UiBoundaryDiagnostic(
                position=pos,
                direction=direction,
                accepted=False,
                element_count=len(elems),
            )

            # Check element support
            if len(elems) < self.MIN_ELEMENT_SUPPORT:
                diag.reason = INSUFFICIENT_ELEMENT_SUPPORT
                diagnostics.append(diag)
                continue

            # Check span
            if direction == "horizontal":
                # Check horizontal span of elements at this y position
                lefts = [e.get("bounding_rect", [0])[0] for e in elems if len(e.get("bounding_rect", [])) >= 4]
                rights = [e.get("bounding_rect", [0, 0, 0])[2] for e in elems if len(e.get("bounding_rect", [])) >= 4]
                if lefts and rights:
                    span = max(rights) - min(lefts)
                    span_ratio = span / window_width
                else:
                    span_ratio = 0
            else:
                tops = [e.get("bounding_rect", [0, 0])[1] for e in elems if len(e.get("bounding_rect", [])) >= 4]
                bottoms = [e.get("bounding_rect", [0, 0, 0])[3] for e in elems if len(e.get("bounding_rect", [])) >= 4]
                if tops and bottoms:
                    span = max(bottoms) - min(tops)
                    span_ratio = span / window_height
                else:
                    span_ratio = 0

            diag.span_ratio = round(span_ratio, 3)

            if span_ratio < self.MIN_SPAN_RATIO:
                diag.reason = INSUFFICIENT_SPAN
                diagnostics.append(diag)
                continue

            # Check density contrast
            if direction == "horizontal":
                top_band = arr[max(0, pos - 8):pos, :, :] if pos > 0 else None
                bottom_band = arr[pos:min(height, pos + 8), :, :] if pos < height else None
            else:
                top_band = arr[:, max(0, pos - 8):pos, :] if pos > 0 else None
                bottom_band = arr[:, pos:min(width, pos + 8), :] if pos < width else None

            if top_band is not None and top_band.size > 0 and bottom_band is not None and bottom_band.size > 0:
                top_std = float(top_band.std())
                bottom_std = float(bottom_band.std())
                density_contrast = abs(top_std - bottom_std) / max(max(top_std, bottom_std), 0.1)
                diag.density_contrast = round(density_contrast, 3)
                if density_contrast < self.DENSITY_CONTRAST_THRESHOLD:
                    diag.reason = DENSITY_CONTRAST_LOW
                    diagnostics.append(diag)
                    continue

            # Check proximity to existing boundaries
            if existing_regions:
                too_close = False
                for reg in existing_regions:
                    for ev in reg.boundary_evidence:
                        if "separator" in ev:
                            # Extract position from evidence
                            # This is a simplified check
                            too_close = True
                            break
                    if too_close:
                        break
                if too_close:
                    diag.reason = TOO_CLOSE_TO_EXISTING
                    diagnostics.append(diag)
                    continue

            # Accepted
            diag.accepted = True
            diagnostics.append(diag)

        return diagnostics

    def _build_regions_from_boundaries(
        self,
        accepted_boundaries: list[UiBoundaryDiagnostic],
        window_width: int,
        window_height: int,
        arr: np.ndarray,
    ) -> list[GeometricRegion]:
        """Build GeometricRegions from accepted UIA boundaries."""
        # Sort by position
        h_boundaries = sorted(
            [b for b in accepted_boundaries if b.direction == "horizontal"],
            key=lambda b: b.position,
        )
        v_boundaries = sorted(
            [b for b in accepted_boundaries if b.direction == "vertical"],
            key=lambda b: b.position,
        )

        # Build horizontal splits
        h_splits = [0] + [b.position for b in h_boundaries] + [window_height]
        v_splits = [0] + [b.position for b in v_boundaries] + [window_width]

        regions: list[GeometricRegion] = []
        region_idx = 0
        min_area = window_width * window_height * self.MIN_REGION_AREA_RATIO

        for i in range(len(h_splits) - 1):
            for j in range(len(v_splits) - 1):
                top = h_splits[i]
                bottom = h_splits[i + 1]
                left = v_splits[j]
                right = v_splits[j + 1]

                w = right - left
                h = bottom - top
                if w <= 0 or h <= 0:
                    continue
                if w * h < min_area:
                    continue

                evidence = []
                if i < len(h_boundaries):
                    evidence.append(f"uia_h_boundary_y{h_splits[i]}")
                if j < len(v_boundaries):
                    evidence.append(f"uia_v_boundary_x{v_splits[j]}")

                regions.append(GeometricRegion(
                    region_id=f"U{region_idx}",
                    bounds=(left, top, right, bottom),
                    boundary_evidence=evidence,
                    geometry_confidence=0.6,
                ))
                region_idx += 1

                if region_idx >= self.MAX_REGIONS:
                    break
            if region_idx >= self.MAX_REGIONS:
                break

        return regions

    def _compute_fragmentation(
        self,
        regions: list[GeometricRegion],
        window_width: int,
        window_height: int,
    ) -> float:
        """Compute over-fragmentation score (0-1, higher = more fragmented)."""
        if not regions:
            return 0.0
        window_area = window_width * window_height
        if window_area <= 0:
            return 0.0

        # Count regions that are very small (< 2% of window)
        small_count = sum(
            1 for r in regions
            if (r.bounds[2] - r.bounds[0]) * (r.bounds[3] - r.bounds[1]) < window_area * 0.02
        )
        return small_count / max(len(regions), 1)
