"""Band Proposal Pass — detect thin top/bottom bands missed by GeometricPartitioner.

Read-only diagnostic pass. Does NOT modify geometric_regions, page_compiler,
ROI selector, query, or backfill.

Outputs band_proposals to artifacts for visual QA and future region engine phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class BandProposal:
    """A proposed thin band region at top or bottom of window."""

    band_id: str  # "TB0", "TB1", "BB0", etc.
    band_type: str  # "top_band", "bottom_band"
    bounds: tuple[int, int, int, int]  # left, top, right, bottom
    height_px: int
    height_ratio: float  # height / window_height
    detection_method: str  # "ocr_density", "pixel_edge", "uia_boundary"
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_id": self.band_id,
            "band_type": self.band_type,
            "bounds": list(self.bounds),
            "height_px": self.height_px,
            "height_ratio": round(self.height_ratio, 4),
            "detection_method": self.detection_method,
            "evidence": list(self.evidence),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class BandProposalDiagnostics:
    """Aggregate diagnostics for band proposal pass."""

    top_bands_found: int = 0
    bottom_bands_found: int = 0
    proposals: list[BandProposal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_bands_found": self.top_bands_found,
            "bottom_bands_found": self.bottom_bands_found,
            "proposals": [p.to_dict() for p in self.proposals],
        }


class BandProposalPass:
    """Detect thin top/bottom bands missed by GeometricPartitioner.

    Uses OCR text density, pixel edge detection, and UIA boundaries.
    Supports band stacks (multiple thin bands at top).
    Does NOT produce semantic labels — all proposals are unknown_structured.
    """

    # Max band height as ratio of window
    MAX_TOP_BAND_RATIO = 0.15
    MAX_BOTTOM_BAND_RATIO = 0.10
    # Min band height in pixels
    MIN_BAND_HEIGHT = 12
    # OCR density threshold for band detection
    OCR_DENSITY_THRESHOLD = 0.3  # fraction of band area covered by OCR blocks

    def detect(
        self,
        screenshot: Image.Image | None,
        ocr_blocks: list[dict[str, Any]] | None = None,
        raw_elements: list[dict[str, Any]] | None = None,
        window_width: int = 0,
        window_height: int = 0,
    ) -> BandProposalDiagnostics:
        """Detect top/bottom band proposals."""
        diag = BandProposalDiagnostics()

        if not screenshot or window_width <= 0 or window_height <= 0:
            return diag

        arr = np.array(screenshot.convert("L"))  # grayscale

        # Detect top bands
        top_bands = self._detect_top_bands(arr, ocr_blocks, raw_elements, window_width, window_height)
        diag.top_bands_found = len(top_bands)
        diag.proposals.extend(top_bands)

        # Detect bottom bands
        bottom_bands = self._detect_bottom_bands(arr, ocr_blocks, raw_elements, window_width, window_height)
        diag.bottom_bands_found = len(bottom_bands)
        diag.proposals.extend(bottom_bands)

        return diag

    def _detect_top_bands(
        self,
        arr: np.ndarray,
        ocr_blocks: list[dict[str, Any]] | None,
        raw_elements: list[dict[str, Any]] | None,
        window_width: int,
        window_height: int,
    ) -> list[BandProposal]:
        """Detect top band stack (title/menu/tab/address/toolbar)."""
        bands: list[BandProposal] = []
        max_top = int(window_height * self.MAX_TOP_BAND_RATIO)

        # Find horizontal density edges in the top region
        edges = self._find_horizontal_edges(arr[:max_top, :], window_width)

        # Find OCR text row positions in top region
        ocr_rows = self._find_ocr_rows(ocr_blocks, top_limit=max_top)

        # Find UIA element boundaries in top region
        uia_boundaries = self._find_uia_boundaries(raw_elements, top_limit=max_top)

        # Merge all boundary signals
        all_boundaries = sorted(set(edges + ocr_rows + uia_boundaries))

        if not all_boundaries:
            # No boundaries found — check if there's any OCR in top 5%
            top_5pct = int(window_height * 0.05)
            ocr_in_top = self._count_ocr_in_band(ocr_blocks, 0, top_5pct, window_width)
            if ocr_in_top >= 2:
                bands.append(BandProposal(
                    band_id="TB0",
                    band_type="top_band",
                    bounds=(0, 0, window_width, top_5pct),
                    height_px=top_5pct,
                    height_ratio=0.05,
                    detection_method="ocr_density",
                    evidence=[f"ocr_count={ocr_in_top}"],
                    confidence=0.3,
                ))
            return bands

        # Build band stack from boundaries
        band_top = 0
        band_idx = 0
        for boundary_y in all_boundaries:
            band_height = boundary_y - band_top
            if band_height < self.MIN_BAND_HEIGHT:
                continue
            if band_height > window_height * self.MAX_TOP_BAND_RATIO:
                break

            # Check if this band has content
            ocr_count = self._count_ocr_in_band(ocr_blocks, band_top, boundary_y, window_width)
            uia_count = self._count_uia_in_band(raw_elements, band_top, boundary_y, window_width)

            if ocr_count >= 1 or uia_count >= 1:
                bands.append(BandProposal(
                    band_id=f"TB{band_idx}",
                    band_type="top_band",
                    bounds=(0, band_top, window_width, boundary_y),
                    height_px=band_height,
                    height_ratio=band_height / window_height,
                    detection_method=self._primary_method(ocr_count, uia_count),
                    evidence=[
                        f"ocr_count={ocr_count}",
                        f"uia_count={uia_count}",
                        f"boundary_y={boundary_y}",
                    ],
                    confidence=min(0.5, 0.2 + ocr_count * 0.05 + uia_count * 0.05),
                ))
                band_idx += 1

            band_top = boundary_y

        # Check remaining top area after last boundary
        if band_top < max_top:
            remaining_height = max_top - band_top
            if remaining_height >= self.MIN_BAND_HEIGHT:
                ocr_count = self._count_ocr_in_band(ocr_blocks, band_top, max_top, window_width)
                uia_count = self._count_uia_in_band(raw_elements, band_top, max_top, window_width)
                if ocr_count >= 1 or uia_count >= 1:
                    bands.append(BandProposal(
                        band_id=f"TB{band_idx}",
                        band_type="top_band",
                        bounds=(0, band_top, window_width, max_top),
                        height_px=remaining_height,
                        height_ratio=remaining_height / window_height,
                        detection_method=self._primary_method(ocr_count, uia_count),
                        evidence=[
                            f"ocr_count={ocr_count}",
                            f"uia_count={uia_count}",
                            f"remaining_after_last_boundary",
                        ],
                        confidence=min(0.4, 0.15 + ocr_count * 0.05),
                    ))

        return bands

    def _detect_bottom_bands(
        self,
        arr: np.ndarray,
        ocr_blocks: list[dict[str, Any]] | None,
        raw_elements: list[dict[str, Any]] | None,
        window_width: int,
        window_height: int,
    ) -> list[BandProposal]:
        """Detect bottom band (status bar, control strip)."""
        bands: list[BandProposal] = []
        min_bottom = int(window_height * (1 - self.MAX_BOTTOM_BAND_RATIO))

        # Find horizontal density edges in the bottom region
        bottom_arr = arr[min_bottom:, :]
        edges = self._find_horizontal_edges(bottom_arr, window_width)
        edges = [y + min_bottom for y in edges]  # offset to window coords

        # Find OCR text rows in bottom region
        ocr_rows = self._find_ocr_rows(ocr_blocks, top_limit=window_height, bottom_limit=min_bottom)

        # Find UIA element boundaries in bottom region
        uia_boundaries = self._find_uia_boundaries(raw_elements, top_limit=window_height, bottom_limit=min_bottom)

        # Merge all boundary signals
        all_boundaries = sorted(set(edges + ocr_rows + uia_boundaries))

        if not all_boundaries:
            # No boundaries — check bottom 5%
            bottom_5pct = int(window_height * 0.95)
            ocr_in_bottom = self._count_ocr_in_band(ocr_blocks, bottom_5pct, window_height, window_width)
            if ocr_in_bottom >= 1:
                bands.append(BandProposal(
                    band_id="BB0",
                    band_type="bottom_band",
                    bounds=(0, bottom_5pct, window_width, window_height),
                    height_px=window_height - bottom_5pct,
                    height_ratio=(window_height - bottom_5pct) / window_height,
                    detection_method="ocr_density",
                    evidence=[f"ocr_count={ocr_in_bottom}"],
                    confidence=0.25,
                ))
            return bands

        # Build band from first boundary above window bottom
        for boundary_y in reversed(all_boundaries):
            band_height = window_height - boundary_y
            if band_height < self.MIN_BAND_HEIGHT:
                continue
            if band_height > window_height * self.MAX_BOTTOM_BAND_RATIO:
                continue

            ocr_count = self._count_ocr_in_band(ocr_blocks, boundary_y, window_height, window_width)
            uia_count = self._count_uia_in_band(raw_elements, boundary_y, window_height, window_width)

            if ocr_count >= 1 or uia_count >= 1:
                bands.append(BandProposal(
                    band_id="BB0",
                    band_type="bottom_band",
                    bounds=(0, boundary_y, window_width, window_height),
                    height_px=band_height,
                    height_ratio=band_height / window_height,
                    detection_method=self._primary_method(ocr_count, uia_count),
                    evidence=[
                        f"ocr_count={ocr_count}",
                        f"uia_count={uia_count}",
                        f"boundary_y={boundary_y}",
                    ],
                    confidence=min(0.5, 0.2 + ocr_count * 0.05 + uia_count * 0.05),
                ))
                break

        return bands

    # --- Helper methods ---

    def _find_horizontal_edges(
        self,
        arr: np.ndarray,
        window_width: int,
    ) -> list[int]:
        """Find strong horizontal edges in a grayscale array."""
        if arr.shape[0] < 3:
            return []
        diff = np.abs(arr[1:, :].astype(float) - arr[:-1, :].astype(float))
        row_mean = diff.mean(axis=1)
        threshold = max(10.0, float(row_mean.mean() + row_mean.std() * 1.5))
        edge_rows = np.where(row_mean > threshold)[0]
        return [int(y) + 1 for y in edge_rows]

    def _find_ocr_rows(
        self,
        ocr_blocks: list[dict[str, Any]] | None,
        top_limit: int = 0,
        bottom_limit: int = 0,
    ) -> list[int]:
        """Find Y positions where OCR text rows start."""
        if not ocr_blocks:
            return []
        rows: list[int] = []
        for block in ocr_blocks:
            bbox = block.get("bbox", [])
            if len(bbox) < 4:
                continue
            _, bt, _, bb = bbox
            cy = (bt + bb) // 2
            if top_limit > 0 and cy >= top_limit:
                continue
            if bottom_limit > 0 and cy < bottom_limit:
                continue
            rows.append(int(bt))
        return sorted(set(rows))

    def _find_uia_boundaries(
        self,
        raw_elements: list[dict[str, Any]] | None,
        top_limit: int = 0,
        bottom_limit: int = 0,
    ) -> list[int]:
        """Find Y positions where UIA elements start/end."""
        if not raw_elements:
            return []
        boundaries: list[int] = []
        for elem in raw_elements:
            bounds = elem.get("bounding_rect")
            if not bounds or len(bounds) < 4:
                continue
            _, et, _, eb = bounds
            if top_limit > 0 and et >= top_limit:
                continue
            if bottom_limit > 0 and et < bottom_limit:
                continue
            boundaries.append(int(et))
            boundaries.append(int(eb))
        return sorted(set(boundaries))

    def _count_ocr_in_band(
        self,
        ocr_blocks: list[dict[str, Any]] | None,
        band_top: int,
        band_bottom: int,
        window_width: int,
    ) -> int:
        """Count OCR blocks whose center falls within band."""
        if not ocr_blocks:
            return 0
        count = 0
        for block in ocr_blocks:
            bbox = block.get("bbox", [])
            if len(bbox) < 4:
                continue
            _, bt, _, bb = bbox
            cy = (bt + bb) // 2
            if band_top <= cy < band_bottom:
                count += 1
        return count

    def _count_uia_in_band(
        self,
        raw_elements: list[dict[str, Any]] | None,
        band_top: int,
        band_bottom: int,
        window_width: int,
    ) -> int:
        """Count UIA elements whose center falls within band."""
        if not raw_elements:
            return 0
        count = 0
        for elem in raw_elements:
            bounds = elem.get("bounding_rect")
            if not bounds or len(bounds) < 4:
                continue
            _, et, _, eb = bounds
            cy = (et + eb) // 2
            if band_top <= cy < band_bottom:
                count += 1
        return count

    def _primary_method(self, ocr_count: int, uia_count: int) -> str:
        """Determine primary detection method."""
        if ocr_count >= uia_count:
            return "ocr_density"
        return "uia_boundary"
