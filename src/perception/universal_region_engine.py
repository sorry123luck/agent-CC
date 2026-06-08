"""UniversalRegionEngine P0: Region Evidence Model.

Builds PageRegion shadow artifacts from multiple skeleton sources.
Does NOT participate in the main observe/query/ROI pipeline.

Skeleton sources (priority order):
1. GeometricRegion (always available)
2. StructuredRegionOverlay groups (always available)
3. band_proposal_pass proposals (always available)
4. U4 candidate_regions (only when OPENCLAW_U4_LAYOUT_SHADOW=1)
5. full_window_fallback (when all above are empty)

Gate: OPENCLAW_URE_P0_SHADOW=1 to enable (default off).
Output: artifacts["universal_region_shadow"]
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.perception.region_evidence_model import (
    SKELETON_SOURCE_BAND_PROPOSAL,
    SKELETON_SOURCE_FULL_WINDOW,
    SKELETON_SOURCE_GEOMETRIC,
    SKELETON_SOURCE_STRUCTURED_GROUP,
    SKELETON_SOURCE_U4_CANDIDATE,
    BoundaryEvidence,
    PageRegion,
    RegionSemanticHint,
    RegionSubstructure,
    compute_evidence_sources,
    compute_evidence_summary,
)

logger = logging.getLogger(__name__)

# Substructure classification constants
_NOISE_TYPES = {"unknown_evidence", "whitespace_block"}


class UniversalRegionEngine:
    """Builds PageRegion shadow artifacts from multiple evidence sources.

    P0: All region_type=UNKNOWN, no merge, no product classification.
    """

    def build(
        self,
        geometric_regions: list[Any],
        structured_overlay: dict[str, Any] | None = None,
        band_proposals: list[dict[str, Any]] | None = None,
        ocr_blocks: list[dict[str, Any]] | None = None,
        vision_candidates: list[dict[str, Any]] | None = None,
        raw_elements: list[dict[str, Any]] | None = None,
        window_width: int = 0,
        window_height: int = 0,
        u4_layout: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build universal_region_shadow artifact.

        Args:
            geometric_regions: GeometricRegion objects from main pipeline.
            structured_overlay: StructuredRegionOverlay dict (for groups).
            band_proposals: BandProposal list (for band proposals).
            ocr_blocks: OCR blocks for evidence.
            vision_candidates: Vision candidates for evidence.
            raw_elements: UIA elements for evidence.
            window_width: Window width.
            window_height: Window height.
            u4_layout: U4 shadow output (optional, only when U4 enabled).

        Returns:
            universal_region_shadow dict for artifacts.
        """
        # Collect skeleton candidates from multiple sources
        skeletons = self._collect_skeletons(
            geometric_regions, structured_overlay, band_proposals,
            u4_layout, window_width, window_height,
        )

        # If no skeletons, use full window fallback (only if window has valid size)
        if not skeletons:
            if window_width > 0 and window_height > 0:
                skeletons = [self._make_full_window_fallback(window_width, window_height)]
            else:
                # No valid skeletons and no valid window size → return empty
                return {
                    "page_regions": [],
                    "total_regions": 0,
                    "skeleton_sources": [],
                    "window_width": window_width,
                    "window_height": window_height,
                    "excluded_reason": "zero_window_size",
                }

        # Build PageRegions with evidence
        page_regions: list[dict[str, Any]] = []
        for skel in skeletons:
            pr = self._build_page_region(
                skel, ocr_blocks, vision_candidates, raw_elements,
                window_width, window_height,
            )
            page_regions.append(pr.to_dict())

        return {
            "page_regions": page_regions,
            "total_regions": len(page_regions),
            "skeleton_sources": list({s["skeleton_source"] for s in skeletons}),
            "window_width": window_width,
            "window_height": window_height,
        }

    def _collect_skeletons(
        self,
        geometric_regions: list[Any],
        structured_overlay: dict[str, Any] | None,
        band_proposals: list[dict[str, Any]] | None,
        u4_layout: dict[str, Any] | None,
        window_width: int,
        window_height: int,
    ) -> list[dict[str, Any]]:
        """Collect skeleton candidates from multiple sources."""
        skeletons: list[dict[str, Any]] = []
        total_area = max(1, window_width * window_height)

        # Source 1: GeometricRegion (priority 1)
        for gr in geometric_regions:
            bounds = self._extract_bounds(gr)
            if bounds:
                area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                skeletons.append({
                    "bounds": bounds,
                    "skeleton_source": SKELETON_SOURCE_GEOMETRIC,
                    "area_ratio": area / total_area,
                    "ocr_count": 0,
                    "vision_count": 0,
                    "uia_count": 0,
                })

        # Source 2: StructuredRegionOverlay groups (priority 2)
        if structured_overlay:
            for group in structured_overlay.get("groups", []):
                bounds = self._extract_bounds_from_dict(group)
                if bounds and not self._overlaps_existing(bounds, skeletons):
                    area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                    skeletons.append({
                        "bounds": bounds,
                        "skeleton_source": SKELETON_SOURCE_STRUCTURED_GROUP,
                        "area_ratio": area / total_area,
                        "ocr_count": 0,
                        "vision_count": 0,
                        "uia_count": 0,
                    })

        # Source 3: band_proposals (priority 3)
        if band_proposals:
            for bp in band_proposals:
                bounds = self._extract_bounds_from_dict(bp)
                if bounds and not self._overlaps_existing(bounds, skeletons):
                    area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                    skeletons.append({
                        "bounds": bounds,
                        "skeleton_source": SKELETON_SOURCE_BAND_PROPOSAL,
                        "area_ratio": area / total_area,
                        "ocr_count": 0,
                        "vision_count": 0,
                        "uia_count": 0,
                    })

        # Source 4: U4 candidate_regions (optional, priority 4)
        if u4_layout:
            for cr in u4_layout.get("candidate_regions", []):
                bounds = self._extract_bounds_from_dict(cr)
                if bounds and not self._overlaps_existing(bounds, skeletons):
                    area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                    skeletons.append({
                        "bounds": bounds,
                        "skeleton_source": SKELETON_SOURCE_U4_CANDIDATE,
                        "area_ratio": area / total_area,
                        "ocr_count": cr.get("ocr_count", 0),
                        "vision_count": cr.get("vision_count", 0),
                        "uia_count": cr.get("uia_count", 0),
                    })

        return skeletons

    def _make_full_window_fallback(
        self, window_width: int, window_height: int,
    ) -> dict[str, Any]:
        """Create a full-window fallback skeleton."""
        return {
            "bounds": (0, 0, window_width, window_height),
            "skeleton_source": SKELETON_SOURCE_FULL_WINDOW,
            "area_ratio": 1.0,
            "ocr_count": 0,
            "vision_count": 0,
            "uia_count": 0,
        }

    def _build_page_region(
        self,
        skeleton: dict[str, Any],
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
        raw_elements: list[dict[str, Any]] | None,
        window_width: int,
        window_height: int,
    ) -> PageRegion:
        """Build a single PageRegion from skeleton + evidence."""
        bounds = skeleton["bounds"]
        l, t, r, b = bounds

        # Collect evidence within this region
        ocr_items = self._items_in_region(ocr_blocks, l, t, r, b, "bbox")
        vis_items = self._items_in_region(
            vision_candidates, l, t, r, b, "bounding_rect",
        )
        uia_items = self._items_in_region(
            raw_elements, l, t, r, b, "bounding_rect",
        )

        # Merge with skeleton's pre-existing evidence counts
        # (e.g., U4 candidates may already have ocr_count/vision_count/uia_count)
        ocr_count = max(len(ocr_items), skeleton.get("ocr_count", 0))
        vis_count = max(len(vis_items), skeleton.get("vision_count", 0))
        uia_count = max(len(uia_items), skeleton.get("uia_count", 0))

        # Build substructures
        substructures = self._classify_substructures(
            ocr_items, vis_items, uia_items,
        )

        # Compute evidence sources
        evidence_sources = compute_evidence_sources(
            ocr_count, vis_count, uia_count,
        )
        evidence_summary = compute_evidence_summary(ocr_count, vis_count, uia_count)

        return PageRegion(
            region_id=f"PR{0}",  # Will be renumbered by caller if needed
            bounds=bounds,
            region_type="unknown",
            confidence=0.0,
            skeleton_source=skeleton["skeleton_source"],
            evidence_sources=evidence_sources,
            substructures=substructures,
            boundary_evidence=[],  # P0: no boundary evidence
            semantic_hints=[],  # P0: no semantic hints
            area_ratio=skeleton["area_ratio"],
            evidence_summary=evidence_summary,
        )

    def _classify_substructures(
        self,
        ocr_items: list[dict[str, Any]],
        vis_items: list[dict[str, Any]],
        uia_items: list[dict[str, Any]],
    ) -> list[RegionSubstructure]:
        """Classify items within a region as substructures."""
        substructures: list[RegionSubstructure] = []
        sub_idx = 0

        # OCR items
        for item in ocr_items:
            bbox = item.get("bbox", [])
            if len(bbox) < 4:
                continue
            text = item.get("text", "").strip()
            sub_type, conf = self._classify_ocr_item(text, bbox)
            substructures.append(RegionSubstructure(
                sub_id=f"S{sub_idx}",
                sub_type=sub_type,
                bounds=tuple(bbox),
                evidence_sources=["ocr"],
                confidence=conf,
                text=text,
            ))
            sub_idx += 1

        # Vision items
        for item in vis_items:
            bbox = item.get("bounding_rect") or item.get("bbox", [])
            if len(bbox) < 4:
                continue
            sub_type, conf = self._classify_vision_item(item, bbox)
            substructures.append(RegionSubstructure(
                sub_id=f"S{sub_idx}",
                sub_type=sub_type,
                bounds=tuple(bbox),
                evidence_sources=["vision"],
                confidence=conf,
                control_type=item.get("label", ""),
            ))
            sub_idx += 1

        # UIA items
        for item in uia_items:
            bbox = item.get("bounding_rect") or item.get("bbox", [])
            if len(bbox) < 4:
                continue
            ct = item.get("control_type", "")
            sub_type, conf = self._classify_uia_item(ct, bbox)
            substructures.append(RegionSubstructure(
                sub_id=f"S{sub_idx}",
                sub_type=sub_type,
                bounds=tuple(bbox),
                evidence_sources=["uia"],
                confidence=conf,
                control_type=ct,
            ))
            sub_idx += 1

        return substructures

    def _classify_ocr_item(
        self, text: str, bbox: list[int],
    ) -> tuple[str, float]:
        """Classify an OCR item as substructure type."""
        if not text or not text.strip():
            return "unknown_evidence", 0.3

        text = text.strip()
        text_len = len(text)
        bbox_h = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
        bbox_w = bbox[2] - bbox[0] if len(bbox) >= 4 else 0

        # Code detection
        code_keywords = [
            "def ", "class ", "import ", "from ", "function ", "var ",
            "const ", "let ", "if ", "for ", "while ", "return ",
            "self.", "this.", "public ", "private ",
        ]
        code_patterns = ["{", "}", "()", "[]", "=>", "->", "::", "://"]
        code_score = 0
        for kw in code_keywords:
            if kw in text.lower() or kw in text:
                code_score += 2
                break
        for pat in code_patterns:
            if pat in text:
                code_score += 1
                break
        if text.startswith("  ") or text.startswith("\t"):
            code_score += 1
        if code_score >= 3:
            return "code_line", min(0.9, 0.5 + code_score * 0.1)

        # List item detection
        import re
        if text_len < 15 and bbox_w < 200:
            return "list_item_like", 0.5
        if re.match(r'^[\d:.\-/]+$', text):
            return "list_item_like", 0.5
        if text_len < 20 and " " not in text:
            return "list_item_like", 0.4

        # Default: text_row
        return "text_row", 0.6

    def _classify_vision_item(
        self, item: dict[str, Any], bbox: list[int],
    ) -> tuple[str, float]:
        """Classify a vision candidate as substructure type."""
        label = (item.get("label", "") or "").lower()
        bbox_w = bbox[2] - bbox[0] if len(bbox) >= 4 else 0
        bbox_h = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
        area = bbox_w * bbox_h

        # Small square → icon
        if area < 2500 and 0.5 < bbox_w / max(bbox_h, 1) < 2.0:
            return "icon_candidate", 0.6

        # Control-related labels
        control_words = ["button", "btn", "icon", "menu", "tab",
                         "toolbar", "scroll", "slider", "toggle"]
        if any(w in label for w in control_words):
            return "control_candidate", 0.7

        # Text-related labels
        text_words = ["text", "label", "title", "heading"]
        if any(w in label for w in text_words):
            return "text_row", 0.5

        # Large item → text
        if area > 10000:
            return "text_row", 0.4

        # Medium → icon
        if area > 1000 and bbox_w < 100 and bbox_h < 100:
            return "icon_candidate", 0.4

        # Small → icon
        if area <= 1000:
            return "icon_candidate", 0.3

        return "unknown_evidence", 0.3

    def _classify_uia_item(
        self, control_type: str, bbox: list[int],
    ) -> tuple[str, float]:
        """Classify a UIA element as substructure type."""
        ct = (control_type or "").lower()
        bbox_w = bbox[2] - bbox[0] if len(bbox) >= 4 else 0
        bbox_h = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
        area = bbox_w * bbox_h

        # Structural containers (large) → noise
        structural_types = ("window", "pane", "document", "group", "custom")
        if any(ct.startswith(t) for t in structural_types):
            if area > 50000:
                return "unknown_evidence", 0.2
            return "control_candidate", 0.4

        # Text
        if ct in ("text", "textblock", "edit", "edittext", "label"):
            return "text_row", 0.7

        # Interactive
        if ct in ("button", "hyperlink", "menuitem", "tabitem",
                  "checkbox", "radiobutton", "togglebutton"):
            return "control_candidate", 0.8

        # Image/icon
        if ct in ("image", "icon", "thumb"):
            return "icon_candidate", 0.7

        # List items
        if ct in ("listitem", "treeitem", "dataitem"):
            return "list_item_like", 0.7

        # List containers
        if ct in ("list", "tree", "datagrid", "listview"):
            return "control_candidate", 0.5

        # Tab
        if ct in ("tab", "tabcontrol", "tabitem"):
            return "control_candidate", 0.7

        # Scroll
        if ct in ("scrollbar",):
            return "control_candidate", 0.6

        # Menu/toolbar
        if ct in ("menu", "menubar", "toolbar", "statusbar"):
            return "control_candidate", 0.7

        return "unknown_evidence", 0.3

    # ── Geometry helpers ────────────────────────────────────────

    def _extract_bounds(self, region: Any) -> tuple[int, int, int, int] | None:
        """Extract bounds from a region object."""
        if hasattr(region, "bounds"):
            b = region.bounds
            if len(b) >= 4:
                return (b[0], b[1], b[2], b[3])
        if hasattr(region, "rect"):
            r = region.rect
            if len(r) >= 4:
                return (r[0], r[1], r[0] + r[2], r[1] + r[3])
        return None

    def _extract_bounds_from_dict(
        self, d: dict[str, Any],
    ) -> tuple[int, int, int, int] | None:
        """Extract bounds from a dict."""
        bounds = d.get("bounds", [])
        if len(bounds) >= 4:
            return (bounds[0], bounds[1], bounds[2], bounds[3])
        rect = d.get("rect", [])
        if len(rect) >= 4:
            return (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
        return None

    def _overlaps_existing(
        self,
        bounds: tuple[int, int, int, int],
        existing: list[dict[str, Any]],
        threshold: float = 0.8,
    ) -> bool:
        """Check if bounds overlaps significantly with any existing skeleton."""
        l, t, r, b = bounds
        area = max(1, (r - l) * (b - t))
        for skel in existing:
            eb = skel["bounds"]
            # Compute intersection
            il = max(l, eb[0])
            it_val = max(t, eb[1])
            ir = min(r, eb[2])
            ib = min(b, eb[3])
            if il < ir and it_val < ib:
                intersection = (ir - il) * (ib - it_val)
                if intersection / area > threshold:
                    return True
        return False

    def _items_in_region(
        self,
        items: list[dict[str, Any]] | None,
        left: int, top: int, right: int, bottom: int,
        bbox_key: str,
    ) -> list[dict[str, Any]]:
        """Get items whose center falls within the region."""
        if not items:
            return []
        result = []
        for item in items:
            bbox = item.get(bbox_key) or item.get("bbox", [])
            if not bbox or len(bbox) < 4:
                continue
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            if left <= cx < right and top <= cy < bottom:
                result.append(item)
        return result
