"""U4.0: Pixel Projection + Whitespace Separator Shadow.

Read-only diagnostic that detects horizontal/vertical whitespace gaps
and density transitions from screenshot pixels. Does NOT use Omni
candidates to determine boundaries. OCR/Omni/UIA only as evidence
counts within proposed regions.

Only runs when invoked explicitly (OPENCLAW_U4_LAYOUT_SHADOW=1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class Separator:
    """A detected whitespace gap or density transition."""

    separator_id: str
    direction: str  # "horizontal" or "vertical"
    position: int  # y for horizontal, x for vertical
    width: int  # gap height for horizontal, gap width for vertical
    span: int  # how far the gap extends (width for h-gap, height for v-gap)
    strength: float  # 0-1, how clean the gap is
    kind: str  # "whitespace" or "density_transition"

    def to_dict(self) -> dict[str, Any]:
        return {
            "separator_id": self.separator_id,
            "direction": self.direction,
            "position": self.position,
            "width": self.width,
            "span": self.span,
            "strength": round(float(self.strength), 3),
            "kind": self.kind,
        }


@dataclass
class CandidateRegion:
    """A proposed layout region bounded by separators."""

    region_id: str
    bounds: tuple[int, int, int, int]
    bounded_by: list[str]  # separator IDs
    ocr_count: int = 0
    vision_count: int = 0
    uia_count: int = 0
    area_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "bounds": list(self.bounds),
            "bounded_by": list(self.bounded_by),
            "ocr_count": self.ocr_count,
            "vision_count": self.vision_count,
            "uia_count": self.uia_count,
            "area_ratio": round(self.area_ratio, 4),
        }


# ── U4.2 Phase A: Substructure / Evidence data model ──────────


class SubstructureType:
    """Substructure type constants for region internal content."""
    TEXT_ROW = "text_row"
    CODE_LINE = "code_line"
    MESSAGE_LIKE = "message_like"
    LIST_ITEM_LIKE = "list_item_like"
    TABLE_ROW_LIKE = "table_row_like"
    ICON_CANDIDATE = "icon_candidate"
    CONTROL_CANDIDATE = "control_candidate"
    WHITESPACE_BLOCK = "whitespace_block"
    UNKNOWN_EVIDENCE = "unknown_evidence"


@dataclass
class RegionSubstructure:
    """A substructure within a region (not a page-level region)."""
    sub_id: str
    sub_type: str  # SubstructureType constant
    bounds: tuple[int, int, int, int]
    evidence_sources: list[str]  # ["ocr", "vision", "uia"]
    confidence: float  # 0-1
    reason: str  # human-readable explanation
    text: str = ""  # OCR text if available
    control_type: str = ""  # UIA control type if available

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub_id": self.sub_id,
            "sub_type": self.sub_type,
            "bounds": list(self.bounds),
            "evidence_sources": self.evidence_sources,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "text": self.text,
            "control_type": self.control_type,
        }


@dataclass
class PageRegionShadow:
    """Shadow output for a candidate region's substructure analysis."""
    region_id: str
    bounds: tuple[int, int, int, int]
    substructures: list[RegionSubstructure] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "bounds": list(self.bounds),
            "substructures": [s.to_dict() for s in self.substructures],
        }


@dataclass
class U4LayoutResult:

    horizontal_separators: list[dict[str, Any]] = field(default_factory=list)
    vertical_separators: list[dict[str, Any]] = field(default_factory=list)
    density_transitions: list[dict[str, Any]] = field(default_factory=list)
    candidate_regions: list[dict[str, Any]] = field(default_factory=list)
    rejected_reasons: dict[str, int] = field(default_factory=dict)
    proposals_count: int = 0
    useful_proposals: int = 0  # Always 0 in U4.0
    neutral_proposals: int = 0
    noisy_proposals: int = 0

    # U4.1 Phase 1: boundary signal diagnostics
    boundary_decisions: list[dict[str, Any]] = field(default_factory=list)
    raw_separator_count: int = 0
    signal_collection_errors: int = 0

    # U4.1 Phase 2: flow-aware merge results
    merged_regions: list[dict[str, Any]] = field(default_factory=list)
    merge_details: list[dict[str, Any]] = field(default_factory=list)
    gap_classifications: dict[str, int] = field(default_factory=dict)

    # U4.2 Phase 1: region container signals
    region_signals: list[dict[str, Any]] = field(default_factory=list)

    # U4.2 Phase A: substructure shadow
    page_region_model_shadow: list[dict[str, Any]] = field(default_factory=list)

    # U4.2 Phase B.1: pairwise flow signals
    pairwise_flow_signals: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizontal_separators": self.horizontal_separators,
            "vertical_separators": self.vertical_separators,
            "density_transitions": self.density_transitions,
            "candidate_regions": self.candidate_regions,
            "rejected_reasons": dict(self.rejected_reasons),
            "proposals_count": self.proposals_count,
            "useful_proposals": self.useful_proposals,
            "neutral_proposals": self.neutral_proposals,
            "noisy_proposals": self.noisy_proposals,
            # U4.1 Phase 1
            "boundary_decisions": self.boundary_decisions,
            "raw_separator_count": self.raw_separator_count,
            "signal_collection_errors": self.signal_collection_errors,
            # U4.1 Phase 2
            "merged_regions": self.merged_regions,
            "merge_details": self.merge_details,
            "gap_classifications": self.gap_classifications,
            # U4.2 Phase 1
            "region_signals": self.region_signals,
            # U4.2 Phase A
            "page_region_model_shadow": self.page_region_model_shadow,
            # U4.2 Phase B.1
            "pairwise_flow_signals": self.pairwise_flow_signals,
        }


class LayoutBoundaryShadowPass:
    """Detect visual layout boundaries from pixel projections."""

    # Whitespace gap thresholds
    MIN_WHITESPACE_GAP = 25  # pixels (content row gaps are typically 5-15px)
    MIN_WHITESPACE_SPAN_RATIO = 0.30  # must span 30% of window

    # Density transition thresholds
    DENSITY_TRANSITION_THRESHOLD = 15.0  # pixel brightness std change
    MIN_TRANSITION_SPAN_RATIO = 0.40

    # Region constraints
    MIN_REGION_AREA_RATIO = 0.02  # 2% of window
    MAX_REGIONS = 15

    # Content stream thresholds
    CONTENT_STREAM_MIN_RUN = 3  # 3+ consecutive same-type regions → stream

    def run(
        self,
        screenshot: Image.Image | None,
        ocr_blocks: list[dict[str, Any]] | None = None,
        vision_candidates: list[dict[str, Any]] | None = None,
        raw_elements: list[dict[str, Any]] | None = None,
        window_width: int = 0,
        window_height: int = 0,
    ) -> U4LayoutResult:
        """Run U4.0 layout boundary detection."""
        result = U4LayoutResult()

        if screenshot is None:
            result.rejected_reasons["screenshot_missing"] = 1
            return result

        if window_width <= 0 or window_height <= 0:
            result.rejected_reasons["invalid_dimensions"] = 1
            return result

        arr = np.asarray(screenshot.convert("L")).astype(np.float32)
        height, width = arr.shape

        # Skip if image is truly blank (very low variance = no content).
        # A white-background app (Notepad, Chrome) still has text with
        # high local contrast, so std > 10. A blank/loading screen has std < 5.
        img_std = float(arr.std())
        if img_std < 5.0:
            result.rejected_reasons["predominantly_whitespace"] = 1
            return result

        # 1. Detect horizontal whitespace gaps (with evidence comparison)
        h_gaps = self._detect_whitespace_gaps(
            arr, "horizontal", width, height, ocr_blocks, vision_candidates,
        )
        result.horizontal_separators = [s.to_dict() for s in h_gaps]

        # 2. Detect vertical whitespace gaps (with evidence comparison)
        v_gaps = self._detect_whitespace_gaps(
            arr, "vertical", width, height, ocr_blocks, vision_candidates,
        )
        result.vertical_separators = [s.to_dict() for s in v_gaps]

        # 3. Detect horizontal density transitions
        h_transitions = self._detect_density_transitions(arr, "horizontal", width, height)
        result.density_transitions = [s.to_dict() for s in h_transitions]

        # 4. Detect vertical density transitions
        v_transitions = self._detect_density_transitions(arr, "vertical", width, height)
        result.density_transitions.extend([s.to_dict() for s in v_transitions])

        # 4b. Filter density transitions that overlap with whitespace gaps.
        # Whitespace gaps are higher-confidence structural boundaries;
        # density transitions near them are redundant and cause fragmentation.
        h_transitions = self._filter_redundant_transitions(
            h_transitions, h_gaps, direction="horizontal",
        )
        v_transitions = self._filter_redundant_transitions(
            v_transitions, v_gaps, direction="vertical",
        )
        result.density_transitions = [
            s.to_dict() for s in h_transitions + v_transitions
        ]

        # 4c. U4.1 Phase 1: Collect boundary signal diagnostics
        all_raw_separators = h_gaps + v_gaps + h_transitions + v_transitions
        result.raw_separator_count = len(all_raw_separators)
        signal_errors = 0
        for sep_list, dir_label in [
            (h_gaps, "horizontal"), (v_gaps, "vertical"),
            (h_transitions, "horizontal"), (v_transitions, "vertical"),
        ]:
            decisions = self.collect_boundary_signals(
                sep_list, dir_label, width, height, arr,
                ocr_blocks, vision_candidates, raw_elements,
            )
            result.boundary_decisions.extend(decisions)
            signal_errors += sum(1 for d in decisions if d.get("error"))
        result.signal_collection_errors = signal_errors

        # 5. Build candidate regions from separators
        all_separators = h_gaps + v_gaps + h_transitions + v_transitions
        candidates = self._build_candidate_regions(
            all_separators, width, height,
            ocr_blocks, vision_candidates, raw_elements,
        )

        # 6. Merge over-fragmented consecutive regions
        candidates = self._merge_fragmented_regions(candidates, width, height)

        result.candidate_regions = [c.to_dict() for c in candidates]

        # 6b. U4.1 Phase 2: Flow-aware merge (conservative)
        merged_cands, merge_details, gap_classes = self.classify_and_merge(
            candidates, result.boundary_decisions, width, height,
        )
        result.merged_regions = [c.to_dict() for c in merged_cands]
        result.merge_details = merge_details
        result.gap_classifications = gap_classes

        # 6c. U4.2 Phase 1: Region container signal collection
        result.region_signals = self.collect_region_signals(
            candidates, ocr_blocks, vision_candidates, raw_elements,
            arr, width, height,
        )

        # 6d. U4.2 Phase A: Substructure shadow
        result.page_region_model_shadow = self.build_page_region_shadow(
            candidates, ocr_blocks, vision_candidates, raw_elements,
        )

        # 6e. U4.2 Phase B.1: Pairwise flow signal collection
        result.pairwise_flow_signals = self.collect_pairwise_signals(
            candidates, result.page_region_model_shadow, width, height,
        )

        # 7. Score proposals (all neutral in U4.0)
        result.proposals_count = len(candidates)
        result.neutral_proposals = len(candidates)
        result.useful_proposals = 0
        result.noisy_proposals = 0

        return result

    def _detect_whitespace_gaps(
        self,
        arr: np.ndarray,
        direction: str,
        width: int,
        height: int,
        ocr_blocks: list[dict[str, Any]] | None = None,
        vision_candidates: list[dict[str, Any]] | None = None,
    ) -> list[Separator]:
        """Detect whitespace gaps from pixel projection.

        Two-pass approach:
        1. Collect all geometrically valid gaps (size + span).
        2. Filter out content-stream gaps: consecutive gaps that separate
           similar content rows (text lines, chat messages, list items).
        """
        # ── Pass 1: collect all geometrically valid gaps ──
        raw_gaps: list[tuple[int, int, int, float]] = []  # (start, end, span, strength)

        if direction == "horizontal":
            row_means = arr.mean(axis=1)
            is_whitespace = row_means > 240
            in_gap = False
            gap_start = 0
            for y in range(height):
                if is_whitespace[y]:
                    if not in_gap:
                        gap_start = y
                        in_gap = True
                else:
                    if in_gap:
                        gap_height = y - gap_start
                        if gap_height >= self.MIN_WHITESPACE_GAP:
                            gap_region = arr[gap_start:y, :]
                            col_whitespace = (gap_region > 240).all(axis=0)
                            span = int(col_whitespace.sum())
                            span_ratio = span / width
                            if span_ratio >= self.MIN_WHITESPACE_SPAN_RATIO:
                                raw_gaps.append((
                                    gap_start, y, span,
                                    min(1.0, span_ratio * 1.5),
                                ))
                        in_gap = False
            if in_gap:
                gap_height = height - gap_start
                if gap_height >= self.MIN_WHITESPACE_GAP:
                    gap_region = arr[gap_start:height, :]
                    col_whitespace = (gap_region > 240).all(axis=0)
                    span = int(col_whitespace.sum())
                    span_ratio = span / width
                    if span_ratio >= self.MIN_WHITESPACE_SPAN_RATIO:
                        raw_gaps.append((
                            gap_start, height, span,
                            min(1.0, span_ratio * 1.5),
                        ))

        else:  # vertical
            col_means = arr.mean(axis=0)
            is_whitespace = col_means > 240
            in_gap = False
            gap_start = 0
            for x in range(width):
                if is_whitespace[x]:
                    if not in_gap:
                        gap_start = x
                        in_gap = True
                else:
                    if in_gap:
                        gap_width = x - gap_start
                        if gap_width >= self.MIN_WHITESPACE_GAP:
                            gap_region = arr[:, gap_start:x]
                            row_whitespace = (gap_region > 240).all(axis=1)
                            span = int(row_whitespace.sum())
                            span_ratio = span / height
                            if span_ratio >= self.MIN_WHITESPACE_SPAN_RATIO:
                                raw_gaps.append((
                                    gap_start, x, span,
                                    min(1.0, span_ratio * 1.5),
                                ))
                        in_gap = False
            if in_gap:
                gap_width = width - gap_start
                if gap_width >= self.MIN_WHITESPACE_GAP:
                    gap_region = arr[:, gap_start:width]
                    row_whitespace = (gap_region > 240).all(axis=1)
                    span = int(row_whitespace.sum())
                    span_ratio = span / height
                    if span_ratio >= self.MIN_WHITESPACE_SPAN_RATIO:
                        raw_gaps.append((
                            gap_start, width, span,
                            min(1.0, span_ratio * 1.5),
                        ))

        if not raw_gaps:
            return []

        # ── Pass 2: filter content-stream gaps ──
        accepted = self._filter_content_stream_gaps(
            raw_gaps, direction, width, height, arr,
            ocr_blocks, vision_candidates,
        )

        # Build Separator objects
        gaps: list[Separator] = []
        for idx, (start, end, span, strength) in enumerate(accepted):
            gaps.append(Separator(
                separator_id=f"{'H' if direction == 'horizontal' else 'V'}S{idx}",
                direction=direction,
                position=start,
                width=end - start,
                span=span,
                strength=strength,
                kind="whitespace",
            ))
        return gaps

    def _filter_content_stream_gaps(
        self,
        raw_gaps: list[tuple[int, int, int, float]],
        direction: str,
        width: int,
        height: int,
        arr: np.ndarray,
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
    ) -> list[tuple[int, int, int, float]]:
        """Filter out gaps that are part of a content stream.

        A content stream is a sequence of 3+ consecutive gaps that separate
        rows/columns with similar evidence (text lines, chat messages, list items).
        Gaps within a content stream are rejected; only structural boundaries survive.
        """
        if len(raw_gaps) < 3:
            # Few gaps → not a content stream, apply per-gap check
            return [
                g for g in raw_gaps
                if self._gap_has_evidence_diff(
                    g[0], g[1], direction, width, height,
                    arr, ocr_blocks, vision_candidates,
                )
            ]

        # Compute evidence signature for each gap
        signatures: list[dict[str, float]] = []
        for start, end, _, _ in raw_gaps:
            sig = self._compute_gap_signature(
                start, end, direction, width, height, arr,
                ocr_blocks, vision_candidates,
            )
            signatures.append(sig)

        # Find content stream runs: consecutive gaps with similar signatures
        in_stream = [False] * len(raw_gaps)
        run_start = 0
        for i in range(1, len(raw_gaps) + 1):
            # Check if gap i is similar to gap i-1
            if i < len(raw_gaps) and self._signatures_similar(
                signatures[i - 1], signatures[i]
            ):
                continue
            # End of a run of similar gaps
            run_length = i - run_start
            if run_length >= 3:
                # This is a content stream — mark all gaps in the run
                for j in range(run_start, i):
                    in_stream[j] = True
            run_start = i

        # Accept gaps that are NOT in a content stream
        # Also accept stream boundaries (first and last gap of a stream)
        # as they may separate the stream from different content
        accepted: list[tuple[int, int, int, float]] = []
        for i, (gap, is_stream) in enumerate(zip(raw_gaps, in_stream)):
            if not is_stream:
                accepted.append(gap)
            else:
                # Check if this is a stream boundary (first or last in run)
                is_first = (i == 0 or not in_stream[i - 1])
                is_last = (i == len(raw_gaps) - 1 or not in_stream[i + 1])
                if is_first or is_last:
                    # Stream boundary: check if it separates different content
                    if self._gap_has_evidence_diff(
                        gap[0], gap[1], direction, width, height,
                        arr, ocr_blocks, vision_candidates,
                    ):
                        accepted.append(gap)
                # Interior stream gaps are rejected

        return accepted

    def _filter_redundant_transitions(
        self,
        transitions: list[Separator],
        gaps: list[Separator],
        direction: str,
    ) -> list[Separator]:
        """Remove density transitions that are near whitespace gaps.

        Whitespace gaps are higher-confidence structural boundaries.
        Density transitions that fall within or adjacent to a gap are
        redundant and cause over-fragmentation of content areas.

        Also remove transitions that are near other transitions in the
        same direction — they typically mark edges of content rows, not
        structural boundaries.
        """
        if not gaps and not transitions:
            return transitions

        # Build set of positions covered by gaps (with generous margin)
        gap_positions: set[int] = set()
        for gap in gaps:
            # Margin = 3× gap width or 20px, whichever is larger
            margin = max(20, gap.width * 3)
            for p in range(gap.position - margin, gap.position + gap.width + margin):
                gap_positions.add(p)

        # Also build set of positions near other transitions
        # (consecutive transitions at similar spacing = content rows)
        transition_positions: set[int] = set()
        sorted_transitions = sorted(transitions, key=lambda t: t.position)
        for i in range(len(sorted_transitions) - 1):
            curr = sorted_transitions[i]
            nxt = sorted_transitions[i + 1]
            spacing = nxt.position - curr.position
            # If spacing is small (< 60px), these are content row edges
            if spacing < 60:
                for p in range(curr.position - 5, curr.position + 6):
                    transition_positions.add(p)
                for p in range(nxt.position - 5, nxt.position + 6):
                    transition_positions.add(p)

        filtered: list[Separator] = []
        for t in transitions:
            if t.position in gap_positions:
                continue  # Near a whitespace gap
            if t.position in transition_positions:
                continue  # Near another transition (content row pattern)
            filtered.append(t)
        return filtered

    # ── U4.1 Phase 1: Signal collection ─────────────────────────

    def collect_boundary_signals(
        self,
        separators: list[Separator],
        direction: str,
        width: int,
        height: int,
        arr: np.ndarray,
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
        raw_elements: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Collect signal scores for each separator. Phase 1: no decisions."""
        decisions: list[dict[str, Any]] = []
        for sep in separators:
            try:
                sig = self._collect_single_gap_signals(
                    sep, direction, width, height, arr,
                    ocr_blocks, vision_candidates, raw_elements,
                    separators,
                )
                decisions.append(sig)
            except Exception:
                decisions.append({
                    "gap_id": sep.separator_id,
                    "direction": direction,
                    "position": sep.position,
                    "width": sep.width,
                    "signals": {},
                    "error": True,
                })
        return decisions

    def _collect_single_gap_signals(
        self,
        gap: Separator,
        direction: str,
        width: int,
        height: int,
        arr: np.ndarray,
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
        raw_elements: list[dict[str, Any]] | None,
        all_separators: list[Separator],
    ) -> dict[str, Any]:
        """Collect 5 signal scores for a single gap."""
        band = max(30, gap.width * 2)

        if direction == "horizontal":
            above_top = max(0, gap.position - band)
            below_bottom = min(height, gap.position + gap.width + band)

            # Signal 1: OCR similarity
            ocr_above = self._count_ocr_in_band(
                ocr_blocks, above_top, gap.position, width,
            )
            ocr_below = self._count_ocr_in_band(
                ocr_blocks, gap.position + gap.width, below_bottom, width,
            )
            ocr_sim = self._similarity_score(ocr_above, ocr_below, tolerance=0.5)

            # Signal 2: Vision similarity
            vis_above = self._count_items_in_band(
                vision_candidates, above_top, gap.position, width, "bounding_rect",
            )
            vis_below = self._count_items_in_band(
                vision_candidates, gap.position + gap.width, below_bottom, width, "bounding_rect",
            )
            vis_sim = self._similarity_score(vis_above, vis_below, tolerance=0.5)

            # Signal 3: UIA same parent (check if elements share same parent)
            uia_above = self._count_items_in_band(
                raw_elements, above_top, gap.position, width, "bounding_rect",
            )
            uia_below = self._count_items_in_band(
                raw_elements, gap.position + gap.width, below_bottom, width, "bounding_rect",
            )
            uia_same = self._uia_same_parent_score(
                raw_elements, above_top, below_bottom, gap.position, gap.width, width,
            )

            # Signal 4: Pixel pattern similarity
            above_region = arr[above_top:gap.position, :]
            below_region = arr[gap.position + gap.width:below_bottom, :]
            pixel_sim = self._pixel_pattern_similarity(above_region, below_region)

            # Signal 5: Gap rhythm regularity
            rhythm = self._gap_rhythm_score(gap, all_separators, direction)

        else:  # vertical
            left_start = max(0, gap.position - band)
            right_end = min(width, gap.position + gap.width + band)

            ocr_left = self._count_ocr_in_band_rect(
                ocr_blocks, left_start, gap.position, 0, height,
            )
            ocr_right = self._count_ocr_in_band_rect(
                ocr_blocks, gap.position + gap.width, right_end, 0, height,
            )
            ocr_sim = self._similarity_score(ocr_left, ocr_right, tolerance=0.5)

            vis_left = self._count_items_in_band_rect(
                vision_candidates, left_start, gap.position, 0, height,
            )
            vis_right = self._count_items_in_band_rect(
                vision_candidates, gap.position + gap.width, right_end, 0, height,
            )
            vis_sim = self._similarity_score(vis_left, vis_right, tolerance=0.5)

            uia_left = self._count_items_in_band_rect(
                raw_elements, left_start, gap.position, 0, height,
            )
            uia_right = self._count_items_in_band_rect(
                raw_elements, gap.position + gap.width, right_end, 0, height,
            )
            uia_same = 0.0  # Vertical UIA same-parent not implemented

            left_region = arr[:, left_start:gap.position]
            right_region = arr[:, gap.position + gap.width:right_end]
            pixel_sim = self._pixel_pattern_similarity(left_region, right_region)

            rhythm = self._gap_rhythm_score(gap, all_separators, direction)

            ocr_above, ocr_below = ocr_left, ocr_right
            vis_above, vis_below = vis_left, vis_right
            uia_above, uia_below = uia_left, uia_right

        return {
            "gap_id": gap.separator_id,
            "direction": direction,
            "position": gap.position,
            "width": gap.width,
            "signals": {
                "ocr_similarity": round(ocr_sim, 3),
                "vision_similarity": round(vis_sim, 3),
                "uia_same_parent": round(uia_same, 3),
                "pixel_pattern_similarity": round(float(pixel_sim), 3),
                "gap_rhythm_regularity": round(float(rhythm), 3),
            },
            "evidence_counts": {
                "ocr_above": ocr_above,
                "ocr_below": ocr_below,
                "vision_above": vis_above,
                "vision_below": vis_below,
                "uia_above": uia_above,
                "uia_below": uia_below,
            },
            "has_ocr_evidence": ocr_above > 0 or ocr_below > 0,
            "has_vision_evidence": vis_above > 0 or vis_below > 0,
            "has_uia_evidence": uia_above > 0 or uia_below > 0,
        }

    def _similarity_score(
        self, count_a: int, count_b: int, tolerance: float = 0.5,
    ) -> float:
        """Score 0-1: how similar two counts are. 1.0 = identical."""
        if count_a == 0 and count_b == 0:
            # Both have no evidence → treat as similar (same empty state)
            # rather than uncertain. This prevents whitespace/empty regions
            # from being falsely classified as different.
            return 1.0
        max_count = max(count_a, count_b)
        diff = abs(count_a - count_b)
        if diff <= max(1, max_count * tolerance):
            return 1.0 - (diff / max(max_count, 1))
        return 0.0

    def _uia_same_parent_score(
        self,
        raw_elements: list[dict[str, Any]] | None,
        above_top: int, below_bottom: int,
        gap_position: int, gap_width: int, window_width: int,
    ) -> float:
        """Check if UIA elements above and below gap share same parent."""
        if not raw_elements:
            return 0.5  # No evidence → uncertain

        above_parents: set[str] = set()
        below_parents: set[str] = set()

        for elem in raw_elements:
            bbox = elem.get("bounding_rect") or elem.get("bbox", [])
            if not bbox or len(bbox) < 4:
                continue
            cy = (bbox[1] + bbox[3]) // 2
            parent_id = elem.get("parent_id") or elem.get("automation_id", "")
            if not parent_id:
                continue

            if above_top <= cy < gap_position:
                above_parents.add(str(parent_id))
            elif gap_position + gap_width <= cy < below_bottom:
                below_parents.add(str(parent_id))

        if not above_parents and not below_parents:
            return 0.5  # No UIA evidence

        # Check overlap
        overlap = above_parents & below_parents
        if overlap:
            return 1.0  # Same parent → likely same flow

        # Check if parents are list-type
        return 0.0  # Different parents → likely structural

    def _pixel_pattern_similarity(
        self, region_a: np.ndarray, region_b: np.ndarray,
    ) -> float:
        """Compare pixel patterns of two regions. 1.0 = identical."""
        if region_a.size == 0 or region_b.size == 0:
            return 0.5

        mean_a = float(region_a.mean())
        mean_b = float(region_b.mean())
        std_a = float(region_a.std())
        std_b = float(region_b.std())

        # Compare mean brightness
        mean_diff = abs(mean_a - mean_b) / 255.0
        # Compare std (texture)
        std_diff = abs(std_a - std_b) / max(std_a, std_b, 1.0)

        # Combined similarity (lower diff = higher similarity)
        sim = 1.0 - (mean_diff * 0.5 + std_diff * 0.5)
        return max(0.0, min(1.0, sim))

    def _gap_rhythm_score(
        self,
        gap: Separator,
        all_separators: list[Separator],
        direction: str,
    ) -> float:
        """Score gap spacing regularity. High = content flow, low = structural."""
        # Find gaps of same direction near this gap
        same_dir = [s for s in all_separators if s.direction == direction]
        if len(same_dir) < 3:
            return 0.5  # Not enough gaps to determine rhythm

        positions = sorted(s.position for s in same_dir)
        spacings = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]

        if len(spacings) < 2:
            return 0.5

        import statistics
        mean_spacing = statistics.mean(spacings)
        if mean_spacing <= 0:
            return 0.5

        try:
            std_spacing = statistics.stdev(spacings)
        except statistics.StatisticsError:
            return 0.5

        # Coefficient of variation: lower = more regular
        cv = std_spacing / mean_spacing
        # Convert to 0-1 score (cv=0 → 1.0, cv≥1 → 0.0)
        score = max(0.0, 1.0 - cv)
        return score

    # ── U4.1 Phase 2: Flow-aware merge ──────────────────────────

    # Conservative thresholds: only merge strong same-flow signals.
    # Phase 1 data shows content flow gaps have OCR sim ≥ 0.56 and
    # Pixel sim ≥ 0.76. We use higher thresholds for conservatism.
    SAME_FLOW_OCR_THRESHOLD = 0.70
    SAME_FLOW_PIXEL_THRESHOLD = 0.80
    SAME_FLOW_MIN_RUN = 3  # 3+ consecutive same-flow gaps → merge

    def classify_and_merge(
        self,
        candidates: list[CandidateRegion],
        boundary_decisions: list[dict[str, Any]],
        width: int,
        height: int,
    ) -> tuple[list[CandidateRegion], list[dict[str, Any]], dict[str, int]]:
        """Phase 2: Classify gaps and merge same-flow regions.

        Returns:
            (merged_candidates, merge_details, gap_classifications)
        """
        if len(candidates) < 2:
            return candidates, [], {"total": 0}

        # Step 1: Classify each gap decision
        classifications = self._classify_gaps(boundary_decisions)

        # Step 2: Build gap→classification lookup
        gap_class_map: dict[str, str] = {}
        for dec, cls in zip(boundary_decisions, classifications):
            gap_class_map[dec.get("gap_id", "")] = cls

        # Step 3: Find which candidate regions are separated by same-flow gaps
        # Sort candidates by position
        sorted_cands = sorted(candidates, key=lambda c: (c.bounds[1], c.bounds[0]))

        # Step 4: Identify merge runs (consecutive regions separated by same-flow gaps)
        merge_runs = self._find_merge_runs(sorted_cands, gap_class_map)

        # Step 5: Merge runs into container regions
        merged, details = self._execute_merges(sorted_cands, merge_runs, width, height)

        # Step 6: Count classifications
        class_counts: dict[str, int] = {"same_flow": 0, "structural": 0, "uncertain": 0}
        for cls in classifications:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        class_counts["total"] = len(classifications)

        return merged, details, class_counts

    def _classify_gaps(
        self, boundary_decisions: list[dict[str, Any]],
    ) -> list[str]:
        """Classify each gap as same_flow / structural / uncertain.

        Conservative: only classify as same_flow when BOTH OCR and Pixel
        signals are high. Everything else is uncertain (neutral evidence).
        """
        classifications: list[str] = []
        for dec in boundary_decisions:
            sig = dec.get("signals", {})
            ocr = sig.get("ocr_similarity", 0.5)
            pixel = sig.get("pixel_pattern_similarity", 0.5)

            if (ocr >= self.SAME_FLOW_OCR_THRESHOLD
                    and pixel >= self.SAME_FLOW_PIXEL_THRESHOLD):
                classifications.append("same_flow")
            else:
                # Conservative: don't classify as structural either.
                # Keep as uncertain → neutral evidence.
                classifications.append("uncertain")

        return classifications

    def _find_merge_runs(
        self,
        sorted_cands: list[CandidateRegion],
        gap_class_map: dict[str, str],
    ) -> list[tuple[int, int, str]]:
        """Find runs of consecutive regions separated by same-flow gaps.

        Groups regions by column first, then finds vertical merge runs
        within each column.

        Returns list of (start_idx, end_idx, flow_type) tuples.
        """
        if len(sorted_cands) < 2:
            return []

        # Step 1: Group regions by horizontal position (column)
        # Regions with ≥50% horizontal overlap go in the same column.
        columns: list[list[int]] = []  # list of lists of indices into sorted_cands
        assigned = [False] * len(sorted_cands)

        for i in range(len(sorted_cands)):
            if assigned[i]:
                continue
            col = [i]
            assigned[i] = True
            ci = sorted_cands[i]

            for j in range(i + 1, len(sorted_cands)):
                if assigned[j]:
                    continue
                cj = sorted_cands[j]
                # Check horizontal overlap
                overlap_l = max(ci.bounds[0], cj.bounds[0])
                overlap_r = min(ci.bounds[2], cj.bounds[2])
                overlap = max(0, overlap_r - overlap_l)
                wi = ci.bounds[2] - ci.bounds[0]
                wj = cj.bounds[2] - cj.bounds[0]
                min_w = min(wi, wj)
                if min_w > 0 and overlap / min_w >= 0.5:
                    col.append(j)
                    assigned[j] = True
            columns.append(col)

        # Step 2: Within each column, find vertical merge runs
        runs: list[tuple[int, int, str]] = []

        for col_indices in columns:
            if len(col_indices) < self.SAME_FLOW_MIN_RUN:
                continue

            # Sort by vertical position
            col_sorted = sorted(col_indices, key=lambda idx: sorted_cands[idx].bounds[1])

            # Check each pair of vertically adjacent regions
            same_flow_flags: list[bool] = []
            for k in range(len(col_sorted) - 1):
                idx_a = col_sorted[k]
                idx_b = col_sorted[k + 1]
                ca = sorted_cands[idx_a]
                cb = sorted_cands[idx_b]

                # Vertical gap
                gap_px = cb.bounds[1] - ca.bounds[3]

                # OCR similarity
                ocr_sim = self._similarity_score(
                    ca.ocr_count, cb.ocr_count, 0.5,
                )
                # Height similarity
                ha = ca.bounds[3] - ca.bounds[1]
                hb = cb.bounds[3] - cb.bounds[1]
                h_sim = 1.0 - abs(ha - hb) / max(ha, hb, 1)

                # Conservative: same-flow if OCR similar AND height similar
                # AND gap is reasonable (≤ 30px or slightly overlapping)
                if ocr_sim >= 0.7 and h_sim >= 0.5 and gap_px <= 30:
                    same_flow_flags.append(True)
                else:
                    same_flow_flags.append(False)

            # Find runs of True
            run_start = 0
            for k in range(len(same_flow_flags)):
                if same_flow_flags[k]:
                    continue  # Still in a run
                # k is False → end of any run that started at run_start
                if same_flow_flags[run_start]:
                    run_length = k - run_start + 1  # includes regions at boundaries
                    if run_length >= self.SAME_FLOW_MIN_RUN:
                        # Use actual indices, not a range
                        run_indices = col_sorted[run_start:k + 1]
                        flow_type = self._determine_flow_type(
                            [sorted_cands[i] for i in run_indices],
                        )
                        runs.append((run_indices, flow_type))
                run_start = k + 1
            # Handle run that extends to the end
            if run_start < len(same_flow_flags) and same_flow_flags[run_start]:
                run_length = len(same_flow_flags) - run_start + 1
                if run_length >= self.SAME_FLOW_MIN_RUN:
                    run_indices = col_sorted[run_start:]
                    flow_type = self._determine_flow_type(
                        [sorted_cands[i] for i in run_indices],
                    )
                    runs.append((run_indices, flow_type))

        return runs

        # Find runs of True in is_same_flow
        runs: list[tuple[int, int, str]] = []
        run_start = 0
        for i in range(1, len(is_same_flow) + 1):
            if i < len(is_same_flow) and is_same_flow[i]:
                continue
            # End of run
            if is_same_flow[run_start]:
                run_length = i - run_start + 1
                if run_length >= self.SAME_FLOW_MIN_RUN:
                    # Determine flow type from region evidence
                    flow_type = self._determine_flow_type(
                        sorted_cands[run_start:i + 1],
                    )
                    runs.append((run_start, i, flow_type))
            run_start = i

        return runs

    def _determine_flow_type(
        self, regions: list[CandidateRegion],
    ) -> str:
        """Determine the flow type from a set of regions."""
        total_ocr = sum(r.ocr_count for r in regions)
        total_vis = sum(r.vision_count for r in regions)
        avg_height = sum(r.bounds[3] - r.bounds[1] for r in regions) / len(regions)

        if total_ocr > total_vis * 2:
            return "dense_text_area"
        elif total_vis > total_ocr * 2:
            if avg_height < 50:
                return "list_region"
            return "content_stream"
        else:
            if avg_height < 40:
                return "list_region"
            return "dense_text_area"

    def _execute_merges(
        self,
        sorted_cands: list[CandidateRegion],
        runs: list[tuple[list[int], str]],
        width: int,
        height: int,
    ) -> tuple[list[CandidateRegion], list[dict[str, Any]]]:
        """Execute merges for identified runs.

        Args:
            runs: List of (index_list, flow_type) tuples where index_list
                  contains the sorted_cands indices to merge.
        """
        if not runs:
            return sorted_cands, []

        # Mark which indices are part of a merge run
        merged_indices: set[int] = set()
        for run_indices, _ in runs:
            for i in run_indices:
                merged_indices.add(i)

        # Build index→run lookup
        index_to_run: dict[int, tuple[list[int], str]] = {}
        for run_indices, flow_type in runs:
            for i in run_indices:
                index_to_run[i] = (run_indices, flow_type)

        result: list[CandidateRegion] = []
        details: list[dict[str, Any]] = []
        processed_runs: set[int] = set()  # track which runs we've already processed

        for i in range(len(sorted_cands)):
            if i not in merged_indices:
                result.append(sorted_cands[i])
                continue

            run_indices, flow_type = index_to_run[i]
            run_id = id(run_indices)  # unique per run object
            if run_id in processed_runs:
                continue  # already merged this run

            processed_runs.add(run_id)
            run_regions = [sorted_cands[j] for j in run_indices]

            # Merge
            merged_bounds = (
                min(r.bounds[0] for r in run_regions),
                min(r.bounds[1] for r in run_regions),
                max(r.bounds[2] for r in run_regions),
                max(r.bounds[3] for r in run_regions),
            )
            merged_region = CandidateRegion(
                region_id=f"M{len(details)}",
                bounds=merged_bounds,
                bounded_by=list(set(
                    b for r in run_regions for b in r.bounded_by
                )),
                ocr_count=sum(r.ocr_count for r in run_regions),
                vision_count=sum(r.vision_count for r in run_regions),
                uia_count=sum(r.uia_count for r in run_regions),
                area_ratio=(
                    (merged_bounds[2] - merged_bounds[0])
                    * (merged_bounds[3] - merged_bounds[1])
                ) / (width * height),
            )
            result.append(merged_region)

            details.append({
                "merged_region_id": merged_region.region_id,
                "flow_type": flow_type,
                "original_regions": [r.region_id for r in run_regions],
                "original_count": len(run_regions),
                "merged_bounds": list(merged_bounds),
                "ocr_count": merged_region.ocr_count,
                "vision_count": merged_region.vision_count,
            })

        return result, details

    # ── U4.2 Phase 1: Region signal collection ──────────────────

    def collect_region_signals(
        self,
        candidates: list[CandidateRegion],
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
        raw_elements: list[dict[str, Any]] | None,
        arr: np.ndarray | None,
        window_width: int,
        window_height: int,
    ) -> list[dict[str, Any]]:
        """Collect container signals for each candidate region.

        Phase 1: signal collection only, no classification.
        """
        signals: list[dict[str, Any]] = []
        for cand in candidates:
            sig = self._collect_single_region_signals(
                cand, ocr_blocks, vision_candidates, raw_elements,
                arr, window_width, window_height,
            )
            signals.append(sig)
        return signals

    def _collect_single_region_signals(
        self,
        cand: CandidateRegion,
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
        raw_elements: list[dict[str, Any]] | None,
        arr: np.ndarray | None,
        window_width: int,
        window_height: int,
    ) -> dict[str, Any]:
        """Collect 8 signals for a single candidate region."""
        l, t, r, b = cand.bounds
        region_w = max(1, r - l)
        region_h = max(1, b - t)
        area = region_w * region_h

        # Items within this region
        ocr_items = self._items_in_region(ocr_blocks, l, t, r, b, "bbox")
        vis_items = self._items_in_region(
            vision_candidates, l, t, r, b, "bounding_rect",
        )
        uia_items = self._items_in_region(
            raw_elements, l, t, r, b, "bounding_rect",
        )

        # Signal 1: text_density (OCR chars per pixel area)
        total_chars = sum(
            len(item.get("text", "")) for item in ocr_items
        )
        text_density = total_chars / area if area > 0 else 0

        # Signal 2: control_density (vision candidates per pixel area)
        control_density = len(vis_items) / area if area > 0 else 0

        # Signal 3: height_ratio
        height_ratio = region_h / window_height if window_height > 0 else 0

        # Signal 4: width_ratio
        width_ratio = region_w / window_width if window_width > 0 else 0

        # Signal 5: ocr_row_count
        ocr_row_count = len(ocr_items)

        # Signal 6: row_height_consistency (CV of OCR block y-gaps)
        row_height_cv = self._compute_row_height_cv(ocr_items)

        # Signal 7: x_alignment_consistency
        x_alignment = self._compute_x_alignment(ocr_items, vis_items)

        # Signal 8: pixel_uniformity (std of brightness within region)
        pixel_uniformity = 0.0
        if arr is not None:
            region_arr = arr[max(0, t):min(arr.shape[0], b),
                            max(0, l):min(arr.shape[1], r)]
            if region_arr.size > 0:
                pixel_uniformity = float(region_arr.std())

        return {
            "region_id": cand.region_id,
            "bounds": list(cand.bounds),
            "area_ratio": round(cand.area_ratio, 4),
            "ocr_count": cand.ocr_count,
            "vision_count": cand.vision_count,
            "uia_count": cand.uia_count,
            "signals": {
                "text_density": round(text_density, 6),
                "control_density": round(control_density, 6),
                "height_ratio": round(height_ratio, 4),
                "width_ratio": round(width_ratio, 4),
                "ocr_row_count": ocr_row_count,
                "row_height_cv": round(row_height_cv, 3),
                "x_alignment": round(x_alignment, 3),
                "pixel_uniformity": round(pixel_uniformity, 1),
            },
        }

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

    def _compute_row_height_cv(
        self, ocr_items: list[dict[str, Any]],
    ) -> float:
        """Compute CV of gaps between consecutive OCR blocks (by y-center).

        Low CV = regular row layout (list/table).
        High CV = irregular layout (document with varying paragraph spacing).
        Returns 0.5 if not enough data.
        """
        if len(ocr_items) < 3:
            return 0.5  # Not enough data

        # Sort by y-center
        y_centers = []
        for item in ocr_items:
            bbox = item.get("bbox", [])
            if len(bbox) >= 4:
                y_centers.append((bbox[1] + bbox[3]) // 2)
        y_centers.sort()

        if len(y_centers) < 3:
            return 0.5

        # Compute gaps between consecutive y-centers
        gaps = [y_centers[i + 1] - y_centers[i]
                for i in range(len(y_centers) - 1)]
        gaps = [g for g in gaps if g > 0]

        if len(gaps) < 2:
            return 0.5

        import statistics
        mean_gap = statistics.mean(gaps)
        if mean_gap <= 0:
            return 0.5

        try:
            std_gap = statistics.stdev(gaps)
        except statistics.StatisticsError:
            return 0.5

        return std_gap / mean_gap

    def _compute_x_alignment(
        self,
        ocr_items: list[dict[str, Any]],
        vis_items: list[dict[str, Any]],
    ) -> float:
        """Compute x-coordinate alignment consistency.

        Measures how many unique x-left values exist relative to item count.
        Low unique_ratio = items align in columns (list/table).
        High unique_ratio = items spread randomly (document).
        Returns 0.5 if not enough data.
        """
        all_lefts: list[int] = []
        for item in ocr_items:
            bbox = item.get("bbox", [])
            if len(bbox) >= 4:
                all_lefts.append(bbox[0])
        for item in vis_items:
            bbox = item.get("bounding_rect") or item.get("bbox", [])
            if len(bbox) >= 4:
                all_lefts.append(bbox[0])

        if len(all_lefts) < 3:
            return 0.5

        # Cluster x-lefts within 10px tolerance
        sorted_lefts = sorted(all_lefts)
        clusters: list[list[int]] = []
        current_cluster = [sorted_lefts[0]]
        for x in sorted_lefts[1:]:
            if x - current_cluster[-1] <= 10:
                current_cluster.append(x)
            else:
                clusters.append(current_cluster)
                current_cluster = [x]
        clusters.append(current_cluster)

        # unique_ratio = clusters / items (lower = more aligned)
        unique_ratio = len(clusters) / len(all_lefts)
        return unique_ratio

    # ── U4.2 Phase A: Substructure classification ───────────────

    def build_page_region_shadow(
        self,
        candidates: list[CandidateRegion],
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
        raw_elements: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Build substructure shadow for each candidate region.

        Phase A: classify internal items as substructures, no page-level
        region type judgment.
        """
        shadows: list[PageRegionShadow] = []
        for cand in candidates:
            shadow = self._classify_region_substructures(
                cand, ocr_blocks, vision_candidates, raw_elements,
            )
            shadows.append(shadow)
        return [s.to_dict() for s in shadows]

    def _classify_region_substructures(
        self,
        cand: CandidateRegion,
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
        raw_elements: list[dict[str, Any]] | None,
    ) -> PageRegionShadow:
        """Classify items within a region as substructures."""
        l, t, r, b = cand.bounds
        shadow = PageRegionShadow(
            region_id=cand.region_id,
            bounds=cand.bounds,
        )

        # Get items within this region
        ocr_items = self._items_in_region(ocr_blocks, l, t, r, b, "bbox")
        vis_items = self._items_in_region(
            vision_candidates, l, t, r, b, "bounding_rect",
        )
        uia_items = self._items_in_region(
            raw_elements, l, t, r, b, "bounding_rect",
        )

        sub_idx = 0

        # Classify OCR items
        for item in ocr_items:
            bbox = item.get("bbox", [])
            if len(bbox) < 4:
                continue
            text = item.get("text", "").strip()
            sub_type, confidence, reason = self._classify_ocr_item(
                text, bbox, l, t, r, b,
            )
            shadow.substructures.append(RegionSubstructure(
                sub_id=f"S{sub_idx}",
                sub_type=sub_type,
                bounds=tuple(bbox),
                evidence_sources=["ocr"],
                confidence=confidence,
                reason=reason,
                text=text,
            ))
            sub_idx += 1

        # Classify vision items
        for item in vis_items:
            bbox = item.get("bounding_rect") or item.get("bbox", [])
            if len(bbox) < 4:
                continue
            sub_type, confidence, reason = self._classify_vision_item(
                item, bbox, l, t, r, b,
            )
            shadow.substructures.append(RegionSubstructure(
                sub_id=f"S{sub_idx}",
                sub_type=sub_type,
                bounds=tuple(bbox),
                evidence_sources=["vision"],
                confidence=confidence,
                reason=reason,
                control_type=item.get("label", ""),
            ))
            sub_idx += 1

        # Classify UIA items
        for item in uia_items:
            bbox = item.get("bounding_rect") or item.get("bbox", [])
            if len(bbox) < 4:
                continue
            control_type = item.get("control_type", "")
            sub_type, confidence, reason = self._classify_uia_item(
                item, control_type, bbox,
            )
            shadow.substructures.append(RegionSubstructure(
                sub_id=f"S{sub_idx}",
                sub_type=sub_type,
                bounds=tuple(bbox),
                evidence_sources=["uia"],
                confidence=confidence,
                reason=reason,
                control_type=control_type,
            ))
            sub_idx += 1

        # If no items found, mark as whitespace or unknown
        if not shadow.substructures:
            region_w = r - l
            region_h = b - t
            area = region_w * region_h
            if area > 0:
                shadow.substructures.append(RegionSubstructure(
                    sub_id=f"S{sub_idx}",
                    sub_type=SubstructureType.WHITESPACE_BLOCK,
                    bounds=cand.bounds,
                    evidence_sources=[],
                    confidence=0.5,
                    reason="No OCR/vision/UIA items in region",
                ))

        return shadow

    def _classify_ocr_item(
        self,
        text: str,
        bbox: list[int],
        reg_l: int, reg_t: int, reg_r: int, reg_b: int,
    ) -> tuple[str, float, str]:
        """Classify an OCR item as a substructure type.

        Classification priority (highest to lowest):
        1. code_line: code-like patterns (keywords, brackets, paths, indentation)
        2. message_like: chat message indicators (timestamp, username, bubble)
        3. list_item_like: list/table row indicators (column layout, short label)
        4. text_row: default for most text content
        5. unknown_evidence: genuinely can't determine
        """
        if not text or not text.strip():
            return SubstructureType.UNKNOWN_EVIDENCE, 0.3, "Empty OCR text"

        text = text.strip()
        text_len = len(text)
        bbox_h = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
        bbox_w = bbox[2] - bbox[0] if len(bbox) >= 4 else 0

        # ── 1. Code line detection ──
        # Code keywords, brackets, assignment, paths, indentation
        code_keywords = [
            "def ", "class ", "import ", "from ", "function ", "var ",
            "const ", "let ", "if ", "for ", "while ", "return ",
            "async ", "await ", "try:", "except:", "finally:",
            "self.", "this.", "public ", "private ", "static ",
            "void ", "int ", "str ", "bool ", "float ",
        ]
        code_patterns = ["{", "}", "()", "[]", "=>", "->", "::", "://",
                         "==", "!=", ">=", "<=", "&&", "||", "++", "--"]
        code_path_patterns = [".py", ".js", ".ts", ".java", ".cpp", ".h",
                              ".json", ".yaml", ".yml", ".xml", ".html",
                              "/", "\\", "C:\\", "D:\\", "/usr/", "/home/"]

        code_score = 0
        code_reasons = []

        # Check code keywords
        for kw in code_keywords:
            if kw in text.lower() or kw in text:
                code_score += 2
                code_reasons.append(f"keyword '{kw.strip()}'")
                break

        # Check code patterns
        for pat in code_patterns:
            if pat in text:
                code_score += 1
                code_reasons.append(f"pattern '{pat}'")
                break

        # Check path patterns
        for pat in code_path_patterns:
            if pat in text:
                code_score += 2
                code_reasons.append(f"path pattern '{pat}'")
                break

        # Check indentation (starts with spaces/tabs)
        if text.startswith("  ") or text.startswith("\t"):
            code_score += 1
            code_reasons.append("indented")

        # Check short line with assignment (= sign, not in URLs)
        if "=" in text and "://" not in text and text_len < 80:
            code_score += 1
            code_reasons.append("assignment")

        if code_score >= 3:
            return SubstructureType.CODE_LINE, min(0.9, 0.5 + code_score * 0.1), \
                f"Code indicators: {', '.join(code_reasons[:3])}"

        # ── 2. Message-like detection (VERY STRICT) ──
        # Only classify as message_like when there are strong chat indicators
        import re

        message_score = 0
        message_reasons = []

        # Timestamp pattern: HH:MM, YYYY-MM-DD, etc.
        if re.search(r'\d{1,2}:\d{2}', text):
            message_score += 2
            message_reasons.append("timestamp")

        # Username-like pattern: starts with @, or "Name:" at beginning
        if re.search(r'^@?\w{2,20}[：:]\s', text):
            message_score += 2
            message_reasons.append("username prefix")

        # Chat bubble indicators: "[图片]", "[表情]", "[文件]", etc.
        if re.search(r'\[.{1,6}\]', text):
            message_score += 1
            message_reasons.append("media indicator")

        # Date-time separator in chat: "上午", "下午", "昨天", "今天"
        if re.search(r'(上午|下午|昨天|今天|星期)', text):
            message_score += 1
            message_reasons.append("date separator")

        if message_score >= 3:
            return SubstructureType.MESSAGE_LIKE, min(0.8, 0.4 + message_score * 0.1), \
                f"Chat message indicators: {', '.join(message_reasons)}"

        # ── 3. List item / table row detection ──
        # Short text that could be a cell in a list/table
        list_score = 0
        list_reasons = []

        # Very short text (< 15 chars) in narrow bbox → likely a cell
        if text_len < 15 and bbox_w < 200:
            list_score += 2
            list_reasons.append(f"short cell text ({text_len} chars)")

        # Numeric text → likely a cell (time, count, score)
        if re.match(r'^[\d:.\-/]+$', text):
            list_score += 2
            list_reasons.append("numeric cell")

        # Single word or very short phrase → likely a cell
        if text_len < 20 and " " not in text.strip():
            list_score += 1
            list_reasons.append("single word/phrase")

        # Bbox aspect ratio: wide and short → table cell
        if bbox_w > bbox_h * 4 and bbox_h < 40:
            list_score += 1
            list_reasons.append("wide-short bbox (cell shape)")

        if list_score >= 3:
            return SubstructureType.LIST_ITEM_LIKE, min(0.7, 0.4 + list_score * 0.1), \
                f"List/table indicators: {', '.join(list_reasons)}"

        # ── 4. Text row (default for most content) ──
        # Paragraphs, labels, titles, descriptions, song names, etc.
        # This is the DEFAULT classification for non-code, non-message text.
        if text_len > 0:
            return SubstructureType.TEXT_ROW, 0.6, \
                f"General text ({text_len} chars)"

        # ── 5. Unknown ──
        return SubstructureType.UNKNOWN_EVIDENCE, 0.3, \
            f"No classification indicators"

    def _classify_vision_item(
        self,
        item: dict[str, Any],
        bbox: list[int],
        reg_l: int, reg_t: int, reg_r: int, reg_b: int,
    ) -> tuple[str, float, str]:
        """Classify a vision candidate as a substructure type.

        Conservative: most vision items are icons or controls.
        Only classify as message_like when there are strong chat indicators.
        """
        label = (item.get("label", "") or "").lower()
        bbox_h = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
        bbox_w = bbox[2] - bbox[0] if len(bbox) >= 4 else 0
        area = bbox_w * bbox_h

        # Small square item → icon
        if area < 2500 and 0.5 < bbox_w / max(bbox_h, 1) < 2.0:
            return SubstructureType.ICON_CANDIDATE, 0.6, \
                f"Small square item ({bbox_w}x{bbox_h}), likely icon"

        # Label contains control-related words
        control_words = ["button", "btn", "icon", "menu", "tab",
                         "toolbar", "scroll", "slider", "toggle",
                         "checkbox", "radio", "switch", "dropdown"]
        if any(w in label for w in control_words):
            return SubstructureType.CONTROL_CANDIDATE, 0.7, \
                f"Label '{label}' suggests control"

        # Label contains text-related words → text_row (not message)
        text_words = ["text", "label", "title", "heading", "paragraph",
                      "textblock", "document", "content"]
        if any(w in label for w in text_words):
            return SubstructureType.TEXT_ROW, 0.5, \
                f"Label '{label}' suggests text"

        # Large item → default to text_row (NOT message_like)
        # Only classify as message_like when label has chat-specific terms
        chat_words = ["message", "chat", "bubble", "comment", "reply"]
        if any(w in label for w in chat_words):
            return SubstructureType.MESSAGE_LIKE, 0.6, \
                f"Label '{label}' suggests chat message"

        # Large item without label → likely text content
        if area > 10000:
            return SubstructureType.TEXT_ROW, 0.4, \
                f"Large vision item ({bbox_w}x{bbox_h}), default text_row"

        # Medium item without label → could be icon or control
        if area > 1000 and bbox_w < 100 and bbox_h < 100:
            return SubstructureType.ICON_CANDIDATE, 0.4, \
                f"Medium vision item ({bbox_w}x{bbox_h}), likely icon"

        # Small item without label → likely icon
        if area <= 1000:
            return SubstructureType.ICON_CANDIDATE, 0.3, \
                f"Small vision item ({bbox_w}x{bbox_h}), likely icon"

        # Default
        return SubstructureType.UNKNOWN_EVIDENCE, 0.3, \
            f"Vision item with label '{label}', unclear type"

    def _classify_uia_item(
        self,
        item: dict[str, Any],
        control_type: str,
        bbox: list[int],
    ) -> tuple[str, float, str]:
        """Classify a UIA element as a substructure type.

        Comprehensive control_type mapping with structural container detection.
        Large structural containers (Window/Pane/Document/Group) are marked
        as noise to prevent them from polluting flow signals.
        """
        ct = (control_type or "").lower()
        bbox_w = bbox[2] - bbox[0] if len(bbox) >= 4 else 0
        bbox_h = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
        area = bbox_w * bbox_h

        # ── Structural containers (large, spanning regions) ──
        # These are NOT content items — they are UIA tree structure.
        # Mark as noise so they don't pollute flow signals.
        structural_types = ("window", "pane", "document", "group",
                            "custom", "semanticzoom", "appbar")
        if any(ct.startswith(t) for t in structural_types):
            if area > 50000:  # Large container → structural noise
                return SubstructureType.UNKNOWN_EVIDENCE, 0.2, \
                    f"UIA '{control_type}' is large structural container ({bbox_w}x{bbox_h})"
            # Small group/pane → could be a content container
            return SubstructureType.CONTROL_CANDIDATE, 0.4, \
                f"UIA '{control_type}' is small structural element"

        # ── Text controls ──
        if ct in ("text", "textblock", "edit", "edittext",
                  "richtext", "label", "heading"):
            return SubstructureType.TEXT_ROW, 0.7, \
                f"UIA '{control_type}' is text"

        # ── Interactive controls ──
        if ct in ("button", "hyperlink", "menuitem", "tabitem",
                  "checkbox", "radiobutton", "togglebutton",
                  "combobox", "spinner", "slider"):
            return SubstructureType.CONTROL_CANDIDATE, 0.8, \
                f"UIA '{control_type}' is interactive"

        # ── Image/icon controls ──
        if ct in ("image", "icon", "thumb"):
            return SubstructureType.ICON_CANDIDATE, 0.7, \
                f"UIA '{control_type}' is visual"

        # ── List/tree items ──
        if ct in ("listitem", "treeitem", "dataitem"):
            return SubstructureType.LIST_ITEM_LIKE, 0.7, \
                f"UIA '{control_type}' is list item"

        # ── List/tree containers ──
        if ct in ("list", "tree", "datagrid", "listview", "treeview"):
            return SubstructureType.CONTROL_CANDIDATE, 0.5, \
                f"UIA '{control_type}' is list container"

        # ── Tab controls ──
        if ct in ("tab", "tabcontrol", "tabitem"):
            return SubstructureType.CONTROL_CANDIDATE, 0.7, \
                f"UIA '{control_type}' is tab"

        # ── Scroll bar ──
        if ct in ("scrollbar", "thumb"):
            return SubstructureType.CONTROL_CANDIDATE, 0.6, \
                f"UIA '{control_type}' is scroll"

        # ── Menu/toolbar ──
        if ct in ("menu", "menubar", "toolbar", "statusbar"):
            return SubstructureType.CONTROL_CANDIDATE, 0.7, \
                f"UIA '{control_type}' is menu/toolbar"

        # ── Progress/tooltip ──
        if ct in ("progressbar", "tooltip", "calendar", "colorpicker"):
            return SubstructureType.CONTROL_CANDIDATE, 0.5, \
                f"UIA '{control_type}' is auxiliary"

        # ── Table rows ──
        if ct in ("datarow", "header", "cell"):
            return SubstructureType.TABLE_ROW_LIKE, 0.7, \
                f"UIA control_type '{control_type}' is table element"

        # Default
        return SubstructureType.UNKNOWN_EVIDENCE, 0.3, \
            f"UIA control_type '{control_type}', unclear type"

    # ── U4.2 Phase B.1: Pairwise Flow Signal Collection ────────

    def collect_pairwise_signals(
        self,
        candidates: list[CandidateRegion],
        page_region_shadow: list[dict[str, Any]],
        window_width: int,
        window_height: int,
    ) -> list[dict[str, Any]]:
        """Collect pairwise flow signals for all adjacent candidate region pairs.

        Phase B.1: signal collection only, no merge, no flow type inference.
        """
        pairs: list[dict[str, Any]] = []

        # Build substructure lookup
        sub_lookup: dict[str, list[dict[str, Any]]] = {}
        for shadow in page_region_shadow:
            sub_lookup[shadow["region_id"]] = shadow.get("substructures", [])

        # Find all adjacent pairs
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                ci = candidates[i]
                cj = candidates[j]

                # Check adjacency (vertical overlap or horizontal overlap)
                is_vert_adjacent, vert_gap = self._is_vertically_adjacent(ci, cj)
                is_horiz_adjacent, horiz_gap = self._is_horizontally_adjacent(ci, cj)

                if not is_vert_adjacent and not is_horiz_adjacent:
                    continue

                # Collect signals
                signals = self._compute_pairwise_signals(
                    ci, cj, sub_lookup, window_width, window_height,
                    is_vert_adjacent, vert_gap, is_horiz_adjacent, horiz_gap,
                )
                pairs.append(signals)

        return pairs

    def _is_vertically_adjacent(
        self, a: CandidateRegion, b: CandidateRegion,
    ) -> tuple[bool, int]:
        """Check if two regions are vertically adjacent."""
        # Check horizontal overlap
        overlap_l = max(a.bounds[0], b.bounds[0])
        overlap_r = min(a.bounds[2], b.bounds[2])
        overlap = max(0, overlap_r - overlap_l)
        wa = a.bounds[2] - a.bounds[0]
        wb = b.bounds[2] - b.bounds[0]
        min_w = min(wa, wb)
        if min_w <= 0 or overlap / min_w < 0.3:
            return False, 0

        # Check vertical gap
        top = min(a.bounds[3], b.bounds[3])
        bottom = max(a.bounds[1], b.bounds[1])
        gap = bottom - top  # negative = overlapping

        if gap <= 50:  # adjacent or overlapping
            return True, gap
        return False, gap

    def _is_horizontally_adjacent(
        self, a: CandidateRegion, b: CandidateRegion,
    ) -> tuple[bool, int]:
        """Check if two regions are horizontally adjacent."""
        # Check vertical overlap
        overlap_t = max(a.bounds[1], b.bounds[1])
        overlap_b = min(a.bounds[3], b.bounds[3])
        overlap = max(0, overlap_b - overlap_t)
        ha = a.bounds[3] - a.bounds[1]
        hb = b.bounds[3] - b.bounds[1]
        min_h = min(ha, hb)
        if min_h <= 0 or overlap / min_h < 0.3:
            return False, 0

        # Check horizontal gap
        left = min(a.bounds[2], b.bounds[2])
        right = max(a.bounds[0], b.bounds[0])
        gap = right - left

        if gap <= 50:
            return True, gap
        return False, gap

    def _compute_pairwise_signals(
        self,
        a: CandidateRegion,
        b: CandidateRegion,
        sub_lookup: dict[str, list[dict[str, Any]]],
        window_width: int,
        window_height: int,
        is_vert_adj: bool,
        vert_gap: int,
        is_horiz_adj: bool,
        horiz_gap: int,
    ) -> dict[str, Any]:
        """Compute pairwise signals with unknown/noise exclusion.

        Key change: dominant_subtype_similarity only considers valid subtypes.
        If both sides have no valid subtypes, similarity is None (insufficient evidence).
        """
        subs_a = sub_lookup.get(a.region_id, [])
        subs_b = sub_lookup.get(b.region_id, [])

        # ── Classify substructures into valid vs noise ──
        # noise = unknown_evidence (unclassified) + whitespace_block (empty)
        # valid = everything else (text_row, list_item_like, icon_candidate,
        #         control_candidate, code_line, message_like)
        noise_types = {"unknown_evidence", "whitespace_block"}

        valid_a = [s for s in subs_a if s.get("sub_type") not in noise_types]
        valid_b = [s for s in subs_b if s.get("sub_type") not in noise_types]

        noise_a = [s for s in subs_a if s.get("sub_type") in noise_types]
        noise_b = [s for s in subs_b if s.get("sub_type") in noise_types]

        total_a = len(subs_a)
        total_b = len(subs_b)
        valid_count_a = len(valid_a)
        valid_count_b = len(valid_b)
        unknown_ratio_a = len(noise_a) / max(total_a, 1)
        unknown_ratio_b = len(noise_b) / max(total_b, 1)

        # ── 1. vertical_gap_px ──
        vertical_gap = vert_gap if is_vert_adj else 0

        # ── 2. horizontal_overlap_ratio ──
        overlap_l = max(a.bounds[0], b.bounds[0])
        overlap_r = min(a.bounds[2], b.bounds[2])
        overlap_w = max(0, overlap_r - overlap_l)
        wa = a.bounds[2] - a.bounds[0]
        wb = b.bounds[2] - b.bounds[0]
        min_w = min(wa, wb)
        horiz_overlap = overlap_w / min_w if min_w > 0 else 0

        # ── 3. row_height_similarity (valid subs only) ──
        heights_a = [s["bounds"][3] - s["bounds"][1] for s in valid_a
                     if len(s.get("bounds", [])) >= 4]
        heights_b = [s["bounds"][3] - s["bounds"][1] for s in valid_b
                     if len(s.get("bounds", [])) >= 4]
        avg_h_a = sum(heights_a) / len(heights_a) if heights_a else 0
        avg_h_b = sum(heights_b) / len(heights_b) if heights_b else 0
        if avg_h_a > 0 and avg_h_b > 0:
            row_height_sim = self._similarity_score(
                int(avg_h_a), int(avg_h_b), tolerance=0.3,
            )
            row_height_insufficient = False
        else:
            row_height_sim = None
            row_height_insufficient = True

        # ── 4. dominant_subtype_similarity (valid subs only) ──
        valid_types_a = [s["sub_type"] for s in valid_a]
        valid_types_b = [s["sub_type"] for s in valid_b]

        if valid_types_a:
            dominant_a = max(set(valid_types_a), key=valid_types_a.count)
        else:
            dominant_a = None

        if valid_types_b:
            dominant_b = max(set(valid_types_b), key=valid_types_b.count)
        else:
            dominant_b = None

        if dominant_a is not None and dominant_b is not None:
            subtype_sim = 1.0 if dominant_a == dominant_b else 0.0
            subtype_insufficient = False
        else:
            subtype_sim = None
            subtype_insufficient = True

        # ── 5. text_density_similarity (valid subs only) ──
        text_a = sum(1 for s in valid_a if s["sub_type"] in ("text_row", "code_line"))
        text_b = sum(1 for s in valid_b if s["sub_type"] in ("text_row", "code_line"))
        area_a = max(1, (a.bounds[2] - a.bounds[0]) * (a.bounds[3] - a.bounds[1]))
        area_b = max(1, (b.bounds[2] - b.bounds[0]) * (b.bounds[3] - b.bounds[1]))

        if text_a >= 2 and text_b >= 2:
            text_dens_a = text_a / area_a * 10000
            text_dens_b = text_b / area_b * 10000
            text_dens_sim = self._similarity_score(
                int(text_dens_a * 100), int(text_dens_b * 100), tolerance=0.5,
            )
            text_dens_insufficient = False
        else:
            text_dens_sim = None
            text_dens_insufficient = True

        # ── 6. icon_density_similarity (valid subs only) ──
        icon_a = sum(1 for s in valid_a if s["sub_type"] == "icon_candidate")
        icon_b = sum(1 for s in valid_b if s["sub_type"] == "icon_candidate")

        if icon_a >= 2 and icon_b >= 2:
            icon_dens_a = icon_a / area_a * 10000
            icon_dens_b = icon_b / area_b * 10000
            icon_dens_sim = self._similarity_score(
                int(icon_dens_a * 100), int(icon_dens_b * 100), tolerance=0.5,
            )
            icon_dens_insufficient = False
        else:
            icon_dens_sim = None
            icon_dens_insufficient = True

        # ── 7. x_alignment_similarity (valid subs only) ──
        lefts_a = [s["bounds"][0] for s in valid_a if len(s.get("bounds", [])) >= 4]
        lefts_b = [s["bounds"][0] for s in valid_b if len(s.get("bounds", [])) >= 4]
        if lefts_a and lefts_b:
            min_left_a = min(lefts_a)
            min_left_b = min(lefts_b)
            x_align_sim = 1.0 if abs(min_left_a - min_left_b) < 20 else 0.0
            x_align_insufficient = False
        else:
            x_align_sim = None
            x_align_insufficient = True

        # ── 8. column_alignment_similarity (valid subs only) ──
        col_a = self._count_x_clusters(lefts_a, tolerance=15)
        col_b = self._count_x_clusters(lefts_b, tolerance=15)
        if col_a > 0 and col_b > 0:
            col_align_sim = self._similarity_score(col_a, col_b, tolerance=0.3)
            col_align_insufficient = False
        else:
            col_align_sim = None
            col_align_insufficient = True

        # ── 9. edge_region_flag ──
        is_edge = (
            a.bounds[1] < window_height * 0.08
            or a.bounds[3] > window_height * 0.92
            or a.bounds[0] < window_width * 0.05
            or a.bounds[2] > window_width * 0.95
            or b.bounds[1] < window_height * 0.08
            or b.bounds[3] > window_height * 0.92
            or b.bounds[0] < window_width * 0.05
            or b.bounds[2] > window_width * 0.95
        )

        # ── 10. icon_sub_ratio (valid subs only) ──
        total_valid = valid_count_a + valid_count_b
        total_icons = icon_a + icon_b
        icon_ratio = total_icons / total_valid if total_valid > 0 else 0

        # ── 11. input_like_flag ──
        has_input = any(
            s.get("control_type", "").lower() in ("edit", "textbox", "input")
            for s in subs_a + subs_b
        )

        # ── Signal reliability ──
        insufficient_count = sum([
            row_height_insufficient, subtype_insufficient,
            text_dens_insufficient, icon_dens_insufficient,
            x_align_insufficient, col_align_insufficient,
        ])
        if insufficient_count >= 4:
            signal_reliability = "low"
        elif insufficient_count >= 2:
            signal_reliability = "medium"
        else:
            signal_reliability = "high"

        return {
            "pair_id": f"{a.region_id}-{b.region_id}",
            "region_a": a.region_id,
            "region_b": b.region_id,
            "direction": "vertical" if is_vert_adj else "horizontal",
            "signals": {
                "vertical_gap_px": vertical_gap,
                "horizontal_overlap_ratio": round(horiz_overlap, 3),
                "row_height_similarity": round(row_height_sim, 3) if row_height_sim is not None else None,
                "dominant_subtype_similarity": round(subtype_sim, 3) if subtype_sim is not None else None,
                "text_density_similarity": round(text_dens_sim, 3) if text_dens_sim is not None else None,
                "icon_density_similarity": round(icon_dens_sim, 3) if icon_dens_sim is not None else None,
                "x_alignment_similarity": round(x_align_sim, 3) if x_align_sim is not None else None,
                "column_alignment_similarity": round(col_align_sim, 3) if col_align_sim is not None else None,
                "edge_region_flag": is_edge,
                "icon_sub_ratio": round(icon_ratio, 3),
                "input_like_flag": has_input,
            },
            "insufficient_evidence": {
                "row_height": row_height_insufficient,
                "dominant_subtype": subtype_insufficient,
                "text_density": text_dens_insufficient,
                "icon_density": icon_dens_insufficient,
                "x_alignment": x_align_insufficient,
                "column_alignment": col_align_insufficient,
            },
            "signal_reliability": signal_reliability,
            "valid_subtype_count_a": valid_count_a,
            "valid_subtype_count_b": valid_count_b,
            "unknown_ratio_a": round(unknown_ratio_a, 3),
            "unknown_ratio_b": round(unknown_ratio_b, 3),
            "region_a_dominant_sub": dominant_a,
            "region_b_dominant_sub": dominant_b,
        }

    def _count_x_clusters(
        self, x_values: list[int], tolerance: int = 15,
    ) -> int:
        """Count distinct x-position clusters."""
        if not x_values:
            return 0
        sorted_x = sorted(set(x_values))
        clusters = 1
        for i in range(1, len(sorted_x)):
            if sorted_x[i] - sorted_x[i - 1] > tolerance:
                clusters += 1
        return clusters

    def _compute_gap_signature(
        self,
        gap_start: int,
        gap_end: int,
        direction: str,
        width: int,
        height: int,
        arr: np.ndarray,
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
    ) -> dict[str, float]:
        """Compute evidence signature for a gap (used for stream detection)."""
        band_height = max(30, gap_end - gap_start)

        if direction == "horizontal":
            above_top = max(0, gap_start - band_height)
            below_bottom = min(height, gap_end + band_height)

            ocr_above = self._count_ocr_in_band(
                ocr_blocks, above_top, gap_start, width,
            )
            ocr_below = self._count_ocr_in_band(
                ocr_blocks, gap_end, below_bottom, width,
            )
            vis_above = self._count_items_in_band(
                vision_candidates, above_top, gap_start, width, "bounding_rect",
            )
            vis_below = self._count_items_in_band(
                vision_candidates, gap_end, below_bottom, width, "bounding_rect",
            )

            above_region = arr[above_top:gap_start, :]
            below_region = arr[gap_end:below_bottom, :]
            density_above = float(above_region.std()) if above_region.size > 0 else 0
            density_below = float(below_region.std()) if below_region.size > 0 else 0

            return {
                "ocr_above": ocr_above,
                "ocr_below": ocr_below,
                "vis_above": vis_above,
                "vis_below": vis_below,
                "density_above": density_above,
                "density_below": density_below,
                "gap_height": gap_end - gap_start,
            }

        # Vertical: use left/right bands
        band_width = max(30, gap_end - gap_start)
        left_start = max(0, gap_start - band_width)
        right_end = min(width, gap_end + band_width)

        ocr_left = self._count_ocr_in_band_rect(
            ocr_blocks, left_start, gap_start, 0, height,
        )
        ocr_right = self._count_ocr_in_band_rect(
            ocr_blocks, gap_end, right_end, 0, height,
        )
        vis_left = self._count_items_in_band_rect(
            vision_candidates, left_start, gap_start, 0, height,
        )
        vis_right = self._count_items_in_band_rect(
            vision_candidates, gap_end, right_end, 0, height,
        )

        left_region = arr[:, left_start:gap_start]
        right_region = arr[:, gap_end:right_end]
        density_left = float(left_region.std()) if left_region.size > 0 else 0
        density_right = float(right_region.std()) if right_region.size > 0 else 0

        return {
            "ocr_above": ocr_left,
            "ocr_below": ocr_right,
            "vis_above": vis_left,
            "vis_below": vis_right,
            "density_above": density_left,
            "density_below": density_right,
            "gap_height": gap_end - gap_start,
        }

    def _signatures_similar(
        self,
        a: dict[str, float],
        b: dict[str, float],
    ) -> bool:
        """Check if two gap signatures indicate similar content rows."""
        # OCR count similarity (within 2x or both ≤1)
        ocr_similar = (
            abs(a["ocr_above"] - b["ocr_above"]) <= max(1, max(a["ocr_above"], b["ocr_above"]) * 0.5)
            and abs(a["ocr_below"] - b["ocr_below"]) <= max(1, max(a["ocr_below"], b["ocr_below"]) * 0.5)
        )
        # Vision count similarity
        vis_similar = (
            abs(a["vis_above"] - b["vis_above"]) <= max(2, max(a["vis_above"], b["vis_above"]) * 0.5)
            and abs(a["vis_below"] - b["vis_below"]) <= max(2, max(a["vis_below"], b["vis_below"]) * 0.5)
        )
        # Density similarity (std within 15)
        density_similar = (
            abs(a["density_above"] - b["density_above"]) < 15
            and abs(a["density_below"] - b["density_below"]) < 15
        )
        # Gap height similarity (within 3x)
        height_similar = (
            min(a["gap_height"], b["gap_height"]) * 3
            >= max(a["gap_height"], b["gap_height"])
        )

        return ocr_similar and vis_similar and density_similar and height_similar

    def _detect_density_transitions(
        self,
        arr: np.ndarray,
        direction: str,
        width: int,
        height: int,
    ) -> list[Separator]:
        """Detect density transitions (brightness std changes)."""
        transitions: list[Separator] = []
        sep_idx = 0

        if direction == "horizontal":
            # Compute per-row std
            row_stds = np.std(arr, axis=1)
            # Find sharp transitions
            diffs = np.abs(np.diff(row_stds))
            for y in range(len(diffs)):
                if diffs[y] >= self.DENSITY_TRANSITION_THRESHOLD:
                    # Check span: how many columns have brightness change
                    if y + 1 < height:
                        row_diff = np.abs(arr[y + 1].astype(float) - arr[y].astype(float))
                        span = int((row_diff > self.DENSITY_TRANSITION_THRESHOLD).sum())
                        span_ratio = span / width
                        if span_ratio >= self.MIN_TRANSITION_SPAN_RATIO:
                            transitions.append(Separator(
                                separator_id=f"HT{sep_idx}",
                                direction="horizontal",
                                position=y,
                                width=1,
                                span=span,
                                strength=min(1.0, diffs[y] / 50.0),
                                kind="density_transition",
                            ))
                            sep_idx += 1

        else:  # vertical
            col_stds = np.std(arr, axis=0)
            diffs = np.abs(np.diff(col_stds))
            for x in range(len(diffs)):
                if diffs[x] >= self.DENSITY_TRANSITION_THRESHOLD:
                    if x + 1 < width:
                        col_diff = np.abs(arr[:, x + 1].astype(float) - arr[:, x].astype(float))
                        span = int((col_diff > self.DENSITY_TRANSITION_THRESHOLD).sum())
                        span_ratio = span / height
                        if span_ratio >= self.MIN_TRANSITION_SPAN_RATIO:
                            transitions.append(Separator(
                                separator_id=f"VT{sep_idx}",
                                direction="vertical",
                                position=x,
                                width=1,
                                span=span,
                                strength=min(1.0, diffs[x] / 50.0),
                                kind="density_transition",
                            ))
                            sep_idx += 1

        return transitions

    def _build_candidate_regions(
        self,
        separators: list[Separator],
        width: int,
        height: int,
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
        raw_elements: list[dict[str, Any]] | None,
    ) -> list[CandidateRegion]:
        """Build candidate regions from separators."""
        # Collect horizontal and vertical separator positions
        h_positions = sorted(set([0] + [s.position for s in separators if s.direction == "horizontal"] + [height]))
        v_positions = sorted(set([0] + [s.position for s in separators if s.direction == "vertical"] + [width]))

        # Build grid cells
        candidates: list[CandidateRegion] = []
        region_idx = 0
        min_area = width * height * self.MIN_REGION_AREA_RATIO

        for i in range(len(h_positions) - 1):
            for j in range(len(v_positions) - 1):
                top = h_positions[i]
                bottom = h_positions[i + 1]
                left = v_positions[j]
                right = v_positions[j + 1]

                rw = right - left
                rh = bottom - top
                if rw <= 0 or rh <= 0:
                    continue
                if rw * rh < min_area:
                    continue

                # Find bounding separators
                bounded_by = []
                for s in separators:
                    if s.direction == "horizontal" and s.position in (top, bottom):
                        bounded_by.append(s.separator_id)
                    elif s.direction == "vertical" and s.position in (left, right):
                        bounded_by.append(s.separator_id)

                # Count evidence within region
                ocr_count = self._count_in_region(left, top, right, bottom, ocr_blocks, "bbox")
                vision_count = self._count_in_region(left, top, right, bottom, vision_candidates, "bounding_rect")
                uia_count = self._count_in_region(left, top, right, bottom, raw_elements, "bounding_rect")

                candidates.append(CandidateRegion(
                    region_id=f"R{region_idx}",
                    bounds=(left, top, right, bottom),
                    bounded_by=bounded_by,
                    ocr_count=ocr_count,
                    vision_count=vision_count,
                    uia_count=uia_count,
                    area_ratio=(rw * rh) / (width * height),
                ))
                region_idx += 1

                if region_idx >= self.MAX_REGIONS:
                    break
            if region_idx >= self.MAX_REGIONS:
                break

        return candidates

    def _count_in_region(
        self,
        left: int, top: int, right: int, bottom: int,
        items: list[dict[str, Any]] | None,
        bbox_key: str,
    ) -> int:
        """Count items whose center falls within the region."""
        if not items:
            return 0
        count = 0
        for item in items:
            bbox = item.get(bbox_key) or item.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            if left <= cx <= right and top <= cy <= bottom:
                count += 1
        return count

    # ── U4.0-fix: Evidence-aware gap filtering ──────────────────────

    def _gap_has_evidence_diff(
        self,
        gap_start: int,
        gap_end: int,
        direction: str,
        width: int,
        height: int,
        arr: np.ndarray,
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
    ) -> bool:
        """Check if gap separates regions with different evidence.

        Returns True if the gap likely separates different content areas.
        Returns False if both sides look similar (text rows, chat messages).
        """
        # Define bands above and below the gap
        band_height = max(30, gap_end - gap_start)  # at least 30px band

        if direction == "horizontal":
            above_top = max(0, gap_start - band_height)
            below_bottom = min(height, gap_end + band_height)

            # OCR density above vs below
            ocr_above = self._count_ocr_in_band(ocr_blocks, above_top, gap_start, width)
            ocr_below = self._count_ocr_in_band(ocr_blocks, gap_end, below_bottom, width)

            # Vision density above vs below
            vis_above = self._count_items_in_band(vision_candidates, above_top, gap_start, width, "bounding_rect")
            vis_below = self._count_items_in_band(vision_candidates, gap_end, below_bottom, width, "bounding_rect")

            # Pixel density above vs below
            above_region = arr[above_top:gap_start, :]
            below_region = arr[gap_end:below_bottom, :]
            density_above = float(above_region.std()) if above_region.size > 0 else 0
            density_below = float(below_region.std()) if below_region.size > 0 else 0

            # If no evidence on either side, accept gap (can't determine similarity)
            if ocr_above == 0 and ocr_below == 0 and vis_above == 0 and vis_below == 0:
                return True

            # Check if sides are similar (reject gap)
            ocr_similar = abs(ocr_above - ocr_below) <= max(1, max(ocr_above, ocr_below) * 0.5)
            vis_similar = abs(vis_above - vis_below) <= max(2, max(vis_above, vis_below) * 0.5)
            density_similar = abs(density_above - density_below) < 15

            if ocr_similar and vis_similar and density_similar:
                return False  # Both sides look similar → reject gap

            return True  # Different evidence → accept gap

        return True  # Vertical gaps always accepted for now

    def _count_ocr_in_band(
        self,
        ocr_blocks: list[dict[str, Any]] | None,
        band_top: int,
        band_bottom: int,
        window_width: int,
    ) -> int:
        """Count OCR blocks in a horizontal band."""
        if not ocr_blocks:
            return 0
        count = 0
        for block in ocr_blocks:
            bbox = block.get("bbox", [])
            if len(bbox) >= 4:
                cy = (bbox[1] + bbox[3]) // 2
                if band_top <= cy < band_bottom:
                    count += 1
        return count

    def _count_items_in_band(
        self,
        items: list[dict[str, Any]] | None,
        band_top: int,
        band_bottom: int,
        window_width: int,
        bbox_key: str = "bounding_rect",
    ) -> int:
        """Count items in a horizontal band."""
        if not items:
            return 0
        count = 0
        for item in items:
            bbox = item.get(bbox_key) or item.get("bbox")
            if bbox and len(bbox) >= 4:
                cy = (bbox[1] + bbox[3]) // 2
                if band_top <= cy < band_bottom:
                    count += 1
        return count

    def _count_ocr_in_band_rect(
        self,
        ocr_blocks: list[dict[str, Any]] | None,
        left: int, right: int, top: int, bottom: int,
    ) -> int:
        """Count OCR blocks whose center falls within a rectangular band."""
        if not ocr_blocks:
            return 0
        count = 0
        for block in ocr_blocks:
            bbox = block.get("bbox", [])
            if len(bbox) >= 4:
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                if left <= cx < right and top <= cy < bottom:
                    count += 1
        return count

    def _count_items_in_band_rect(
        self,
        items: list[dict[str, Any]] | None,
        left: int, right: int, top: int, bottom: int,
    ) -> int:
        """Count items whose center falls within a rectangular band."""
        if not items:
            return 0
        count = 0
        for item in items:
            bbox = item.get("bounding_rect") or item.get("bbox")
            if bbox and len(bbox) >= 4:
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                if left <= cx < right and top <= cy < bottom:
                    count += 1
        return count

    # ── U4.1: Content-stream-aware merging ────────────────────────

    def _merge_fragmented_regions(
        self,
        candidates: list[CandidateRegion],
        width: int,
        height: int,
    ) -> list[CandidateRegion]:
        """Merge consecutive regions with similar evidence into content areas.

        Two-pass merge:
        1. Merge vertically adjacent small regions with similar evidence.
        2. Merge any consecutive regions whose evidence type is the same
           (e.g., all text-heavy or all vision-heavy) into content containers.
        """
        if len(candidates) <= 1:
            return candidates

        # Sort by vertical position, then horizontal
        sorted_cands = sorted(candidates, key=lambda c: (c.bounds[1], c.bounds[0]))

        # ── Pass 1: merge adjacent small regions ──
        merged_pass1: list[CandidateRegion] = []
        current = sorted_cands[0]

        for cand in sorted_cands[1:]:
            curr_bottom = current.bounds[3]
            cand_top = cand.bounds[1]
            gap = cand_top - curr_bottom

            # Evidence similarity
            ocr_similar = abs(current.ocr_count - cand.ocr_count) <= max(
                1, max(current.ocr_count, cand.ocr_count) * 0.5,
            )
            vis_similar = abs(current.vision_count - cand.vision_count) <= max(
                2, max(current.vision_count, cand.vision_count) * 0.5,
            )
            height_similar = abs(
                (current.bounds[3] - current.bounds[1])
                - (cand.bounds[3] - cand.bounds[1]),
            ) < 30

            # Size check
            both_small = current.area_ratio < 0.10 and cand.area_ratio < 0.10

            if (
                gap >= 0 and gap <= 20
                and ocr_similar and vis_similar and height_similar
                and both_small
            ):
                current = self._merge_two_regions(current, cand)
            else:
                merged_pass1.append(current)
                current = cand
        merged_pass1.append(current)

        # ── Pass 2: merge content streams ──
        # If 3+ consecutive regions all have the same dominant evidence type
        # (text-heavy or vision-heavy), merge them into one content area.
        return self._merge_content_streams(merged_pass1, width, height)

    def _merge_two_regions(
        self, a: CandidateRegion, b: CandidateRegion,
    ) -> CandidateRegion:
        """Merge two regions into one."""
        return CandidateRegion(
            region_id=a.region_id,
            bounds=(
                min(a.bounds[0], b.bounds[0]),
                min(a.bounds[1], b.bounds[1]),
                max(a.bounds[2], b.bounds[2]),
                max(a.bounds[3], b.bounds[3]),
            ),
            bounded_by=list(set(a.bounded_by + b.bounded_by)),
            ocr_count=a.ocr_count + b.ocr_count,
            vision_count=a.vision_count + b.vision_count,
            uia_count=a.uia_count + b.uia_count,
            area_ratio=a.area_ratio + b.area_ratio,
        )

    def _classify_evidence_type(self, region: CandidateRegion) -> str:
        """Classify a region's dominant evidence type.

        Returns: 'text' (OCR-heavy), 'vision' (vision-heavy),
                 'mixed' (both), 'empty' (neither).
        """
        has_ocr = region.ocr_count >= 1
        has_vision = region.vision_count >= 1
        if has_ocr and has_vision:
            return "mixed"
        if has_ocr:
            return "text"
        if has_vision:
            return "vision"
        return "empty"

    def _merge_content_streams(
        self,
        candidates: list[CandidateRegion],
        width: int,
        height: int,
    ) -> list[CandidateRegion]:
        """Merge consecutive regions with the same evidence type.

        If 3+ consecutive regions all share the same dominant evidence type,
        they form a content stream (text rows, message rows, list items).
        Merge them into a single content container region.

        For empty-evidence regions (no OCR/vision), also merge if they have
        similar height — a strong signal of repeated content rows.
        """
        if len(candidates) < 2:
            return candidates

        # Classify each region
        types = [self._classify_evidence_type(c) for c in candidates]

        # Find runs of same type
        result: list[CandidateRegion] = []
        run_start = 0

        for i in range(1, len(candidates) + 1):
            continues_run = False
            if i < len(candidates):
                # Check vertical adjacency (gap ≤ 20px)
                prev_bottom = candidates[i - 1].bounds[3]
                curr_top = candidates[i].bounds[1]
                adjacent = (curr_top - prev_bottom) <= 20

                if adjacent and types[i] == types[run_start]:
                    continues_run = True
                elif adjacent and types[run_start] == "empty" and types[i] == "empty":
                    # Empty-evidence regions: merge if similar height
                    h_prev = candidates[i - 1].bounds[3] - candidates[i - 1].bounds[1]
                    h_curr = candidates[i].bounds[3] - candidates[i].bounds[1]
                    if min(h_prev, h_curr) * 3 >= max(h_prev, h_curr):
                        continues_run = True

            if continues_run:
                continue

            # End of run
            run_length = i - run_start
            should_merge = False
            if run_length >= 3 and types[run_start] != "empty":
                # Also check that all regions in the run share the same
                # horizontal extent (same column) — don't merge across
                # vertical separators.
                lefts = [candidates[j].bounds[0] for j in range(run_start, i)]
                rights = [candidates[j].bounds[2] for j in range(run_start, i)]
                same_column = (
                    max(lefts) - min(lefts) < 20
                    and max(rights) - min(rights) < 20
                )
                if same_column:
                    should_merge = True
            elif run_length >= 3 and types[run_start] == "empty":
                # Empty-evidence run: merge if regions have similar height
                # AND share the same horizontal extent (same column)
                lefts = [candidates[j].bounds[0] for j in range(run_start, i)]
                rights = [candidates[j].bounds[2] for j in range(run_start, i)]
                same_column = (
                    max(lefts) - min(lefts) < 20
                    and max(rights) - min(rights) < 20
                )
                if same_column:
                    heights = [
                        candidates[j].bounds[3] - candidates[j].bounds[1]
                        for j in range(run_start, i)
                    ]
                    avg_h = sum(heights) / len(heights)
                    if all(min(h, avg_h) * 2 >= max(h, avg_h) for h in heights):
                        should_merge = True

            if should_merge:
                merged = candidates[run_start]
                for j in range(run_start + 1, i):
                    merged = self._merge_two_regions(merged, candidates[j])
                result.append(merged)
            else:
                result.extend(candidates[run_start:i])

            run_start = i

        return result
