"""U3-shadow: OCR/Visual Density Region Proposal.

Read-only diagnostic that proposes structural regions from OCR text rows,
whitespace gaps, and basic density patterns. Does NOT modify main regions,
element.region_id, ROI, query, or backfill.

Only runs when invoked explicitly (QA script or OPENCLAW_U3_OCR_SHADOW=1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Proposal types
PROPOSAL_TYPES = {
    "ocr_text_row",
    "top_band_candidate",
    "bottom_band_candidate",
    "side_panel_candidate",
    "dense_text_area",
    "input_like_band",
}

# Classification categories
_USEFUL_TYPES = {
    "top_bar", "toolbar", "status_region", "control_strip",
    "media_control_bar", "input_region", "side_rail",
}
_NEUTRAL_TYPES = {
    "dense_text_area", "unknown_structured",
}

# Rejection reasons
NO_OCR_BLOCKS = "no_ocr_blocks"
SCREENSHOT_MISSING = "screenshot_missing"
TOO_NARROW = "too_narrow"
TOO_SHORT = "too_short"
LOW_DENSITY = "low_density"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class OcrTextRow:
    """A horizontal row of OCR text blocks."""

    y_top: int
    y_bottom: int
    x_left: int
    x_right: int
    text_count: int
    texts: list[str]
    avg_confidence: float

    @property
    def height(self) -> int:
        return max(self.y_bottom - self.y_top, 1)

    @property
    def width(self) -> int:
        return max(self.x_right - self.x_left, 1)

    @property
    def center_y(self) -> int:
        return (self.y_top + self.y_bottom) // 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "y_top": self.y_top,
            "y_bottom": self.y_bottom,
            "x_left": self.x_left,
            "x_right": self.x_right,
            "height": self.height,
            "width": self.width,
            "text_count": self.text_count,
            "texts": self.texts[:5],
            "avg_confidence": round(self.avg_confidence, 3),
        }


@dataclass
class RegionProposal:
    """A proposed structural region from OCR/visual evidence."""

    proposal_id: str
    proposal_type: str  # from PROPOSAL_TYPES
    bounds: tuple[int, int, int, int]
    inferred_structure_type: str = "unknown_structured"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    ocr_text_count: int = 0
    ocr_texts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type,
            "bounds": list(self.bounds),
            "inferred_structure_type": self.inferred_structure_type,
            "confidence": round(self.confidence, 3),
            "evidence": list(self.evidence),
            "ocr_text_count": self.ocr_text_count,
            "ocr_texts": self.ocr_texts[:5],
        }


@dataclass
class WhitespaceGap:
    """A detected whitespace gap (vertical or horizontal)."""

    direction: str  # "horizontal" or "vertical"
    position: int  # y for horizontal, x for vertical
    span: int  # width for horizontal, height for vertical
    gap_size: int  # height for horizontal gap, width for vertical gap

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "position": self.position,
            "span": self.span,
            "gap_size": self.gap_size,
        }


@dataclass
class OcrVisualShadowResult:
    """Result of U3-shadow pass."""

    ocr_text_rows_detected: int = 0
    proposals_count: int = 0
    proposal_type_counts: dict[str, int] = field(default_factory=dict)
    useful_proposals: int = 0
    neutral_proposals: int = 0
    noisy_proposals: int = 0
    over_fragmentation: float = 0.0
    rejected_reasons: dict[str, int] = field(default_factory=dict)
    ocr_text_rows: list[dict[str, Any]] = field(default_factory=list)
    whitespace_gaps: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ocr_text_rows_detected": self.ocr_text_rows_detected,
            "proposals_count": self.proposals_count,
            "proposal_type_counts": dict(self.proposal_type_counts),
            "useful_proposals": self.useful_proposals,
            "neutral_proposals": self.neutral_proposals,
            "noisy_proposals": self.noisy_proposals,
            "over_fragmentation": round(self.over_fragmentation, 3),
            "rejected_reasons": dict(self.rejected_reasons),
            "ocr_text_rows": [r.to_dict() for r in self._rows] if hasattr(self, '_rows') else self.ocr_text_rows,
            "whitespace_gaps": self.whitespace_gaps,
            "proposals": self.proposals,
        }


class OcrVisualShadowPass:
    """Propose structural regions from OCR text rows and whitespace gaps."""

    # OCR row clustering
    ROW_GAP_THRESHOLD = 5  # pixels; rows closer than this are merged
    MIN_ROW_WIDTH_RATIO = 0.15  # row must span at least 15% of window width

    # Whitespace gap detection
    MIN_WHITESPACE_GAP = 15  # pixels
    MIN_WHITESPACE_SPAN_RATIO = 0.3  # gap must span at least 30% of window

    # Proposal scoring
    _USEFUL_TYPES = {
        "top_bar", "toolbar", "status_region", "control_strip",
        "media_control_bar", "input_region", "side_rail",
    }

    def run(
        self,
        ocr_blocks: list[dict[str, Any]] | None,
        window_width: int = 0,
        window_height: int = 0,
    ) -> OcrVisualShadowResult:
        """Run U3-shadow pass."""
        result = OcrVisualShadowResult()

        if not ocr_blocks:
            result.rejected_reasons[NO_OCR_BLOCKS] = 1
            return result

        if window_width <= 0 or window_height <= 0:
            result.rejected_reasons[SCREENSHOT_MISSING] = 1
            return result

        # 1. Cluster OCR blocks into text rows
        rows = self._cluster_ocr_rows(ocr_blocks, window_width, window_height)
        result.ocr_text_rows_detected = len(rows)
        result.ocr_text_rows = [r.to_dict() for r in rows]
        result._rows = rows

        # 2. Detect whitespace gaps
        gaps = self._detect_whitespace_gaps(ocr_blocks, window_width, window_height)
        result.whitespace_gaps = [g.to_dict() for g in gaps]

        # 3. Generate proposals from rows + gaps
        proposals = self._generate_proposals(rows, gaps, ocr_blocks, window_width, window_height)
        result.proposals_count = len(proposals)
        result.proposals = [p.to_dict() for p in proposals]

        # 4. Count by type
        for p in proposals:
            result.proposal_type_counts[p.proposal_type] = result.proposal_type_counts.get(p.proposal_type, 0) + 1

        # 5. Score useful / neutral / noisy
        for p in proposals:
            if p.inferred_structure_type in _USEFUL_TYPES:
                result.useful_proposals += 1
            elif p.inferred_structure_type in _NEUTRAL_TYPES:
                result.neutral_proposals += 1
            else:
                result.noisy_proposals += 1

        # 6. Over-fragmentation
        if proposals:
            small_count = sum(
                1 for p in proposals
                if (p.bounds[2] - p.bounds[0]) * (p.bounds[3] - p.bounds[1]) < window_width * window_height * 0.02
            )
            result.over_fragmentation = small_count / len(proposals)

        return result

    def _cluster_ocr_rows(
        self,
        ocr_blocks: list[dict[str, Any]],
        window_width: int,
        window_height: int,
    ) -> list[OcrTextRow]:
        """Cluster OCR blocks into horizontal text rows."""
        # Extract valid blocks with bbox
        valid = []
        for block in ocr_blocks:
            bbox = block.get("bbox", [])
            if len(bbox) >= 4:
                text = str(block.get("text", "")).strip()
                if text:
                    valid.append({
                        "bbox": bbox,
                        "text": text,
                        "confidence": float(block.get("confidence", 0)),
                    })

        if not valid:
            return []

        # Sort by Y center
        for b in valid:
            b["cy"] = (b["bbox"][1] + b["bbox"][3]) // 2
        valid.sort(key=lambda b: b["cy"])

        # Cluster by Y proximity
        rows: list[OcrTextRow] = []
        current_cluster: list[dict] = [valid[0]]

        for block in valid[1:]:
            prev_cy = current_cluster[-1]["cy"]
            curr_cy = block["cy"]
            if curr_cy - prev_cy <= self.ROW_GAP_THRESHOLD:
                current_cluster.append(block)
            else:
                rows.append(self._build_row(current_cluster))
                current_cluster = [block]

        if current_cluster:
            rows.append(self._build_row(current_cluster))

        # Filter by minimum width
        min_width = window_width * self.MIN_ROW_WIDTH_RATIO
        return [r for r in rows if r.width >= min_width]

    def _build_row(self, cluster: list[dict]) -> OcrTextRow:
        """Build an OcrTextRow from a cluster of OCR blocks."""
        y_top = min(b["bbox"][1] for b in cluster)
        y_bottom = max(b["bbox"][3] for b in cluster)
        x_left = min(b["bbox"][0] for b in cluster)
        x_right = max(b["bbox"][2] for b in cluster)
        texts = [b["text"] for b in cluster]
        avg_conf = sum(b["confidence"] for b in cluster) / len(cluster)
        return OcrTextRow(
            y_top=y_top, y_bottom=y_bottom,
            x_left=x_left, x_right=x_right,
            text_count=len(cluster), texts=texts,
            avg_confidence=avg_conf,
        )

    def _detect_whitespace_gaps(
        self,
        ocr_blocks: list[dict[str, Any]],
        window_width: int,
        window_height: int,
    ) -> list[WhitespaceGap]:
        """Detect horizontal whitespace gaps between OCR text rows."""
        # Build Y-coverage histogram
        y_coverage = [0] * window_height
        for block in ocr_blocks:
            bbox = block.get("bbox", [])
            if len(bbox) >= 4:
                for y in range(max(0, bbox[1]), min(window_height, bbox[3])):
                    y_coverage[y] += 1

        # Find horizontal gaps (consecutive Y rows with 0 coverage)
        gaps: list[WhitespaceGap] = []
        gap_start = -1
        min_span = int(window_width * self.MIN_WHITESPACE_SPAN_RATIO)

        for y in range(window_height):
            if y_coverage[y] == 0:
                if gap_start < 0:
                    gap_start = y
            else:
                if gap_start >= 0:
                    gap_size = y - gap_start
                    if gap_size >= self.MIN_WHITESPACE_GAP:
                        gaps.append(WhitespaceGap(
                            direction="horizontal",
                            position=gap_start,
                            span=window_width,
                            gap_size=gap_size,
                        ))
                    gap_start = -1

        # Final gap at bottom
        if gap_start >= 0:
            gap_size = window_height - gap_start
            if gap_size >= self.MIN_WHITESPACE_GAP:
                gaps.append(WhitespaceGap(
                    direction="horizontal",
                    position=gap_start,
                    span=window_width,
                    gap_size=gap_size,
                ))

        return gaps

    def _generate_proposals(
        self,
        rows: list[OcrTextRow],
        gaps: list[WhitespaceGap],
        ocr_blocks: list[dict[str, Any]],
        window_width: int,
        window_height: int,
    ) -> list[RegionProposal]:
        """Generate region proposals from rows and gaps."""
        proposals: list[RegionProposal] = []
        proposal_idx = 0

        # Use gaps to define region boundaries
        y_boundaries = sorted(set([0] + [g.position for g in gaps] + [g.position + g.gap_size for g in gaps] + [window_height]))
        y_boundaries = [y for y in y_boundaries if 0 <= y <= window_height]

        for i in range(len(y_boundaries) - 1):
            y_top = y_boundaries[i]
            y_bottom = y_boundaries[i + 1]
            band_height = y_bottom - y_top
            if band_height < 10:
                continue

            # Count OCR blocks in this band
            band_blocks = []
            for block in ocr_blocks:
                bbox = block.get("bbox", [])
                if len(bbox) >= 4:
                    cy = (bbox[1] + bbox[3]) // 2
                    if y_top <= cy < y_bottom:
                        band_blocks.append(block)

            # Count rows in this band
            band_rows = [r for r in rows if y_top <= r.center_y < y_bottom]

            # Infer structure type based on position + evidence
            rel_top = y_top / window_height
            rel_bottom = y_bottom / window_height
            height_ratio = band_height / window_height

            inferred_type = "unknown_structured"
            confidence = 0.0
            evidence = []

            # Check for top band
            if rel_top < 0.15 and len(band_rows) >= 1:
                h_row_score = self._compute_h_row_score(band_rows)
                if h_row_score > 0.3:
                    inferred_type = "top_bar"
                    confidence = min(0.6, 0.3 + h_row_score * 0.3)
                    evidence.append(f"h_row_score={h_row_score:.2f}")

            # Check for toolbar (top 15-25% with buttons/icons)
            elif rel_top < 0.25 and len(band_blocks) >= 2:
                short_texts = [b for b in band_blocks if len(str(b.get("text", ""))) <= 8]
                if len(short_texts) >= 2:
                    inferred_type = "toolbar"
                    confidence = 0.5
                    evidence.append(f"short_texts={len(short_texts)}")

            # Check for status region (bottom 10% with short text)
            elif rel_bottom > 0.90 and len(band_blocks) >= 1:
                short_texts = [b for b in band_blocks if len(str(b.get("text", ""))) <= 12]
                if len(short_texts) >= 1:
                    inferred_type = "status_region"
                    confidence = 0.5
                    evidence.append(f"bottom_short_texts={len(short_texts)}")

            # Check for input region (bottom 20-30% with edit-like)
            elif rel_bottom > 0.70 and height_ratio < 0.25:
                if any("edit" in str(b.get("control_type", "")).lower() for b in band_blocks):
                    inferred_type = "input_region"
                    confidence = 0.5
                    evidence.append("edit_control_in_bottom")

            # Check for dense text area
            elif len(band_blocks) >= 5 and height_ratio > 0.10:
                inferred_type = "dense_text_area"
                confidence = 0.4
                evidence.append(f"block_count={len(band_blocks)}")

            # Check for side panel (narrow + tall)
            if inferred_type == "unknown_structured":
                for row in band_rows:
                    if row.width < window_width * 0.3 and row.height > window_height * 0.3:
                        inferred_type = "side_panel_candidate"
                        confidence = 0.4
                        evidence.append(f"narrow_tall_row")
                        break

            if inferred_type != "unknown_structured" or len(band_blocks) >= 2:
                proposals.append(RegionProposal(
                    proposal_id=f"P{proposal_idx}",
                    proposal_type="ocr_text_row" if band_rows else "dense_text_area",
                    bounds=(0, y_top, window_width, y_bottom),
                    inferred_structure_type=inferred_type,
                    confidence=confidence,
                    evidence=evidence,
                    ocr_text_count=len(band_blocks),
                    ocr_texts=[str(b.get("text", ""))[:20] for b in band_blocks[:5]],
                ))
                proposal_idx += 1

        return proposals

    def _compute_h_row_score(self, rows: list[OcrTextRow]) -> float:
        """Compute horizontal alignment score for rows."""
        if len(rows) < 2:
            return 0.0
        centers_y = [r.center_y for r in rows]
        mean_y = sum(centers_y) / len(centers_y)
        std_y = (sum((y - mean_y) ** 2 for y in centers_y) / len(centers_y)) ** 0.5
        avg_height = sum(r.height for r in rows) / len(rows)
        if avg_height <= 0:
            return 0.0
        ratio = std_y / avg_height
        if ratio < 0.5:
            return 1.0
        if ratio < 1.0:
            return 0.7
        if ratio < 2.0:
            return 0.4
        return 0.0
