"""Evidence-Aware Region Grouping (Phase R3.2).

Groups adjacent geometric regions into candidate structural regions
based on spatial relationships, alignment, and evidence similarity.

Does NOT merge incompatible structures (top_bar with content, side_rail
with main content, etc.).

Output: list of RegionGroup, each containing merged bounds + child region IDs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.perception.geometric_partitioner import GeometricRegion


# Minimum overlap ratio to consider regions as adjacent
_ADJACENCY_THRESHOLD = 0.02  # 2% of region perimeter

# Maximum gap between regions to still consider them groupable (as ratio of window)
_MAX_GAP_RATIO = 0.03  # 3% of window dimension


@dataclass
class RegionGroup:
    """A group of geometric regions that share structural characteristics."""

    group_id: str
    child_region_ids: list[str] = field(default_factory=list)
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)
    band: str = "center"  # "top", "bottom", "left", "right", "center"
    alignment: str = "none"  # "horizontal", "vertical", "none"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "child_region_ids": list(self.child_region_ids),
            "bounds": list(self.bounds),
            "band": self.band,
            "alignment": self.alignment,
            "reason": self.reason,
        }


class RegionGrouper:
    """Group adjacent geometric regions by evidence-aware rules."""

    def group(
        self,
        geometric_regions: list[GeometricRegion],
        window_width: int,
        window_height: int,
        ocr_blocks: list[dict[str, Any]] | None = None,
        vision_candidates: list[dict[str, Any]] | None = None,
    ) -> list[RegionGroup]:
        """Group geometric regions into structural candidates."""
        if not geometric_regions or window_width <= 0 or window_height <= 0:
            return []

        # Build region index with computed properties
        region_props = self._compute_region_properties(
            geometric_regions, window_width, window_height, ocr_blocks, vision_candidates
        )

        # Build adjacency graph
        adjacency = self._build_adjacency_graph(geometric_regions, region_props, window_width, window_height)

        # Find groups via connected components with compatibility checks
        groups = self._find_groups(geometric_regions, region_props, adjacency, window_width, window_height)

        return groups

    def _compute_region_properties(
        self,
        regions: list[GeometricRegion],
        window_width: int,
        window_height: int,
        ocr_blocks: list[dict[str, Any]] | None,
        vision_candidates: list[dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        """Compute properties for each region."""
        props: dict[str, dict[str, Any]] = {}
        for gr in regions:
            l, t, r, b = gr.bounds
            rw = max(r - l, 1)
            rh = max(b - t, 1)
            cx, cy = (l + r) // 2, (t + b) // 2

            rel_top = t / window_height
            rel_bottom = b / window_height
            rel_left = l / window_width
            rel_right = r / window_width

            # Band classification (grouping semantics — uses rel_bottom for top)
            from src.perception.position_band import classify_grouping_band
            band = classify_grouping_band(gr.bounds, window_width, window_height)

            # OCR count in this region
            ocr_count = 0
            ocr_centers_y: list[float] = []
            ocr_centers_x: list[float] = []
            if ocr_blocks:
                for block in ocr_blocks:
                    bbox = block.get("bbox", [])
                    if len(bbox) >= 4:
                        bcx, bcy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                        if l <= bcx <= r and t <= bcy <= b:
                            ocr_count += 1
                            ocr_centers_x.append(bcx)
                            ocr_centers_y.append(bcy)

            # Vision count in this region
            vision_count = 0
            if vision_candidates:
                for vc in vision_candidates:
                    bbox = vc.get("bounding_rect") or vc.get("bbox", [])
                    if len(bbox) >= 4:
                        bcx, bcy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                        if l <= bcx <= r and t <= bcy <= b:
                            vision_count += 1

            # Horizontal alignment score (OCR blocks in same row)
            h_score = 0.0
            if len(ocr_centers_y) >= 2:
                mean_y = sum(ocr_centers_y) / len(ocr_centers_y)
                std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ocr_centers_y) / len(ocr_centers_y))
                h_score = max(0, 1.0 - std_y / max(rh, 1))

            # Vertical alignment score (OCR blocks in same column)
            v_score = 0.0
            if len(ocr_centers_x) >= 2:
                mean_x = sum(ocr_centers_x) / len(ocr_centers_x)
                std_x = math.sqrt(sum((x - mean_x) ** 2 for x in ocr_centers_x) / len(ocr_centers_x))
                v_score = max(0, 1.0 - std_x / max(rw, 1))

            props[gr.region_id] = {
                "bounds": gr.bounds,
                "center": (cx, cy),
                "band": band,
                "rel_top": rel_top,
                "rel_bottom": rel_bottom,
                "rel_left": rel_left,
                "rel_right": rel_right,
                "width": rw,
                "height": rh,
                "aspect": rw / rh,
                "ocr_count": ocr_count,
                "vision_count": vision_count,
                "h_align_score": h_score,
                "v_align_score": v_score,
                "has_separator": any("separator" in e for e in gr.boundary_evidence),
            }
        return props

    def _build_adjacency_graph(
        self,
        regions: list[GeometricRegion],
        props: dict[str, dict[str, Any]],
        window_width: int,
        window_height: int,
    ) -> dict[str, set[str]]:
        """Build adjacency graph: which regions are near each other."""
        adjacency: dict[str, set[str]] = {gr.region_id: set() for gr in regions}
        max_gap_x = window_width * _MAX_GAP_RATIO
        max_gap_y = window_height * _MAX_GAP_RATIO

        for i, gr_a in enumerate(regions):
            pa = props[gr_a.region_id]
            al, at, ar, ab = pa["bounds"]
            for j in range(i + 1, len(regions)):
                gr_b = regions[j]
                pb = props[gr_b.region_id]
                bl, bt, br, bb = pb["bounds"]

                # Check adjacency: gap between regions < threshold
                gap_x = max(0, max(al - br, bl - ar))
                gap_y = max(0, max(at - bb, bt - ab))

                if gap_x <= max_gap_x and gap_y <= max_gap_y:
                    adjacency[gr_a.region_id].add(gr_b.region_id)
                    adjacency[gr_b.region_id].add(gr_a.region_id)

        return adjacency

    def _find_groups(
        self,
        regions: list[GeometricRegion],
        props: dict[str, dict[str, Any]],
        adjacency: dict[str, set[str]],
        window_width: int,
        window_height: int,
    ) -> list[RegionGroup]:
        """Find groups via BFS with compatibility checks."""
        visited: set[str] = set()
        groups: list[RegionGroup] = []
        group_counter = 0

        for gr in regions:
            if gr.region_id in visited:
                continue

            # BFS to find connected compatible regions
            queue = [gr.region_id]
            group_members: list[str] = []
            while queue:
                rid = queue.pop(0)
                if rid in visited:
                    continue
                visited.add(rid)
                group_members.append(rid)

                for neighbor_id in adjacency.get(rid, set()):
                    if neighbor_id not in visited:
                        if self._are_compatible(rid, neighbor_id, props):
                            queue.append(neighbor_id)

            # Create group
            if len(group_members) == 1:
                # Single region — keep as-is
                p = props[group_members[0]]
                groups.append(RegionGroup(
                    group_id=f"G{group_counter}",
                    child_region_ids=group_members,
                    bounds=p["bounds"],
                    band=p["band"],
                    alignment="none",
                    reason="single_region",
                ))
            else:
                # Multiple regions — merge bounds and determine group properties
                merged_bounds = self._merge_bounds([props[rid]["bounds"] for rid in group_members])
                group_band = self._determine_group_band(group_members, props)
                group_alignment = self._determine_group_alignment(group_members, props)
                reason = self._determine_group_reason(group_members, props, group_alignment)

                groups.append(RegionGroup(
                    group_id=f"G{group_counter}",
                    child_region_ids=group_members,
                    bounds=merged_bounds,
                    band=group_band,
                    alignment=group_alignment,
                    reason=reason,
                ))
            group_counter += 1

        return groups

    def _are_compatible(
        self,
        rid_a: str,
        rid_b: str,
        props: dict[str, dict[str, Any]],
    ) -> bool:
        """Check if two regions can be grouped together."""
        pa = props[rid_a]
        pb = props[rid_b]

        # Rule 1: Don't merge if a strong separator exists between them
        if pa.get("has_separator") and pb.get("has_separator"):
            return False

        # Rule 2: Same band → compatible
        if pa["band"] == pb["band"] and pa["band"] != "center":
            return True

        # Rule 3: Both center → check alignment
        if pa["band"] == "center" and pb["band"] == "center":
            # Check if they share horizontal or vertical alignment
            h_compat = self._are_horizontally_aligned(pa, pb)
            v_compat = self._are_vertically_aligned(pa, pb)
            return h_compat or v_compat

        # Rule 4: Adjacent bands with alignment → compatible
        # (e.g., two top regions side by side)
        if pa["band"] in ("top", "bottom") and pb["band"] in ("top", "bottom"):
            return True
        if pa["band"] in ("left", "right") and pb["band"] in ("left", "right"):
            return True

        # Rule 5: Different structural bands → NOT compatible
        # (top vs bottom, left vs right, etc.)
        return False

    def _are_horizontally_aligned(self, pa: dict, pb: dict) -> bool:
        """Check if two regions are in the same horizontal row."""
        _, ay, _, aby = pa["bounds"]
        _, by, _, bby = pb["bounds"]
        a_cy = (ay + aby) / 2
        b_cy = (by + bby) / 2
        a_h = pa["height"]
        b_h = pb["height"]
        avg_h = (a_h + b_h) / 2
        return abs(a_cy - b_cy) < avg_h * 0.8

    def _are_vertically_aligned(self, pa: dict, pb: dict) -> bool:
        """Check if two regions are in the same vertical column."""
        ax, _, arx, _ = pa["bounds"]
        bx, _, brx, _ = pb["bounds"]
        a_cx = (ax + arx) / 2
        b_cx = (bx + brx) / 2
        a_w = pa["width"]
        b_w = pb["width"]
        avg_w = (a_w + b_w) / 2
        return abs(a_cx - b_cx) < avg_w * 0.8

    def _merge_bounds(self, bounds_list: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
        """Merge multiple bounds into one encompassing bound."""
        lefts = [b[0] for b in bounds_list]
        tops = [b[1] for b in bounds_list]
        rights = [b[2] for b in bounds_list]
        bottoms = [b[3] for b in bounds_list]
        return (min(lefts), min(tops), max(rights), max(bottoms))

    def _determine_group_band(self, members: list[str], props: dict[str, dict[str, Any]]) -> str:
        """Determine the band for a group of regions."""
        bands = [props[rid]["band"] for rid in members]
        # Majority vote
        from collections import Counter
        counter = Counter(bands)
        return counter.most_common(1)[0][0]

    def _determine_group_alignment(self, members: list[str], props: dict[str, dict[str, Any]]) -> str:
        """Determine if group is horizontally or vertically aligned."""
        if len(members) < 2:
            return "none"

        h_scores = [props[rid]["h_align_score"] for rid in members]
        v_scores = [props[rid]["v_align_score"] for rid in members]
        avg_h = sum(h_scores) / len(h_scores)
        avg_v = sum(v_scores) / len(v_scores)

        if avg_h > 0.5 and avg_h > avg_v:
            return "horizontal"
        if avg_v > 0.5 and avg_v > avg_h:
            return "vertical"
        return "none"

    def _determine_group_reason(self, members: list[str], props: dict[str, dict[str, Any]], alignment: str) -> str:
        """Generate a human-readable reason for the grouping."""
        bands = set(props[rid]["band"] for rid in members)
        total_ocr = sum(props[rid]["ocr_count"] for rid in members)
        total_vision = sum(props[rid]["vision_count"] for rid in members)

        parts = [f"{len(members)}regions"]
        if len(bands) == 1:
            parts.append(f"band={bands.pop()}")
        if alignment != "none":
            parts.append(f"align={alignment}")
        if total_ocr > 0:
            parts.append(f"ocr={total_ocr}")
        if total_vision > 0:
            parts.append(f"vision={total_vision}")
        return ", ".join(parts)
