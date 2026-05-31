"""Pixel-based visual layout diagnostics.

This module intentionally emits semantic-only region evidence. It does not
create actionable controls and must not be used directly for clicks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VisualRegionCandidate:
    region_id: str
    role_hint: str
    bounds: tuple[int, int, int, int]
    geometry_confidence: float
    evidence: list[str] = field(default_factory=list)
    source: str = "visual_layout"
    actionability: str = "semantic_only"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bounds"] = list(self.bounds)
        return data


class VisualLayoutSegmenter:
    """Detect coarse visual regions from screenshot pixels.

    The first pass is deliberately conservative: separators, large low-texture
    panes, and bordered input-like rectangles. These are diagnostic hints only.
    """

    def __init__(
        self,
        *,
        min_region_area_ratio: float = 0.006,
        separator_threshold: float = 10.0,
    ) -> None:
        self.min_region_area_ratio = min_region_area_ratio
        self.separator_threshold = separator_threshold

    def segment(self, image: Image.Image | None) -> list[VisualRegionCandidate]:
        if image is None:
            return []
        rgb = image.convert("RGB")
        arr = np.asarray(rgb).astype(np.int16)
        height, width = arr.shape[:2]
        if width <= 1 or height <= 1:
            return []

        regions: list[VisualRegionCandidate] = []
        regions.extend(self._detect_vertical_separators(arr))
        regions.extend(self._detect_horizontal_separators(arr))
        regions.extend(self._detect_app_shell_regions(arr))
        regions.extend(self._detect_bordered_rectangles(arr))
        regions.extend(self._detect_large_low_texture_blocks(arr))
        regions.extend(self._detect_bottom_input_strips(arr))
        regions.extend(self._detect_content_like_regions(arr))
        return self._dedupe_and_clip(regions, width, height)

    def _detect_vertical_separators(self, arr: np.ndarray) -> list[VisualRegionCandidate]:
        height, width = arr.shape[:2]
        col_mean = arr.mean(axis=0)
        diff = np.abs(col_mean[1:] - col_mean[:-1]).mean(axis=1)
        candidates = self._runs(diff >= self.separator_threshold)
        regions: list[VisualRegionCandidate] = []
        for idx, (start, end) in enumerate(candidates):
            x1 = max(0, start - 1)
            x2 = min(width, end + 2)
            if x2 - x1 > max(8, width * 0.05):
                continue
            score = min(0.95, float(diff[start:end].mean() if end > start else diff[start]) / 35.0)
            regions.append(
                VisualRegionCandidate(
                    region_id=f"visual_v_separator_{idx}",
                    role_hint="separator",
                    bounds=(x1, 0, x2, height),
                    geometry_confidence=max(0.45, score),
                    evidence=["vertical_separator", "color_transition"],
                )
            )
        return regions

    def _detect_horizontal_separators(self, arr: np.ndarray) -> list[VisualRegionCandidate]:
        height, width = arr.shape[:2]
        row_mean = arr.mean(axis=1)
        diff = np.abs(row_mean[1:] - row_mean[:-1]).mean(axis=1)
        candidates = self._runs(diff >= self.separator_threshold)
        regions: list[VisualRegionCandidate] = []
        for idx, (start, end) in enumerate(candidates):
            y1 = max(0, start - 1)
            y2 = min(height, end + 2)
            if y2 - y1 > max(8, height * 0.05):
                continue
            score = min(0.95, float(diff[start:end].mean() if end > start else diff[start]) / 35.0)
            regions.append(
                VisualRegionCandidate(
                    region_id=f"visual_h_separator_{idx}",
                    role_hint="separator",
                    bounds=(0, y1, width, y2),
                    geometry_confidence=max(0.45, score),
                    evidence=["horizontal_separator", "color_transition"],
                )
            )
        return regions

    def _detect_app_shell_regions(self, arr: np.ndarray) -> list[VisualRegionCandidate]:
        """Detect common app-shell geometry: side rail, top bands, card grids, floating actions.

        This is intentionally geometry-only. It names structural shapes, not app semantics.
        """
        height, width = arr.shape[:2]
        gray = arr.mean(axis=2)
        regions: list[VisualRegionCandidate] = []

        row_mean = arr.mean(axis=1)
        row_diff = np.abs(row_mean[1:] - row_mean[:-1]).mean(axis=1)
        title_candidates = np.where(row_diff[: max(2, int(height * 0.12))] >= self.separator_threshold)[0]
        title_bottom = int(title_candidates[0] + 1) if title_candidates.size else 0

        col_mean = arr.mean(axis=0)
        col_diff = np.abs(col_mean[1:] - col_mean[:-1]).mean(axis=1)
        vertical_edge = np.abs(arr[:, 1:] - arr[:, :-1]).mean(axis=2)
        vertical_scan = vertical_edge[max(0, title_bottom):, :]
        vertical_coverage = (vertical_scan >= 8.0).mean(axis=0) if vertical_scan.size else np.zeros_like(col_diff)
        partition_candidates = self._partition_positions(
            col_diff,
            vertical_coverage,
            min_position=int(width * 0.035),
            max_position=int(width * 0.46),
            coverage_threshold=0.62,
            prefer_coverage=False,
        )
        left_candidates = [x for x in partition_candidates if x <= int(width * 0.24)]
        preferred_left_candidates = [
            x for x in left_candidates
            if x >= max(44, int(width * 0.045))
        ]
        side_right = preferred_left_candidates[0] if preferred_left_candidates else (left_candidates[0] if left_candidates else 0)
        if side_right >= max(36, int(width * 0.045)):
            regions.append(
                VisualRegionCandidate(
                    region_id="visual_shell_side_panel_0",
                    role_hint="side_panel_region",
                    bounds=(0, max(0, title_bottom), side_right, height),
                    geometry_confidence=0.74,
                    evidence=["left_vertical_band", "app_shell_partition"],
                )
            )

        list_right = 0
        if side_right:
            min_list_width = max(120, int(width * 0.12))
            max_list_right = int(width * 0.46)
            list_candidates = [
                x
                for x in partition_candidates
                if side_right + min_list_width <= x <= max_list_right
            ]
            if list_candidates:
                list_right = list_candidates[0]
                regions.append(
                    VisualRegionCandidate(
                        region_id="visual_shell_list_panel_0",
                        role_hint="list_panel_region",
                        bounds=(side_right, max(0, title_bottom), list_right, height),
                        geometry_confidence=0.70,
                        evidence=["middle_vertical_band", "app_shell_partition"],
                    )
                )

        content_left = list_right or side_right or 0
        top_start = max(0, title_bottom)
        horizontal_edge = np.abs(arr[1:] - arr[:-1]).mean(axis=2)
        horizontal_scan = horizontal_edge[:, max(0, content_left):]
        horizontal_coverage = (
            (horizontal_scan >= 8.0).mean(axis=1)
            if horizontal_scan.size
            else np.zeros_like(row_diff)
        )
        horizontal_candidates = self._partition_positions(
            row_diff,
            horizontal_coverage,
            min_position=top_start + 8,
            max_position=int(height * 0.28),
            coverage_threshold=0.55,
            prefer_coverage=True,
        )
        toolbar_bottom = horizontal_candidates[0] if horizontal_candidates else min(height, top_start + max(48, int(height * 0.09)))
        tab_bottom = (
            horizontal_candidates[1]
            if len(horizontal_candidates) > 1
            else toolbar_bottom
        )
        if content_left < width - 80 and toolbar_bottom - top_start >= 24:
            regions.append(
                VisualRegionCandidate(
                    region_id="visual_shell_toolbar_0",
                    role_hint="toolbar_like_region",
                    bounds=(content_left, top_start, width, toolbar_bottom),
                    geometry_confidence=0.58,
                    evidence=["top_horizontal_band", "app_shell_partition"],
                )
            )
        if len(horizontal_candidates) > 1 and content_left < width - 80 and tab_bottom - toolbar_bottom >= 20:
            regions.append(
                VisualRegionCandidate(
                    region_id="visual_shell_tab_bar_0",
                    role_hint="tab_bar_like_region",
                    bounds=(content_left, toolbar_bottom, width, tab_bottom),
                    geometry_confidence=0.62,
                    evidence=["top_tab_band", "app_shell_partition"],
                )
            )

        bottom_input = (
            self._detect_bright_bottom_input_region(arr, content_left=content_left)
            if content_left >= max(36, int(width * 0.04))
            else None
        )
        content_bottom = bottom_input.bounds[1] if bottom_input is not None else height
        if content_left < width - 80 and content_bottom - tab_bottom >= max(120, int(height * 0.18)):
            regions.append(
                VisualRegionCandidate(
                    region_id="visual_shell_content_0",
                    role_hint="content_like_region",
                    bounds=(content_left, tab_bottom, width, content_bottom),
                    geometry_confidence=0.54,
                    evidence=["main_content_band", "app_shell_partition"],
                )
            )
        if bottom_input is not None:
            regions.append(bottom_input)

        card_grid = self._detect_repeated_card_grid(arr, content_left, tab_bottom)
        if card_grid is not None:
            regions.append(card_grid)

        floating = self._detect_floating_action(arr)
        if floating is not None:
            regions.append(floating)

        return regions

    def _partition_positions(
        self,
        mean_diff: np.ndarray,
        coverage: np.ndarray,
        *,
        min_position: int,
        max_position: int,
        coverage_threshold: float,
        prefer_coverage: bool,
    ) -> list[int]:
        """Return region boundary positions, favoring long boundaries over local edges."""
        if mean_diff.size == 0:
            return []
        max_index = min(mean_diff.size, max(0, max_position))
        min_index = max(0, min_position - 1)
        if min_index >= max_index:
            return []
        coverage = coverage[: mean_diff.size] if coverage.size >= mean_diff.size else np.pad(coverage, (0, mean_diff.size - coverage.size))
        coverage_mask = coverage[min_index:max_index] >= coverage_threshold
        if prefer_coverage and coverage_mask.any():
            active = coverage_mask
        else:
            active = coverage_mask | (mean_diff[min_index:max_index] >= self.separator_threshold)
        positions: list[int] = []
        for start, end in self._runs(active):
            abs_start = min_index + start
            abs_end = min_index + end
            scores = coverage[abs_start:abs_end] * 100.0 + np.minimum(mean_diff[abs_start:abs_end], 40.0)
            best_offset = int(np.argmax(scores)) if scores.size else 0
            positions.append(abs_start + best_offset + 1)
        return sorted(set(positions))

    def _detect_bright_bottom_input_region(
        self,
        arr: np.ndarray,
        *,
        content_left: int = 0,
    ) -> VisualRegionCandidate | None:
        """Detect large bright bottom input surfaces inside the main content area."""
        height, width = arr.shape[:2]
        gray = arr.mean(axis=2)
        x_start = max(0, min(width - 1, content_left))
        y_start = int(height * 0.54)
        crop = gray[y_start:, x_start:]
        if crop.size == 0:
            return None
        threshold = max(248.0, float(np.percentile(crop, 82)))
        mask = crop >= threshold
        components = self._connected_components(mask)
        best: tuple[int, int, int, int, int] | None = None
        for x1, y1, x2, y2, count in components:
            box_w = x2 - x1
            box_h = y2 - y1
            abs_x1 = x_start + x1
            abs_y1 = y_start + y1
            abs_x2 = x_start + x2
            abs_y2 = y_start + y2
            if abs_y1 < height * 0.58:
                continue
            if not (width * 0.38 <= box_w <= width * 0.92):
                continue
            if not (36 <= box_h <= height * 0.22):
                continue
            fill_ratio = count / max(box_w * box_h, 1)
            if fill_ratio < 0.45:
                continue
            if best is None or (box_w * box_h, abs_y1) > ((best[2] - best[0]) * (best[3] - best[1]), best[1]):
                best = (abs_x1, abs_y1, abs_x2, abs_y2, count)
        if best is None:
            return None
        x1, y1, x2, y2, _count = best
        row_mean = arr.mean(axis=1)
        row_diff = np.abs(row_mean[1:] - row_mean[:-1]).mean(axis=1)
        search_start = max(0, y1 - 80, int(height * 0.78))
        boundary_candidates = [
            int(y + 1)
            for y in np.where(row_diff[search_start:y1] >= max(6.0, self.separator_threshold * 0.55))[0].tolist()
        ]
        if boundary_candidates:
            y1 = search_start + boundary_candidates[0]
        return VisualRegionCandidate(
            region_id="visual_shell_bottom_input_0",
            role_hint="bottom_input_strip",
            bounds=(x1, y1, x2, y2),
            geometry_confidence=0.58,
            evidence=["bottom_bright_input_surface", "app_shell_partition"],
        )

    def _detect_repeated_card_grid(
        self,
        arr: np.ndarray,
        content_left: int,
        content_top: int,
    ) -> VisualRegionCandidate | None:
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
            if x2 <= content_left or y2 <= content_top:
                continue
            if not (int(width * 0.12) <= box_w <= int(width * 0.45)):
                continue
            if not (45 <= box_h <= int(height * 0.22)):
                continue
            density = count / max(box_w * box_h, 1)
            if density > 0.45:
                continue
            boxes.append((x1, y1, x2, y2))

        if len(boxes) < 4:
            return None

        boxes.sort(key=lambda b: (b[1], b[0]))
        left = min(b[0] for b in boxes)
        top = min(b[1] for b in boxes)
        right = max(b[2] for b in boxes)
        bottom = max(b[3] for b in boxes)
        if (right - left) * (bottom - top) < width * height * 0.12:
            return None
        return VisualRegionCandidate(
            region_id="visual_shell_card_grid_0",
            role_hint="card_grid_region",
            bounds=(left, top, right, bottom),
            geometry_confidence=min(0.86, 0.52 + len(boxes) * 0.035),
            evidence=["repeated_bordered_cards", f"card_count:{len(boxes)}"],
        )

    def _detect_floating_action(self, arr: np.ndarray) -> VisualRegionCandidate | None:
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
            return None
        x1, y1, x2, y2, _count = best
        return VisualRegionCandidate(
            region_id="visual_shell_floating_0",
            role_hint="floating_region",
            bounds=(x_start + x1, y_start + y1, x_start + x2, y_start + y2),
            geometry_confidence=0.70,
            evidence=["bottom_right_floating_component", "high_contrast_blob"],
        )

    def _detect_bordered_rectangles(self, arr: np.ndarray) -> list[VisualRegionCandidate]:
        height, width = arr.shape[:2]
        gray = arr.mean(axis=2)
        # Compare against the dominant light UI background. This catches subtle
        # borders while ignoring pure white empty areas.
        background = float(np.percentile(gray, 92))
        edge_mask = (background - gray) > 12
        components = self._connected_components(edge_mask)
        regions: list[VisualRegionCandidate] = []
        min_area = width * height * self.min_region_area_ratio
        for idx, (x1, y1, x2, y2, count) in enumerate(components):
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w <= 8 or box_h <= 8:
                continue
            if box_w * box_h < min_area:
                continue
            if count / max(box_w * box_h, 1) > 0.65:
                # Filled color block, not just a bordered field.
                continue
            aspect = box_w / max(box_h, 1)
            bottom_half = y1 >= height * 0.45
            looks_input = aspect >= 2.2 and box_h <= height * 0.35
            role = "input_like_region" if looks_input and bottom_half else "bordered_region"
            evidence = ["bordered_rectangle"]
            if looks_input:
                evidence.append("input_like_aspect")
            regions.append(
                VisualRegionCandidate(
                    region_id=f"visual_bordered_{idx}",
                    role_hint=role,
                    bounds=(x1, y1, x2, y2),
                    geometry_confidence=0.72 if looks_input else 0.55,
                    evidence=evidence,
                )
            )
        return regions

    def _detect_large_low_texture_blocks(self, arr: np.ndarray) -> list[VisualRegionCandidate]:
        height, width = arr.shape[:2]
        gray = arr.mean(axis=2)
        block_w = max(24, width // 16)
        block_h = max(24, height // 12)
        regions: list[VisualRegionCandidate] = []
        idx = 0
        for y in range(0, height, block_h):
            for x in range(0, width, block_w):
                x2 = min(width, x + block_w)
                y2 = min(height, y + block_h)
                block = gray[y:y2, x:x2]
                if block.size == 0:
                    continue
                if float(block.std()) <= 4.0 and (x2 - x) * (y2 - y) >= width * height * 0.01:
                    regions.append(
                        VisualRegionCandidate(
                            region_id=f"visual_low_texture_{idx}",
                            role_hint="low_texture_area",
                            bounds=(x, y, x2, y2),
                            geometry_confidence=0.35,
                            evidence=["low_texture_area"],
                        )
                    )
                    idx += 1
        return regions[:24]

    def _detect_bottom_input_strips(self, arr: np.ndarray) -> list[VisualRegionCandidate]:
        """Detect bottom input strips for borderless input areas (chat apps, etc.).

        Heuristics (geometric role_hint only, not application semantics):
        - Located in the bottom 18% of the image
        - Width >= 40% of window width
        - Height in 28-90px range
        - Internal texture variance higher than pure background
        - A horizontal separator or clear boundary above the strip strengthens the signal
        """
        height, width = arr.shape[:2]
        bright_region = self._detect_bright_bottom_input_region(arr)
        if bright_region is not None:
            return [bright_region]
        gray = arr.mean(axis=2)
        bottom_cutoff = int(height * 0.82)
        min_strip_width = int(width * 0.40)
        regions: list[VisualRegionCandidate] = []
        idx = 0

        scan_start = max(0, min(bottom_cutoff, height - 120))
        scan_end = max(scan_start + 1, height - 27)
        for y_start in range(scan_start, scan_end):
            strip_height = min(90, height - y_start)
            if strip_height < 28:
                continue
            strip = gray[y_start:y_start + strip_height, :]
            if strip.size == 0:
                continue
            row_std = np.std(strip, axis=1)
            textured_rows = int(np.sum(row_std > 6.0))
            texture_ratio = textured_rows / max(strip_height, 1)

            if texture_ratio < 0.18:
                continue

            non_bg_cols = np.sum(np.abs(strip - float(np.median(strip))) > 8, axis=0)
            active_span = int(np.sum(non_bg_cols > (strip_height * 0.08)))
            if active_span < min_strip_width:
                continue

            col_active = (non_bg_cols > (strip_height * 0.08)).astype(np.int32)
            runs = self._runs(col_active)
            for run_start, run_end in runs:
                run_width = run_end - run_start
                if run_width < min_strip_width:
                    continue
                x1 = max(0, run_start - 4)
                x2 = min(width, run_end + 4)
                y1 = max(0, y_start - 4)
                y2 = min(height, y_start + strip_height + 4)
                evidence = ["bottom_strip_texture", "horizontal_band"]
                sep_above = any(
                    vr.role_hint == "separator"
                    and vr.bounds[1] >= y1 - 12
                    and vr.bounds[3] <= y1 + 6
                    and vr.bounds[2] - vr.bounds[0] >= width * 0.3
                    for vr in regions
                )
                if sep_above:
                    evidence.append("separator_above")
                regions.append(
                    VisualRegionCandidate(
                        region_id=f"visual_bottom_strip_{idx}",
                        role_hint="bottom_input_strip",
                        bounds=(x1, y1, x2, y2),
                        geometry_confidence=0.42 if sep_above else 0.32,
                        evidence=evidence,
                    )
                )
                idx += 1
            break
        return regions[:3]

    def _detect_content_like_regions(self, arr: np.ndarray) -> list[VisualRegionCandidate]:
        """Detect large content-like regions as structural geometry hints.

        Covers regions with moderate texture variance that occupy substantial
        portions of the window — typical of message streams, document views,
        file lists, and card grids. These are geometry-only hints; final
        semantics (message_stream / document / list / grid) come from fusion + OCR + VLM.
        """
        height, width = arr.shape[:2]
        gray = arr.mean(axis=2)
        min_area = width * height * 0.08
        regions: list[VisualRegionCandidate] = []
        idx = 0

        block_h = max(40, height // 8)
        block_w = max(60, width // 6)

        for y in range(0, height - block_h, block_h // 2):
            for x in range(0, width - block_w, block_w // 2):
                x2 = min(width, x + block_w)
                y2 = min(height, y + block_h)
                block = gray[y:y2, x:x2]
                if block.size == 0:
                    continue
                area = (x2 - x) * (y2 - y)
                if area < min_area:
                    continue
                std_val = float(block.std())
                if 6.0 <= std_val <= 42.0:
                    right_half = x >= width * 0.45
                    role = "content_like_region"
                    evidence = ["moderate_texture_block"]
                    if right_half:
                        evidence.append("right_panel_placement")
                    regions.append(
                        VisualRegionCandidate(
                            region_id=f"visual_content_{idx}",
                            role_hint=role,
                            bounds=(x, y, x2, y2),
                            geometry_confidence=0.38,
                            evidence=evidence,
                        )
                    )
                    idx += 1
        return regions[:12]

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

    def _dedupe_and_clip(
        self,
        regions: list[VisualRegionCandidate],
        width: int,
        height: int,
    ) -> list[VisualRegionCandidate]:
        kept: list[VisualRegionCandidate] = []
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
            normalized = VisualRegionCandidate(
                region_id=region.region_id,
                role_hint=region.role_hint,
                bounds=clipped,
                geometry_confidence=region.geometry_confidence,
                evidence=list(region.evidence),
                source=region.source,
                actionability=region.actionability,
            )
            if any(self._iou(normalized.bounds, existing.bounds) > 0.9 for existing in kept):
                continue
            kept.append(normalized)
        return kept
