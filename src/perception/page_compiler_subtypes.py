"""Subtype-specific region builders for InteractionCanvasEngine."""

from __future__ import annotations

from typing import Any, Callable

from src.perception.page_compiler_models import (
    ContentAreaSubtype,
    Region,
    SemanticRole,
)
from src.perception.geometric_partitioner import GeometricRegion


class InteractionCanvasEngineSubtypeBuilder:
    """Build subtype-specific child regions and subtype evidence."""

    def __init__(
        self,
        infer_semantic_role: Callable[[str, str, str | None], SemanticRole],
    ) -> None:
        self._infer_semantic_role = infer_semantic_role

    def build_chat_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        inferred_region_map: dict[str, str],
        geometric_regions: list[GeometricRegion] | None = None,
    ) -> list[Region]:
        _, content_top, _, content_bottom = content_region.bounds or (0, 0, 0, 0)
        content_height = max(content_bottom - content_top, 1)
        composer_threshold = content_top + int(content_height * 0.72)
        top_strip_threshold = content_top + int(content_height * 0.20)

        composer_items: list[tuple[str, tuple[int, int, int, int]]] = []
        message_items: list[tuple[str, tuple[int, int, int, int]]] = []
        bottom_band_items: list[tuple[str, tuple[int, int, int, int]]] = []

        for element_id, element, bounds in content_elements:
            left, top, right, bottom = bounds
            center_y = (top + bottom) // 2
            control_type = (element.get("control_type") or "").lower()
            text = (
                ((element.get("text") or "") + " " + (element.get("name") or ""))
                .lower()
            )
            semantic_role = self._infer_semantic_role(
                element.get("control_type", ""),
                element.get("text", "") or element.get("name", ""),
                element.get("name"),
            )
            if center_y <= top_strip_threshold and semantic_role in (
                SemanticRole.TAB,
                SemanticRole.MENU_ITEM,
                SemanticRole.NAV_ITEM,
                SemanticRole.SEARCH_INPUT,
                SemanticRole.TOOLBAR,
            ):
                continue
            in_composer_band = center_y >= composer_threshold
            looks_like_composer = semantic_role in (
                SemanticRole.MESSAGE_INPUT,
                SemanticRole.TEXT_INPUT,
                SemanticRole.SEARCH_INPUT,
                SemanticRole.SEND_BUTTON,
                SemanticRole.SUBMIT_BUTTON,
            ) or "edit" in control_type or "button" in control_type or "input" in text or "发送" in text
            if in_composer_band:
                bottom_band_items.append((element_id, bounds))
            target_items = (
                composer_items if in_composer_band and looks_like_composer else message_items
            )
            target_items.append((element_id, bounds))

        if not composer_items and bottom_band_items:
            composer_items = [
                (element_id, bounds)
                for element_id, bounds in bottom_band_items
                if element_id not in {item_id for item_id, _bounds in message_items}
            ]

        if not message_items:
            return []
        if not composer_items:
            composer_bounds = (
                content_region.bounds[0],
                composer_threshold,
                content_region.bounds[2],
                content_region.bounds[3],
            )
            composer_region = Region(
                region_id="region_content_composer_0",
                role="composer_area",
                subtype=ContentAreaSubtype.UNKNOWN,
                parent_region_id=content_region.region_id,
                bounds=composer_bounds,
                element_ids=[],
                attributes={"source": "chat_fallback_geometry"},
            )
        else:
            composer_region = self.make_child_region(
                region_id="region_content_composer_0",
                role="composer_area",
                parent_region=content_region,
                items=composer_items,
            )

        message_region = self.make_child_region(
            region_id="region_content_messages_0",
            role="message_stream",
            parent_region=content_region,
            items=message_items,
        )
        for element_id, _bounds in composer_items:
            inferred_region_map[element_id] = composer_region.region_id
        for element_id, _bounds in message_items:
            inferred_region_map[element_id] = message_region.region_id
        return [message_region, composer_region]

    def build_form_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        inferred_region_map: dict[str, str],
    ) -> list[Region]:
        _, content_top, _, content_bottom = content_region.bounds or (0, 0, 0, 0)
        content_height = max(content_bottom - content_top, 1)
        action_threshold = content_top + int(content_height * 0.70)

        field_items: list[tuple[str, tuple[int, int, int, int]]] = []
        action_items: list[tuple[str, tuple[int, int, int, int]]] = []

        for element_id, element, bounds in content_elements:
            _, top, _, bottom = bounds
            center_y = (top + bottom) // 2
            semantic_role = self._infer_semantic_role(
                element.get("control_type", ""),
                element.get("text", "") or element.get("name", ""),
                element.get("name"),
            )
            if center_y >= action_threshold and semantic_role in (
                SemanticRole.SUBMIT_BUTTON,
                SemanticRole.CANCEL_BUTTON,
                SemanticRole.BUTTON,
            ):
                action_items.append((element_id, bounds))
            else:
                field_items.append((element_id, bounds))

        if not action_items or not field_items:
            return []

        fields_region = self.make_child_region(
            region_id="region_content_form_fields_0",
            role="form_fields",
            parent_region=content_region,
            items=field_items,
        )
        actions_region = self.make_child_region(
            region_id="region_content_form_actions_0",
            role="form_actions",
            parent_region=content_region,
            items=action_items,
        )
        for element_id, _bounds in field_items:
            inferred_region_map[element_id] = fields_region.region_id
        for element_id, _bounds in action_items:
            inferred_region_map[element_id] = actions_region.region_id
        return [fields_region, actions_region]

    def build_list_detail_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        inferred_region_map: dict[str, str],
    ) -> list[Region]:
        left, _, right, _ = content_region.bounds or (0, 0, 0, 0)
        split_x = left + int((right - left) * 0.40)

        list_items: list[tuple[str, tuple[int, int, int, int]]] = []
        detail_items: list[tuple[str, tuple[int, int, int, int]]] = []

        for element_id, _element, bounds in content_elements:
            elem_left, _, elem_right, _ = bounds
            center_x = (elem_left + elem_right) // 2
            if center_x <= split_x:
                list_items.append((element_id, bounds))
            else:
                detail_items.append((element_id, bounds))

        if not list_items or not detail_items:
            return []

        list_region = self.make_child_region(
            region_id="region_content_list_0",
            role="list_panel",
            parent_region=content_region,
            items=list_items,
        )
        detail_region = self.make_child_region(
            region_id="region_content_detail_0",
            role="detail_panel",
            parent_region=content_region,
            items=detail_items,
        )
        for element_id, _bounds in list_items:
            inferred_region_map[element_id] = list_region.region_id
        for element_id, _bounds in detail_items:
            inferred_region_map[element_id] = detail_region.region_id
        return [list_region, detail_region]

    def build_editor_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        inferred_region_map: dict[str, str],
    ) -> list[Region]:
        toolbar_items, side_items, viewport_items, action_items = (
            self.partition_workspace_regions(content_region, content_elements)
        )
        return self.materialize_partitioned_regions(
            content_region,
            inferred_region_map,
            toolbar_items=toolbar_items,
            side_items=side_items,
            viewport_items=viewport_items,
            action_items=action_items,
        )

    def build_dashboard_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        inferred_region_map: dict[str, str],
    ) -> list[Region]:
        toolbar_items, side_items, viewport_items, action_items = (
            self.partition_workspace_regions(
                content_region,
                content_elements,
                promote_filters=True,
            )
        )
        filter_items = self.extract_filter_items(content_region, content_elements)
        regions = self.materialize_partitioned_regions(
            content_region,
            inferred_region_map,
            toolbar_items=toolbar_items,
            filter_items=filter_items,
            side_items=side_items,
            viewport_items=viewport_items,
            action_items=action_items,
        )
        self.annotate_dashboard_regions(regions, content_elements)
        return regions

    def build_grid_table_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        inferred_region_map: dict[str, str],
    ) -> list[Region]:
        toolbar_items, side_items, viewport_items, action_items = (
            self.partition_workspace_regions(
                content_region,
                content_elements,
                grid_priority=True,
            )
        )
        filter_items = self.extract_filter_items(content_region, content_elements)
        regions = self.materialize_partitioned_regions(
            content_region,
            inferred_region_map,
            toolbar_items=toolbar_items,
            filter_items=filter_items,
            side_items=side_items,
            viewport_items=viewport_items,
            action_items=action_items,
        )
        self.annotate_grid_table_regions(regions, content_elements)
        return regions

    def build_canvas_doc_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        inferred_region_map: dict[str, str],
    ) -> list[Region]:
        toolbar_items, side_items, viewport_items, action_items = (
            self.partition_workspace_regions(
                content_region,
                content_elements,
                doc_viewer_priority=True,
            )
        )
        return self.materialize_partitioned_regions(
            content_region,
            inferred_region_map,
            toolbar_items=toolbar_items,
            side_items=side_items,
            viewport_items=viewport_items,
            action_items=action_items,
        )

    def partition_workspace_regions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        promote_filters: bool = False,
        grid_priority: bool = False,
        doc_viewer_priority: bool = False,
    ) -> tuple[
        list[tuple[str, tuple[int, int, int, int]]],
        list[tuple[str, tuple[int, int, int, int]]],
        list[tuple[str, tuple[int, int, int, int]]],
        list[tuple[str, tuple[int, int, int, int]]],
    ]:
        content_left, content_top, content_right, content_bottom = (
            content_region.bounds or (0, 0, 0, 0)
        )
        content_width = max(content_right - content_left, 1)
        content_height = max(content_bottom - content_top, 1)
        toolbar_items: list[tuple[str, tuple[int, int, int, int]]] = []
        side_items: list[tuple[str, tuple[int, int, int, int]]] = []
        viewport_items: list[tuple[str, tuple[int, int, int, int]]] = []
        action_items: list[tuple[str, tuple[int, int, int, int]]] = []

        for element_id, element, bounds in content_elements:
            left, top, right, bottom = bounds
            width = max(right - left, 1)
            height = max(bottom - top, 1)
            center_y = (top + bottom) // 2
            semantic_role = self._infer_semantic_role(
                element.get("control_type", ""),
                element.get("text", "") or element.get("name", ""),
                element.get("name"),
            )
            control_type = (element.get("control_type") or "").lower()
            text_name = f"{element.get('name', '')} {element.get('text', '')}".lower()

            if center_y <= content_top + int(content_height * 0.20) and (
                semantic_role
                in {
                    SemanticRole.TAB,
                    SemanticRole.MENU_ITEM,
                    SemanticRole.NAV_ITEM,
                    SemanticRole.TOOLBAR,
                }
                or any(token in control_type for token in ("toolbar", "menu", "tab"))
            ):
                toolbar_items.append((element_id, bounds))
                continue

            if (
                width <= int(content_width * 0.24)
                and height >= int(content_height * 0.28)
                and (
                    any(token in control_type for token in ("list", "tree", "pane"))
                    or semantic_role
                    in {
                        SemanticRole.LIST_ITEM,
                        SemanticRole.TREE_ITEM,
                        SemanticRole.SIDEBAR,
                    }
                )
            ):
                side_items.append((element_id, bounds))
                continue

            if center_y >= content_bottom - int(content_height * 0.18) and (
                semantic_role
                in {
                    SemanticRole.BUTTON,
                    SemanticRole.SUBMIT_BUTTON,
                    SemanticRole.CANCEL_BUTTON,
                    SemanticRole.TOGGLE_BUTTON,
                }
                or any(
                    token in text_name
                    for token in ("next", "prev", "zoom", "apply", "save")
                )
            ):
                action_items.append((element_id, bounds))
                continue

            if (
                width
                >= int(
                    content_width
                    * (
                        0.45
                        if grid_priority
                        else 0.30
                        if promote_filters
                        else 0.50
                    )
                )
                and height
                >= int(
                    content_height
                    * (
                        0.35
                        if doc_viewer_priority
                        else 0.18
                        if promote_filters
                        else 0.28
                    )
                )
                and (
                    any(
                        token in control_type
                        for token in (
                            "document",
                            "custom",
                            "pane",
                            "image",
                            "data",
                            "grid",
                            "table",
                            "edit",
                            "list",
                        )
                    )
                    or semantic_role
                    in {
                        SemanticRole.TEXT_INPUT,
                        SemanticRole.LIST_ITEM,
                        SemanticRole.TEXT,
                        SemanticRole.IMAGE,
                    }
                )
            ):
                viewport_items.append((element_id, bounds))
                continue

            if promote_filters and center_y <= content_top + int(content_height * 0.22):
                if "filter" in text_name or "search" in text_name:
                    toolbar_items.append((element_id, bounds))

        return toolbar_items, side_items, viewport_items, action_items

    def extract_filter_items(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        content_top = (content_region.bounds or (0, 0, 0, 0))[1]
        content_height = max((content_region.bounds or (0, 0, 0, 1))[3] - content_top, 1)
        filter_items: list[tuple[str, tuple[int, int, int, int]]] = []
        for element_id, element, bounds in content_elements:
            _, top, _, bottom = bounds
            center_y = (top + bottom) // 2
            text_name = f"{element.get('name', '')} {element.get('text', '')}".lower()
            control_type = (element.get("control_type") or "").lower()
            if center_y <= content_top + int(content_height * 0.22) and (
                "filter" in text_name or "search" in text_name or "combo" in control_type
            ):
                filter_items.append((element_id, bounds))
        return filter_items

    def materialize_partitioned_regions(
        self,
        content_region: Region,
        inferred_region_map: dict[str, str],
        toolbar_items: list[tuple[str, tuple[int, int, int, int]]] | None = None,
        filter_items: list[tuple[str, tuple[int, int, int, int]]] | None = None,
        side_items: list[tuple[str, tuple[int, int, int, int]]] | None = None,
        viewport_items: list[tuple[str, tuple[int, int, int, int]]] | None = None,
        action_items: list[tuple[str, tuple[int, int, int, int]]] | None = None,
    ) -> list[Region]:
        regions: list[Region] = []
        definitions = [
            ("toolbar", "region_content_toolbar_0", toolbar_items or []),
            ("filter_bar", "region_content_filter_bar_0", filter_items or []),
            ("side_panel", "region_content_side_panel_0", side_items or []),
            ("viewport", "region_content_viewport_0", viewport_items or []),
            ("action_bar", "region_content_action_bar_0", action_items or []),
        ]
        for role, region_id, items in definitions:
            if not items:
                continue
            region = self.make_child_region(
                region_id=region_id,
                role=role,
                parent_region=content_region,
                items=items,
            )
            regions.append(region)
            for element_id, _bounds in items:
                inferred_region_map[element_id] = region.region_id
        return regions

    def annotate_dashboard_regions(
        self,
        regions: list[Region],
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
    ) -> None:
        viewport_region = next((region for region in regions if region.role == "viewport"), None)
        if viewport_region is None or viewport_region.bounds is None:
            return
        left, _top, right, _bottom = viewport_region.bounds
        width = max(right - left, 1)
        widget_groups: dict[str, list[str]] = {"left": [], "center": [], "right": []}
        widget_like_ids: list[str] = []
        for element_id, element, bounds in content_elements:
            if element_id not in viewport_region.element_ids:
                continue
            elem_left, elem_top, elem_right, elem_bottom = bounds
            elem_width = max(elem_right - elem_left, 1)
            elem_height = max(elem_bottom - elem_top, 1)
            control_type = (element.get("control_type") or "").lower()
            if elem_width < int(width * 0.18) or elem_height < 100:
                continue
            if not any(token in control_type for token in ("pane", "custom", "image", "document", "list")):
                continue
            widget_like_ids.append(element_id)
            center_x = (elem_left + elem_right) // 2
            if center_x <= left + int(width * 0.33):
                widget_groups["left"].append(element_id)
            elif center_x >= left + int(width * 0.66):
                widget_groups["right"].append(element_id)
            else:
                widget_groups["center"].append(element_id)
        widget_groups = {key: value for key, value in widget_groups.items() if value}
        viewport_region.attributes["widget_groups"] = widget_groups
        viewport_region.attributes["widget_group_count"] = len(widget_groups)
        viewport_region.attributes["widget_like_ids"] = widget_like_ids

    def annotate_grid_table_regions(
        self,
        regions: list[Region],
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
    ) -> None:
        viewport_region = next((region for region in regions if region.role == "viewport"), None)
        action_region = next((region for region in regions if region.role == "action_bar"), None)
        if viewport_region is None or viewport_region.bounds is None:
            return
        left, top, right, bottom = viewport_region.bounds
        height = max(bottom - top, 1)
        header_threshold = top + int(height * 0.16)
        pagination_threshold = bottom - int(height * 0.16)
        header_ids: list[str] = []
        row_ids: list[str] = []
        pagination_ids: list[str] = []
        for element_id, element, bounds in content_elements:
            center_x = (bounds[0] + bounds[2]) // 2
            center_y = (bounds[1] + bounds[3]) // 2
            if not (left <= center_x <= right and top <= center_y <= bottom):
                continue
            control_type = (element.get("control_type") or "").lower()
            text_name = f"{element.get('name', '')} {element.get('text', '')}".lower()
            if center_y <= header_threshold and (
                any(token in control_type for token in ("header", "text", "custom", "data", "grid", "table"))
                or any(token in text_name for token in ("name", "status", "date", "type", "sort"))
            ):
                header_ids.append(element_id)
            elif center_y >= pagination_threshold and (
                any(token in text_name for token in ("page", "next", "prev", "rows", "1/", "2/"))
                or any(token in control_type for token in ("button", "text"))
            ):
                pagination_ids.append(element_id)
            else:
                row_ids.append(element_id)
        viewport_region.attributes["grid_header_ids"] = header_ids
        viewport_region.attributes["grid_row_ids"] = row_ids
        viewport_region.attributes["grid_pagination_ids"] = pagination_ids
        viewport_region.attributes["row_group_count"] = self.estimate_row_group_count(
            row_ids,
            content_elements,
        )
        if action_region is not None:
            action_region.attributes["secondary_actions"] = [
                element_id
                for element_id in pagination_ids
                if element_id in action_region.element_ids
            ]

    def estimate_row_group_count(
        self,
        row_ids: list[str],
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
    ) -> int:
        if not row_ids:
            return 0
        tops: list[int] = []
        for element_id, _element, bounds in content_elements:
            if element_id in row_ids:
                tops.append(bounds[1])
        if not tops:
            return 0
        tops.sort()
        groups = 1
        last_top = tops[0]
        for top in tops[1:]:
            if abs(top - last_top) > 36:
                groups += 1
                last_top = top
        return groups

    def make_child_region(
        self,
        region_id: str,
        role: str,
        parent_region: Region,
        items: list[tuple[str, tuple[int, int, int, int]]],
    ) -> Region:
        rects = [item[1] for item in items]
        bounds = (
            min(rect[0] for rect in rects),
            min(rect[1] for rect in rects),
            max(rect[2] for rect in rects),
            max(rect[3] for rect in rects),
        )
        return Region(
            region_id=region_id,
            role=role,
            subtype=ContentAreaSubtype.UNKNOWN,
            parent_region_id=parent_region.region_id,
            bounds=bounds,
            element_ids=[item[0] for item in items],
        )
