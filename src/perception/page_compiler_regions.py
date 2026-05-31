"""Region assembly helpers for InteractionCanvasEngine."""

from __future__ import annotations

from typing import Any, Callable

from src.perception.geometric_partitioner import GeometricRegion
from src.perception.page_compiler_models import ContentAreaSubtype, Region, SemanticRole, SurfaceType
from src.perception.page_compiler_subtypes import InteractionCanvasEngineSubtypeBuilder
from src.perception.zone_partitioner import WindowZoneStructure, ZoneType


class InteractionCanvasEngineRegionBuilder:
    """Build layout-first regions and child-region structures."""

    def __init__(
        self,
        infer_semantic_role: Callable[[str, str, str | None], SemanticRole],
        get_element_bounds: Callable[[dict[str, Any]], tuple[int, int, int, int] | None],
        coerce_bbox: Callable[[Any], tuple[int, int, int, int] | None],
        subtype_builder: InteractionCanvasEngineSubtypeBuilder,
    ) -> None:
        self._infer_semantic_role = infer_semantic_role
        self._get_element_bounds = get_element_bounds
        self._coerce_bbox = coerce_bbox
        self._subtype_builder = subtype_builder

    def zone_type_to_region_id(self, zone_type: str) -> str:
        mapping = {
            "title_bar": "region_title_bar",
            "menu_bar": "region_menu_bar",
            "tool_bar": "region_tool_bar",
            "side_bar": "region_side_bar",
            "content_area": "region_content_0",
            "status_bar": "region_status_bar",
            "unknown": "region_unknown",
        }
        return mapping.get(zone_type, f"region_{zone_type}")

    def infer_layout_regions(
        self,
        raw_elements: list[dict[str, Any]],
        content_subtype: ContentAreaSubtype,
        vision_layout_regions: list[dict[str, Any]] | None = None,
        geometric_regions: list[GeometricRegion] | None = None,
        fusion_diagnostics: dict[str, Any] | None = None,
    ) -> tuple[list[Region], dict[str, str]]:
        classified: dict[str, list[tuple[str, tuple[int, int, int, int]]]] = {
            "title_bar": [],
            "tool_bar": [],
            "side_bar": [],
            "content_area": [],
            "status_bar": [],
        }
        all_rects: list[tuple[int, int, int, int]] = []
        for index, element in enumerate(raw_elements):
            bounds = self._get_element_bounds(element)
            if not bounds:
                continue
            all_rects.append(bounds)
            classified["content_area"].append((element.get("element_id", f"elem_{index}"), bounds))

        if not all_rects:
            return [
                Region(
                    region_id="region_content_0",
                    role="content_area",
                    subtype=content_subtype,
                    bounds=None,
                    element_ids=[],
                )
            ], {}

        window_left = min(rect[0] for rect in all_rects)
        window_top = min(rect[1] for rect in all_rects)
        window_right = max(rect[2] for rect in all_rects)
        window_bottom = max(rect[3] for rect in all_rects)
        window_width = max(window_right - window_left, 1)
        window_height = max(window_bottom - window_top, 1)
        toolbar_band = min(max(int(window_height * 0.22), 96), 220)
        status_band = min(max(int(window_height * 0.10), 40), 120)
        side_band = min(max(int(window_width * 0.22), 180), 420)

        classified = {key: [] for key in classified}
        inferred_region_map: dict[str, str] = {}
        for index, element in enumerate(raw_elements):
            bounds = self._get_element_bounds(element)
            if not bounds:
                continue
            element_id = element.get("element_id", f"elem_{index}")
            left, top, right, bottom = bounds
            width = max(right - left, 1)
            height = max(bottom - top, 1)
            center_x = left + width // 2
            center_y = top + height // 2
            control_type = (element.get("control_type") or "").lower()
            text_name = f"{element.get('name', '')} {element.get('text', '')}".lower()

            role = "content_area"
            if "title" in control_type or "titlebar" in control_type or "title" in text_name:
                role = "title_bar"
            elif (
                "status" in control_type
                or "statusbar" in text_name
                or (
                    center_y >= window_bottom - status_band
                    and width >= int(window_width * 0.55)
                    and height <= max(56, int(window_height * 0.10))
                    and "button" not in control_type
                    and "edit" not in control_type
                )
            ):
                role = "status_bar"
            elif (
                (center_x <= window_left + side_band or center_x >= window_right - side_band)
                and width <= int(window_width * 0.40)
                and height >= int(window_height * 0.35)
                and any(token in text_name for token in ("sidebar", "channel", "contact"))
            ):
                role = "side_bar"
            elif (
                center_y <= window_top + toolbar_band
                and height <= max(64, int(window_height * 0.12))
                and (
                    any(token in control_type for token in ("toolbar", "menu", "tab"))
                    or any(token in text_name for token in ("toolbar", "menu", "tools", "search"))
                )
                and width >= int(window_width * 0.68)
            ):
                role = "tool_bar"
            classified[role].append((element_id, bounds))

        if not classified["content_area"]:
            largest = max(
                (
                    (element.get("element_id", f"elem_{index}"), self._get_element_bounds(element))
                    for index, element in enumerate(raw_elements)
                    if self._get_element_bounds(element) is not None
                ),
                key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
            )
            classified["content_area"].append(largest)
            for role_name in ("title_bar", "tool_bar", "side_bar", "status_bar"):
                classified[role_name] = [item for item in classified[role_name] if item[0] != largest[0]]

        regions: list[Region] = []
        for role in ["title_bar", "tool_bar", "side_bar", "content_area", "status_bar"]:
            items = classified[role]
            if not items:
                continue
            rects = [item[1] for item in items]
            bounds = (
                min(rect[0] for rect in rects),
                min(rect[1] for rect in rects),
                max(rect[2] for rect in rects),
                max(rect[3] for rect in rects),
            )
            region_id = self.zone_type_to_region_id(role)
            subtype = content_subtype if role == "content_area" else ContentAreaSubtype.UNKNOWN
            element_ids = [item[0] for item in items]
            regions.append(Region(region_id=region_id, role=role, subtype=subtype, bounds=bounds, element_ids=element_ids))
            for element_id in element_ids:
                inferred_region_map[element_id] = region_id

        if "region_content_0" not in {region.region_id for region in regions}:
            regions.append(
                Region(
                    region_id="region_content_0",
                    role="content_area",
                    subtype=content_subtype,
                    bounds=(window_left, window_top, window_right, window_bottom),
                    element_ids=[],
                )
            )

        regions = self.add_content_subregions(
            regions, raw_elements, content_subtype, inferred_region_map,
            vision_layout_regions or [], geometric_regions, fusion_diagnostics,
        )
        return regions, inferred_region_map

    def add_content_subregions(
        self,
        regions: list[Region],
        raw_elements: list[dict[str, Any]],
        content_subtype: ContentAreaSubtype,
        inferred_region_map: dict[str, str],
        vision_layout_regions: list[dict[str, Any]],
        geometric_regions: list[GeometricRegion] | None = None,
        fusion_diagnostics: dict[str, Any] | None = None,
    ) -> list[Region]:
        content_region = next((region for region in regions if region.region_id == "region_content_0"), None)
        if content_region is None:
            return regions
        element_lookup = {element.get("element_id", f"elem_{index}"): element for index, element in enumerate(raw_elements)}
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]] = []
        for element_id in content_region.element_ids:
            element = element_lookup.get(element_id)
            if element is None:
                continue
            bounds = self._get_element_bounds(element)
            if bounds is not None:
                content_elements.append((element_id, element, bounds))
        if content_region.bounds is None:
            return regions
        if not content_elements:
            return regions

        effective_subtype = self.resolve_structural_content_subtype(content_subtype, content_elements, content_region.bounds)
        content_region.subtype = effective_subtype
        dialog_regions = self.build_dialog_subregions(content_region, content_elements, [])
        dialog_dominant = self.is_dialog_layout_dominant(dialog_regions, content_elements)

        child_regions: list[Region] = []
        if not dialog_dominant and effective_subtype == ContentAreaSubtype.CHAT:
            child_regions = self._subtype_builder.build_chat_subregions(
                content_region, content_elements, inferred_region_map, geometric_regions,
            )
        elif not dialog_dominant and effective_subtype == ContentAreaSubtype.FORM:
            child_regions = self._subtype_builder.build_form_subregions(content_region, content_elements, inferred_region_map)
        elif not dialog_dominant and effective_subtype == ContentAreaSubtype.LIST_DETAIL:
            child_regions = self._subtype_builder.build_list_detail_subregions(content_region, content_elements, inferred_region_map)
        elif not dialog_dominant and effective_subtype == ContentAreaSubtype.EDITOR:
            child_regions = self._subtype_builder.build_editor_subregions(content_region, content_elements, inferred_region_map)
        elif not dialog_dominant and effective_subtype == ContentAreaSubtype.DASHBOARD:
            child_regions = self._subtype_builder.build_dashboard_subregions(content_region, content_elements, inferred_region_map)
        elif not dialog_dominant and effective_subtype == ContentAreaSubtype.GRID_TABLE:
            child_regions = self._subtype_builder.build_grid_table_subregions(content_region, content_elements, inferred_region_map)
        elif not dialog_dominant and effective_subtype == ContentAreaSubtype.CANVAS_DOC_VIEWER:
            child_regions = self._subtype_builder.build_canvas_doc_subregions(content_region, content_elements, inferred_region_map)

        child_regions = self.merge_candidate_regions(child_regions, dialog_regions)
        child_regions = self.merge_candidate_regions(
            child_regions,
            self.build_vision_guided_regions(content_region, content_elements, vision_layout_regions, inferred_region_map),
        )
        standard_regions = self.build_standard_content_regions(content_region, content_elements, child_regions)
        child_regions.extend([region for region in standard_regions if region.region_id not in {item.region_id for item in child_regions}])
        if not child_regions:
            return regions
        child_regions = self.order_child_regions(child_regions)
        self.refresh_child_region_assignments(content_region, child_regions, inferred_region_map)
        content_region.child_region_ids = [region.region_id for region in child_regions]
        return regions + child_regions

    def build_standard_content_regions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        existing_regions: list[Region],
    ) -> list[Region]:
        existing_roles = {region.role for region in existing_regions}
        regions: list[Region] = []
        _, content_top, _, content_bottom = content_region.bounds or (0, 0, 0, 0)
        content_height = max(content_bottom - content_top, 1)
        content_left, _, content_right, _ = content_region.bounds or (0, 0, 0, 0)
        content_width = max(content_region.bounds[2] - content_region.bounds[0], 1)
        toolbar_items: list[tuple[str, tuple[int, int, int, int]]] = []
        filter_items: list[tuple[str, tuple[int, int, int, int]]] = []
        action_items: list[tuple[str, tuple[int, int, int, int]]] = []
        side_items: list[tuple[str, tuple[int, int, int, int]]] = []
        viewport_items: list[tuple[str, tuple[int, int, int, int]]] = []
        dialog_items: list[tuple[str, tuple[int, int, int, int]]] = []

        for element_id, element, bounds in content_elements:
            left, top, right, bottom = bounds
            width = max(right - left, 1)
            height = max(bottom - top, 1)
            center_y = (top + bottom) // 2
            center_x = (left + right) // 2
            semantic_role = self._infer_semantic_role(
                element.get("control_type", ""),
                element.get("text", "") or element.get("name", ""),
                element.get("name"),
            )
            control_type = (element.get("control_type") or "").lower()
            text_name = f"{element.get('name', '')} {element.get('text', '')}".lower()
            edge_distance = min(abs(center_x - content_left), abs(content_right - center_x))
            is_button_like = semantic_role in {
                SemanticRole.BUTTON,
                SemanticRole.SEND_BUTTON,
                SemanticRole.SUBMIT_BUTTON,
                SemanticRole.CANCEL_BUTTON,
                SemanticRole.TOGGLE_BUTTON,
                SemanticRole.ICON_BUTTON,
                SemanticRole.TAB,
                SemanticRole.MENU_ITEM,
                SemanticRole.NAV_ITEM,
            }
            is_input_like = semantic_role in {
                SemanticRole.MESSAGE_INPUT,
                SemanticRole.TEXT_INPUT,
                SemanticRole.SEARCH_INPUT,
            }
            is_list_like = semantic_role in {
                SemanticRole.LIST_ITEM,
                SemanticRole.CHAT_ITEM,
                SemanticRole.TREE_ITEM,
                SemanticRole.SIDEBAR,
            }
            is_top_band = center_y <= content_top + int(content_height * 0.20)
            is_bottom_band = center_y >= content_bottom - int(content_height * 0.22)

            if is_top_band and (
                semantic_role in {SemanticRole.TAB, SemanticRole.MENU_ITEM, SemanticRole.TOOLBAR, SemanticRole.NAV_ITEM}
                or any(token in control_type for token in ("toolbar", "menu", "tab"))
            ):
                toolbar_items.append((element_id, bounds))
            if is_top_band and (
                semantic_role == SemanticRole.SEARCH_INPUT
                or "filter" in text_name
                or "search" in text_name
            ):
                filter_items.append((element_id, bounds))
            if is_bottom_band and is_button_like:
                action_items.append((element_id, bounds))
            if (
                width <= int((content_region.bounds[2] - content_region.bounds[0]) * 0.35)
                and height >= int(content_height * 0.30)
                and (is_list_like or any(token in control_type for token in ("list", "tree", "pane")))
            ):
                side_items.append((element_id, bounds))
            if (
                width >= int((content_region.bounds[2] - content_region.bounds[0]) * 0.45)
                and height >= int(content_height * 0.35)
                and not is_top_band
                and (
                    any(token in control_type for token in ("pane", "document", "custom", "list"))
                    or (not is_input_like and not is_button_like and edge_distance >= int((content_right - content_left) * 0.12))
                )
            ):
                viewport_items.append((element_id, bounds))
            if (
                (
                    (
                        int(content_width * 0.50) <= width <= int(content_width * 0.82)
                        and int(content_height * 0.30) <= height <= int(content_height * 0.80)
                        and abs(center_x - (content_region.bounds[0] + content_width // 2)) <= int(content_width * 0.18)
                    )
                    or any(token in text_name for token in ("dialog", "modal", "settings", "confirm", "alert"))
                )
                and any(token in control_type for token in ("pane", "window", "custom", "document"))
                and (
                    any(token in text_name for token in ("dialog", "modal", "settings", "confirm", "alert"))
                    or (width <= int(content_width * 0.82) and height <= int(content_height * 0.82))
                )
            ):
                dialog_items.append((element_id, bounds))

        for role, region_id, items in [
            ("toolbar", "region_content_toolbar_0", toolbar_items),
            ("filter_bar", "region_content_filter_bar_0", filter_items),
            ("action_bar", "region_content_action_bar_0", action_items),
            ("side_panel", "region_content_side_panel_0", side_items),
            ("viewport", "region_content_viewport_0", viewport_items),
            ("dialog_body", "region_content_dialog_body_0", dialog_items),
        ]:
            if role in existing_roles or not items:
                continue
            regions.append(
                self._subtype_builder.make_child_region(
                    region_id=region_id,
                    role=role,
                    parent_region=content_region,
                    items=items,
                )
            )
        return regions

    def build_dialog_subregions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        existing_regions: list[Region],
    ) -> list[Region]:
        existing_roles = {region.role for region in existing_regions}
        if "dialog_body" in existing_roles or content_region.bounds is None:
            return []
        content_left, content_top, content_right, content_bottom = content_region.bounds
        content_width = max(content_right - content_left, 1)
        content_height = max(content_bottom - content_top, 1)
        center_x = content_left + content_width // 2
        candidates: list[tuple[str, tuple[int, int, int, int]]] = []
        for element_id, element, bounds in content_elements:
            left, top, right, bottom = bounds
            width = max(right - left, 1)
            height = max(bottom - top, 1)
            elem_center_x = (left + right) // 2
            control_type = (element.get("control_type") or "").lower()
            text_name = f"{element.get('name', '')} {element.get('text', '')}".lower()
            if (
                int(content_width * 0.45) <= width <= int(content_width * 1.0)
                and int(content_height * 0.25) <= height <= int(content_height * 1.0)
                and abs(elem_center_x - center_x) <= int(content_width * 0.18)
                and (
                    any(token in control_type for token in ("pane", "window", "custom", "document"))
                    or any(token in text_name for token in ("dialog", "modal", "settings", "confirm", "alert"))
                )
                and (
                    any(token in text_name for token in ("dialog", "modal", "settings", "confirm", "alert"))
                    or (width <= int(content_width * 0.92) and height <= int(content_height * 0.92))
                )
            ):
                candidates.append((element_id, bounds))

        if not candidates:
            return []
        dialog_host_id, dialog_bounds = max(candidates, key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]))
        dialog_member_ids = self.select_region_members_from_bounds(content_elements, dialog_bounds)
        if dialog_host_id not in dialog_member_ids:
            dialog_member_ids.append(dialog_host_id)
        member_button_count = sum(
            1
            for element_id, element, _bounds in content_elements
            if element_id in dialog_member_ids
            and self._infer_semantic_role(
                element.get("control_type", ""),
                element.get("text", "") or element.get("name", ""),
                element.get("name"),
            ) in {
                SemanticRole.BUTTON,
                SemanticRole.SUBMIT_BUTTON,
                SemanticRole.CANCEL_BUTTON,
                SemanticRole.SEND_BUTTON,
                SemanticRole.TOGGLE_BUTTON,
            }
        )
        host_text = next(
            (
                f"{element.get('name', '')} {element.get('text', '')}".lower()
                for element_id, element, _bounds in content_elements
                if element_id == dialog_host_id
            ),
            "",
        )
        if len(dialog_member_ids) < 2 or (member_button_count == 0 and not any(token in host_text for token in ("dialog", "modal", "settings", "confirm", "alert"))):
            return []
        dialog_region = Region(
            region_id="region_content_dialog_body_0",
            role="dialog_body",
            subtype=ContentAreaSubtype.UNKNOWN,
            parent_region_id=content_region.region_id,
            bounds=dialog_bounds,
            element_ids=dialog_member_ids,
            attributes={"source": "dialog_parser", "host_element_id": dialog_host_id},
        )
        self.annotate_dialog_region(dialog_region, content_elements)
        footer_items: list[tuple[str, tuple[int, int, int, int]]] = []
        footer_top = dialog_bounds[1] + int((dialog_bounds[3] - dialog_bounds[1]) * 0.72)
        for element_id, element, bounds in content_elements:
            if element_id not in dialog_member_ids:
                continue
            _, top, _, bottom = bounds
            center_y = (top + bottom) // 2
            semantic_role = self._infer_semantic_role(
                element.get("control_type", ""),
                element.get("text", "") or element.get("name", ""),
                element.get("name"),
            )
            if center_y >= footer_top and semantic_role in {
                SemanticRole.BUTTON,
                SemanticRole.SUBMIT_BUTTON,
                SemanticRole.CANCEL_BUTTON,
                SemanticRole.SEND_BUTTON,
                SemanticRole.TOGGLE_BUTTON,
            }:
                footer_items.append((element_id, bounds))
        regions = [dialog_region]
        if footer_items and "action_bar" not in existing_roles:
            regions.append(
                self._subtype_builder.make_child_region(
                    region_id="region_content_action_bar_0",
                    role="action_bar",
                    parent_region=content_region,
                    items=footer_items,
                )
            )
        return regions

    def annotate_dialog_region(
        self,
        dialog_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
    ) -> None:
        if dialog_region.bounds is None:
            return
        _, top, _, bottom = dialog_region.bounds
        height = max(bottom - top, 1)
        footer_threshold = top + int(height * 0.72)
        title_threshold = top + int(height * 0.18)
        field_ids: list[str] = []
        title_ids: list[str] = []
        viewport_ids: list[str] = []
        footer_ids: list[str] = []
        for element_id, element, bounds in content_elements:
            if element_id not in dialog_region.element_ids:
                continue
            _left, elem_top, _right, elem_bottom = bounds
            center_y = (elem_top + elem_bottom) // 2
            semantic_role = self._infer_semantic_role(
                element.get("control_type", ""),
                element.get("text", "") or element.get("name", ""),
                element.get("name"),
            )
            control_type = (element.get("control_type") or "").lower()
            if center_y <= title_threshold and (
                semantic_role in {SemanticRole.TEXT, SemanticRole.TAB, SemanticRole.NAV_ITEM}
                or any(token in control_type for token in ("text", "tab"))
            ):
                title_ids.append(element_id)
                continue
            if center_y >= footer_threshold and semantic_role in {
                SemanticRole.BUTTON,
                SemanticRole.SUBMIT_BUTTON,
                SemanticRole.CANCEL_BUTTON,
                SemanticRole.SEND_BUTTON,
                SemanticRole.TOGGLE_BUTTON,
            }:
                footer_ids.append(element_id)
                continue
            if semantic_role in {
                SemanticRole.TEXT_INPUT,
                SemanticRole.MESSAGE_INPUT,
                SemanticRole.SEARCH_INPUT,
                SemanticRole.PASSWORD_INPUT,
                SemanticRole.FILE_INPUT,
            } or "edit" in control_type:
                field_ids.append(element_id)
            else:
                viewport_ids.append(element_id)
        dialog_region.attributes["internal_roles"] = {
            "title": title_ids,
            "form_fields": field_ids,
            "viewport": viewport_ids,
            "footer_actions": footer_ids,
        }
        dialog_region.attributes["dialog_kind"] = "form_dialog" if field_ids else "viewer_dialog"
        dialog_region.attributes["field_count"] = len(field_ids)
        dialog_region.attributes["footer_action_count"] = len(footer_ids)

    def is_dialog_layout_dominant(
        self,
        dialog_regions: list[Region],
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
    ) -> bool:
        if not dialog_regions:
            return False
        dialog_region = next((region for region in dialog_regions if region.role == "dialog_body"), None)
        if dialog_region is None:
            return False
        dialog_ratio = len(dialog_region.element_ids) / max(len(content_elements), 1)
        source = dialog_region.attributes.get("source")
        host_name = str(dialog_region.attributes.get("host_element_id") or "")
        return dialog_ratio >= 0.5 or source == "dialog_parser" or bool(host_name)

    def order_child_regions(self, child_regions: list[Region]) -> list[Region]:
        role_order = {
            "toolbar": 0,
            "filter_bar": 1,
            "side_panel": 2,
            "list_panel": 3,
            "message_stream": 4,
            "form_fields": 5,
            "detail_panel": 6,
            "viewport": 7,
            "dialog_body": 8,
            "composer_area": 9,
            "form_actions": 10,
            "action_bar": 11,
            "geometry_region": 12,
        }
        return sorted(child_regions, key=lambda region: (role_order.get(region.role, 99), region.region_id))

    def refresh_child_region_assignments(
        self,
        content_region: Region,
        child_regions: list[Region],
        inferred_region_map: dict[str, str],
    ) -> None:
        priority_by_role = {
            "toolbar": 10,
            "filter_bar": 15,
            "side_panel": 20,
            "viewport": 25,
            "action_bar": 65,
            "list_panel": 40,
            "message_stream": 40,
            "form_fields": 45,
            "detail_panel": 50,
            "dialog_body": 60,
            "composer_area": 80,
            "form_actions": 85,
            "geometry_region": 30,
        }
        for element_id in content_region.element_ids:
            inferred_region_map[element_id] = content_region.region_id
        chosen_priority = {element_id: 0 for element_id in content_region.element_ids}
        for region in child_regions:
            priority = priority_by_role.get(region.role, 1)
            for element_id in region.element_ids:
                if priority >= chosen_priority.get(element_id, 0):
                    inferred_region_map[element_id] = region.region_id
                    chosen_priority[element_id] = priority

    def build_vision_guided_regions(
        self,
        content_region: Region,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        vision_layout_regions: list[dict[str, Any]],
        inferred_region_map: dict[str, str],
    ) -> list[Region]:
        if content_region.bounds is None or not vision_layout_regions:
            return []
        valid_roles = {
            "message_stream",
            "composer_area",
            "form_fields",
            "action_bar",
            "list_panel",
            "detail_panel",
            "toolbar",
            "viewport",
            "filter_bar",
            "side_panel",
            "dialog_body",
            "form_actions",
        }
        regions: list[Region] = []
        for index, item in enumerate(vision_layout_regions):
            role = self.normalize_region_role(item.get("role") or item.get("kind") or item.get("semantic_label"))
            bounds = self._coerce_bbox(item.get("bbox") or item.get("bounding_box"))
            if role not in valid_roles or bounds is None:
                continue
            clipped = self.clip_bbox_to_region(bounds, content_region.bounds)
            if clipped is None:
                continue
            member_ids = self.select_region_members_from_bounds(content_elements, clipped)
            if not member_ids:
                continue
            region_id = self.region_id_for_role(role, index)
            regions.append(
                Region(
                    region_id=region_id,
                    role=role,
                    subtype=ContentAreaSubtype.UNKNOWN,
                    parent_region_id=content_region.region_id,
                    bounds=clipped,
                    element_ids=member_ids,
                    attributes={
                        "source": "vision_layout",
                        "structure_evidence_score": item.get("structure_evidence_score"),
                        "vision_label": item.get("role") or item.get("kind") or item.get("semantic_label"),
                    },
                )
            )
            for element_id in member_ids:
                inferred_region_map[element_id] = region_id
        return regions

    def merge_candidate_regions(self, existing_regions: list[Region], preferred_regions: list[Region]) -> list[Region]:
        by_role = {region.role: region for region in existing_regions}
        ordered_roles = [region.role for region in existing_regions]
        for region in preferred_regions:
            if region.role == "geometry_region":
                continue
            current = by_role.get(region.role)
            if current is None:
                by_role[region.role] = region
                ordered_roles.append(region.role)
                continue
            if len(region.element_ids) >= len(current.element_ids):
                by_role[region.role] = region
        return [by_role[role] for role in ordered_roles if role in by_role]

    def region_id_for_role(self, role: str, index: int) -> str:
        mapping = {
            "message_stream": "region_content_messages_0",
            "composer_area": "region_content_composer_0",
            "form_fields": "region_content_form_fields_0",
            "form_actions": "region_content_form_actions_0",
            "action_bar": "region_content_action_bar_0",
            "list_panel": "region_content_list_0",
            "detail_panel": "region_content_detail_0",
            "toolbar": "region_content_toolbar_0",
            "viewport": "region_content_viewport_0",
            "filter_bar": "region_content_filter_bar_0",
            "side_panel": "region_content_side_panel_0",
            "dialog_body": "region_content_dialog_body_0",
        }
        return mapping.get(role, f"region_content_{role}_{index}")

    def normalize_region_role(self, value: Any) -> str | None:
        if value is None:
            return None
        raw = str(value).strip().lower()
        mapping = {
            "messages": "message_stream",
            "message_list": "message_stream",
            "composer": "composer_area",
            "input_bar": "composer_area",
            "fields": "form_fields",
            "actions": "action_bar",
            "filters": "filter_bar",
            "sidebar": "side_panel",
            "modal": "dialog_body",
            "dialog": "dialog_body",
            "detail": "detail_panel",
            "list": "list_panel",
        }
        return mapping.get(raw, raw)

    def clip_bbox_to_region(
        self,
        bbox: tuple[int, int, int, int],
        region_bounds: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int] | None:
        left = max(bbox[0], region_bounds[0])
        top = max(bbox[1], region_bounds[1])
        right = min(bbox[2], region_bounds[2])
        bottom = min(bbox[3], region_bounds[3])
        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom

    def select_region_members_from_bounds(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        region_bounds: tuple[int, int, int, int],
    ) -> list[str]:
        member_ids: list[str] = []
        for element_id, _element, bounds in content_elements:
            center_x = (bounds[0] + bounds[2]) // 2
            center_y = (bounds[1] + bounds[3]) // 2
            if region_bounds[0] <= center_x <= region_bounds[2] and region_bounds[1] <= center_y <= region_bounds[3]:
                member_ids.append(element_id)
        return member_ids

    def resolve_structural_content_subtype(
        self,
        content_subtype: ContentAreaSubtype,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> ContentAreaSubtype:
        has_message_stream = self.has_chat_stream_area(content_elements, content_bounds)
        has_chat_conversation = self.has_chat_conversation_layout(content_elements, content_bounds)
        has_form_actions = self.has_bottom_action_band(content_elements, content_bounds)
        has_split_detail = self.has_list_detail_split(content_elements, content_bounds)
        has_editor_shell = self.has_editor_shell(content_elements, content_bounds)
        has_grid_layout = self.has_grid_table_layout(content_elements, content_bounds)
        has_canvas_doc = self.has_canvas_doc_layout(content_elements, content_bounds)
        has_dashboard = self.has_dashboard_widget_layout(content_elements, content_bounds)
        input_like_count = sum(
            1
            for _element_id, element, _bounds in content_elements
            if "edit" in (element.get("control_type") or "").lower()
            or self._infer_semantic_role(
                element.get("control_type", ""),
                element.get("text", "") or element.get("name", ""),
                element.get("name"),
            )
            in {
                SemanticRole.MESSAGE_INPUT,
                SemanticRole.TEXT_INPUT,
                SemanticRole.SEARCH_INPUT,
            }
        )
        grid_like_count = sum(
            1
            for _element_id, element, _bounds in content_elements
            if any(token in (element.get("control_type") or "").lower() for token in ("data", "grid", "table"))
        )
        explicit_datagrid = any(
            "datagrid" in (element.get("control_type") or "").lower()
            for _element_id, element, _bounds in content_elements
        )

        if content_subtype in {
            ContentAreaSubtype.UNKNOWN,
            ContentAreaSubtype.LIST_DETAIL,
            ContentAreaSubtype.DASHBOARD,
        } and has_dashboard and not has_grid_layout and not has_chat_conversation:
            return ContentAreaSubtype.DASHBOARD
        if content_subtype in {
            ContentAreaSubtype.UNKNOWN,
            ContentAreaSubtype.FORM,
            ContentAreaSubtype.LIST_DETAIL,
            ContentAreaSubtype.EDITOR,
            ContentAreaSubtype.CHAT,
        } and (has_grid_layout or explicit_datagrid or (grid_like_count >= 1 and input_like_count >= 1)):
            return ContentAreaSubtype.GRID_TABLE
        if content_subtype in {
            ContentAreaSubtype.UNKNOWN,
            ContentAreaSubtype.CANVAS_DOC_VIEWER,
            ContentAreaSubtype.DASHBOARD,
            ContentAreaSubtype.CHAT,
            ContentAreaSubtype.LIST_DETAIL,
        } and has_canvas_doc and input_like_count == 0:
            return ContentAreaSubtype.CANVAS_DOC_VIEWER
        if content_subtype in {
            ContentAreaSubtype.UNKNOWN,
            ContentAreaSubtype.EDITOR,
        } and has_editor_shell and not has_split_detail:
            return ContentAreaSubtype.EDITOR
        if content_subtype == ContentAreaSubtype.CHAT and has_split_detail and not has_form_actions and not has_chat_conversation:
            return ContentAreaSubtype.LIST_DETAIL
        if content_subtype == ContentAreaSubtype.CHAT and not has_message_stream and has_form_actions:
            return ContentAreaSubtype.FORM
        if content_subtype == ContentAreaSubtype.UNKNOWN:
            if has_chat_conversation:
                return ContentAreaSubtype.CHAT
            if has_split_detail:
                return ContentAreaSubtype.LIST_DETAIL
            if has_message_stream:
                return ContentAreaSubtype.CHAT
            if has_form_actions:
                return ContentAreaSubtype.FORM
        if content_subtype in {
            ContentAreaSubtype.UNKNOWN,
            ContentAreaSubtype.LIST_DETAIL,
            ContentAreaSubtype.CANVAS_DOC_VIEWER,
        } and has_chat_conversation:
            return ContentAreaSubtype.CHAT
        return content_subtype

    def has_editor_shell(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> bool:
        content_left, content_top, content_right, content_bottom = content_bounds
        content_width = max(content_right - content_left, 1)
        content_height = max(content_bottom - content_top, 1)
        top_controls = 0
        main_editor_area = 0
        for _element_id, element, bounds in content_elements:
            left, top, right, bottom = bounds
            width = max(right - left, 1)
            height = max(bottom - top, 1)
            center_y = (top + bottom) // 2
            control_type = (element.get("control_type") or "").lower()
            if center_y <= content_top + int(content_height * 0.20) and (
                any(token in control_type for token in ("tab", "toolbar", "menu"))
                or "editor" in f"{element.get('name', '')} {element.get('text', '')}".lower()
            ):
                top_controls += 1
            if (
                width >= int(content_width * 0.55)
                and height >= int(content_height * 0.45)
                and any(token in control_type for token in ("document", "edit", "custom"))
            ):
                main_editor_area += 1
        return top_controls >= 1 and main_editor_area >= 1

    def has_grid_table_layout(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> bool:
        _content_left, content_top, _content_right, content_bottom = content_bounds
        content_height = max(content_bottom - content_top, 1)
        grid_like = 0
        top_filters = 0
        for _element_id, element, bounds in content_elements:
            _, top, _, bottom = bounds
            center_y = (top + bottom) // 2
            control_type = (element.get("control_type") or "").lower()
            text_name = f"{element.get('name', '')} {element.get('text', '')}".lower()
            if any(token in control_type for token in ("data", "grid", "table")):
                grid_like += 1
            if center_y <= content_top + int(content_height * 0.20) and (
                "search" in text_name or "filter" in text_name or "combo" in control_type
            ):
                top_filters += 1
        return grid_like >= 1 and top_filters >= 1

    def has_canvas_doc_layout(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> bool:
        content_left, _content_top, content_right, _content_bottom = content_bounds
        content_width = max(content_right - content_left, 1)
        large_viewer = 0
        side_strip = 0
        for _element_id, element, bounds in content_elements:
            left, top, right, bottom = bounds
            width = max(right - left, 1)
            height = max(bottom - top, 1)
            control_type = (element.get("control_type") or "").lower()
            text_name = f"{element.get('name', '')} {element.get('text', '')}".lower()
            if (
                width >= int(content_width * 0.60)
                and height >= 220
                and (
                    any(token in control_type for token in ("pane", "document", "image", "custom"))
                    or any(token in text_name for token in ("page", "preview", "viewer", "canvas", "document"))
                )
            ):
                large_viewer += 1
            if width <= int(content_width * 0.22) and any(token in control_type for token in ("list", "tree", "pane")):
                side_strip += 1
        return large_viewer >= 1 and side_strip >= 1

    def has_dashboard_widget_layout(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> bool:
        content_left, _content_top, content_right, _content_bottom = content_bounds
        content_width = max(content_right - content_left, 1)
        widget_count = 0
        unique_types: set[str] = set()
        for _element_id, element, bounds in content_elements:
            left, top, right, bottom = bounds
            width = max(right - left, 1)
            height = max(bottom - top, 1)
            control_type = (element.get("control_type") or "").lower()
            unique_types.add(control_type or "unknown")
            if width >= int(content_width * 0.20) and height >= 120:
                if any(token in control_type for token in ("pane", "custom", "image", "document", "list")):
                    widget_count += 1
        return widget_count >= 3 and len(unique_types) >= 4

    def has_chat_stream_area(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> bool:
        _, content_top, _, content_bottom = content_bounds
        content_height = max(content_bottom - content_top, 1)
        for _element_id, element, bounds in content_elements:
            left, top, right, bottom = bounds
            width = max(right - left, 1)
            height = max(bottom - top, 1)
            control_type = (element.get("control_type") or "").lower()
            if (
                top <= content_top + int(content_height * 0.20)
                and height >= int(content_height * 0.45)
                and width >= int((content_bounds[2] - content_bounds[0]) * 0.55)
                and any(token in control_type for token in ("pane", "list", "document", "custom"))
            ):
                return True
            semantic_role = self._infer_semantic_role(
                element.get("control_type", ""),
                element.get("text", "") or element.get("name", ""),
                element.get("name"),
            )
            center_y = (top + bottom) // 2
            if semantic_role in {SemanticRole.TEXT, SemanticRole.CHAT_ITEM, SemanticRole.LIST_ITEM} and (
                content_top + int(content_height * 0.10) <= center_y <= content_bottom - int(content_height * 0.18)
                and width >= int((content_bounds[2] - content_bounds[0]) * 0.18)
            ):
                return True
        return False

    def has_chat_conversation_layout(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> bool:
        left, top, right, bottom = content_bounds
        width = max(right - left, 1)
        height = max(bottom - top, 1)
        split_x = left + int(width * 0.36)
        left_count = 0
        right_count = 0
        bottom_input_like = 0
        header_like = 0
        for _element_id, element, bounds in content_elements:
            elem_left, elem_top, elem_right, elem_bottom = bounds
            center_x = (elem_left + elem_right) // 2
            center_y = (elem_top + elem_bottom) // 2
            text_name = f"{element.get('name', '')} {element.get('text', '')}"
            role = self._infer_semantic_role(
                element.get("control_type", ""),
                element.get("text", "") or element.get("name", ""),
                element.get("name"),
            )
            if center_x <= split_x and center_y >= top + int(height * 0.10):
                if role in {SemanticRole.CHAT_ITEM, SemanticRole.LIST_ITEM, SemanticRole.SEARCH_INPUT}:
                    left_count += 1
            else:
                if role in {
                    SemanticRole.TEXT,
                    SemanticRole.CHAT_ITEM,
                    SemanticRole.TEXT_INPUT,
                    SemanticRole.MESSAGE_INPUT,
                    SemanticRole.SEND_BUTTON,
                    SemanticRole.BUTTON,
                }:
                    right_count += 1
            if center_y >= bottom - int(height * 0.22) and role in {
                SemanticRole.MESSAGE_INPUT,
                SemanticRole.TEXT_INPUT,
                SemanticRole.SEND_BUTTON,
                SemanticRole.SUBMIT_BUTTON,
                SemanticRole.BUTTON,
            }:
                bottom_input_like += 1
            if center_y <= top + int(height * 0.20) and center_x > split_x and text_name.strip():
                header_like += 1
        return left_count >= 2 and right_count >= 2 and (bottom_input_like >= 1 or header_like >= 1)

    def has_bottom_action_band(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> bool:
        _, content_top, _, content_bottom = content_bounds
        content_height = max(content_bottom - content_top, 1)
        action_threshold = content_top + int(content_height * 0.70)
        input_count = 0
        action_count = 0
        for _element_id, element, bounds in content_elements:
            _, top, _, bottom = bounds
            center_y = (top + bottom) // 2
            semantic_role = self._infer_semantic_role(
                element.get("control_type", ""),
                element.get("text", "") or element.get("name", ""),
                element.get("name"),
            )
            if semantic_role in (SemanticRole.MESSAGE_INPUT, SemanticRole.TEXT_INPUT, SemanticRole.SEARCH_INPUT):
                input_count += 1
            if center_y >= action_threshold and semantic_role in (
                SemanticRole.SUBMIT_BUTTON,
                SemanticRole.CANCEL_BUTTON,
                SemanticRole.BUTTON,
            ):
                action_count += 1
        return input_count >= 1 and action_count >= 1

    def has_list_detail_split(
        self,
        content_elements: list[tuple[str, dict[str, Any], tuple[int, int, int, int]]],
        content_bounds: tuple[int, int, int, int],
    ) -> bool:
        left, _, right, _ = content_bounds
        split_x = left + int((right - left) * 0.40)
        left_count = 0
        right_count = 0
        for _element_id, _element, bounds in content_elements:
            elem_left, _, elem_right, _ = bounds
            center_x = (elem_left + elem_right) // 2
            if center_x <= split_x:
                left_count += 1
            else:
                right_count += 1
        return left_count >= 2 and right_count >= 1

    def build_regions(
        self,
        raw_elements: list[dict[str, Any]],
        zone_structure: WindowZoneStructure | None,
        surface_type: SurfaceType,
        content_subtype: ContentAreaSubtype,
        element_region_map: dict[str, str],
        fallback_regions: list[Region] | None = None,
        vision_layout_regions: list[dict[str, Any]] | None = None,
        geometric_regions: list[GeometricRegion] | None = None,
        fusion_diagnostics: dict[str, Any] | None = None,
    ) -> list[Region]:
        del surface_type
        if zone_structure is not None:
            return self.build_regions_from_zone_structure(zone_structure, content_subtype)
        return self.build_fallback_region(
            raw_elements, content_subtype, element_region_map,
            fallback_regions, vision_layout_regions, geometric_regions, fusion_diagnostics,
        )

    def build_regions_from_zone_structure(
        self,
        zone_structure: WindowZoneStructure,
        content_subtype: ContentAreaSubtype,
    ) -> list[Region]:
        regions: list[Region] = []
        region_id_counter: dict[str, int] = {}
        for zone in zone_structure.zones:
            base_id = self.zone_type_to_region_id(zone.zone_type.value)
            counter = region_id_counter.get(base_id, 0)
            region_id = f"{base_id}_{counter}" if counter > 0 else base_id
            region_id_counter[base_id] = counter + 1
            element_ids: list[str] = []
            for merged_elem in zone.elements:
                element_ids.extend(merged_elem.original_element_ids)
                element_ids.append(merged_elem.element_id)
            element_ids = list(dict.fromkeys(element_ids))
            subtype = content_subtype if zone.zone_type == ZoneType.CONTENT_AREA else ContentAreaSubtype.UNKNOWN
            regions.append(
                Region(
                    region_id=region_id,
                    role=zone.zone_type.value,
                    subtype=subtype,
                    bounds=zone.bounding_rect,
                    element_ids=element_ids,
                )
            )
        return regions

    def build_fallback_region(
        self,
        raw_elements: list[dict[str, Any]],
        content_subtype: ContentAreaSubtype,
        element_region_map: dict[str, str],
        fallback_regions: list[Region] | None = None,
        vision_layout_regions: list[dict[str, Any]] | None = None,
        geometric_regions: list[GeometricRegion] | None = None,
        fusion_diagnostics: dict[str, Any] | None = None,
    ) -> list[Region]:
        if fallback_regions is None:
            regions, inferred_region_map = self.infer_layout_regions(
                raw_elements, content_subtype, vision_layout_regions or [],
                geometric_regions,
                fusion_diagnostics,
            )
        else:
            regions = fallback_regions
            inferred_region_map: dict[str, str] = {}
            for region in fallback_regions:
                for element_id in region.element_ids:
                    inferred_region_map[element_id] = region.region_id
        element_region_map.update(inferred_region_map)
        return regions
