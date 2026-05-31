"""Locator, anchor, and relation builders for InteractionCanvasEngine."""

from __future__ import annotations

from typing import Any

from src.perception.page_compiler_models import (
    Anchor,
    AnchorKind,
    CoordinateSpace,
    ElementRelation,
    Locator,
    LocatorKind,
    LocatorStatus,
    Candidate,
    Region,
    RelationType,
    SemanticRole,
    SurfaceType,
)


class InteractionCanvasEngineLocatorBuilder:
    """Builds locators, anchors, and relations for compiled page snapshots."""

    def __init__(
        self,
        locator_priority_by_surface: dict[SurfaceType, list[LocatorKind]],
    ) -> None:
        self._locator_priority_by_surface = locator_priority_by_surface

    def generate_locators(
        self,
        elements: list[Candidate],
        surface_type: SurfaceType,
        has_dom_bridge: bool = False,
    ) -> list[Locator]:
        """Generate minimal locator sets for all elements."""
        locators: list[Locator] = []
        priority_by_kind = self._locator_priority_by_surface.get(
            surface_type,
            self._locator_priority_by_surface[SurfaceType.UNKNOWN],
        )

        for element in elements:
            element_locators = self.generate_element_locators(
                element,
                surface_type,
                priority_by_kind,
                has_dom_bridge,
            )
            locators.extend(element_locators)
            element.locator_ids = [locator.locator_id for locator in element_locators]

        return locators

    def generate_element_locators(
        self,
        element: Candidate,
        surface_type: SurfaceType,
        priority_by_kind: list[LocatorKind],
        has_dom_bridge: bool = False,
    ) -> list[Locator]:
        """Generate locators for one element."""
        locators: list[Locator] = []
        automation_id = element.attributes.get("automation_id")

        if surface_type in (SurfaceType.NATIVE_UIA, SurfaceType.ELECTRON_WEBVIEW):
            if automation_id:
                locators.append(
                    Locator(
                        locator_id=f"loc_{element.element_id}_uia",
                        element_ref=element.element_id,
                        kind=LocatorKind.UIA,
                        priority=priority_by_kind.index(LocatorKind.UIA),
                        status=LocatorStatus.CANDIDATE,
                        coordinate_space=CoordinateSpace.WINDOW,
                        selector={"automation_id": automation_id},
                        expected={"control_type": element.control_type},
                        confidence=0.8,
                        durability_score=0.7,
                        cost_score=0.1,
                    )
                )

        if has_dom_bridge and surface_type in (
            SurfaceType.BROWSER,
            SurfaceType.ELECTRON_WEBVIEW,
        ):
            if element.name or element.text:
                selector: dict[str, Any] = {}
                if element.name:
                    selector["name"] = element.name
                elif element.text:
                    selector["text"] = element.text[:50]
                locators.append(
                    Locator(
                        locator_id=f"loc_{element.element_id}_dom",
                        element_ref=element.element_id,
                        kind=LocatorKind.DOM,
                        priority=priority_by_kind.index(LocatorKind.DOM),
                        status=LocatorStatus.CANDIDATE,
                        coordinate_space=CoordinateSpace.WINDOW,
                        selector=selector,
                        expected={"control_type": element.control_type},
                        confidence=0.7,
                        durability_score=0.6,
                        cost_score=0.2,
                    )
                )

        if element.bounds and element.text:
            locators.append(
                Locator(
                    locator_id=f"loc_{element.element_id}_ocr",
                    element_ref=element.element_id,
                    kind=LocatorKind.OCR,
                    priority=priority_by_kind.index(LocatorKind.OCR),
                    status=LocatorStatus.CANDIDATE,
                    coordinate_space=CoordinateSpace.WINDOW,
                    selector={
                        "text": element.text[:80],
                        "bbox": list(element.bounds),
                    },
                    expected={"control_type": element.control_type},
                    confidence=float(element.attributes.get("ocr_confidence", 0.55)),
                    durability_score=0.4,
                    cost_score=0.4,
                )
            )

        if element.bounds and element.interactable:
            left, top, right, bottom = element.bounds
            locators.append(
                Locator(
                    locator_id=f"loc_{element.element_id}_vision",
                    element_ref=element.element_id,
                    kind=LocatorKind.VISION_BBOX,
                    priority=priority_by_kind.index(LocatorKind.VISION_BBOX),
                    status=LocatorStatus.CANDIDATE,
                    coordinate_space=CoordinateSpace.WINDOW,
                    selector={"bbox": [left, top, right, bottom]},
                    expected={"bbox": [left, top, right, bottom]},
                    confidence=0.5,
                    durability_score=0.3,
                    cost_score=0.5,
                )
            )

        if element.bounds and element.interactable:
            left, top, right, bottom = element.bounds
            center_x = left + (right - left) // 2
            center_y = top + (bottom - top) // 2
            locators.append(
                Locator(
                    locator_id=f"loc_{element.element_id}_coord",
                    element_ref=element.element_id,
                    kind=LocatorKind.EPHEMERAL_COORD,
                    priority=priority_by_kind.index(LocatorKind.EPHEMERAL_COORD),
                    status=LocatorStatus.CANDIDATE,
                    coordinate_space=CoordinateSpace.WINDOW,
                    selector={"x": center_x, "y": center_y},
                    expected={"x": center_x, "y": center_y},
                    confidence=0.3,
                    durability_score=0.1,
                    cost_score=0.3,
                )
            )

        return locators

    def generate_anchors(
        self,
        elements: list[Candidate],
        regions: list[Region],
    ) -> list[Anchor]:
        """Generate minimal anchors from stable key elements."""
        anchors: list[Anchor] = []

        name_counts: dict[str, int] = {}
        for element in elements:
            if element.name:
                name_counts[element.name] = name_counts.get(element.name, 0) + 1

        all_areas: list[int] = []
        for element in elements:
            if element.bounds:
                left, top, right, bottom = element.bounds
                all_areas.append((right - left) * (bottom - top))

        key_roles = {
            SemanticRole.MESSAGE_INPUT,
            SemanticRole.TEXT_INPUT,
            SemanticRole.SEARCH_INPUT,
            SemanticRole.SEND_BUTTON,
            SemanticRole.SUBMIT_BUTTON,
            SemanticRole.SIDEBAR,
            SemanticRole.TOOLBAR,
        }

        for element in elements:
            if element.semantic_role not in key_roles or not element.bounds:
                continue
            left, top, right, bottom = element.bounds
            area = (right - left) * (bottom - top)
            anchor_id = f"anchor_{element.element_id}"
            anchor = Anchor(
                anchor_id=anchor_id,
                kind=AnchorKind.SINGLE,
                element_refs=[element.element_id],
                signature={
                    "semantic_role": element.semantic_role.value,
                    "text": element.text[:30] if element.text else "",
                    "bounds": [left, top, right, bottom],
                },
                stability_score=self.compute_anchor_stability(
                    element,
                    name_counts,
                    area,
                    all_areas,
                ),
                region_id=element.region_id,
            )
            anchors.append(anchor)
            element.anchor_ids.append(anchor_id)

        for region in regions:
            region_anchors = [
                anchor for anchor in anchors if anchor.region_id == region.region_id
            ]
            if len(region_anchors) < 2:
                continue
            average_stability = sum(
                anchor.stability_score for anchor in region_anchors
            ) / len(region_anchors)
            compound_anchor = Anchor(
                anchor_id=f"anchor_{region.region_id}_main",
                kind=AnchorKind.COMPOUND,
                element_refs=[
                    element_ref
                    for anchor in region_anchors
                    for element_ref in anchor.element_refs
                ],
                signature={
                    "type": f"{region.role}_compound",
                    "region_role": region.role,
                    "sub_anchor_count": len(region_anchors),
                },
                stability_score=average_stability,
                region_id=region.region_id,
            )
            anchors.append(compound_anchor)
            region.anchor_ids.append(compound_anchor.anchor_id)

        return anchors

    def attach_anchor_refs_to_locators(
        self,
        elements: list[Candidate],
        locators: list[Locator],
    ) -> None:
        """Attach element anchor refs to locators."""
        element_anchor_map = {
            element.element_id: list(element.anchor_ids) for element in elements
        }
        for locator in locators:
            locator.anchor_refs = list(
                element_anchor_map.get(locator.element_ref, [])
            )

    def compute_anchor_stability(
        self,
        element: Candidate,
        name_counts: dict[str, int],
        area: int,
        all_areas: list[int],
    ) -> float:
        """Compute anchor stability in the range [0.0, 1.0]."""
        score = 0.5
        if element.attributes.get("automation_id"):
            score += 0.2

        if element.name:
            name_frequency = name_counts.get(element.name, 1)
            if name_frequency == 1:
                score += 0.1
            else:
                score -= 0.1

        if all_areas:
            median_area = sorted(all_areas)[len(all_areas) // 2]
            if median_area > 0:
                area_ratio = area / median_area
                if area_ratio < 0.3:
                    score -= 0.1
                elif area_ratio > 5.0:
                    score -= 0.05

        return max(0.0, min(1.0, round(score, 2)))

    def generate_relations(
        self,
        elements: list[Candidate],
        regions: list[Region],
    ) -> list[ElementRelation]:
        """Generate minimal layout relations between elements and regions."""
        relations: list[ElementRelation] = []

        inputs = [
            element
            for element in elements
            if element.semantic_role
            in (
                SemanticRole.MESSAGE_INPUT,
                SemanticRole.TEXT_INPUT,
                SemanticRole.SEARCH_INPUT,
            )
        ]
        buttons = [
            element
            for element in elements
            if element.semantic_role
            in (
                SemanticRole.SEND_BUTTON,
                SemanticRole.SUBMIT_BUTTON,
                SemanticRole.BUTTON,
            )
        ]

        for input_element in inputs:
            if not input_element.bounds:
                continue
            inp_left, inp_top, inp_right, inp_bottom = input_element.bounds
            for button in buttons:
                if button.element_id == input_element.element_id or not button.bounds:
                    continue
                btn_left, btn_top, btn_right, btn_bottom = button.bounds
                if btn_left > inp_right:
                    relations.append(
                        ElementRelation(
                            relation_id=f"rel_{input_element.element_id}_right_of_{button.element_id}",
                            from_id=input_element.element_id,
                            to_id=button.element_id,
                            type=RelationType.RIGHT_OF,
                            strength=0.8,
                            offset=(
                                btn_left - inp_right,
                                (btn_top + btn_bottom) // 2
                                - (inp_top + inp_bottom) // 2,
                            ),
                        )
                    )
                elif btn_right < inp_left:
                    relations.append(
                        ElementRelation(
                            relation_id=f"rel_{input_element.element_id}_left_of_{button.element_id}",
                            from_id=input_element.element_id,
                            to_id=button.element_id,
                            type=RelationType.LEFT_OF,
                            strength=0.8,
                            offset=(
                                inp_left - btn_right,
                                (btn_top + btn_bottom) // 2
                                - (inp_top + inp_bottom) // 2,
                            ),
                        )
                    )

        for element in elements:
            if not element.region_id or not element.bounds:
                continue
            region = next(
                (region for region in regions if region.region_id == element.region_id),
                None,
            )
            if not region or not region.bounds:
                continue
            elem_left, elem_top, elem_right, elem_bottom = element.bounds
            reg_left, reg_top, reg_right, reg_bottom = region.bounds
            if (
                elem_left >= reg_left
                and elem_right <= reg_right
                and elem_top >= reg_top
                and elem_bottom <= reg_bottom
            ):
                relations.append(
                    ElementRelation(
                        relation_id=f"rel_{element.element_id}_inside_{element.region_id}",
                        from_id=element.element_id,
                        to_id=element.region_id,
                        type=RelationType.INSIDE,
                        strength=1.0,
                    )
                )

        ordered_roles = ["title_bar", "tool_bar", "content_area", "status_bar"]
        ordered_regions = sorted(
            [region for region in regions if region.role in ordered_roles],
            key=lambda region: ordered_roles.index(region.role),
        )
        for index in range(len(ordered_regions) - 1):
            upper = ordered_regions[index]
            lower = ordered_regions[index + 1]
            if upper.bounds and lower.bounds:
                relations.append(
                    ElementRelation(
                        relation_id=f"rel_{upper.region_id}_above_{lower.region_id}",
                        from_id=upper.region_id,
                        to_id=lower.region_id,
                        type=RelationType.ABOVE,
                        strength=1.0,
                    )
                )

        return relations
