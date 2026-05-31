"""Action verification helpers for runtime execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import win32con
import win32gui

from src.perception.page_compiler_models import LocatorKind, InteractionCanvas, SemanticRole
from src.windows.window_enum import WindowEnumService


class VerificationMethod(Enum):
    """Verification methods used by the runtime."""

    WINDOW_STATE = "window_state"
    WINDOW_TITLE = "window_title"
    ELEMENT_PRESENT = "element_present"
    SCREENSHOT_CHANGE = "screenshot_change"
    SNAPSHOT_TARGET = "snapshot_target"
    SNAPSHOT_LOCATOR = "snapshot_locator"
    SNAPSHOT_DIFF = "snapshot_diff"
    NO_CHANGE = "no_change"


@dataclass
class VerificationResult:
    """Verification result."""

    verified: bool
    method: VerificationMethod
    message: str
    details: dict | None = None


class ActionVerifier:
    """Provides small, composable verification helpers."""

    def __init__(self) -> None:
        self._window_enum = WindowEnumService()

    def verify_window_state(
        self,
        hwnd: int,
        expected_state: Literal["active", "minimized", "maximized"],
        timeout: float = 2.0,
    ) -> VerificationResult:
        import time

        start = time.time()
        while time.time() - start < timeout:
            try:
                if not win32gui.IsWindow(hwnd):
                    return VerificationResult(
                        False,
                        VerificationMethod.WINDOW_STATE,
                        "window_not_found",
                        {"hwnd": hwnd},
                    )

                if expected_state == "active":
                    foreground = self._window_enum.get_foreground_window()
                    if foreground and foreground.hwnd == hwnd:
                        return VerificationResult(
                            True,
                            VerificationMethod.WINDOW_STATE,
                            "window_active",
                            {"hwnd": hwnd},
                        )
                elif expected_state == "minimized" and win32gui.IsIconic(hwnd):
                    return VerificationResult(
                        True,
                        VerificationMethod.WINDOW_STATE,
                        "window_minimized",
                        {"hwnd": hwnd},
                    )
                elif expected_state == "maximized":
                    placement = win32gui.GetWindowPlacement(hwnd)
                    if placement[1] == win32con.SW_SHOWMAXIMIZED:
                        return VerificationResult(
                            True,
                            VerificationMethod.WINDOW_STATE,
                            "window_maximized",
                            {"hwnd": hwnd},
                        )

                time.sleep(0.1)
            except Exception as exc:
                return VerificationResult(
                    False,
                    VerificationMethod.WINDOW_STATE,
                    f"window_state_error: {exc}",
                    {"hwnd": hwnd, "expected_state": expected_state},
                )

        return VerificationResult(
            False,
            VerificationMethod.WINDOW_STATE,
            "window_state_timeout",
            {"hwnd": hwnd, "expected_state": expected_state},
        )

    def verify_window_title(
        self,
        hwnd: int,
        expected_title_contains: str,
    ) -> VerificationResult:
        try:
            windows = self._window_enum.enumerate_all(refresh=True)
            window = next((item for item in windows if item.hwnd == hwnd), None)
            if window is None:
                return VerificationResult(
                    False,
                    VerificationMethod.WINDOW_TITLE,
                    "window_not_found",
                    {"hwnd": hwnd},
                )

            title = window.title or ""
            if expected_title_contains.lower() in title.lower():
                return VerificationResult(
                    True,
                    VerificationMethod.WINDOW_TITLE,
                    "window_title_matched",
                    {"title": title},
                )
            return VerificationResult(
                False,
                VerificationMethod.WINDOW_TITLE,
                "window_title_mismatch",
                {"expected": expected_title_contains, "actual": title},
            )
        except Exception as exc:
            return VerificationResult(
                False,
                VerificationMethod.WINDOW_TITLE,
                f"window_title_error: {exc}",
                {"hwnd": hwnd},
            )

    def verify_element_present(
        self,
        hwnd: int,
        automation_id: str | None = None,
        name: str | None = None,
        control_type: str | None = None,
    ) -> VerificationResult:
        try:
            from src.perception.uia_client import UIAClient

            for element in UIAClient(hwnd).find_all():
                if automation_id and element.automation_id == automation_id:
                    return VerificationResult(
                        True,
                        VerificationMethod.ELEMENT_PRESENT,
                        "element_present",
                        {"automation_id": automation_id},
                    )
                if name and element.name == name:
                    return VerificationResult(
                        True,
                        VerificationMethod.ELEMENT_PRESENT,
                        "element_present",
                        {"name": name},
                    )
                if control_type and element.control_type == control_type:
                    return VerificationResult(
                        True,
                        VerificationMethod.ELEMENT_PRESENT,
                        "element_present",
                        {"control_type": control_type},
                    )

            return VerificationResult(
                False,
                VerificationMethod.ELEMENT_PRESENT,
                "element_not_found",
                {
                    "automation_id": automation_id,
                    "name": name,
                    "control_type": control_type,
                },
            )
        except Exception as exc:
            return VerificationResult(
                False,
                VerificationMethod.ELEMENT_PRESENT,
                f"element_present_error: {exc}",
                {"hwnd": hwnd},
            )

    def verify_snapshot_target(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        locator_kind: LocatorKind | None = None,
    ) -> VerificationResult:
        """Validate that a target remains executable from the snapshot view."""
        element = snapshot.get_element(element_id)
        if element is None:
            return VerificationResult(
                False,
                VerificationMethod.SNAPSHOT_TARGET,
                "snapshot_element_missing",
                {"element_id": element_id},
            )

        if locator_kind is not None:
            locator = next(
                (
                    snapshot.get_locator(locator_id)
                    for locator_id in element.locator_ids
                    if snapshot.get_locator(locator_id)
                    and snapshot.get_locator(locator_id).kind == locator_kind
                ),
                None,
            )
            if locator is None:
                return VerificationResult(
                    False,
                    VerificationMethod.SNAPSHOT_LOCATOR,
                    "snapshot_locator_missing",
                    {"element_id": element_id, "locator_kind": locator_kind.value},
                )
            return self._verify_locator_snapshot(snapshot, element_id, locator)

        if not element.locator_ids:
            return VerificationResult(
                False,
                VerificationMethod.SNAPSHOT_TARGET,
                "snapshot_target_has_no_locators",
                {"element_id": element_id},
            )

        return VerificationResult(
            True,
            VerificationMethod.SNAPSHOT_TARGET,
            "snapshot_target_present",
            {
                "element_id": element_id,
                "locator_count": len(element.locator_ids),
                "region_id": element.region_id,
                "anchor_count": len(element.anchor_ids),
                "content_group_id": element.content_group_id,
                "scroll_context_id": self._get_region_scroll_context(snapshot, element.region_id),
            },
        )

    def compare_snapshots(
        self,
        before_snapshot: InteractionCanvas,
        after_snapshot: InteractionCanvas,
        target_element_id: str | None = None,
    ) -> VerificationResult:
        """Compare two snapshots and report whether a meaningful page change occurred."""
        before_elements = len(before_snapshot.elements)
        after_elements = len(after_snapshot.elements)
        before_regions = len(before_snapshot.regions)
        after_regions = len(after_snapshot.regions)
        before_locators = len(before_snapshot.locators)
        after_locators = len(after_snapshot.locators)
        before_anchors = len(before_snapshot.anchors)
        after_anchors = len(after_snapshot.anchors)
        before_scroll_contexts = len(before_snapshot.scroll_contexts)
        after_scroll_contexts = len(after_snapshot.scroll_contexts)
        before_child_regions = sum(len(region.child_region_ids) for region in before_snapshot.regions)
        after_child_regions = sum(len(region.child_region_ids) for region in after_snapshot.regions)
        before_dialog_evidence = self._collect_dialog_evidence(before_snapshot)
        after_dialog_evidence = self._collect_dialog_evidence(after_snapshot)
        before_widget_evidence = self._collect_widget_evidence(before_snapshot)
        after_widget_evidence = self._collect_widget_evidence(after_snapshot)
        before_grid_evidence = self._collect_grid_evidence(before_snapshot)
        after_grid_evidence = self._collect_grid_evidence(after_snapshot)
        before_editor_evidence = self._collect_editor_evidence(before_snapshot)
        after_editor_evidence = self._collect_editor_evidence(after_snapshot)
        before_canvas_evidence = self._collect_canvas_doc_evidence(before_snapshot)
        after_canvas_evidence = self._collect_canvas_doc_evidence(after_snapshot)
        before_scroll_evidence = self._collect_scroll_evidence(before_snapshot)
        after_scroll_evidence = self._collect_scroll_evidence(after_snapshot)
        page_changed = (
            before_snapshot.page.page_class != after_snapshot.page.page_class
            or before_snapshot.surface.surface_type != after_snapshot.surface.surface_type
        )
        structure_changed = (
            before_elements != after_elements
            or before_regions != after_regions
            or before_locators != after_locators
            or before_anchors != after_anchors
            or before_scroll_contexts != after_scroll_contexts
            or before_child_regions != after_child_regions
            or before_dialog_evidence != after_dialog_evidence
            or before_widget_evidence != after_widget_evidence
            or before_grid_evidence != after_grid_evidence
            or before_editor_evidence != after_editor_evidence
            or before_canvas_evidence != after_canvas_evidence
            or before_scroll_evidence != after_scroll_evidence
        )
        target_changed = False
        target_state_changed = False
        target_content_changed = False
        target_structure_role_changed = False
        target_region_before = None
        target_region_after = None

        if target_element_id is not None:
            before_target = before_snapshot.get_element(target_element_id)
            after_target = after_snapshot.get_element(target_element_id)
            target_region_before = before_target.region_id if before_target is not None else None
            target_region_after = after_target.region_id if after_target is not None else None
            target_state_changed = (
                before_target is not None
                and after_target is not None
                and before_target.state != after_target.state
            )
            target_content_changed = (
                before_target is not None
                and after_target is not None
                and (
                    before_target.text != after_target.text
                    or before_target.name != after_target.name
                    or before_target.value != after_target.value
                    or before_target.placeholder != after_target.placeholder
                    or before_target.interactable != after_target.interactable
                    or before_target.content_group_id != after_target.content_group_id
                )
            )
            target_structure_role_changed = (
                before_target is not None
                and after_target is not None
                and (
                    before_target.attributes.get("dialog_slot") != after_target.attributes.get("dialog_slot")
                    or before_target.attributes.get("widget_group_id") != after_target.attributes.get("widget_group_id")
                    or before_target.attributes.get("grid_role") != after_target.attributes.get("grid_role")
                )
            )
            target_changed = (
                before_target is None
                or after_target is None
                or before_target.bounds != after_target.bounds
                or before_target.region_id != after_target.region_id
                or before_target.locator_ids != after_target.locator_ids
                or before_target.anchor_ids != after_target.anchor_ids
                or target_state_changed
                or target_content_changed
                or target_structure_role_changed
            )

        changed = page_changed or structure_changed or target_changed

        return VerificationResult(
            verified=changed,
            method=VerificationMethod.SNAPSHOT_DIFF,
            message="snapshot_changed" if changed else "snapshot_unchanged",
            details={
                "before_page_class": before_snapshot.page.page_class,
                "after_page_class": after_snapshot.page.page_class,
                "before_surface_type": before_snapshot.surface.surface_type.value,
                "after_surface_type": after_snapshot.surface.surface_type.value,
                "before_element_count": before_elements,
                "after_element_count": after_elements,
                "before_region_count": before_regions,
                "after_region_count": after_regions,
                "before_locator_count": before_locators,
                "after_locator_count": after_locators,
                "before_anchor_count": before_anchors,
                "after_anchor_count": after_anchors,
                "before_scroll_context_count": before_scroll_contexts,
                "after_scroll_context_count": after_scroll_contexts,
                "before_child_region_count": before_child_regions,
                "after_child_region_count": after_child_regions,
                "before_dialog_evidence": before_dialog_evidence,
                "after_dialog_evidence": after_dialog_evidence,
                "before_widget_evidence": before_widget_evidence,
                "after_widget_evidence": after_widget_evidence,
                "before_grid_evidence": before_grid_evidence,
                "after_grid_evidence": after_grid_evidence,
                "before_editor_evidence": before_editor_evidence,
                "after_editor_evidence": after_editor_evidence,
                "before_canvas_evidence": before_canvas_evidence,
                "after_canvas_evidence": after_canvas_evidence,
                "before_scroll_evidence": before_scroll_evidence,
                "after_scroll_evidence": after_scroll_evidence,
                "target_element_id": target_element_id,
                "page_changed": page_changed,
                "structure_changed": structure_changed,
                "target_changed": target_changed,
                "target_state_changed": target_state_changed,
                "target_content_changed": target_content_changed,
                "target_structure_role_changed": target_structure_role_changed,
                "target_region_before": target_region_before,
                "target_region_after": target_region_after,
            },
        )

    def verify_action_effect(
        self,
        before_snapshot: InteractionCanvas,
        after_snapshot: InteractionCanvas,
        target_element_id: str | None,
        action: str,
    ) -> VerificationResult:
        """Interpret snapshot differences using target semantic role."""
        diff_result = self.compare_snapshots(
            before_snapshot,
            after_snapshot,
            target_element_id=target_element_id,
        )
        details = dict(diff_result.details or {})
        target_element = (
            before_snapshot.get_element(target_element_id)
            if target_element_id is not None
            else None
        )
        semantic_role = (
            target_element.semantic_role
            if target_element is not None
            else SemanticRole.UNKNOWN
        )
        region = (
            before_snapshot.get_region(target_element.region_id)
            if target_element is not None and target_element.region_id
            else None
        )
        region_role = region.role if region is not None else None
        content_subtype = self._resolve_content_subtype(before_snapshot, region)
        interaction_hints = (
            dict(target_element.attributes.get("interaction_hints") or {})
            if target_element is not None
            else {}
        )
        structure_evidence_score = (
            float(
                target_element.attributes.get(
                    "structure_evidence_score",
                    before_snapshot.artifacts.get("structure_evidence_score", 0.0),
                )
                or 0.0
            )
            if target_element is not None
            else float(before_snapshot.artifacts.get("structure_evidence_score", 0.0) or 0.0)
        )

        policy = "strict_snapshot_change"
        verified = diff_result.verified
        message = diff_result.message
        dialog_slot = target_element.attributes.get("dialog_slot") if target_element is not None else None
        widget_group_id = target_element.attributes.get("widget_group_id") if target_element is not None else None
        grid_role = target_element.attributes.get("grid_role") if target_element is not None else None
        region_attributes = dict(region.attributes) if region is not None else {}

        if action in {"click", "double_click", "right_click"}:
            if semantic_role in {
                SemanticRole.SEND_BUTTON,
                SemanticRole.SUBMIT_BUTTON,
                SemanticRole.MENU_ITEM,
                SemanticRole.NAV_ITEM,
                SemanticRole.LINK,
            }:
                policy = "require_page_or_structure_change"
                verified = bool(
                    details.get("page_changed")
                    or details.get("structure_changed")
                    or details.get("target_changed")
                )
                message = (
                    "semantic_effect_detected"
                    if verified
                    else "semantic_effect_missing"
                )
            elif semantic_role in {
                SemanticRole.TAB,
                SemanticRole.TOGGLE_BUTTON,
                SemanticRole.LIST_ITEM,
                SemanticRole.CHAT_ITEM,
            }:
                policy = "require_target_or_page_change"
                verified = bool(
                    details.get("target_changed")
                    or details.get("page_changed")
                    or details.get("structure_changed")
                )
                message = (
                    "semantic_effect_detected"
                    if verified
                    else "semantic_effect_missing"
                )
            elif region_role in {"viewport", "detail_panel", "list_panel", "message_stream", "dialog_body"}:
                policy = "require_target_or_structure_change_in_viewport"
                verified = bool(
                    details.get("target_changed")
                    or details.get("structure_changed")
                    or details.get("target_state_changed")
                    or details.get("target_content_changed")
                )
                message = "viewport_effect_detected" if verified else "viewport_effect_missing"
            if content_subtype == "list_detail" and region_role in {"list_panel", "detail_panel"}:
                policy = "require_list_detail_target_or_structure_change"
                if interaction_hints.get("expected_effect") in {"open_detail", "navigate"}:
                    policy = "require_list_detail_navigation_change"
                verified = bool(
                    details.get("target_changed")
                    or details.get("target_content_changed")
                    or details.get("structure_changed")
                    or (
                        region_role == "list_panel"
                        and details.get("target_region_before") != details.get("target_region_after")
                    )
                )
                message = "list_detail_effect_detected" if verified else "list_detail_effect_missing"
            if content_subtype == "editor" and region_role in {"toolbar", "viewport", "side_panel"}:
                policy = "require_editor_target_or_structure_change"
                if interaction_hints.get("expected_effect") in {"run_code", "open_panel", "navigate"}:
                    policy = f"require_editor_{interaction_hints.get('expected_effect')}_change"
                verified = bool(
                    details.get("target_changed")
                    or details.get("target_content_changed")
                    or details.get("structure_changed")
                    or details.get("page_changed")
                    or (
                        region_role == "toolbar"
                        and interaction_hints.get("expected_effect") in {"run_code", "open_panel", "navigate"}
                        and details.get("target_changed")
                    )
                    or details.get("before_editor_evidence") != details.get("after_editor_evidence")
                )
                message = "editor_effect_detected" if verified else "editor_effect_missing"
            if content_subtype == "canvas_doc_viewer" and region_role in {"viewport", "side_panel", "action_bar", "toolbar"}:
                policy = "require_canvas_doc_navigation_or_structure_change"
                if interaction_hints.get("expected_effect") in {"navigate", "next_page", "prev_page", "zoom_in", "zoom_out"}:
                    policy = f"require_canvas_doc_{interaction_hints.get('expected_effect')}_change"
                verified = bool(
                    details.get("page_changed")
                    or details.get("structure_changed")
                    or details.get("target_changed")
                    or details.get("target_content_changed")
                    or details.get("before_canvas_evidence") != details.get("after_canvas_evidence")
                    or (
                        region_role == "action_bar"
                        and interaction_hints.get("expected_effect") in {"navigate", "open_panel"}
                        and details.get("target_content_changed")
                    )
                )
                message = "canvas_doc_effect_detected" if verified else "canvas_doc_effect_missing"
            if region_role == "dialog_body" or dialog_slot:
                policy = "require_dialog_structure_or_target_change"
                verified = bool(
                    details.get("target_changed")
                    or details.get("target_structure_role_changed")
                    or details.get("structure_changed")
                    or details.get("before_dialog_evidence") != details.get("after_dialog_evidence")
                )
                message = "dialog_effect_detected" if verified else "dialog_effect_missing"
            if widget_group_id:
                policy = "require_widget_group_or_structure_change"
                verified = bool(
                    details.get("target_changed")
                    or details.get("target_structure_role_changed")
                    or details.get("before_widget_evidence") != details.get("after_widget_evidence")
                    or details.get("structure_changed")
                )
                message = "widget_effect_detected" if verified else "widget_effect_missing"
            if grid_role == "pagination":
                policy = "require_grid_pagination_change"
                verified = bool(
                    details.get("page_changed")
                    or details.get("structure_changed")
                    or details.get("target_content_changed")
                    or details.get("before_grid_evidence") != details.get("after_grid_evidence")
                )
                message = "grid_effect_detected" if verified else "grid_effect_missing"
            elif grid_role in {"header", "row"}:
                if grid_role == "header":
                    policy = "require_grid_header_or_sort_change"
                else:
                    policy = "require_grid_row_or_structure_change"
                verified = bool(
                    details.get("target_changed")
                    or details.get("target_content_changed")
                    or details.get("target_structure_role_changed")
                    or details.get("before_grid_evidence") != details.get("after_grid_evidence")
                )
                message = "grid_effect_detected" if verified else "grid_effect_missing"

        expected_effect = interaction_hints.get("expected_effect")
        can_override_with_interaction_hint = not (
            str(policy).startswith("require_editor_")
            or str(policy).startswith("require_canvas_doc_")
            or str(policy).startswith("require_grid_")
            or str(policy).startswith("require_list_detail_")
            or str(policy).startswith("require_dialog_")
            or str(policy).startswith("require_widget_")
        )
        if expected_effect in {"open_panel", "open_dialog", "navigate", "send_message"} and can_override_with_interaction_hint:
            policy = f"interaction_hint/{expected_effect}"
            verified = bool(
                details.get("page_changed")
                or details.get("structure_changed")
                or details.get("target_changed")
                or details.get("before_dialog_evidence") != details.get("after_dialog_evidence")
            )
            message = "interaction_hint_effect_detected" if verified else "interaction_hint_effect_missing"
        elif expected_effect in {"select_item", "toggle_state", "focus_input"} and can_override_with_interaction_hint:
            policy = f"interaction_hint/{expected_effect}"
            verified = bool(
                details.get("target_changed")
                or details.get("target_state_changed")
                or details.get("target_content_changed")
            )
            message = "interaction_hint_effect_detected" if verified else "interaction_hint_effect_missing"

        if structure_evidence_score >= 0.75 and not verified:
            details["high_structure_evidence"] = True
            details["verification_hint"] = "high_confidence_structure_but_effect_missing"

        details["action"] = action
        details["semantic_role"] = semantic_role.value
        details["region_role"] = region_role
        details["interaction_hints"] = interaction_hints
        details["structure_evidence_score"] = structure_evidence_score
        details["policy"] = policy
        details["dialog_slot"] = dialog_slot
        details["widget_group_id"] = widget_group_id
        details["grid_role"] = grid_role
        details["content_subtype"] = content_subtype
        details["region_attributes"] = region_attributes
        return VerificationResult(
            verified=verified,
            method=VerificationMethod.SNAPSHOT_DIFF,
            message=message,
            details=details,
        )

    def _verify_locator_snapshot(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        locator,
    ) -> VerificationResult:
        """Validate locator plus structural context inside snapshot."""
        element = snapshot.get_element(element_id)
        if element is None:
            return VerificationResult(
                False,
                VerificationMethod.SNAPSHOT_LOCATOR,
                "snapshot_element_missing",
                {"element_id": element_id},
            )
        if locator.kind == LocatorKind.UIA:
            automation_id = (locator.selector or {}).get("automation_id")
            if not automation_id:
                return VerificationResult(
                    False,
                    VerificationMethod.SNAPSHOT_LOCATOR,
                    "snapshot_uia_locator_missing_selector",
                    {"element_id": element_id},
                )
            return self.verify_element_present(hwnd=snapshot.window.hwnd, automation_id=automation_id)

        missing_anchors = [
            anchor_id for anchor_id in getattr(locator, "anchor_refs", []) if self._get_anchor(snapshot, anchor_id) is None
        ]
        region = snapshot.get_region(element.region_id) if element.region_id else None
        details = {
            "element_id": element_id,
            "locator_kind": locator.kind.value,
            "region_id": element.region_id,
            "region_role": region.role if region else None,
            "anchor_count": len(element.anchor_ids),
            "content_group_id": element.content_group_id,
            "scroll_context_id": self._get_region_scroll_context(snapshot, element.region_id),
            "missing_anchor_refs": missing_anchors,
            "interaction_hints": dict(element.attributes.get("interaction_hints") or {}),
            "structure_evidence_score": float(
                element.attributes.get(
                    "structure_evidence_score",
                    snapshot.artifacts.get("structure_evidence_score", 0.0),
                )
                or 0.0
            ),
            "dialog_slot": element.attributes.get("dialog_slot"),
            "widget_group_id": element.attributes.get("widget_group_id"),
            "grid_role": element.attributes.get("grid_role"),
            "region_attributes": dict(region.attributes) if region else {},
        }
        if missing_anchors:
            return VerificationResult(
                False,
                VerificationMethod.SNAPSHOT_LOCATOR,
                "snapshot_anchor_missing",
                details,
            )
        return VerificationResult(
            True,
            VerificationMethod.SNAPSHOT_LOCATOR,
            "snapshot_locator_ready",
            details,
        )

    def _get_region_scroll_context(
        self,
        snapshot: InteractionCanvas,
        region_id: str | None,
    ) -> str | None:
        if region_id is None:
            return None
        region = next((item for item in snapshot.regions if item.region_id == region_id), None)
        return region.scroll_context_id if region else None

    def _get_anchor(self, snapshot: InteractionCanvas, anchor_id: str):
        return next((anchor for anchor in snapshot.anchors if anchor.anchor_id == anchor_id), None)

    def _resolve_content_subtype(self, snapshot: InteractionCanvas, region) -> str | None:
        if region is None:
            return None
        if region.subtype is not None and getattr(region.subtype, "value", None) not in {None, "unknown"}:
            return region.subtype.value
        current = region
        visited: set[str] = set()
        while current is not None and current.region_id not in visited:
            visited.add(current.region_id)
            if current.subtype is not None and getattr(current.subtype, "value", None) not in {None, "unknown"}:
                return current.subtype.value
            parent_id = current.parent_region_id
            current = snapshot.get_region(parent_id) if parent_id else None
        return None

    def _collect_dialog_evidence(self, snapshot: InteractionCanvas) -> dict:
        regions = [region for region in snapshot.regions if region.role == "dialog_body"]
        return {
            "dialog_count": len(regions),
            "dialog_kinds": sorted(str(region.attributes.get("dialog_kind") or "") for region in regions),
            "field_counts": sorted(int(region.attributes.get("field_count") or 0) for region in regions),
            "footer_action_counts": sorted(int(region.attributes.get("footer_action_count") or 0) for region in regions),
        }

    def _collect_widget_evidence(self, snapshot: InteractionCanvas) -> dict:
        widget_groups: list[str] = []
        widget_counts: list[int] = []
        for region in snapshot.regions:
            if region.role != "viewport":
                continue
            groups = region.attributes.get("widget_groups") or {}
            if isinstance(groups, dict):
                widget_groups.extend(sorted(groups.keys()))
            widget_counts.append(int(region.attributes.get("widget_group_count") or 0))
        return {
            "widget_group_keys": sorted(widget_groups),
            "widget_group_counts": sorted(widget_counts),
        }

    def _collect_grid_evidence(self, snapshot: InteractionCanvas) -> dict:
        header_counts: list[int] = []
        row_counts: list[int] = []
        pagination_counts: list[int] = []
        row_group_counts: list[int] = []
        for region in snapshot.regions:
            if region.role != "viewport":
                continue
            header_counts.append(len(region.attributes.get("grid_header_ids") or []))
            row_counts.append(len(region.attributes.get("grid_row_ids") or []))
            pagination_counts.append(len(region.attributes.get("grid_pagination_ids") or []))
            row_group_counts.append(int(region.attributes.get("row_group_count") or 0))
        return {
            "grid_header_counts": sorted(header_counts),
            "grid_row_counts": sorted(row_counts),
            "grid_pagination_counts": sorted(pagination_counts),
            "row_group_counts": sorted(row_group_counts),
        }

    def _collect_editor_evidence(self, snapshot: InteractionCanvas) -> dict:
        toolbar_labels: list[str] = []
        side_panel_count = 0
        viewport_texts: list[str] = []
        for region in snapshot.regions:
            if region.role not in {"toolbar", "side_panel", "viewport"}:
                continue
            elements = [
                snapshot.get_element(element_id)
                for element_id in region.element_ids
                if snapshot.get_element(element_id) is not None
            ]
            if region.role == "toolbar":
                toolbar_labels.extend(
                    sorted(
                        str(element.text or element.name or "").strip().lower()
                        for element in elements
                        if (element.text or element.name)
                    )
                )
            elif region.role == "side_panel":
                side_panel_count += len(elements)
            elif region.role == "viewport":
                viewport_texts.extend(
                    sorted(
                        str(element.text or element.name or "").strip().lower()
                        for element in elements
                        if (element.text or element.name)
                    )[:5]
                )
        return {
            "toolbar_labels": toolbar_labels,
            "side_panel_count": side_panel_count,
            "viewport_texts": sorted(viewport_texts),
        }

    def _collect_canvas_doc_evidence(self, snapshot: InteractionCanvas) -> dict:
        side_panel_count = 0
        action_labels: list[str] = []
        viewport_labels: list[str] = []
        for region in snapshot.regions:
            if region.role not in {"side_panel", "action_bar", "viewport", "toolbar"}:
                continue
            elements = [
                snapshot.get_element(element_id)
                for element_id in region.element_ids
                if snapshot.get_element(element_id) is not None
            ]
            if region.role == "side_panel":
                side_panel_count += len(elements)
            elif region.role in {"action_bar", "toolbar"}:
                action_labels.extend(
                    sorted(
                        str(element.text or element.name or "").strip().lower()
                        for element in elements
                        if (element.text or element.name)
                    )
                )
            elif region.role == "viewport":
                viewport_labels.extend(
                    sorted(
                        str(element.text or element.name or "").strip().lower()
                        for element in elements
                        if (element.text or element.name)
                    )[:5]
                )
        return {
            "side_panel_count": side_panel_count,
            "action_labels": sorted(action_labels),
            "viewport_labels": sorted(viewport_labels),
        }

    def _collect_scroll_evidence(self, snapshot: InteractionCanvas) -> dict:
        return {
            "scroll_context_count": len(snapshot.scroll_contexts),
            "scroll_offsets": sorted(
                int(context.scroll_offset or 0)
                for context in snapshot.scroll_contexts
            ),
            "viewport_heights": sorted(
                int(context.viewport_height or 0)
                for context in snapshot.scroll_contexts
            ),
            "total_content_heights": sorted(
                int(context.total_content_height or 0)
                for context in snapshot.scroll_contexts
            ),
        }

    def verify_screenshot_change(
        self,
        hwnd: int,
        before_screenshot,
        pixel_threshold: float = 0.01,
    ) -> VerificationResult:
        try:
            from src.windows.screenshot_service import ScreenshotService

            after_screenshot = ScreenshotService().capture(mode="window", target=hwnd)
            before_pixels = self._count_different_pixels(before_screenshot, after_screenshot)
            total_pixels = before_screenshot.width * before_screenshot.height
            changed_ratio = before_pixels / total_pixels if total_pixels > 0 else 0

            return VerificationResult(
                changed_ratio >= pixel_threshold,
                VerificationMethod.SCREENSHOT_CHANGE,
                "screenshot_changed" if changed_ratio >= pixel_threshold else "screenshot_unchanged",
                {"changed_ratio": changed_ratio, "changed_pixels": before_pixels},
            )
        except Exception as exc:
            return VerificationResult(
                False,
                VerificationMethod.SCREENSHOT_CHANGE,
                f"screenshot_change_error: {exc}",
                {"hwnd": hwnd},
            )

    def _count_different_pixels(self, img1, img2) -> int:
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)

        import numpy as np

        arr1 = np.array(img1.convert("RGB"))
        arr2 = np.array(img2.convert("RGB"))
        diff = np.abs(arr1.astype(int) - arr2.astype(int))
        different = np.sum(diff > 10, axis=2)
        return int(np.sum(different))
