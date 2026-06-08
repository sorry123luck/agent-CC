"""Support builders for InteractionCanvasEngine semantic role inference and provider trace."""

from __future__ import annotations

from typing import Any

from src.perception.page_compiler_models import (
    LocatorKind,
    ProviderTrace,
    SemanticRole,
)
from src.perception.zone_partitioner import WindowZoneStructure


_SEND_TOKENS = ("\u53d1\u9001",)
_CANCEL_TOKENS = ("\u53d6\u6d88",)
_SUBMIT_TOKENS = ("\u63d0\u4ea4", "\u786e\u8ba4", "\u786e\u5b9a")
_SEARCH_TOKENS = ("\u641c\u7d22", "\u67e5\u627e", "\u7b5b\u9009")
_CHAT_ITEM_TOKENS = (
    "\u4f1a\u8bdd",
    "\u804a\u5929",
    "\u8054\u7cfb\u4eba",
    "\u7fa4\u804a",
    "\u6587\u4ef6\u4f20\u8f93\u52a9\u624b",
)


class InteractionCanvasEngineSupportBuilder:
    """Builds semantic-role and provider-trace support structures."""

    def build_provider_trace(
        self,
        uia_element_count: int,
        has_dom_bridge: bool,
        visual_score: float | None,
        zone_structure: WindowZoneStructure | None,
        element_source_stats: dict[str, int] | None = None,
        locators: list | None = None,
        ocr_blocks: list[dict[str, Any]] | None = None,
        vision_candidates: list[dict[str, Any]] | None = None,
        vision_layout_regions: list[dict[str, Any]] | None = None,
        vision_control_groups: list[dict[str, Any]] | None = None,
        vision_interaction_hints: list[dict[str, Any]] | None = None,
        structure_evidence_score: float | None = None,
    ) -> ProviderTrace:
        """Build provider trace with real source usage details."""
        dom_locator_count = 0
        ocr_locator_count = 0
        if locators:
            for locator in locators:
                if getattr(locator, "kind", None) == LocatorKind.DOM:
                    selector = getattr(locator, "selector", {}) or {}
                    if selector.get("name") or selector.get("text"):
                        dom_locator_count += 1
                if getattr(locator, "kind", None) == LocatorKind.OCR:
                    ocr_locator_count += 1

        dom_actually_used = dom_locator_count > 0
        ocr_blocks = ocr_blocks or []
        vision_candidates = vision_candidates or []
        vision_layout_regions = vision_layout_regions or []
        vision_control_groups = vision_control_groups or []
        vision_interaction_hints = vision_interaction_hints or []

        provider_details: dict[str, Any] = {
            "uia": {
                "used": uia_element_count > 0,
                "element_count": uia_element_count,
                "zone_partition_used": zone_structure is not None,
            },
            "dom": {
                "used": dom_actually_used,
                "dom_locator_count": dom_locator_count,
                "status": (
                    "ready"
                    if dom_actually_used
                    else (
                        "not_available"
                        if not has_dom_bridge
                        else "bridge_exists_no_dom_elements"
                    )
                ),
            },
            "vision": {
                "used": visual_score is not None or len(vision_candidates) > 0,
                "score": visual_score,
                "candidate_count": len(vision_candidates),
                "layout_region_count": len(vision_layout_regions),
                "control_group_count": len(vision_control_groups),
                "interaction_hint_count": len(vision_interaction_hints),
                "structure_evidence_score": structure_evidence_score,
            },
            "ocr": {
                "used": len(ocr_blocks) > 0 or ocr_locator_count > 0,
                "block_count": len(ocr_blocks),
                "ocr_locator_count": ocr_locator_count,
            },
        }

        if element_source_stats:
            provider_details["element_sources"] = element_source_stats

        if zone_structure is not None:
            zone_summary: dict[str, Any] = {}
            for zone in zone_structure.zones:
                zone_summary[zone.zone_type.value] = {
                    "element_count": sum(element.original_count for element in zone.elements),
                    "zone_element_count": len(zone.elements),
                }
            provider_details["zone_partition"] = {
                "used": True,
                "zones": zone_summary,
            }

        return ProviderTrace(
            uia_used=uia_element_count > 0,
            ocr_used=len(ocr_blocks) > 0 or ocr_locator_count > 0,
            vision_used=visual_score is not None or len(vision_candidates) > 0,
            dom_used=dom_actually_used,
            provider_details=provider_details,
        )

    def resolve_semantic_role(
        self,
        control_type: str,
        text: str,
        name: str | None,
        vision_match: dict[str, Any] | None = None,
    ) -> SemanticRole:
        native_role = self.infer_semantic_role(control_type, text, name)
        if native_role != SemanticRole.UNKNOWN:
            return native_role
        if not vision_match:
            return native_role
        return self.semantic_role_from_vision_label(
            vision_match.get("kind") or vision_match.get("semantic_label")
        )

    def semantic_role_from_vision_label(self, value: Any) -> SemanticRole:
        label = str(value or "").strip().lower()
        mapping = {
            "send_button": SemanticRole.SEND_BUTTON,
            "submit_button": SemanticRole.SUBMIT_BUTTON,
            "cancel_button": SemanticRole.CANCEL_BUTTON,
            "toggle_button": SemanticRole.TOGGLE_BUTTON,
            "icon_button": SemanticRole.ICON_BUTTON,
            "button": SemanticRole.BUTTON,
            "search_input": SemanticRole.SEARCH_INPUT,
            "message_input": SemanticRole.MESSAGE_INPUT,
            "text_input": SemanticRole.TEXT_INPUT,
            "menu_item": SemanticRole.MENU_ITEM,
            "nav_item": SemanticRole.NAV_ITEM,
            "tab": SemanticRole.TAB,
            "link": SemanticRole.LINK,
            "list_item": SemanticRole.LIST_ITEM,
            "chat_item": SemanticRole.CHAT_ITEM,
            "tree_item": SemanticRole.TREE_ITEM,
            "toolbar": SemanticRole.TOOLBAR,
            "sidebar": SemanticRole.SIDEBAR,
            "title_bar": SemanticRole.TITLE_BAR,
            "status_bar": SemanticRole.STATUS_BAR,
        }
        return mapping.get(label, SemanticRole.UNKNOWN)

    def infer_semantic_role(
        self,
        control_type: str,
        text: str,
        name: str | None,
    ) -> SemanticRole:
        """Infer semantic role from control type and visible text."""
        control_type_lower = str(control_type or "").lower()
        text_value = str(text or "")
        name_value = str(name or "")
        merged_text = f"{text_value} {name_value}".strip()
        merged_text_lower = merged_text.lower()
        has_cjk = any("\u4e00" <= char <= "\u9fff" for char in merged_text)

        if "button" in control_type_lower:
            if any(token in merged_text for token in _SEND_TOKENS) or "send" in merged_text_lower:
                return SemanticRole.SEND_BUTTON
            if any(token in merged_text for token in _CANCEL_TOKENS) or "cancel" in merged_text_lower:
                return SemanticRole.CANCEL_BUTTON
            if any(token in merged_text for token in _SUBMIT_TOKENS) or any(
                token in merged_text_lower for token in ("submit", "confirm")
            ):
                return SemanticRole.SUBMIT_BUTTON
            return SemanticRole.BUTTON

        if "edit" in control_type_lower:
            if any(token in merged_text for token in _SEARCH_TOKENS) or any(
                token in merged_text_lower for token in ("search", "find", "filter")
            ):
                return SemanticRole.SEARCH_INPUT
            if text_value or name_value:
                return SemanticRole.TEXT_INPUT
            return SemanticRole.MESSAGE_INPUT

        if "text" in control_type_lower:
            if any(token in merged_text for token in _SEARCH_TOKENS) or any(
                token in merged_text_lower for token in ("search", "find", "filter")
            ):
                return SemanticRole.SEARCH_INPUT
            # CJK text with explicit chat token → CHAT_ITEM
            # (e.g. "文件传输助手", "会话列表", "联系人" from OCR)
            if has_cjk and any(token in merged_text for token in _CHAT_ITEM_TOKENS):
                return SemanticRole.CHAT_ITEM
            return SemanticRole.TEXT

        if "list" in control_type_lower or "listitem" in control_type_lower:
            if any(token in merged_text for token in _CHAT_ITEM_TOKENS):
                return SemanticRole.CHAT_ITEM
            return SemanticRole.LIST_ITEM
        if "tree" in control_type_lower or "treeitem" in control_type_lower:
            return SemanticRole.TREE_ITEM
        if "menu" in control_type_lower or "menuitem" in control_type_lower:
            return SemanticRole.MENU_ITEM
        if "tab" in control_type_lower or "tabitem" in control_type_lower:
            return SemanticRole.TAB
        if "image" in control_type_lower:
            return SemanticRole.IMAGE
        if "link" in control_type_lower or "hyperlink" in control_type_lower:
            return SemanticRole.LINK
        if "window" in control_type_lower:
            return SemanticRole.CONTAINER
        if "title" in control_type_lower:
            return SemanticRole.TITLE_BAR
        if "toolbar" in control_type_lower or "tool" in control_type_lower:
            return SemanticRole.TOOLBAR
        if "sidebar" in control_type_lower or "sidebar" in merged_text_lower:
            return SemanticRole.SIDEBAR
        if "pane" in control_type_lower or "panel" in control_type_lower:
            return SemanticRole.LAYOUT

        if any(token in merged_text for token in _SEND_TOKENS):
            return SemanticRole.SEND_BUTTON
        if any(token in merged_text for token in _SEARCH_TOKENS):
            return SemanticRole.SEARCH_INPUT
        if any(token in merged_text for token in _CHAT_ITEM_TOKENS):
            return SemanticRole.CHAT_ITEM
        if merged_text:
            return SemanticRole.TEXT

        return SemanticRole.UNKNOWN

    def is_interactable(self, control_type: str) -> bool:
        control_type_lower = control_type.lower()
        non_interactable = {
            "text",
            "image",
            "pane",
            "panel",
            "scrollbar",
            "thumb",
            "title",
            "window",
            "group",
            "semantic",
            "split",
        }
        return control_type_lower not in non_interactable
