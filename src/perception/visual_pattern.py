"""Lightweight visual pattern classification for perception results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VisualPatternResult:
    """A coarse UI mode inferred from local evidence."""

    mode: str = "unknown_pattern"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


class VisualPatternClassifier:
    """Classify a canvas into a small set of cross-app visual modes."""

    _APP_MODE_HINTS = {
        "chatgpt": "chat_document",
        "lark": "collaboration_inbox",
        "feishu": "collaboration_inbox",
        "cloudmusic": "media_home",
        "网易云": "media_home",
        "adspower": "list_management",
        "bitbrowser": "list_management",
        "比特浏览器": "list_management",
        "flclash": "control_dashboard",
        "voicemeeter": "control_matrix",
        "winrar": "archive_file_manager",
        "hipsmain": "security_dashboard",
        "bilibili": "video_feed_home",
        "哔哩": "video_feed_home",
        "baidunetdisk": "loading_state",
        "百度网盘": "loading_state",
        "everything": "file_search",
        "localsend": "local_transfer_dashboard",
        "qq": "account_switcher",
        "hubstudio": "browser_profile_manager",
        "teamviewer": "remote_access_dashboard",
        "qyclient": "media_video_home",
        "iqiyi": "media_video_home",
        "爱奇艺": "media_video_home",
    }

    _LOADING_WORDS = ("启动中", "加载中", "loading", "starting", "please wait")

    def classify(self, canvas: Any) -> VisualPatternResult:
        page_class = str(getattr(canvas, "page_class", "") or "").lower()
        app_name = str(getattr(getattr(canvas, "app", None), "process_name", "") or "").lower()
        text_blob = " ".join(
            str(getattr(e, "text", "") or getattr(e, "name", "") or "")
            for e in getattr(canvas, "elements", []) or []
        ).lower()
        evidence: list[str] = []

        if self._has_loading_text(text_blob):
            return VisualPatternResult("loading_state", 0.85, ["loading_text"])

        search_state = self._classify_search_state(canvas, page_class, app_name, text_blob)
        if search_state.mode != "unknown_pattern":
            return search_state

        if "wechat" in page_class or "weixin" in page_class or "wechat" in app_name or "weixin" in app_name:
            if "发送" in text_blob or "send" in text_blob or self._has_chat_composer_geometry(canvas):
                return VisualPatternResult(
                    "chat_workspace",
                    0.72,
                    ["app_hint:wechat", "composer_geometry"],
                )
            return VisualPatternResult(
                "chat_app_browse_page",
                0.58,
                ["app_hint:wechat", "wechat_without_composer_geometry"],
            )

        if "qq" in page_class or "qq" in app_name:
            if "发送" in text_blob or "rich text editor" in text_blob or self._has_chat_composer_geometry(canvas):
                variant = self._classify_qq_chat_variant(canvas)
                evidence = ["app_hint:qq", "qq_chat_evidence"]
                if variant:
                    evidence.append(f"qq_variant:{variant}")
                return VisualPatternResult(
                    "chat_workspace",
                    0.72,
                    evidence,
                )
            return VisualPatternResult("account_switcher", 0.68, ["app_hint:qq"])

        if "lark" in page_class or "feishu" in page_class or "lark" in app_name or "feishu" in app_name:
            if self._has_collaboration_composer_geometry(canvas):
                return VisualPatternResult(
                    "collaboration_inbox",
                    0.72,
                    ["app_hint:lark", "collaboration_composer_geometry"],
                )
            return VisualPatternResult(
                "collaboration_browse_page",
                0.58,
                ["app_hint:lark", "collaboration_without_composer_geometry"],
            )

        for hint, mode in self._APP_MODE_HINTS.items():
            if hint in page_class or hint in app_name:
                evidence.append(f"app_hint:{hint}")
                confidence = 0.78 if mode == "loading_state" else 0.68
                return VisualPatternResult(mode, confidence, evidence)

        role_counts = self._role_counts(getattr(canvas, "elements", []) or [])
        if role_counts.get("search_input", 0) and (
            role_counts.get("list_item", 0) + role_counts.get("chat_item", 0) >= 2
        ):
            return VisualPatternResult(
                "list_management",
                0.72,
                ["search_input", "repeated_list_items"],
            )

        if "chatgpt" in page_class or "chatgpt" in app_name:
            return VisualPatternResult("chat_document", 0.76, ["app_hint:chatgpt"])

        if any(word in text_blob for word in ("播放", "play", "歌曲", "music")):
            return VisualPatternResult("media_home", 0.62, ["media_text"])

        return VisualPatternResult("unknown_pattern", 0.2, ["insufficient_evidence"])

    def _has_loading_text(self, text_blob: str) -> bool:
        return any(word in text_blob for word in self._LOADING_WORDS)

    def _role_counts(self, elements: list[Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for elem in elements:
            role = getattr(elem, "semantic_role", "unknown")
            role_value = getattr(role, "value", str(role))
            counts[role_value] = counts.get(role_value, 0) + 1
        return counts

    def _classify_search_state(
        self,
        canvas: Any,
        page_class: str,
        app_name: str,
        text_blob: str,
    ) -> VisualPatternResult:
        if "search-command-bar" in text_blob:
            return VisualPatternResult(
                "collaboration_search_overlay",
                0.82,
                ["search_state:feishu_command_bar"],
            )
        if "qq" in page_class or "qq" in app_name:
            if "进入全网搜索" in text_blob or "查找用户、群聊" in text_blob or "enter global search" in text_blob:
                return VisualPatternResult(
                    "chat_search_results",
                    0.82,
                    ["app_hint:qq", "search_state:qq_results"],
                )
        if "wechat" in page_class or "weixin" in page_class or "wechat" in app_name or "weixin" in app_name:
            if self._has_wechat_search_dropdown_geometry(canvas):
                return VisualPatternResult(
                    "chat_search_results",
                    0.72,
                    ["app_hint:wechat", "search_state:wechat_dropdown"],
                )
        return VisualPatternResult("unknown_pattern", 0.0, [])

    def _has_wechat_search_dropdown_geometry(self, canvas: Any) -> bool:
        close_seen = False
        plus_seen = False
        result_avatar_seen = False
        result_info_seen = False
        query_glyph_seen = False
        query_text_seen = False
        for elem in getattr(canvas, "elements", []) or []:
            bounds = getattr(elem, "bounds", None)
            if not bounds or len(bounds) != 4:
                continue
            left, top, right, bottom = [int(value) for value in bounds]
            box_w = right - left
            box_h = bottom - top
            if 35 <= top <= 80 and 220 <= left <= 260 and 14 <= box_w <= 35 and 14 <= box_h <= 42:
                close_seen = True
            if 35 <= top <= 80 and 250 <= left <= 300 and 18 <= box_w <= 45 and 18 <= box_h <= 42:
                plus_seen = True
            if 35 <= top <= 75 and 70 <= left <= 130 and 14 <= box_w <= 34 and 18 <= box_h <= 34:
                query_glyph_seen = True
            if 35 <= top <= 75 and 90 <= left <= 130 and 12 <= box_w <= 36 and 18 <= box_h <= 34:
                query_text_seen = True
            if 95 <= top <= 165 and 70 <= left <= 100 and 25 <= box_w <= 65 and 25 <= box_h <= 70:
                result_avatar_seen = True
            if 105 <= top <= 165 and 345 <= left <= 390 and 12 <= box_w <= 35 and 12 <= box_h <= 40:
                result_info_seen = True
        return close_seen and plus_seen and query_glyph_seen and query_text_seen and (result_avatar_seen or result_info_seen)

    def _has_chat_composer_geometry(self, canvas: Any) -> bool:
        width, height = self._canvas_size(canvas)
        if width <= 0 or height <= 0:
            return False
        bottom_y = int(height * 0.66)
        content_x = int(width * 0.30)
        controls = 0
        for elem in getattr(canvas, "elements", []) or []:
            bounds = getattr(elem, "bounds", None)
            if not bounds or len(bounds) != 4:
                continue
            left, top, right, bottom = [int(value) for value in bounds]
            if top < bottom_y or left < content_x:
                continue
            box_w = max(0, right - left)
            box_h = max(0, bottom - top)
            if 14 <= box_w <= 90 and 14 <= box_h <= 60:
                controls += 1
        return controls >= 3

    def _classify_qq_chat_variant(self, canvas: Any) -> str:
        """Best-effort split for QQ private chat vs group chat.

        This is intentionally conservative: group chat requires explicit group
        affordance evidence in the active chat pane. Otherwise a normal QQ chat
        with a top title is tagged as private for diagnostics only.
        """
        width, _ = self._canvas_size(canvas)
        if width <= 0:
            return "unknown"
        main_left = int(width * 0.30)
        top_title_seen = False
        group_terms = ("群公告", "群文件", "群相册", "群成员", "群应用", "群课堂", "群聊", "成员", "人在线")
        ignored_toolbar_terms = {"发起群聊", "更多", "语音通话", "视频通话", "屏幕共享", "远程协助"}
        for elem in getattr(canvas, "elements", []) or []:
            bounds = getattr(elem, "bounds", None)
            if not bounds or len(bounds) != 4:
                continue
            left, top, right, bottom = [int(value) for value in bounds]
            text = str(getattr(elem, "text", "") or getattr(elem, "name", "") or "").strip()
            if not text or left < main_left:
                continue
            if text in ignored_toolbar_terms:
                continue
            if any(term in text for term in group_terms):
                return "group_chat"
            if top <= 80 and (right - left) >= 12:
                top_title_seen = True
        return "private_chat" if top_title_seen else "unknown"

    def _has_collaboration_composer_geometry(self, canvas: Any) -> bool:
        width, height = self._canvas_size(canvas)
        if width <= 0 or height <= 0:
            return False
        controls = 0
        for elem in getattr(canvas, "elements", []) or []:
            bounds = getattr(elem, "bounds", None)
            if not bounds or len(bounds) != 4:
                continue
            left, top, right, bottom = [int(value) for value in bounds]
            if top < int(height * 0.70) or left < int(width * 0.45):
                continue
            box_w = max(0, right - left)
            box_h = max(0, bottom - top)
            if 12 <= box_w <= 90 and 12 <= box_h <= 60:
                controls += 1
        return controls >= 2

    def _canvas_size(self, canvas: Any) -> tuple[int, int]:
        artifacts = getattr(canvas, "artifacts", None) or {}
        size = artifacts.get("screenshot_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            return int(size[0]), int(size[1])
        max_right = 0
        max_bottom = 0
        for elem in getattr(canvas, "elements", []) or []:
            bounds = getattr(elem, "bounds", None)
            if not bounds or len(bounds) != 4:
                continue
            max_right = max(max_right, int(bounds[2]))
            max_bottom = max(max_bottom, int(bounds[3]))
        return max_right, max_bottom
