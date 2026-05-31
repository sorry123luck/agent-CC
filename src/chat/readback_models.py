"""Data contract for OCR-first chat message readback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Bounds = tuple[int, int, int, int]
Sender = Literal["me", "peer", "unknown"]
MessageType = Literal["text", "image", "voice", "file", "link", "card", "unknown"]
ReadbackSource = Literal[
    "uia",
    "ocr_crop",
    "full_detail_ocr",
    "local_attachment_shape",
    "vlm_crop",
    "manual_review",
]


@dataclass(frozen=True)
class MessageAttachment:
    attachment_type: MessageType
    bounds: Bounds
    thumbnail_crop_path: str | None = None
    needs_vlm: bool = False
    text_hint: str | None = None


@dataclass(frozen=True)
class ChatMessageEvent:
    message_id: str
    sender: Sender
    message_type: MessageType
    text: str | None
    bounds: Bounds
    confidence: float
    source: ReadbackSource
    attachments: list[MessageAttachment] = field(default_factory=list)
    style: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatReadbackResult:
    window_id: str
    app_process: str
    events: list[ChatMessageEvent] = field(default_factory=list)
    status: Literal["pass", "warn", "fail"] = "pass"
    warnings: list[str] = field(default_factory=list)

    def has_text(self, expected: str) -> bool:
        needle = expected.strip()
        if not needle:
            return False
        normalized_needle = _compact_text(needle)
        return any(
            needle in (event.text or "")
            or (bool(normalized_needle) and normalized_needle in _compact_text(event.text or ""))
            for event in self.events
        )


def _compact_text(value: str) -> str:
    return "".join(value.split())
