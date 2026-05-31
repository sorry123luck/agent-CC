"""OCR-first chat message readback worker."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from PIL import Image

from src.chat.message_stream_crop import classify_message_side
from src.chat.readback_models import ChatMessageEvent, ChatReadbackResult, MessageAttachment
from src.perception.ocr_service import get_ocr_service


class ChatReadbackWorker:
    """Convert a chat message crop into structured readback events."""

    def __init__(self, ocr_service: Any | None = None) -> None:
        self._ocr_service = ocr_service or get_ocr_service()

    def read_crop(
        self,
        *,
        image_path: str | Path | None = None,
        image: Image.Image | None = None,
        app_process: str,
        window_id: str,
        stream_bounds: tuple[int, int, int, int],
        crop_origin: tuple[int, int] = (0, 0),
        local_visual_candidates: list[dict[str, Any]] | None = None,
        expected_self_texts: list[str] | None = None,
        sender_style_hints: dict[str, Any] | None = None,
    ) -> ChatReadbackResult:
        source_image = _load_source_image(image=image, image_path=image_path)
        blocks = self._run_ocr(image=image, image_path=image_path)
        events = [
            self._event_from_ocr_block(
                block=block,
                index=index,
                stream_bounds=stream_bounds,
                crop_origin=crop_origin,
                expected_self_texts=expected_self_texts or [],
                image=source_image,
                sender_style_hints=sender_style_hints or {},
            )
            for index, block in enumerate(blocks)
            if _should_keep_ocr_block(block, expected_self_texts or [])
        ]
        events.extend(
            self._attachment_event_from_candidate(candidate, index, stream_bounds=stream_bounds)
            for index, candidate in enumerate(local_visual_candidates or [])
            if _attachment_type_for_candidate(candidate) is not None
        )
        warnings = _readback_warnings(events)
        status = "warn" if warnings else "pass"
        return ChatReadbackResult(window_id=window_id, app_process=app_process, events=events, status=status, warnings=warnings)

    def _run_ocr(self, *, image: Image.Image | None, image_path: str | Path | None) -> list[Any]:
        if hasattr(self._ocr_service, "recognize_image"):
            return list(self._ocr_service.recognize_image(image if image is not None else image_path))
        if image is None:
            if image_path is None:
                return []
            image = Image.open(image_path)
        if hasattr(self._ocr_service, "extract"):
            return list(
                self._ocr_service.extract(
                    image=image,
                    min_text_length=1,
                    filter_pure_digits=False,
                    filter_pure_symbols=False,
                )
            )
        return []

    def _event_from_ocr_block(
        self,
        *,
        block: Any,
        index: int,
        stream_bounds: tuple[int, int, int, int],
        crop_origin: tuple[int, int],
        expected_self_texts: list[str],
        image: Image.Image | None,
        sender_style_hints: dict[str, Any],
    ) -> ChatMessageEvent:
        local_bounds = _block_bounds(block)
        bounds = _offset_bounds(local_bounds, crop_origin)
        text = _block_text(block)
        style = _bubble_style(image=image, local_bounds=local_bounds)
        style_sender = _sender_from_style(style, sender_style_hints)
        if _matches_expected_self_text(text, expected_self_texts):
            sender = "me"
        elif style_sender:
            sender = style_sender
            style["sender_style_match"] = style_sender
        else:
            sender = classify_message_side(
                bounds=bounds,
                stream_bounds=stream_bounds,
            )
        return ChatMessageEvent(
            message_id=f"ocr_{index}",
            sender=sender,  # type: ignore[arg-type]
            message_type="text",
            text=text,
            bounds=bounds,
            confidence=_block_confidence(block),
            source="ocr_crop",
            style=style,
        )

    def _attachment_event_from_candidate(
        self,
        candidate: dict[str, Any],
        index: int,
        *,
        stream_bounds: tuple[int, int, int, int],
    ) -> ChatMessageEvent:
        attachment_type = _attachment_type_for_candidate(candidate) or "unknown"
        bounds = _candidate_bounds(candidate)
        sender = classify_message_side(bounds=bounds, stream_bounds=stream_bounds)
        needs_vlm = attachment_type in {"image", "card", "unknown"}
        attachment = MessageAttachment(
            attachment_type=attachment_type,  # type: ignore[arg-type]
            bounds=bounds,
            thumbnail_crop_path=None,
            needs_vlm=needs_vlm,
            text_hint=_candidate_text_hint(candidate),
        )
        return ChatMessageEvent(
            message_id=f"attachment_{index}",
            sender=sender,  # type: ignore[arg-type]
            message_type=attachment_type,  # type: ignore[arg-type]
            text=attachment.text_hint,
            bounds=bounds,
            confidence=0.75,
            source="local_attachment_shape",
            attachments=[attachment],
        )


def _block_text(block: Any) -> str:
    if isinstance(block, dict):
        return str(block.get("text") or "").strip()
    return str(getattr(block, "text", "") or "").strip()


def _block_bounds(block: Any) -> tuple[int, int, int, int]:
    if isinstance(block, dict):
        value = block.get("bounds") or block.get("bbox") or (0, 0, 1, 1)
    else:
        value = getattr(block, "bounds", None) or getattr(block, "bbox", None) or (0, 0, 1, 1)
    try:
        left, top, right, bottom = [int(item) for item in value]
    except (TypeError, ValueError):
        return (0, 0, 1, 1)
    if right <= left or bottom <= top:
        return (0, 0, 1, 1)
    return (left, top, right, bottom)


def _block_confidence(block: Any) -> float:
    if isinstance(block, dict):
        value = block.get("confidence")
    else:
        value = getattr(block, "confidence", None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.8


def _offset_bounds(bounds: tuple[int, int, int, int], origin: tuple[int, int]) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    offset_x, offset_y = origin
    return (left + offset_x, top + offset_y, right + offset_x, bottom + offset_y)


def _matches_expected_self_text(text: str, expected_self_texts: list[str]) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    compact_normalized = _compact_text(normalized)
    return any(
        expected.strip()
        and (
            expected.strip() in normalized
            or _compact_text(expected.strip()) in compact_normalized
        )
        for expected in expected_self_texts
    )


def _compact_text(value: str) -> str:
    return "".join(value.split())


def _should_keep_ocr_block(block: Any, expected_self_texts: list[str]) -> bool:
    text = _block_text(block)
    if not text:
        return False
    if _matches_expected_self_text(text, expected_self_texts):
        return True
    if _looks_like_time_separator(text):
        return False
    if _looks_like_chat_metadata(text):
        return False
    confidence = _block_confidence(block)
    if confidence < 0.75 and len(text.strip()) <= 1:
        return False
    if confidence < 0.6 and _looks_like_symbol_noise(text):
        return False
    return True


def _looks_like_time_separator(text: str) -> bool:
    normalized = text.strip()
    return bool(re.fullmatch(r"\d{1,2}:\d{2}", normalized))


def _looks_like_chat_metadata(text: str) -> bool:
    normalized = text.strip()
    return bool(re.fullmatch(r"LV\d+.*", normalized, flags=re.IGNORECASE))


def _looks_like_symbol_noise(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return True
    if len(normalized) <= 2 and not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in normalized):
        return True
    return False


def _readback_warnings(events: list[ChatMessageEvent]) -> list[str]:
    if events and all(_looks_like_loading_indicator(event.text or "") for event in events):
        return ["message_stream_loading"]
    return []


def _looks_like_loading_indicator(text: str) -> bool:
    normalized = text.strip().lower().replace(" ", "")
    return normalized in {"正在加载", "正在加载...", "loading", "loading..."}


def _load_source_image(*, image: Image.Image | None, image_path: str | Path | None) -> Image.Image | None:
    if image is not None:
        return image.convert("RGB")
    if image_path is None:
        return None
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


def _bubble_style(*, image: Image.Image | None, local_bounds: tuple[int, int, int, int]) -> dict[str, Any]:
    if image is None:
        return {}
    left, top, right, bottom = local_bounds
    padding = 8
    sample_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(image.width, right + padding),
        min(image.height, bottom + padding),
    )
    if sample_box[2] <= sample_box[0] or sample_box[3] <= sample_box[1]:
        return {}
    crop = image.crop(sample_box)
    pixels = list(crop.getdata())
    if not pixels:
        return {}
    r = round(sum(pixel[0] for pixel in pixels) / len(pixels))
    g = round(sum(pixel[1] for pixel in pixels) / len(pixels))
    b = round(sum(pixel[2] for pixel in pixels) / len(pixels))
    return {"bubble_rgb": [r, g, b], "style_source": "crop_sample"}


def _sender_from_style(style: dict[str, Any], sender_style_hints: dict[str, Any]) -> str | None:
    rgb = style.get("bubble_rgb")
    if not isinstance(rgb, list) or len(rgb) != 3:
        return None
    distances: list[tuple[float, str]] = []
    for sender in ("me", "peer"):
        hint = sender_style_hints.get(sender)
        hint_rgb = _rgb_triplet(hint)
        if hint_rgb is None:
            continue
        distance = sum((int(rgb[index]) - hint_rgb[index]) ** 2 for index in range(3)) ** 0.5
        distances.append((distance, sender))
    if not distances:
        return None
    distances.sort()
    best_distance, best_sender = distances[0]
    if best_distance > 35:
        return None
    if len(distances) > 1 and distances[1][0] - best_distance < 12:
        return None
    return best_sender


def _rgb_triplet(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        rgb = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None
    if any(item < 0 or item > 255 for item in rgb):
        return None
    return rgb  # type: ignore[return-value]


def _attachment_type_for_candidate(candidate: dict[str, Any]) -> str | None:
    role = str(candidate.get("semantic_role") or candidate.get("role") or "").lower()
    control_type = str(candidate.get("control_type") or "").lower()
    name = str(candidate.get("name") or candidate.get("text") or "").lower()
    if "image" in role or control_type == "imagecontrol":
        return "image"
    if "voice" in role or "microphone" in role or "语音" in name:
        return "voice"
    if "file" in role or "attachment" in role or "文件" in name:
        return "file"
    if "link" in role or name.startswith(("http://", "https://")):
        return "link"
    if "card" in role:
        return "card"
    return None


def _candidate_bounds(candidate: dict[str, Any]) -> tuple[int, int, int, int]:
    return _block_bounds({"bounds": candidate.get("bounds") or candidate.get("bbox") or candidate.get("rect")})


def _candidate_text_hint(candidate: dict[str, Any]) -> str | None:
    for key in ("text", "name", "label", "ocr_text"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return value
    return None
