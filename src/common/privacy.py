"""Privacy utilities for field-level redaction of InteractionCanvas data.

Supports different redaction levels:
- realtime: No redaction (Agent needs real data to work)
- memory: Sensitive text replaced with placeholders
- ocpack: Sensitive text replaced, screenshots optional (default excluded)
- log: Sensitive fields redacted before logging
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

REDACTED_PLACEHOLDER = "[REDACTED]"

# Patterns that likely contain sensitive information
_SENSITIVE_TEXT_PATTERNS = [
    re.compile(r"\b\d{16,19}\b"),  # credit card numbers
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN format
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),  # email
    re.compile(r"\b(?:password|密码|口令|pin)\s*[:=]\s*\S+", re.IGNORECASE),
]

# Window title patterns that may leak user info
_WINDOW_TITLE_SENSITIVE = [
    re.compile(r"(?:C|D|E|F):\\Users\\[^\\]+", re.IGNORECASE),  # user path
]


@dataclass(frozen=True)
class PrivacyPolicy:
    """Privacy policy for redaction."""
    level: str = "realtime"  # realtime / memory / ocpack / log
    redact_text: bool = False
    redact_window_title: bool = False
    include_screenshots: bool = False
    sensitive_keywords: list[str] = field(default_factory=lambda: [
        "password", "密码", "token", "secret", "api_key",
        "credit_card", "信用卡", "身份证",
    ])


def _is_sensitive_text(text: str, keywords: list[str] | None = None) -> bool:
    """Check if text likely contains sensitive information."""
    if not text:
        return False
    lower = text.lower()
    kw_list = keywords or PrivacyPolicy().sensitive_keywords
    if any(kw.lower() in lower for kw in kw_list):
        return True
    for pattern in _SENSITIVE_TEXT_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _redact_text_value(text: str) -> str:
    """Redact a single text value if it's sensitive."""
    if _is_sensitive_text(text):
        return REDACTED_PLACEHOLDER
    return text


def _redact_window_title(title: str) -> str:
    """Redact sensitive parts of a window title."""
    result = title
    for pattern in _WINDOW_TITLE_SENSITIVE:
        result = pattern.sub(REDACTED_PLACEHOLDER, result)
    return result


def redact_candidate_text(text: str, policy: PrivacyPolicy) -> str:
    """Redact candidate text based on policy."""
    if policy.level == "realtime":
        return text
    if policy.redact_text:
        return _redact_text_value(text)
    return text


def redact_canvas_dict(
    canvas_data: dict[str, Any],
    policy: PrivacyPolicy,
) -> dict[str, Any]:
    """Redact sensitive fields in a canvas dictionary (serialized form).

    Works on dict representation so it can handle both InteractionCanvas
    and arbitrary nested structures.
    """
    if policy.level == "realtime":
        return canvas_data

    result = dict(canvas_data)

    # Redact window title
    if policy.redact_window_title and "window" in result:
        window = dict(result["window"])
        if "title" in window:
            window["title"] = _redact_window_title(str(window["title"]))
        result["window"] = window

    # Redact candidate text
    if policy.redact_text and "candidates" in result:
        redacted_candidates = []
        for c in result["candidates"]:
            c = dict(c)
            if "text" in c and c["text"]:
                c["text"] = _redact_text_value(str(c["text"]))
            redacted_candidates.append(c)
        result["candidates"] = redacted_candidates

    # Strip screenshots unless explicitly included
    if not policy.include_screenshots:
        result.pop("screenshot_path", None)
        result.pop("screenshot", None)

    return result
