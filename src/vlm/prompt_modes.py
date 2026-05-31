"""VLM prompt task modes."""

from __future__ import annotations

from typing import Literal

PromptMode = Literal[
    "candidate_annotation",
    "region_understanding",
    "icon_crop_understanding",
    "missing_audit",
    "full_page_recognition",
]

CANDIDATE_ANNOTATION: PromptMode = "candidate_annotation"
REGION_UNDERSTANDING: PromptMode = "region_understanding"
ICON_CROP_UNDERSTANDING: PromptMode = "icon_crop_understanding"
MISSING_AUDIT: PromptMode = "missing_audit"
FULL_PAGE_RECOGNITION: PromptMode = "full_page_recognition"

SUPPORTED_PROMPT_MODES: set[str] = {
    CANDIDATE_ANNOTATION,
    REGION_UNDERSTANDING,
    ICON_CROP_UNDERSTANDING,
    MISSING_AUDIT,
    FULL_PAGE_RECOGNITION,
}
