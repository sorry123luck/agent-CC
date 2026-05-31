from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_DEFAULT_FAST_PROFILE = {
    "profile": "fast",
    "max_candidate_ids": 4,
    "image_max_edge": 320,
    "max_tokens": 256,
}

_DENSE_CONTROL_PROFILE = {
    "profile": "fast",
    "max_candidate_ids": 8,
    "image_max_edge": 384,
    "max_tokens": 384,
}

_CHAT_PROFILE = {
    "profile": "fast",
    "max_candidate_ids": 4,
    "image_max_edge": 384,
    "max_tokens": 384,
}


def roi_vlm_profile_for_jobs(jobs: Sequence[Mapping[str, Any]]) -> dict[str, int | str]:
    """Choose one compact provider budget for a batch of similar ROI jobs."""
    if any(_is_dense_control_job(job) for job in jobs):
        return dict(_DENSE_CONTROL_PROFILE)
    if any(_is_chat_or_collaboration_job(job) for job in jobs):
        return dict(_CHAT_PROFILE)
    return dict(_DEFAULT_FAST_PROFILE)


def roi_vlm_profile_for_job(job: Mapping[str, Any]) -> dict[str, int | str]:
    """Choose the provider budget for a single ROI job.

    Live sample batches can mix dense control ROIs with ordinary toolbar/list
    ROIs. Per-ROI selection keeps dense regions from inflating unrelated calls.
    """
    if _is_dense_control_job(job):
        return dict(_DENSE_CONTROL_PROFILE)
    if _is_chat_or_collaboration_job(job):
        return dict(_CHAT_PROFILE)
    return dict(_DEFAULT_FAST_PROFILE)


def _is_dense_control_job(job: Mapping[str, Any]) -> bool:
    mode = str(job.get("mode") or "")
    purpose = str(job.get("purpose") or "")
    return mode == "control_matrix" or purpose.endswith("_control_columns")


def _is_chat_or_collaboration_job(job: Mapping[str, Any]) -> bool:
    mode = str(job.get("mode") or "")
    return mode in {"chat_workspace", "chat_document", "collaboration_inbox"}
