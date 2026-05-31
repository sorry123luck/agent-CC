"""PermanenceStateMachine — fixed_anchor 状态机引擎。

在 observe 管线末尾调用，根据 ConfidenceProfile、VisualAnchor 结果、
verify_count 等数据自动升降级 permanence_state。

状态定义：
  new → provisional → stable → fixed_anchor
                                  ↓
                                stale → degraded → retired

状态转换规则见 plan file section F。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from src.storage.schema import StableCandidateKey as StableCandidateKeyModel

_logger = logging.getLogger(__name__)


class PermanenceState(str, Enum):
    NEW = "new"
    PROVISIONAL = "provisional"
    STABLE = "stable"
    FIXED_ANCHOR = "fixed_anchor"
    STALE = "stale"
    DEGRADED = "degraded"
    RETIRED = "retired"


# Valid transitions (from → set of allowed targets)
_VALID_TRANSITIONS: dict[PermanenceState, set[PermanenceState]] = {
    PermanenceState.NEW: {PermanenceState.PROVISIONAL, PermanenceState.RETIRED},
    PermanenceState.PROVISIONAL: {PermanenceState.STABLE, PermanenceState.RETIRED},
    PermanenceState.STABLE: {
        PermanenceState.FIXED_ANCHOR,
        PermanenceState.STALE,
        PermanenceState.RETIRED,
    },
    PermanenceState.FIXED_ANCHOR: {PermanenceState.STALE, PermanenceState.RETIRED},
    PermanenceState.STALE: {
        PermanenceState.STABLE,
        PermanenceState.DEGRADED,
        PermanenceState.RETIRED,
    },
    PermanenceState.DEGRADED: {PermanenceState.STABLE, PermanenceState.RETIRED},
    PermanenceState.RETIRED: set(),  # terminal state
}

# Rank for upgrade/downgrade determination (higher = more established)
_STATE_RANK: dict[PermanenceState, int] = {
    PermanenceState.RETIRED: 0,
    PermanenceState.DEGRADED: 1,
    PermanenceState.STALE: 2,
    PermanenceState.NEW: 3,
    PermanenceState.PROVISIONAL: 4,
    PermanenceState.STABLE: 5,
    PermanenceState.FIXED_ANCHOR: 6,
}


@dataclass(frozen=True)
class StateTransition:
    """Immutable record of a state transition."""

    stable_key_id: str
    from_state: PermanenceState
    to_state: PermanenceState
    reason: str
    timestamp: str


class PermanenceStateMachine:
    """Permanence state machine for stable_candidate_keys.

    Called once per observe cycle per element with stable_key_id.
    Reads from element.attributes (written by EvidenceFusionEngine + VisualAnchorEngine).
    Writes permanence_state back to DB row.
    """

    def __init__(self) -> None:
        self._transitions: list[StateTransition] = []

    def evaluate_and_update(
        self,
        session,
        stable_key_id: str,
        element_attributes: dict,
        verify_count: int = 1,
        seen_count: int = 1,
    ) -> PermanenceState | None:
        """Evaluate state transition for one stable_key_id.

        Args:
            session: SQLAlchemy session
            stable_key_id: The key to evaluate
            element_attributes: element.attributes dict (contains confidence_profile, visual_anchor_*)
            verify_count: From StableCandidateKey.verify_count
            seen_count: Same as verify_count (they track together)

        Returns:
            New PermanenceState if transitioned, None if no change.
        """
        row = (
            session.query(StableCandidateKeyModel)
            .filter(StableCandidateKeyModel.key_id == stable_key_id)
            .first()
        )
        if row is None:
            return None

        current = PermanenceState(getattr(row, "permanence_state", None) or "new")

        # Extract signals from element_attributes
        confidence_profile = element_attributes.get("confidence_profile", {})
        fused_confidence = confidence_profile.get("fused_confidence", 0.0)
        source_diversity = confidence_profile.get("source_diversity", 0)
        source_count = confidence_profile.get("source_count", 0)

        visual_anchor_confidence = element_attributes.get("visual_anchor_confidence", 0.0)
        visual_anchor_status = element_attributes.get("visual_anchor_status", "")

        coordinate_drift = element_attributes.get("coordinate_drift", 0.0)
        bounds_valid = element_attributes.get("bounds_valid", True)
        fail_count = getattr(row, "fail_count", 0) or 0

        # Determine target state
        new_state = _compute_target_state(
            current=current,
            fused_confidence=fused_confidence,
            source_diversity=source_diversity,
            source_count=source_count,
            visual_anchor_confidence=visual_anchor_confidence,
            visual_anchor_status=visual_anchor_status,
            verify_count=verify_count,
            seen_count=seen_count,
            coordinate_drift=coordinate_drift,
            bounds_valid=bounds_valid,
            fail_count=fail_count,
        )

        if new_state == current:
            # Update coordinate_drift even if state unchanged
            row.coordinate_drift = coordinate_drift
            return None

        # Log warning for invalid transitions but still apply
        if new_state not in _VALID_TRANSITIONS.get(current, set()):
            _logger.warning(
                "Invalid state transition %s → %s for key=%s, forcing",
                current.value,
                new_state.value,
                stable_key_id,
            )

        now = datetime.now(timezone.utc).isoformat()
        old_state = current
        row.permanence_state = new_state.value
        row.state_changed_at = now
        row.coordinate_drift = coordinate_drift

        # Reset fail_count on upgrade; increment on downgrade
        if _is_upgrade(old_state, new_state):
            row.fail_count = 0
        elif _is_downgrade(old_state, new_state):
            row.fail_count = fail_count + 1

        transition = StateTransition(
            stable_key_id=stable_key_id,
            from_state=old_state,
            to_state=new_state,
            reason=_build_reason(old_state, new_state, fused_confidence, visual_anchor_confidence),
            timestamp=now,
        )
        self._transitions.append(transition)
        _logger.info(
            "Permanence state: %s → %s for key=%s (%s)",
            old_state.value,
            new_state.value,
            stable_key_id,
            transition.reason,
        )
        return new_state

    @property
    def transitions(self) -> list[StateTransition]:
        """All transitions recorded in this session."""
        return list(self._transitions)


# ---------------------------------------------------------------------------
# Pure functions (testable without DB)
# ---------------------------------------------------------------------------


def _compute_target_state(
    *,
    current: PermanenceState,
    fused_confidence: float,
    source_diversity: int,
    source_count: int,
    visual_anchor_confidence: float,
    visual_anchor_status: str,
    verify_count: int,
    seen_count: int,
    coordinate_drift: float,
    bounds_valid: bool,
    fail_count: int,
) -> PermanenceState:
    """Compute the target state based on signals. Pure function, no side effects."""

    # --- Retired check (any state → retired) ---
    if fail_count >= 5:
        return PermanenceState.RETIRED
    if current in (PermanenceState.DEGRADED, PermanenceState.STALE):
        if visual_anchor_status == "no_match" and fail_count >= 3:
            return PermanenceState.RETIRED

    # --- Downgrade checks (higher priority than upgrade) ---
    if current == PermanenceState.FIXED_ANCHOR:
        if visual_anchor_confidence < 0.80 or coordinate_drift > 10:
            return PermanenceState.STALE
        if visual_anchor_status == "no_match":
            return PermanenceState.STALE

    if current == PermanenceState.STALE:
        if visual_anchor_status == "no_match" or fused_confidence < 0.50:
            return PermanenceState.DEGRADED

    # --- Recovery checks ---
    if current == PermanenceState.STALE:
        if (
            visual_anchor_confidence >= 0.80
            and fused_confidence >= 0.60
            and visual_anchor_status != "no_match"
        ):
            return PermanenceState.STABLE

    if current == PermanenceState.DEGRADED:
        if (
            visual_anchor_confidence >= 0.80
            and fused_confidence >= 0.60
            and source_diversity >= 2
            and visual_anchor_status != "no_match"
        ):
            return PermanenceState.STABLE

    # --- Upgrade checks ---
    if current == PermanenceState.NEW:
        if source_count >= 1 and bounds_valid:
            return PermanenceState.PROVISIONAL

    if current == PermanenceState.PROVISIONAL:
        if (
            source_count >= 3
            and source_diversity >= 2
            and fused_confidence >= 0.60
            and bounds_valid
        ):
            return PermanenceState.STABLE

    if current == PermanenceState.STABLE:
        if (
            seen_count >= 5
            and verify_count >= 3
            and source_diversity >= 2
            and fused_confidence >= 0.85
            and visual_anchor_confidence >= 0.90
            and fail_count == 0
            and coordinate_drift < 5
            and bounds_valid
        ):
            return PermanenceState.FIXED_ANCHOR

    return current


def _is_upgrade(old: PermanenceState, new: PermanenceState) -> bool:
    return _STATE_RANK.get(new, 0) > _STATE_RANK.get(old, 0)


def _is_downgrade(old: PermanenceState, new: PermanenceState) -> bool:
    return _STATE_RANK.get(new, 0) < _STATE_RANK.get(old, 0)


def _build_reason(
    old: PermanenceState,
    new: PermanenceState,
    fused: float,
    va_conf: float,
) -> str:
    direction = "upgrade" if _is_upgrade(old, new) else "downgrade"
    return f"{direction}: fused={fused:.2f}, va_conf={va_conf:.2f}"
