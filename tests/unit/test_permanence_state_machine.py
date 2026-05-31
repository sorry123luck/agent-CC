"""Tests for PermanenceStateMachine — fixed_anchor 状态机引擎。"""

import pytest

from src.memory.permanence_state_machine import (
    PermanenceState,
    PermanenceStateMachine,
    StateTransition,
    _build_reason,
    _compute_target_state,
    _is_downgrade,
    _is_upgrade,
)
from src.storage.db import Database
from src.storage.migration_evidence import migrate
from src.storage.schema import StableCandidateKey as SCKModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_all()
    migrate(db)
    return db


@pytest.fixture
def sm():
    return PermanenceStateMachine()


def _seed_key(
    db,
    key_id="k1",
    app_id="app1",
    page_class="main",
    permanence_state="new",
    verify_count=1,
    fail_count=0,
    coordinate_drift=0.0,
):
    """Insert a StableCandidateKey row for testing."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    with db.session() as session:
        row = SCKModel(
            key_id=key_id,
            app_id=app_id,
            page_class=page_class,
            canonical_region="r1",
            canonical_text="text",
            canonical_role="Button",
            created_at=now,
            last_seen_at=now,
            verify_count=verify_count,
            permanence_state=permanence_state,
            state_changed_at=now,
            fail_count=fail_count,
            coordinate_drift=coordinate_drift,
        )
        session.add(row)
        session.flush()


def _make_attrs(
    fused_confidence=0.0,
    source_diversity=0,
    source_count=0,
    visual_anchor_confidence=0.0,
    visual_anchor_status="",
    coordinate_drift=0.0,
    bounds_valid=True,
):
    """Build a fake element.attributes dict."""
    return {
        "confidence_profile": {
            "fused_confidence": fused_confidence,
            "source_diversity": source_diversity,
            "source_count": source_count,
        },
        "visual_anchor_confidence": visual_anchor_confidence,
        "visual_anchor_status": visual_anchor_status,
        "coordinate_drift": coordinate_drift,
        "bounds_valid": bounds_valid,
    }


# ---------------------------------------------------------------------------
# Test: Pure function — _compute_target_state
# ---------------------------------------------------------------------------


class TestComputeTargetState:
    def test_new_to_provisional(self):
        result = _compute_target_state(
            current=PermanenceState.NEW,
            fused_confidence=0.0,
            source_diversity=0,
            source_count=1,
            visual_anchor_confidence=0.0,
            visual_anchor_status="",
            verify_count=1,
            seen_count=1,
            coordinate_drift=0.0,
            bounds_valid=True,
            fail_count=0,
        )
        assert result == PermanenceState.PROVISIONAL

    def test_new_stays_new_without_source(self):
        result = _compute_target_state(
            current=PermanenceState.NEW,
            fused_confidence=0.0,
            source_diversity=0,
            source_count=0,
            visual_anchor_confidence=0.0,
            visual_anchor_status="",
            verify_count=1,
            seen_count=1,
            coordinate_drift=0.0,
            bounds_valid=True,
            fail_count=0,
        )
        assert result == PermanenceState.NEW

    def test_new_stays_new_invalid_bounds(self):
        result = _compute_target_state(
            current=PermanenceState.NEW,
            fused_confidence=0.0,
            source_diversity=0,
            source_count=1,
            visual_anchor_confidence=0.0,
            visual_anchor_status="",
            verify_count=1,
            seen_count=1,
            coordinate_drift=0.0,
            bounds_valid=False,
            fail_count=0,
        )
        assert result == PermanenceState.NEW

    def test_provisional_to_stable(self):
        result = _compute_target_state(
            current=PermanenceState.PROVISIONAL,
            fused_confidence=0.60,
            source_diversity=2,
            source_count=3,
            visual_anchor_confidence=0.0,
            visual_anchor_status="",
            verify_count=3,
            seen_count=3,
            coordinate_drift=0.0,
            bounds_valid=True,
            fail_count=0,
        )
        assert result == PermanenceState.STABLE

    def test_provisional_stays_insufficient_sources(self):
        result = _compute_target_state(
            current=PermanenceState.PROVISIONAL,
            fused_confidence=0.60,
            source_diversity=1,  # need ≥2
            source_count=3,
            visual_anchor_confidence=0.0,
            visual_anchor_status="",
            verify_count=3,
            seen_count=3,
            coordinate_drift=0.0,
            bounds_valid=True,
            fail_count=0,
        )
        assert result == PermanenceState.PROVISIONAL

    def test_stable_to_fixed_anchor(self):
        result = _compute_target_state(
            current=PermanenceState.STABLE,
            fused_confidence=0.85,
            source_diversity=2,
            source_count=5,
            visual_anchor_confidence=0.90,
            visual_anchor_status="exact_match",
            verify_count=3,
            seen_count=5,
            coordinate_drift=2.0,
            bounds_valid=True,
            fail_count=0,
        )
        assert result == PermanenceState.FIXED_ANCHOR

    def test_fixed_anchor_to_stale_low_va_conf(self):
        result = _compute_target_state(
            current=PermanenceState.FIXED_ANCHOR,
            fused_confidence=0.90,
            source_diversity=3,
            source_count=10,
            visual_anchor_confidence=0.79,  # < 0.80
            visual_anchor_status="weak_match",
            verify_count=10,
            seen_count=10,
            coordinate_drift=2.0,
            bounds_valid=True,
            fail_count=0,
        )
        assert result == PermanenceState.STALE

    def test_fixed_anchor_to_stale_high_drift(self):
        result = _compute_target_state(
            current=PermanenceState.FIXED_ANCHOR,
            fused_confidence=0.90,
            source_diversity=3,
            source_count=10,
            visual_anchor_confidence=0.85,
            visual_anchor_status="drift_detected",
            verify_count=10,
            seen_count=10,
            coordinate_drift=11.0,  # > 10
            bounds_valid=True,
            fail_count=0,
        )
        assert result == PermanenceState.STALE

    def test_fixed_anchor_to_stale_no_match(self):
        result = _compute_target_state(
            current=PermanenceState.FIXED_ANCHOR,
            fused_confidence=0.90,
            source_diversity=3,
            source_count=10,
            visual_anchor_confidence=0.0,
            visual_anchor_status="no_match",
            verify_count=10,
            seen_count=10,
            coordinate_drift=0.0,
            bounds_valid=True,
            fail_count=0,
        )
        assert result == PermanenceState.STALE

    def test_stale_to_degraded_no_match(self):
        result = _compute_target_state(
            current=PermanenceState.STALE,
            fused_confidence=0.70,
            source_diversity=2,
            source_count=5,
            visual_anchor_confidence=0.0,
            visual_anchor_status="no_match",
            verify_count=5,
            seen_count=5,
            coordinate_drift=0.0,
            bounds_valid=True,
            fail_count=1,
        )
        assert result == PermanenceState.DEGRADED

    def test_stale_to_degraded_low_fused(self):
        result = _compute_target_state(
            current=PermanenceState.STALE,
            fused_confidence=0.49,  # < 0.50
            source_diversity=2,
            source_count=5,
            visual_anchor_confidence=0.80,
            visual_anchor_status="strong_match",
            verify_count=5,
            seen_count=5,
            coordinate_drift=0.0,
            bounds_valid=True,
            fail_count=1,
        )
        assert result == PermanenceState.DEGRADED

    def test_any_state_to_retired_fail_count_5(self):
        for state in PermanenceState:
            if state == PermanenceState.RETIRED:
                continue
            result = _compute_target_state(
                current=state,
                fused_confidence=0.90,
                source_diversity=3,
                source_count=10,
                visual_anchor_confidence=0.95,
                visual_anchor_status="exact_match",
                verify_count=10,
                seen_count=10,
                coordinate_drift=0.0,
                bounds_valid=True,
                fail_count=5,
            )
            assert result == PermanenceState.RETIRED, f"{state} should go to retired"

    def test_stale_recovery_to_stable(self):
        result = _compute_target_state(
            current=PermanenceState.STALE,
            fused_confidence=0.60,
            source_diversity=2,
            source_count=5,
            visual_anchor_confidence=0.80,
            visual_anchor_status="strong_match",
            verify_count=5,
            seen_count=5,
            coordinate_drift=0.0,
            bounds_valid=True,
            fail_count=1,
        )
        assert result == PermanenceState.STABLE

    def test_degraded_recovery_to_stable(self):
        result = _compute_target_state(
            current=PermanenceState.DEGRADED,
            fused_confidence=0.60,
            source_diversity=2,
            source_count=5,
            visual_anchor_confidence=0.80,
            visual_anchor_status="strong_match",
            verify_count=5,
            seen_count=5,
            coordinate_drift=0.0,
            bounds_valid=True,
            fail_count=2,
        )
        assert result == PermanenceState.STABLE

    def test_degraded_stays_without_diversity(self):
        result = _compute_target_state(
            current=PermanenceState.DEGRADED,
            fused_confidence=0.60,
            source_diversity=1,  # need ≥2
            source_count=5,
            visual_anchor_confidence=0.80,
            visual_anchor_status="strong_match",
            verify_count=5,
            seen_count=5,
            coordinate_drift=0.0,
            bounds_valid=True,
            fail_count=2,
        )
        assert result == PermanenceState.DEGRADED


# ---------------------------------------------------------------------------
# Test: Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_exact_threshold_fused_0_60(self):
        """fused=0.60 → stable (boundary passes)."""
        result = _compute_target_state(
            current=PermanenceState.PROVISIONAL,
            fused_confidence=0.60,
            source_diversity=2,
            source_count=3,
            visual_anchor_confidence=0.0,
            visual_anchor_status="",
            verify_count=3,
            seen_count=3,
            coordinate_drift=0.0,
            bounds_valid=True,
            fail_count=0,
        )
        assert result == PermanenceState.STABLE

    def test_exact_threshold_fused_0_59(self):
        """fused=0.59 → not stable (boundary fails)."""
        result = _compute_target_state(
            current=PermanenceState.PROVISIONAL,
            fused_confidence=0.59,
            source_diversity=2,
            source_count=3,
            visual_anchor_confidence=0.0,
            visual_anchor_status="",
            verify_count=3,
            seen_count=3,
            coordinate_drift=0.0,
            bounds_valid=True,
            fail_count=0,
        )
        assert result == PermanenceState.PROVISIONAL

    def test_fail_count_5_retired(self):
        result = _compute_target_state(
            current=PermanenceState.STABLE,
            fused_confidence=0.90,
            source_diversity=3,
            source_count=10,
            visual_anchor_confidence=0.95,
            visual_anchor_status="exact_match",
            verify_count=10,
            seen_count=10,
            coordinate_drift=0.0,
            bounds_valid=True,
            fail_count=5,
        )
        assert result == PermanenceState.RETIRED

    def test_fail_count_4_not_retired(self):
        result = _compute_target_state(
            current=PermanenceState.STABLE,
            fused_confidence=0.90,
            source_diversity=3,
            source_count=10,
            visual_anchor_confidence=0.95,
            visual_anchor_status="exact_match",
            verify_count=10,
            seen_count=10,
            coordinate_drift=0.0,
            bounds_valid=True,
            fail_count=4,
        )
        # fail_count=4 < 5, so not retired; but fail_count != 0 blocks fixed_anchor upgrade
        assert result == PermanenceState.STABLE

    def test_coordinate_drift_10_stale(self):
        result = _compute_target_state(
            current=PermanenceState.FIXED_ANCHOR,
            fused_confidence=0.90,
            source_diversity=3,
            source_count=10,
            visual_anchor_confidence=0.85,
            visual_anchor_status="drift_detected",
            verify_count=10,
            seen_count=10,
            coordinate_drift=10.0,  # boundary: not > 10, so no stale
            bounds_valid=True,
            fail_count=0,
        )
        # drift=10.0 is NOT > 10, so stays fixed_anchor
        assert result == PermanenceState.FIXED_ANCHOR

    def test_coordinate_drift_10_01_stale(self):
        result = _compute_target_state(
            current=PermanenceState.FIXED_ANCHOR,
            fused_confidence=0.90,
            source_diversity=3,
            source_count=10,
            visual_anchor_confidence=0.85,
            visual_anchor_status="drift_detected",
            verify_count=10,
            seen_count=10,
            coordinate_drift=10.01,  # > 10
            bounds_valid=True,
            fail_count=0,
        )
        assert result == PermanenceState.STALE

    def test_va_conf_0_80_not_stale(self):
        result = _compute_target_state(
            current=PermanenceState.FIXED_ANCHOR,
            fused_confidence=0.90,
            source_diversity=3,
            source_count=10,
            visual_anchor_confidence=0.80,  # not < 0.80
            visual_anchor_status="strong_match",
            verify_count=10,
            seen_count=10,
            coordinate_drift=2.0,
            bounds_valid=True,
            fail_count=0,
        )
        assert result == PermanenceState.FIXED_ANCHOR

    def test_va_conf_0_79_stale(self):
        result = _compute_target_state(
            current=PermanenceState.FIXED_ANCHOR,
            fused_confidence=0.90,
            source_diversity=3,
            source_count=10,
            visual_anchor_confidence=0.79,  # < 0.80
            visual_anchor_status="strong_match",
            verify_count=10,
            seen_count=10,
            coordinate_drift=2.0,
            bounds_valid=True,
            fail_count=0,
        )
        assert result == PermanenceState.STALE


# ---------------------------------------------------------------------------
# Test: Upgrade/downgrade helpers
# ---------------------------------------------------------------------------


class TestUpgradeDowngrade:
    def test_upgrade_new_to_provisional(self):
        assert _is_upgrade(PermanenceState.NEW, PermanenceState.PROVISIONAL) is True

    def test_upgrade_stable_to_fixed(self):
        assert _is_upgrade(PermanenceState.STABLE, PermanenceState.FIXED_ANCHOR) is True

    def test_downgrade_fixed_to_stale(self):
        assert _is_downgrade(PermanenceState.FIXED_ANCHOR, PermanenceState.STALE) is True

    def test_downgrade_stale_to_degraded(self):
        assert _is_downgrade(PermanenceState.STALE, PermanenceState.DEGRADED) is True

    def test_same_state_not_upgrade(self):
        assert _is_upgrade(PermanenceState.STABLE, PermanenceState.STABLE) is False

    def test_same_state_not_downgrade(self):
        assert _is_downgrade(PermanenceState.STABLE, PermanenceState.STABLE) is False

    def test_recovery_is_upgrade(self):
        assert _is_upgrade(PermanenceState.STALE, PermanenceState.STABLE) is True

    def test_build_reason_upgrade(self):
        reason = _build_reason(PermanenceState.NEW, PermanenceState.PROVISIONAL, 0.5, 0.0)
        assert "upgrade" in reason
        assert "fused=0.50" in reason

    def test_build_reason_downgrade(self):
        reason = _build_reason(PermanenceState.FIXED_ANCHOR, PermanenceState.STALE, 0.9, 0.7)
        assert "downgrade" in reason
        assert "va_conf=0.70" in reason


# ---------------------------------------------------------------------------
# Test: DB integration — evaluate_and_update
# ---------------------------------------------------------------------------


class TestEvaluateAndUpdate:
    def test_new_to_provisional_db(self, db, sm):
        _seed_key(db, "k1", permanence_state="new")
        attrs = _make_attrs(source_count=1)
        with db.session() as session:
            result = sm.evaluate_and_update(session, "k1", attrs)
        assert result == PermanenceState.PROVISIONAL

    def test_no_change_returns_none(self, db, sm):
        _seed_key(db, "k1", permanence_state="stable")
        attrs = _make_attrs(fused_confidence=0.50, source_diversity=1, source_count=2)
        with db.session() as session:
            result = sm.evaluate_and_update(session, "k1", attrs)
        assert result is None

    def test_coordinate_drift_updated_on_no_change(self, db, sm):
        _seed_key(db, "k1", permanence_state="stable", coordinate_drift=0.0)
        attrs = _make_attrs(
            fused_confidence=0.50,
            source_diversity=1,
            source_count=2,
            coordinate_drift=3.5,
        )
        with db.session() as session:
            sm.evaluate_and_update(session, "k1", attrs)
            row = session.query(SCKModel).filter(SCKModel.key_id == "k1").first()
            assert row.coordinate_drift == 3.5

    def test_nonexistent_key_returns_none(self, db, sm):
        attrs = _make_attrs()
        with db.session() as session:
            result = sm.evaluate_and_update(session, "nonexistent", attrs)
        assert result is None

    def test_transition_recorded(self, db, sm):
        _seed_key(db, "k1", permanence_state="new")
        attrs = _make_attrs(source_count=1)
        with db.session() as session:
            sm.evaluate_and_update(session, "k1", attrs)
        assert len(sm.transitions) == 1
        assert sm.transitions[0].from_state == PermanenceState.NEW
        assert sm.transitions[0].to_state == PermanenceState.PROVISIONAL
        assert sm.transitions[0].stable_key_id == "k1"

    def test_fail_count_reset_on_upgrade(self, db, sm):
        _seed_key(db, "k1", permanence_state="new", fail_count=2)
        attrs = _make_attrs(source_count=1)
        with db.session() as session:
            sm.evaluate_and_update(session, "k1", attrs)
            row = session.query(SCKModel).filter(SCKModel.key_id == "k1").first()
            assert row.fail_count == 0  # reset on upgrade

    def test_fail_count_increments_on_downgrade(self, db, sm):
        _seed_key(db, "k1", permanence_state="fixed_anchor", fail_count=0)
        attrs = _make_attrs(
            fused_confidence=0.90,
            source_diversity=3,
            source_count=10,
            visual_anchor_confidence=0.79,  # triggers stale
            visual_anchor_status="weak_match",
        )
        with db.session() as session:
            sm.evaluate_and_update(session, "k1", attrs)
            row = session.query(SCKModel).filter(SCKModel.key_id == "k1").first()
            assert row.fail_count == 1

    def test_permanence_state_persisted(self, db, sm):
        _seed_key(db, "k1", permanence_state="new")
        attrs = _make_attrs(source_count=1)
        with db.session() as session:
            sm.evaluate_and_update(session, "k1", attrs)
        # Read back in new session
        with db.session() as session:
            row = session.query(SCKModel).filter(SCKModel.key_id == "k1").first()
            assert row.permanence_state == "provisional"
            assert row.state_changed_at is not None


# ---------------------------------------------------------------------------
# Test: Full lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_new_to_fixed_anchor(self, db, sm):
        """Simulate full lifecycle: new → provisional → stable → fixed_anchor."""
        _seed_key(db, "k1", permanence_state="new", verify_count=1)

        # Step 1: new → provisional
        with db.session() as session:
            result = sm.evaluate_and_update(
                session, "k1", _make_attrs(source_count=1), verify_count=1,
            )
        assert result == PermanenceState.PROVISIONAL

        # Step 2: provisional → stable
        with db.session() as session:
            result = sm.evaluate_and_update(
                session,
                "k1",
                _make_attrs(
                    fused_confidence=0.60,
                    source_diversity=2,
                    source_count=3,
                ),
                verify_count=3,
            )
        assert result == PermanenceState.STABLE

        # Step 3: stable → fixed_anchor
        with db.session() as session:
            result = sm.evaluate_and_update(
                session,
                "k1",
                _make_attrs(
                    fused_confidence=0.85,
                    source_diversity=2,
                    source_count=5,
                    visual_anchor_confidence=0.90,
                    visual_anchor_status="exact_match",
                    coordinate_drift=2.0,
                ),
                verify_count=3,
                seen_count=5,
            )
        assert result == PermanenceState.FIXED_ANCHOR

        # Verify final state in DB
        with db.session() as session:
            row = session.query(SCKModel).filter(SCKModel.key_id == "k1").first()
            assert row.permanence_state == "fixed_anchor"
            assert row.fail_count == 0

    def test_downgrade_and_recovery(self, db, sm):
        """fixed_anchor → stale → stable."""
        _seed_key(db, "k1", permanence_state="fixed_anchor", verify_count=10)

        # Step 1: fixed_anchor → stale (low va_conf)
        with db.session() as session:
            result = sm.evaluate_and_update(
                session,
                "k1",
                _make_attrs(
                    fused_confidence=0.90,
                    source_diversity=3,
                    source_count=10,
                    visual_anchor_confidence=0.79,
                    visual_anchor_status="weak_match",
                ),
                verify_count=10,
            )
        assert result == PermanenceState.STALE

        # Step 2: stale → stable (recovery)
        with db.session() as session:
            result = sm.evaluate_and_update(
                session,
                "k1",
                _make_attrs(
                    fused_confidence=0.70,
                    source_diversity=2,
                    source_count=5,
                    visual_anchor_confidence=0.85,
                    visual_anchor_status="strong_match",
                ),
                verify_count=10,
            )
        assert result == PermanenceState.STABLE

    def test_stale_to_degraded_to_retired(self, db, sm):
        """stale → degraded → retired."""
        _seed_key(db, "k1", permanence_state="stale", verify_count=5, fail_count=1)

        # Step 1: stale → degraded
        with db.session() as session:
            result = sm.evaluate_and_update(
                session,
                "k1",
                _make_attrs(
                    fused_confidence=0.40,
                    visual_anchor_status="no_match",
                ),
                verify_count=5,
            )
        assert result == PermanenceState.DEGRADED

        # Step 2: degraded → retired (fail_count accumulates)
        # Need fail_count to reach 5
        with db.session() as session:
            # Simulate multiple failures
            row = session.query(SCKModel).filter(SCKModel.key_id == "k1").first()
            row.fail_count = 4  # one more downgrade will make it 5
            session.flush()
            result = sm.evaluate_and_update(
                session,
                "k1",
                _make_attrs(
                    fused_confidence=0.40,
                    visual_anchor_status="no_match",
                ),
                verify_count=5,
            )
        assert result == PermanenceState.RETIRED


# ---------------------------------------------------------------------------
# Test: VLM frequency reduction (should_analyze)
# ---------------------------------------------------------------------------


class TestVLMFrequency:
    def test_fixed_anchor_skips_vlm(self):
        """fixed_anchor + observe_count%3≠0 → skip."""
        from unittest.mock import MagicMock
        from src.vlm.modeler import VLMSemanticModeler

        modeler = MagicMock(spec=VLMSemanticModeler)
        # Call the real method
        modeler.should_analyze = VLMSemanticModeler.should_analyze.__get__(modeler)
        modeler._config = MagicMock()
        modeler._config.enabled = True

        allowed, reason = modeler.should_analyze(
            permanence_state="fixed_anchor",
            observe_count=2,  # 2 % 3 ≠ 0
        )
        assert allowed is False
        assert reason == "fixed_anchor_reduced_frequency"

    def test_fixed_anchor_allows_vlm(self):
        """fixed_anchor + observe_count%3==0 → allow."""
        from unittest.mock import MagicMock
        from src.vlm.modeler import VLMSemanticModeler

        modeler = MagicMock(spec=VLMSemanticModeler)
        modeler.should_analyze = VLMSemanticModeler.should_analyze.__get__(modeler)
        modeler._config = MagicMock()
        modeler._config.enabled = True
        modeler._primary = MagicMock()
        modeler._primary.is_available.return_value = True
        modeler._budget = MagicMock()
        modeler._budget.check_budget.return_value = (True, None)

        allowed, reason = modeler.should_analyze(
            permanence_state="fixed_anchor",
            observe_count=3,  # 3 % 3 == 0
        )
        assert allowed is True

    def test_fixed_anchor_drift_forces_vlm(self):
        """fixed_anchor + drift_detected → always allow."""
        from unittest.mock import MagicMock
        from src.vlm.modeler import VLMSemanticModeler

        modeler = MagicMock(spec=VLMSemanticModeler)
        modeler.should_analyze = VLMSemanticModeler.should_analyze.__get__(modeler)
        modeler._config = MagicMock()
        modeler._config.enabled = True

        allowed, reason = modeler.should_analyze(
            permanence_state="fixed_anchor",
            observe_count=2,
            drift_detected=True,
        )
        assert allowed is True
        assert reason == "drift_detected"

    def test_fixed_anchor_force_forces_vlm(self):
        """fixed_anchor + force → always allow."""
        from unittest.mock import MagicMock
        from src.vlm.modeler import VLMSemanticModeler

        modeler = MagicMock(spec=VLMSemanticModeler)
        modeler.should_analyze = VLMSemanticModeler.should_analyze.__get__(modeler)
        modeler._config = MagicMock()
        modeler._config.enabled = True

        allowed, reason = modeler.should_analyze(
            permanence_state="fixed_anchor",
            observe_count=2,
            force=True,
        )
        assert allowed is True
        assert reason == "forced"

    def test_non_fixed_anchor_ignores_observe_count(self):
        """stable state → no frequency reduction."""
        from unittest.mock import MagicMock
        from src.vlm.modeler import VLMSemanticModeler

        modeler = MagicMock(spec=VLMSemanticModeler)
        modeler.should_analyze = VLMSemanticModeler.should_analyze.__get__(modeler)
        modeler._config = MagicMock()
        modeler._config.enabled = True
        modeler._primary = MagicMock()
        modeler._primary.is_available.return_value = True
        modeler._budget = MagicMock()
        modeler._budget.check_budget.return_value = (True, None)

        allowed, reason = modeler.should_analyze(
            permanence_state="stable",
            observe_count=2,
        )
        assert allowed is True
        assert reason == "ok"
