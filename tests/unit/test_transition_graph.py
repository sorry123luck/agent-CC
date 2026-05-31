"""TransitionGraphManager 单元测试。"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from src.memory.transition_graph import TransitionGraphManager, TransitionInfo
from src.storage.schema import TransitionEdge


class FakeSession:
    """最小化 Session 模拟，用于单元测试。"""

    def __init__(self):
        self._edges: list[TransitionEdge] = []
        self._committed = False

    def query(self, model):
        return FakeQuery(self._edges, model)

    def add(self, obj):
        self._edges.append(obj)

    def commit(self):
        self._committed = True

    def flush(self):
        pass


class FakeQuery:
    def __init__(self, edges, model):
        self._edges = edges
        self._model = model
        self._filters = {}

    def filter_by(self, **kwargs):
        q = FakeQuery(self._edges, self._model)
        q._filters = kwargs
        return q

    def first(self):
        for e in self._edges:
            if self._matches(e):
                return e
        return None

    def all(self):
        return [e for e in self._edges if self._matches(e)]

    def _matches(self, edge):
        for k, v in self._filters.items():
            if getattr(edge, k, None) != v:
                return False
        return True


def _make_manager():
    return TransitionGraphManager(FakeSession())


class TestRecordTransition:
    def test_creates_new_edge(self):
        mgr = _make_manager()
        mgr.record_transition("page_a", "page_b", action="click", success=True)
        session = mgr._session
        assert len(session._edges) == 1
        edge = session._edges[0]
        assert edge.from_page_class == "page_a"
        assert edge.to_page_class == "page_b"
        assert edge.trigger_action == "click"
        assert edge.observe_count == 1
        assert edge.success_count == 1

    def test_increments_existing(self):
        mgr = _make_manager()
        mgr.record_transition("a", "b", action="click")
        mgr.record_transition("a", "b", action="click")
        session = mgr._session
        assert len(session._edges) == 1
        assert session._edges[0].observe_count == 2
        assert session._edges[0].success_count == 2

    def test_failure_no_success_count(self):
        mgr = _make_manager()
        mgr.record_transition("a", "b", action="click", success=False)
        assert mgr._session._edges[0].success_count == 0


class TestRecordVlmTransitions:
    def test_basic(self):
        mgr = _make_manager()
        transitions = [
            {"from_state": "chat", "trigger": "send_message", "to_state": "chat", "confidence": 0.9},
            {"from_state": "chat", "trigger": "open_settings", "to_state": "settings", "confidence": 0.7},
        ]
        count = mgr.record_vlm_transitions(transitions, "pm1")
        assert count == 2
        assert len(mgr._session._edges) == 2
        assert mgr._session._edges[0].trigger_action == "vlm:send_message"
        assert mgr._session._edges[1].trigger_action == "vlm:open_settings"

    def test_dedup(self):
        mgr = _make_manager()
        transitions = [
            {"from_state": "a", "trigger": "x", "to_state": "b", "confidence": 0.8},
            {"from_state": "a", "trigger": "x", "to_state": "b", "confidence": 0.9},
        ]
        count = mgr.record_vlm_transitions(transitions, "pm1")
        assert count == 2
        # But only 1 edge (deduped by record_transition)
        assert len(mgr._session._edges) == 1
        assert mgr._session._edges[0].observe_count == 2

    def test_empty_noop(self):
        mgr = _make_manager()
        count = mgr.record_vlm_transitions([], "pm1")
        assert count == 0
        assert len(mgr._session._edges) == 0

    def test_skips_invalid(self):
        mgr = _make_manager()
        transitions = [
            {"from_state": "", "trigger": "x", "to_state": "b"},
            {"from_state": "a", "trigger": "x", "to_state": ""},
            {"from_state": "a", "trigger": "x", "to_state": "b"},
        ]
        count = mgr.record_vlm_transitions(transitions, "pm1")
        assert count == 1

    def test_no_trigger_uses_inferred(self):
        mgr = _make_manager()
        transitions = [
            {"from_state": "a", "trigger": "", "to_state": "b"},
        ]
        mgr.record_vlm_transitions(transitions, "pm1")
        assert mgr._session._edges[0].trigger_action == "vlm:inferred"
