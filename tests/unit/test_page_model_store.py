"""Tests for page_model_store — 页面模型存储 CRUD."""

import pytest

from src.memory.page_identity import (
    LayoutFingerprint,
    infer_app_from_process,
    get_app_display_name,
    KNOWN_PROCESSES,
    WORKFLOW_DISPLAY,
)
from src.memory.page_model_store import PageModelStore, _guess_display_name
from src.storage.db import Database


@pytest.fixture
def db(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_all()
    return db


@pytest.fixture
def store():
    return PageModelStore()


def _make_fp(
    input_state: str = "unknown",
    has_dialog: bool = False,
    has_menu: bool = False,
) -> LayoutFingerprint:
    return LayoutFingerprint(
        fixed_roles=("button", "menu_item"),
        region_structure=(("content_area", 10),),
        state_flags=(("editable", True),),
        element_count_bucket="5-20",
        input_state=input_state,
        has_dialog=has_dialog,
        has_menu=has_menu,
    )


# =============================================================================
# Test: PageModel CRUD
# =============================================================================


class TestPageModelCRUD:
    def test_create_new(self, db, store):
        with db.session() as session:
            pm, status = store.find_or_create_page_model(
                session, "wechat", "wechat/chat/main/default",
            )
            assert status == "new"
            assert pm.app_id == "wechat"
            assert pm.page_class_prefix == "wechat/chat"
            assert pm.observe_count == 1

    def test_reuse_same_prefix(self, db, store):
        with db.session() as session:
            store.find_or_create_page_model(session, "wechat", "wechat/chat/main/default")
        with db.session() as session:
            pm, status = store.find_or_create_page_model(
                session, "wechat", "wechat/chat/main/default",
            )
            assert status == "reused"
            assert pm.observe_count == 2

    def test_different_workflow_different_model(self, db, store):
        with db.session() as session:
            pm1, _ = store.find_or_create_page_model(session, "wechat", "wechat/chat/main/default")
            pm2, _ = store.find_or_create_page_model(session, "wechat", "wechat/contacts/main/default")
        assert pm1.page_model_id != pm2.page_model_id
        assert pm1.page_class_prefix == "wechat/chat"
        assert pm2.page_class_prefix == "wechat/contacts"

    def test_unknown_page_class_fallback(self, db, store):
        """unknown page_class + 相同 process_name + surface_type → 同一 PageModel。"""
        with db.session() as session:
            pm1, _ = store.find_or_create_page_model(
                session, "notepad", "unknown/document/main/wide",
                process_name="notepad.exe", surface_type="native_uia",
                layout_signature="abc123",
            )
        with db.session() as session:
            pm2, status = store.find_or_create_page_model(
                session, "notepad", "unknown/document/main/wide",
                process_name="notepad.exe", surface_type="native_uia",
                layout_signature="abc123",
            )
        assert status == "reused"
        assert pm1.page_model_id == pm2.page_model_id


# =============================================================================
# Test: StateTemplate CRUD
# =============================================================================


class TestStateTemplateCRUD:
    def test_create_new(self, db, store):
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(session, "wechat", "wechat/chat/main/default")
            fp = _make_fp(input_state="empty")
            st, status = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp, total_element_count=15, fixed_element_count=5,
                state_label="输入框为空",
            )
            assert status == "new"
            assert st.state_label == "输入框为空"

    def test_exact_match_reuse(self, db, store):
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(session, "wechat", "wechat/chat/main/default")
            fp = _make_fp(input_state="empty")
            store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp, total_element_count=15, fixed_element_count=5,
            )
        with db.session() as session:
            st, status = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp, total_element_count=15, fixed_element_count=5,
            )
        assert status == "reused"
        assert st.verify_count == 2

    def test_exact_match_reuse_refreshes_element_counts(self, db, store):
        """A sparse first observe must not permanently cap model/state counts."""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(session, "wechat", "wechat/chat/main/default")
            fp = _make_fp(input_state="empty")
            store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp, total_element_count=1, fixed_element_count=1,
            )

        with db.session() as session:
            st, status = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp, total_element_count=56, fixed_element_count=20,
            )

        assert status == "reused"
        assert st.total_element_count == 56
        assert st.fixed_element_count == 20

    def test_similarity_match_reuse(self, db, store):
        """状态变化（input_state）→ "new"。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(session, "wechat", "wechat/chat/main/default")
            fp_a = _make_fp(input_state="empty")
            store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp_a, total_element_count=15, fixed_element_count=5,
            )
        with db.session() as session:
            fp_b = _make_fp(input_state="filled")
            st, status = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp_b, total_element_count=16, fixed_element_count=5,
            )
        # 状态变化（input_state）创建新 StateTemplate
        assert status == "new"

    def test_threshold_merge(self, db, store):
        """状态变化（input_state, has_dialog）→ "new"（不再 merge）。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(session, "wechat", "wechat/chat/main/default")
            fp_a = _make_fp(input_state="empty", has_dialog=False)
            store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp_a, total_element_count=15, fixed_element_count=5,
            )
        with db.session() as session:
            fp_b = _make_fp(input_state="filled", has_dialog=True)
            st, status = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp_b, total_element_count=25, fixed_element_count=8,
            )
        # 状态变化（input_state, has_dialog）创建新 StateTemplate
        assert status == "new"

    def test_completely_different_new(self, db, store):
        """完全不同的指纹 → "new"。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(session, "wechat", "wechat/chat/main/default")
            fp_a = _make_fp(input_state="empty")
            store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp_a, total_element_count=15, fixed_element_count=5,
            )
        with db.session() as session:
            fp_b = LayoutFingerprint(
                fixed_roles=("toolbar", "sidebar", "menu_bar"),
                region_structure=(("sidebar", 30), ("toolbar", 15),),
                state_flags=(("dialog_open", True),),
                element_count_bucket="50+",
                input_state="filled",
                has_dialog=True,
                has_menu=True,
            )
            st, status = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/dialog",
                fp_b, total_element_count=60, fixed_element_count=20,
            )
        assert status == "new"


# =============================================================================
# Test: CanvasSnapshot 保留策略
# =============================================================================


class TestCanvasSnapshotRetention:
    def test_retention_policy(self, db, store):
        """超过 max_snapshots 时删除最旧的。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(session, "test", "test/gui/main/default")
            fp = _make_fp()
            st, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "test", "test/gui/main/default",
                fp, total_element_count=10, fixed_element_count=5,
            )
            for i in range(12):
                store.record_canvas_snapshot(
                    session, f"canvas_{i}", pm.page_model_id,
                    st.state_template_id, 10, False,
                )

        with db.session() as session:
            snapshots = store.get_canvas_snapshots(
                session, state_template_id=st.state_template_id,
            )
            assert len(snapshots) == 10

    def test_record_canvas_snapshot_upserts_existing_canvas_id(self, db, store):
        """Background enhancement refreshes the original canvas row instead of duplicating/staling it."""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(session, "test", "test/gui/main/default")
            fp = _make_fp()
            st, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "test", "test/gui/main/default",
                fp, total_element_count=1, fixed_element_count=1,
            )
            first = store.record_canvas_snapshot(
                session, "canvas_same", pm.page_model_id,
                st.state_template_id, 1, False,
            )
            second = store.record_canvas_snapshot(
                session, "canvas_same", pm.page_model_id,
                st.state_template_id, 56, True,
            )

        with db.session() as session:
            snapshots = store.get_canvas_snapshots(
                session, state_template_id=st.state_template_id,
            )

        assert second.snapshot_id == first.snapshot_id
        assert len(snapshots) == 1
        assert snapshots[0].element_count == 56
        assert snapshots[0].has_screenshot is True


# =============================================================================
# Test: CandidateState 关联
# =============================================================================


class TestCandidateState:
    def test_record_and_query(self, db, store):
        from src.storage.schema import StableCandidateKey as StableCandidateKeyModel

        with db.session() as session:
            session.add(StableCandidateKeyModel(
                key_id="test-key-1",
                app_id="test",
                page_class="test/gui/main/default",
                canonical_region="content_area",
                canonical_text="button",
                canonical_role="button",
                created_at="2026-01-01T00:00:00",
                last_seen_at="2026-01-01T00:00:00",
            ))
            session.flush()

            pm, _ = store.find_or_create_page_model(session, "test", "test/gui/main/default")
            fp = _make_fp()
            st, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "test", "test/gui/main/default",
                fp, total_element_count=10, fixed_element_count=5,
            )
            store.record_candidate_state(session, "test-key-1", st.state_template_id, pm.page_model_id)

        with db.session() as session:
            states = store.get_candidate_states(session, "test-key-1")
            assert len(states) == 1
            assert states[0] == st.state_template_id


# =============================================================================
# Test: 页面命名（App Name Inference）
# =============================================================================


class TestPageModelNaming:
    """测试从 process_name 推断 app_id 和中文显示名。"""

    def test_wechat_process_infers_app_id(self, db, store):
        """wechat.exe + unknown app_id → app_id='wechat', display_name 含 '微信'。"""
        with db.session() as session:
            pm, status = store.find_or_create_page_model(
                session, "unknown", "unknown/chat/main/default",
                process_name="wechat.exe", surface_type="native_uia",
            )
        assert pm.app_id == "wechat"
        assert pm.display_name is not None
        assert "微信" in pm.display_name

    def test_notepad_process_infers_app_id(self, db, store):
        """notepad.exe + unknown app_id → app_id='notepad', display_name 含 '记事本'。"""
        with db.session() as session:
            pm, status = store.find_or_create_page_model(
                session, "unknown", "unknown/document/main/wide",
                process_name="notepad.exe", surface_type="native_uia",
            )
        assert pm.app_id == "notepad"
        assert pm.display_name is not None
        assert "记事本" in pm.display_name

    def test_chrome_process_infers_app_id(self, db, store):
        """chrome.exe + unknown app_id → app_id='chrome', display_name 含 'Chrome'。"""
        with db.session() as session:
            pm, status = store.find_or_create_page_model(
                session, "unknown", "unknown/web/main/default",
                process_name="chrome.exe", surface_type="browser",
            )
        assert pm.app_id == "chrome"
        assert pm.display_name is not None
        assert "Chrome" in pm.display_name

    def test_known_app_id_unchanged(self, db, store):
        """已知 app_id（非 unknown）→ 不改变 app_id，仍正常生成 display_name。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(
                session, "wechat", "wechat/chat/main/default",
                process_name="wechat.exe",
            )
        assert pm.app_id == "wechat"
        assert pm.display_name is not None
        assert "微信" in pm.display_name

    def test_unknown_process_infers_from_name(self, db, store):
        """未知进程名 + unknown app_id → 从进程名推导 app_id。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(
                session, "unknown", "unknown/gui/main/default",
                process_name="some_random_app.exe", surface_type="native_uia",
            )
        assert pm.app_id == "some_random_app"
        assert pm.display_name is not None

    def test_weixin_process_maps_to_wechat(self, db, store):
        """weixin.exe → app_id='wechat'（别名映射）。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(
                session, "unknown", "unknown/chat/main/default",
                process_name="weixin.exe", surface_type="native_uia",
            )
        assert pm.app_id == "wechat"
        assert pm.display_name is not None
        assert "微信" in pm.display_name

    def test_reuse_after_inference(self, db, store):
        """推断后的 app_id 能正确复用已有 PageModel。"""
        with db.session() as session:
            pm1, _ = store.find_or_create_page_model(
                session, "unknown", "unknown/chat/main/default",
                process_name="wechat.exe", surface_type="native_uia",
            )
        with db.session() as session:
            pm2, status = store.find_or_create_page_model(
                session, "unknown", "unknown/chat/main/default",
                process_name="wechat.exe", surface_type="native_uia",
            )
        assert status == "reused"
        assert pm1.page_model_id == pm2.page_model_id

    def test_display_name_format_has_app_and_workflow(self, db, store):
        """display_name 格式为 {app_name}{workflow_name}。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(
                session, "unknown", "unknown/chat/main/default",
                process_name="wechat.exe", surface_type="native_uia",
            )
        # "微信" + "聊天页面"
        assert pm.display_name == "微信聊天页面"

    def test_terminal_process_infers_app_id(self, db, store):
        """windowsterminal.exe → app_id='terminal'。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(
                session, "unknown", "unknown/gui/main/default",
                process_name="windowsterminal.exe", surface_type="native_uia",
            )
        assert pm.app_id == "terminal"
        assert pm.display_name is not None
        assert "终端" in pm.display_name


# =============================================================================
# Test: 纯函数 — infer_app_from_process / get_app_display_name
# =============================================================================


class TestInferenceFunctions:
    """测试 page_identity 中的推断纯函数。"""

    def test_infer_known_process(self):
        result = infer_app_from_process("wechat.exe")
        assert result is not None
        assert result == ("wechat", "微信")

    def test_infer_case_insensitive(self):
        result = infer_app_from_process("WeChat.EXE")
        assert result is not None
        assert result == ("wechat", "微信")

    def test_infer_unknown_process(self):
        result = infer_app_from_process("random_app.exe")
        assert result is not None
        assert result == ("random_app", "random_app")

    def test_infer_none(self):
        result = infer_app_from_process(None)
        assert result is None

    def test_get_display_name_from_process(self):
        name = get_app_display_name("unknown", "notepad.exe")
        assert name == "记事本"

    def test_get_display_name_from_app_id(self):
        name = get_app_display_name("wechat")
        assert name == "微信"

    def test_get_display_name_unknown(self):
        name = get_app_display_name("foobar")
        assert name == "foobar"  # 兜底用 app_id 本身

    def test_guess_display_name_with_process(self):
        name = _guess_display_name("wechat", "wechat/chat", "wechat.exe")
        assert name == "微信聊天页面"

    def test_guess_display_name_without_process(self):
        name = _guess_display_name("wechat", "wechat/chat")
        assert name == "微信聊天页面"

    def test_guess_display_name_unknown_app_with_process(self):
        name = _guess_display_name("unknown", "unknown/gui", "notepad.exe")
        assert name == "记事本主界面"
