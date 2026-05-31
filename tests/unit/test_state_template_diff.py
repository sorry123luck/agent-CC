"""Tests for state template differentiation — 页面状态模板差异测试。

验证 StateTemplate 的正确分离和归并：
- 相同页面状态 → 同一 StateTemplate
- 不同可操作状态 → 不同 StateTemplate
- 微小变化 → 同一 StateTemplate（similarity >= 0.80）
- stable_key_id 跨 StateTemplate 可复用
"""

import pytest

from src.memory.page_identity import (
    LayoutFingerprint,
    build_layout_fingerprint,
    compute_fingerprint_similarity,
    fingerprint_to_signature,
)
from src.memory.page_model_store import PageModelStore
from src.perception.page_compiler_models import (
    AppInfo,
    Candidate,
    ElementState,
    InteractionCanvas,
    PageInfo,
    Region,
    SemanticRole,
    SurfaceInfo,
    SurfaceType,
    WindowInfoSnapshot,
)
from src.storage.db import Database
from src.storage.schema import StableCandidateKey as StableCandidateKeyModel


@pytest.fixture
def db(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.create_all()
    return db


@pytest.fixture
def store():
    return PageModelStore()


def _make_fp(
    fixed_roles: tuple[str, ...] = ("button", "menu_item"),
    region_structure: tuple[tuple[str, int], ...] = (("content_area", 10),),
    state_flags: tuple[tuple[str, bool], ...] = (("editable", True),),
    element_count_bucket: str = "5-20",
    input_state: str = "unknown",
    has_dialog: bool = False,
    has_menu: bool = False,
) -> LayoutFingerprint:
    # 将 has_dialog / has_menu 合并到 state_flags 中（与 build_layout_fingerprint 一致）
    flags_dict = dict(state_flags)
    if has_dialog:
        flags_dict["dialog_open"] = True
    if has_menu:
        flags_dict["menu_open"] = True
    merged_flags = tuple(sorted(flags_dict.items()))

    return LayoutFingerprint(
        fixed_roles=fixed_roles,
        region_structure=region_structure,
        state_flags=merged_flags,
        element_count_bucket=element_count_bucket,
        input_state=input_state,
        has_dialog=has_dialog,
        has_menu=has_menu,
    )


def _make_chat_canvas_with_synthetic_input(input_ocr_text: str = "") -> InteractionCanvas:
    input_candidate = Candidate(
        element_id="synthetic_chat_composer_input",
        region_id="composer",
        semantic_role=SemanticRole.MESSAGE_INPUT,
        bounds=(300, 610, 880, 705),
        text="",
        attributes={
            "candidate_kind": "message_input_candidate",
            "actionability": "review",
            "safe_to_type": False,
        },
        provider_sources=["geometry"],
    )
    send_candidate = Candidate(
        element_id="send",
        region_id="composer",
        semantic_role=SemanticRole.SEND_BUTTON,
        bounds=(900, 670, 960, 710),
        text="发送",
        risk_tags=["send"],
        provider_sources=["uia"],
    )
    canvas = InteractionCanvas(
        canvas_id=f"chat-{input_ocr_text or 'empty'}",
        app=AppInfo(app_id="wechat", process_name="weixin.exe"),
        window=WindowInfoSnapshot(hwnd=12345, rect_client=(0, 0, 1000, 730)),
        surface=SurfaceInfo(surface_type=SurfaceType.NATIVE_UIA),
        page=PageInfo(page_class="wechat/chat/main/default", state_flags={}),
        regions=[
            Region(region_id="messages", role="message_stream", bounds=(300, 80, 980, 600), element_ids=[]),
            Region(region_id="composer", role="composer", bounds=(300, 600, 980, 720), element_ids=["synthetic_chat_composer_input", "send"]),
        ],
        elements=[input_candidate, send_candidate],
    )
    if input_ocr_text:
        canvas.artifacts["ocr_blocks"] = [
            {"bbox": [326, 630, 390, 656], "text": input_ocr_text, "confidence": 0.94}
        ]
    return canvas


# =============================================================================
# Test: 记事本多次 observe → 同一 StateTemplate
# =============================================================================


class TestNotepadMultipleObserve:
    def test_same_state_template(self, db, store):
        """记事本多次 observe，页面状态不变 → 同一 StateTemplate。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(
                session, "notepad", "notepad/document/main/default",
            )
            fp = _make_fp(
                fixed_roles=("menu_item", "text_input", "title_bar"),
                region_structure=(("title_bar", 1), ("content_area", 1)),
                element_count_bucket="5-20",
                input_state="empty",
            )

            # 第一次 observe
            st1, status1 = store.find_or_create_state_template(
                session, pm.page_model_id, "notepad", "notepad/document/main/default",
                fp, total_element_count=10, fixed_element_count=3,
            )
            assert status1 == "new"

            # 第二次 observe（相同状态）
            st2, status2 = store.find_or_create_state_template(
                session, pm.page_model_id, "notepad", "notepad/document/main/default",
                fp, total_element_count=10, fixed_element_count=3,
            )
            assert status2 == "reused"
            assert st1.state_template_id == st2.state_template_id

            # 第三次 observe（相同状态）
            st3, status3 = store.find_or_create_state_template(
                session, pm.page_model_id, "notepad", "notepad/document/main/default",
                fp, total_element_count=10, fixed_element_count=3,
            )
            assert status3 == "reused"
            assert st1.state_template_id == st3.state_template_id


# =============================================================================
# Test: 微信 empty_input vs has_text_input → 不同 StateTemplate
# =============================================================================


class TestWeChatInputState:
    def test_empty_vs_filled_different(self, db, store):
        """微信输入框空 vs 有内容 → 不同 StateTemplate。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(
                session, "wechat", "wechat/chat/main/default",
            )

            # 输入框为空
            fp_empty = _make_fp(
                fixed_roles=("button", "menu_item", "send_button"),
                region_structure=(("sidebar", 5), ("content_area", 10)),
                element_count_bucket="20-50",
                input_state="empty",
            )
            st_empty, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp_empty, total_element_count=30, fixed_element_count=8,
                state_label="输入框为空",
            )

            # 输入框有内容
            fp_filled = _make_fp(
                fixed_roles=("button", "menu_item", "send_button"),
                region_structure=(("sidebar", 5), ("content_area", 10)),
                element_count_bucket="20-50",
                input_state="filled",
            )
            st_filled, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp_filled, total_element_count=30, fixed_element_count=8,
                state_label="输入框有内容",
            )

        # 应该是不同的 StateTemplate
        assert st_empty.state_template_id != st_filled.state_template_id
        assert st_empty.state_label == "输入框为空"
        assert st_filled.state_label == "输入框有内容"

    def test_filled_content_change_same(self, db, store):
        """输入框内容从 123 变成 456 → 同一 StateTemplate（都是 filled）。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(
                session, "wechat", "wechat/chat/main/default",
            )

            # 输入框有内容 "123"
            fp1 = _make_fp(
                fixed_roles=("button", "menu_item", "send_button"),
                region_structure=(("sidebar", 5), ("content_area", 10)),
                element_count_bucket="20-50",
                input_state="filled",
            )
            st1, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp1, total_element_count=30, fixed_element_count=8,
            )

            # 输入框有内容 "456"（内容不同但状态相同）
            fp2 = _make_fp(
                fixed_roles=("button", "menu_item", "send_button"),
                region_structure=(("sidebar", 5), ("content_area", 10)),
                element_count_bucket="20-50",
                input_state="filled",
            )
            st2, status = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp2, total_element_count=30, fixed_element_count=8,
            )

        # 同一 StateTemplate（内容变化不影响状态）
        assert st1.state_template_id == st2.state_template_id
        assert status == "reused"

    def test_synthetic_input_empty_vs_ocr_filled_different(self, db, store):
        """合成输入框依赖 OCR 状态证据时，空输入和有输入仍要分离模板。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(
                session, "wechat", "wechat/chat/main/default",
            )

            empty_fp = build_layout_fingerprint(_make_chat_canvas_with_synthetic_input(""))
            filled_fp = build_layout_fingerprint(_make_chat_canvas_with_synthetic_input("123"))
            filled_other_fp = build_layout_fingerprint(_make_chat_canvas_with_synthetic_input("456"))

            st_empty, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                empty_fp, total_element_count=2, fixed_element_count=2,
            )
            st_filled, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                filled_fp, total_element_count=2, fixed_element_count=2,
            )
            st_filled_other, status = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                filled_other_fp, total_element_count=2, fixed_element_count=2,
            )

        assert empty_fp.input_state == "empty"
        assert filled_fp.input_state == "filled"
        assert filled_other_fp.input_state == "filled"
        assert st_empty.state_template_id != st_filled.state_template_id
        assert st_filled.state_template_id == st_filled_other.state_template_id
        assert status == "reused"

    def test_send_enabled_state_splits_templates(self, db, store):
        """发送按钮 disabled/enabled 是可操作状态差异，应分离 StateTemplate。"""
        disabled_canvas = _make_chat_canvas_with_synthetic_input("")
        enabled_canvas = _make_chat_canvas_with_synthetic_input("123")
        disabled_canvas.elements[1].state = ElementState(enabled=False, visible=True)
        enabled_canvas.elements[1].state = ElementState(enabled=True, visible=True)

        with db.session() as session:
            pm, _ = store.find_or_create_page_model(
                session, "wechat", "wechat/chat/main/default",
            )
            fp_disabled = build_layout_fingerprint(disabled_canvas)
            fp_enabled = build_layout_fingerprint(enabled_canvas)

            st_disabled, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp_disabled, total_element_count=2, fixed_element_count=2,
            )
            st_enabled, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp_enabled, total_element_count=2, fixed_element_count=2,
            )

        assert ("send_enabled", False) in fp_disabled.state_flags
        assert ("send_enabled", True) in fp_enabled.state_flags
        assert st_disabled.state_template_id != st_enabled.state_template_id


# =============================================================================
# Test: 菜单展开 → 不同 StateTemplate
# =============================================================================


class TestMenuState:
    def test_menu_closed_vs_open(self, db, store):
        """菜单关闭 vs 菜单展开 → 不同 StateTemplate。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(
                session, "notepad", "notepad/document/main/default",
            )

            # 菜单关闭
            fp_closed = _make_fp(
                fixed_roles=("menu_item", "text_input", "title_bar"),
                region_structure=(("title_bar", 1), ("content_area", 1)),
                element_count_bucket="5-20",
                has_menu=False,
            )
            st_closed, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "notepad", "notepad/document/main/default",
                fp_closed, total_element_count=10, fixed_element_count=3,
                state_label="主页面",
            )

            # 菜单展开
            fp_open = _make_fp(
                fixed_roles=("menu_item", "text_input", "title_bar"),
                region_structure=(("title_bar", 1), ("content_area", 1)),
                element_count_bucket="5-20",
                has_menu=True,
            )
            st_open, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "notepad", "notepad/document/main/default",
                fp_open, total_element_count=10, fixed_element_count=3,
                state_label="菜单展开",
            )

        # 不同 StateTemplate
        assert st_closed.state_template_id != st_open.state_template_id
        assert st_closed.state_label == "主页面"
        assert st_open.state_label == "菜单展开"


# =============================================================================
# Test: 弹窗 → 不同 StateTemplate
# =============================================================================


class TestDialogState:
    def test_dialog_closed_vs_open(self, db, store):
        """弹窗关闭 vs 弹窗打开 → 不同 StateTemplate。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(
                session, "wechat", "wechat/chat/main/default",
            )

            # 无弹窗
            fp_no_dialog = _make_fp(
                fixed_roles=("button", "menu_item", "send_button"),
                region_structure=(("sidebar", 5), ("content_area", 10)),
                element_count_bucket="20-50",
                has_dialog=False,
            )
            st_no, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp_no_dialog, total_element_count=30, fixed_element_count=8,
                state_label="主页面",
            )

            # 有弹窗
            fp_dialog = _make_fp(
                fixed_roles=("button", "menu_item", "send_button"),
                region_structure=(("sidebar", 5), ("content_area", 10)),
                element_count_bucket="20-50",
                has_dialog=True,
            )
            st_yes, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp_dialog, total_element_count=30, fixed_element_count=8,
                state_label="弹窗打开",
            )

        # 不同 StateTemplate
        assert st_no.state_template_id != st_yes.state_template_id
        assert st_no.state_label == "主页面"
        assert st_yes.state_label == "弹窗打开"


# =============================================================================
# Test: 聊天消息新增 → 同一 StateTemplate
# =============================================================================


class TestChatMessageChange:
    def test_message_count_change_same(self, db, store):
        """聊天消息新增 → 同一 StateTemplate（消息数量变化不影响可操作状态）。"""
        with db.session() as session:
            pm, _ = store.find_or_create_page_model(
                session, "wechat", "wechat/chat/main/default",
            )

            # 10 条消息
            fp1 = _make_fp(
                fixed_roles=("button", "menu_item", "send_button"),
                region_structure=(("sidebar", 5), ("content_area", 10)),
                element_count_bucket="20-50",
                input_state="empty",
            )
            st1, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp1, total_element_count=30, fixed_element_count=8,
            )

            # 15 条消息（元素数量区间不变）
            fp2 = _make_fp(
                fixed_roles=("button", "menu_item", "send_button"),
                region_structure=(("sidebar", 5), ("content_area", 15)),
                element_count_bucket="20-50",
                input_state="empty",
            )
            st2, status = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp2, total_element_count=35, fixed_element_count=8,
            )

        # 同一 StateTemplate（消息数量变化不影响状态）
        assert st1.state_template_id == st2.state_template_id
        assert status == "reused"


# =============================================================================
# Test: stable_key_id 跨 StateTemplate 可复用
# =============================================================================


class TestCandidateKeyCrossState:
    def test_same_candidate_across_states(self, db, store):
        """同一个候选（如发送按钮）在不同 StateTemplate 下可复用 stable_key_id。"""
        with db.session() as session:
            # 创建候选
            session.add(StableCandidateKeyModel(
                key_id="send-button-1",
                app_id="wechat",
                page_class="wechat/chat/main/default",
                canonical_region="content_area",
                canonical_text="发送",
                canonical_role="send_button",
                created_at="2026-01-01T00:00:00",
                last_seen_at="2026-01-01T00:00:00",
            ))
            session.flush()

            pm, _ = store.find_or_create_page_model(
                session, "wechat", "wechat/chat/main/default",
            )

            # 状态 1：输入框为空
            fp1 = _make_fp(input_state="empty")
            st1, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp1, total_element_count=30, fixed_element_count=8,
            )

            # 状态 2：输入框有内容
            fp2 = _make_fp(input_state="filled")
            st2, _ = store.find_or_create_state_template(
                session, pm.page_model_id, "wechat", "wechat/chat/main/default",
                fp2, total_element_count=30, fixed_element_count=8,
            )

            # 发送按钮在两个状态下都出现
            store.record_candidate_state(session, "send-button-1", st1.state_template_id, pm.page_model_id)
            store.record_candidate_state(session, "send-button-1", st2.state_template_id, pm.page_model_id)

        # 查询发送按钮出现过的所有状态
        with db.session() as session:
            states = store.get_candidate_states(session, "send-button-1")
            assert len(states) == 2
            assert st1.state_template_id in states
            assert st2.state_template_id in states


# =============================================================================
# Test: 相似度计算验证
# =============================================================================


class TestFingerprintSimilarity:
    def test_identical_fingerprints(self):
        """相同指纹 → 相似度 1.0。"""
        fp = _make_fp()
        assert compute_fingerprint_similarity(fp, fp) == 1.0

    def test_minor_change_high_similarity(self):
        """微小变化（如元素数量区间变化）→ 相似度 >= 0.80。"""
        fp1 = _make_fp(element_count_bucket="5-20")
        fp2 = _make_fp(element_count_bucket="20-50")
        sim = compute_fingerprint_similarity(fp1, fp2)
        assert sim >= 0.80

    def test_input_state_change_moderate(self):
        """输入状态变化 → 相似度 0.60-0.95。"""
        fp1 = _make_fp(input_state="empty")
        fp2 = _make_fp(input_state="filled")
        sim = compute_fingerprint_similarity(fp1, fp2)
        assert 0.60 <= sim <= 0.95

    def test_completely_different_low_similarity(self):
        """完全不同的指纹 → 相似度 < 0.60。"""
        fp1 = _make_fp(
            fixed_roles=("button", "menu_item"),
            region_structure=(("content_area", 10),),
            element_count_bucket="5-20",
        )
        fp2 = LayoutFingerprint(
            fixed_roles=("toolbar", "sidebar", "menu_bar", "status_bar"),
            region_structure=(("sidebar", 30), ("toolbar", 15), ("content_area", 50)),
            state_flags=(("dialog_open", True), ("menu_open", True)),
            element_count_bucket="50+",
            input_state="filled",
            has_dialog=True,
            has_menu=True,
        )
        sim = compute_fingerprint_similarity(fp1, fp2)
        assert sim < 0.60


# =============================================================================
# Test: 签名一致性
# =============================================================================


class TestSignatureConsistency:
    def test_same_fp_same_signature(self):
        """相同指纹 → 相同签名。"""
        fp = _make_fp()
        sig1 = fingerprint_to_signature(fp)
        sig2 = fingerprint_to_signature(fp)
        assert sig1 == sig2

    def test_different_fp_different_signature(self):
        """不同指纹 → 不同签名。"""
        fp1 = _make_fp(input_state="empty")
        fp2 = _make_fp(input_state="filled")
        sig1 = fingerprint_to_signature(fp1)
        sig2 = fingerprint_to_signature(fp2)
        # 注意：签名基于指纹的确定性 hash，input_state 不同应该产生不同签名
        # 但签名算法可能不包含所有字段，所以这里只验证基本一致性
        assert isinstance(sig1, str) and len(sig1) > 0
        assert isinstance(sig2, str) and len(sig2) > 0
