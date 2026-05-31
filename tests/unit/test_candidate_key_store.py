"""Tests for candidate_key_store — key store CRUD + 隐私验证."""

import pytest

from src.memory.candidate_identity import CandidateSignature
from src.memory.candidate_key_store import CandidateKeyStore
from src.storage.db import Database
from src.storage.schema import CandidateTemplate, StableCandidateKey


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def db(tmp_path):
    """创建临时数据库。"""
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)
    db.create_all()
    yield db


@pytest.fixture
def store():
    return CandidateKeyStore()


def _make_sig(
    region: str = "right_panel_bottom",
    role: str = "button",
    rel_bounds: tuple[float, float, float, float] = (0.9, 0.9, 0.97, 0.96),
    text: str = "发送",
    visual_type: str = "button",
    confidence: float = 0.8,
    is_fixed: bool = True,
) -> CandidateSignature:
    # 自动推断 has_valid_bounds
    x1, y1, x2, y2 = rel_bounds
    has_valid = not (x1 == 0.0 and y1 == 0.0 and x2 == 0.0 and y2 == 0.0) and x2 > x1 and y2 > y1
    return CandidateSignature(
        region_id=region,
        region_role=role,
        relative_bounds=rel_bounds,
        size_ratio=0.01 if has_valid else 0.0,
        text_normalized=text,
        visual_type=visual_type,
        role_label=None,
        semantic_tags=(),
        crop_hash="",
        provider_sources=("uia",),
        confidence=confidence,
        is_fixed_control=is_fixed,
        has_valid_bounds=has_valid,
    )


# =============================================================================
# Test: 首次 find_or_create → "new"
# =============================================================================


class TestFirstCreate:
    def test_first_create_returns_new(self, db, store):
        sig = _make_sig()
        with db.session() as session:
            key, status = store.find_or_create(session, "wechat", "chat_main", sig)

        assert status == "new"
        assert key is not None
        assert key.app_id == "wechat"
        assert key.page_class == "chat_main"
        assert key.canonical_text == "发送"


# =============================================================================
# Test: 相似签名再次 find_or_create → "reused"
# =============================================================================


class TestReuse:
    def test_similar_signature_reuses_key(self, db, store):
        sig1 = _make_sig(rel_bounds=(0.90, 0.92, 0.97, 0.96))
        sig2 = _make_sig(rel_bounds=(0.90, 0.92, 0.97, 0.96))

        with db.session() as session:
            key1, status1 = store.find_or_create(session, "wechat", "chat_main", sig1)

        with db.session() as session:
            key2, status2 = store.find_or_create(session, "wechat", "chat_main", sig2)

        assert status1 == "new"
        assert status2 == "reused"
        assert key1.key_id == key2.key_id

    def test_verify_count_increments(self, db, store):
        sig = _make_sig()

        with db.session() as session:
            store.find_or_create(session, "wechat", "chat_main", sig)

        with db.session() as session:
            store.find_or_create(session, "wechat", "chat_main", sig)

        with db.session() as session:
            row = session.query(StableCandidateKey).first()
            assert row.verify_count == 2  # new(1) + reused(+1)


# =============================================================================
# Test: 不同签名 → "new"
# =============================================================================


class TestDifferentSignature:
    def test_different_signature_creates_new_key(self, db, store):
        sig1 = _make_sig(text="发送", region="right_panel_bottom")
        sig2 = _make_sig(text="文件", region="sidebar_top")

        with db.session() as session:
            key1, status1 = store.find_or_create(session, "wechat", "chat_main", sig1)

        with db.session() as session:
            key2, status2 = store.find_or_create(session, "wechat", "chat_main", sig2)

        assert status1 == "new"
        assert status2 == "new"
        assert key1.key_id != key2.key_id


# =============================================================================
# Test: transient candidates 不持久化
# =============================================================================


class TestTransient:
    def test_transient_not_persisted(self, db, store):
        sig = _make_sig(text="你好世界", is_fixed=False)

        with db.session() as session:
            key, status = store.find_or_create(session, "wechat", "chat_main", sig)

        assert status == "transient"
        assert key is None

        with db.session() as session:
            count = session.query(StableCandidateKey).count()
            assert count == 0

    def test_transient_force_persists(self, db, store):
        sig = _make_sig(text="你好世界", is_fixed=False)

        with db.session() as session:
            key, status = store.find_or_create(
                session, "wechat", "chat_main", sig, force=True,
            )

        assert status == "new"
        assert key is not None


# =============================================================================
# Test: 隐私验证 — 动态内容模板中无原始文字
# =============================================================================


class TestPrivacy:
    def test_dynamic_content_no_raw_text(self, db, store):
        sig = _make_sig(text="abc123hash", is_fixed=False)

        with db.session() as session:
            store.find_or_create(
                session, "wechat", "chat_main", sig, force=True,
            )

        with db.session() as session:
            template = session.query(CandidateTemplate).first()
            assert template is not None
            assert template.text_normalized != "你好世界"
            assert template.text_hash is not None
            assert template.text_hash == template.text_normalized

    def test_fixed_control_keeps_raw_text(self, db, store):
        sig = _make_sig(text="发送", is_fixed=True)

        with db.session() as session:
            store.find_or_create(session, "wechat", "chat_main", sig)

        with db.session() as session:
            template = session.query(CandidateTemplate).first()
            assert template is not None
            assert template.text_normalized == "发送"
            assert template.text_hash is None


# =============================================================================
# Test: get_templates_for_key
# =============================================================================


class TestGetTemplates:
    def test_get_templates(self, db, store):
        sig = _make_sig()

        with db.session() as session:
            key, _ = store.find_or_create(session, "wechat", "chat_main", sig)

        with db.session() as session:
            templates = store.get_templates_for_key(session, key.key_id)
            assert len(templates) == 1
            assert templates[0].region_id == "right_panel_bottom"


# =============================================================================
# Test: 无效 bounds 不创建 stable key
# =============================================================================


class TestInvalidBounds:
    def test_zero_bounds_not_persisted(self, db, store):
        """全零 bounds → transient，不创建 stable key。"""
        sig = _make_sig(rel_bounds=(0.0, 0.0, 0.0, 0.0))

        with db.session() as session:
            key, status = store.find_or_create(session, "wechat", "chat_main", sig)

        assert status == "transient"
        assert key is None

        with db.session() as session:
            count = session.query(StableCandidateKey).count()
            assert count == 0

    def test_zero_area_not_persisted(self, db, store):
        """零面积 bounds → transient。"""
        sig = _make_sig(rel_bounds=(0.5, 0.5, 0.5, 0.8))

        with db.session() as session:
            key, status = store.find_or_create(session, "wechat", "chat_main", sig)

        assert status == "transient"
        assert key is None

    def test_has_valid_bounds_false_not_persisted(self, db, store):
        """has_valid_bounds=False → transient，即使其他字段正常。"""
        sig = CandidateSignature(
            region_id="right_panel_bottom",
            region_role="button",
            relative_bounds=(0.0, 0.0, 0.0, 0.0),
            size_ratio=0.0,
            text_normalized="发送",
            visual_type="button",
            role_label=None,
            semantic_tags=(),
            crop_hash="",
            provider_sources=("uia",),
            confidence=0.8,
            is_fixed_control=True,
            has_valid_bounds=False,
        )

        with db.session() as session:
            key, status = store.find_or_create(session, "wechat", "chat_main", sig)

        assert status == "transient"
        assert key is None

    def test_normal_button_still_reuses(self, db, store):
        """正常 bounds 的按钮仍然可以复用 stable key。"""
        sig1 = _make_sig(rel_bounds=(0.90, 0.92, 0.97, 0.96))
        sig2 = _make_sig(rel_bounds=(0.90, 0.92, 0.97, 0.96))

        with db.session() as session:
            key1, status1 = store.find_or_create(session, "wechat", "chat_main", sig1)

        with db.session() as session:
            key2, status2 = store.find_or_create(session, "wechat", "chat_main", sig2)

        assert status1 == "new"
        assert status2 == "reused"
        assert key1.key_id == key2.key_id


# =============================================================================
# Test: _bounds_close — 零 bounds 不认为接近
# =============================================================================


class TestBoundsClose:
    def test_zero_bounds_not_close(self):
        """两个零 bounds → 不认为接近。"""
        from src.memory.candidate_key_store import _bounds_close
        assert _bounds_close((0, 0, 0, 0), (0, 0, 0, 0)) is False

    def test_one_zero_bounds_not_close(self):
        """一个零 bounds 一个正常 → 不认为接近。"""
        from src.memory.candidate_key_store import _bounds_close
        assert _bounds_close((0, 0, 0, 0), (0.9, 0.9, 0.97, 0.96)) is False
        assert _bounds_close((0.9, 0.9, 0.97, 0.96), (0, 0, 0, 0)) is False

    def test_close_bounds(self):
        """接近的 bounds → 认为接近。"""
        from src.memory.candidate_key_store import _bounds_close
        assert _bounds_close((0.90, 0.92, 0.97, 0.96), (0.91, 0.92, 0.97, 0.96)) is True

    def test_far_bounds_not_close(self):
        """距离远的 bounds → 不认为接近。"""
        from src.memory.candidate_key_store import _bounds_close
        assert _bounds_close((0.1, 0.1, 0.2, 0.2), (0.9, 0.9, 0.97, 0.96)) is False
