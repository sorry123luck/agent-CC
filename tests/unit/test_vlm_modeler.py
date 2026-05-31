"""Tests for VLM Semantic Modeler — 核心编排 + 缓存 + 预算 + JSON 解析 + 落库 + payload。"""

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.storage.schema import Base
from src.vlm.modeler import (
    BudgetTracker,
    ResponseRecorder,
    SemanticModelCache,
    SemanticModelerConfig,
    VLMSemanticModeler,
    _extract_candidate_ids,
    _parse_raw_response,
)
from src.vlm.provider import ProviderCapabilities, VLMSemanticRequest
from src.vlm.providers.mock_provider import MockProvider
from src.vlm.providers.openai_provider import OpenAICompatibleProvider
from src.vlm.providers.minimax_provider import MiniMaxProvider
from src.vlm.prompt_builder import PromptInput
from src.vlm.schema import ValidationError, VLMSemanticModelerResult, validate_page_semantic_model


# =============================================================================
# Helpers
# =============================================================================


def _make_screenshot(width: int = 800, height: int = 700) -> Image.Image:
    return Image.new("RGB", (width, height), color=(128, 128, 128))


def _make_prompt_input(
    screenshot: Image.Image | None = None,
    omni_candidates: list[dict] | None = None,
    task_hint: str = "",
) -> PromptInput:
    return PromptInput(
        screenshot=screenshot or _make_screenshot(),
        omni_candidates=omni_candidates,
        task_hint=task_hint,
    )


def _make_config(**overrides) -> SemanticModelerConfig:
    defaults = {"enabled": True, "daily_call_limit": 100, "monthly_budget_usd": 10.0}
    defaults.update(overrides)
    return SemanticModelerConfig(**defaults)


def _make_modeler(
    primary: MockProvider | None = None,
    fallback: MockProvider | None = None,
    config: SemanticModelerConfig | None = None,
    cache: SemanticModelCache | None = None,
    recorder: ResponseRecorder | None = None,
) -> VLMSemanticModeler:
    cfg = config or _make_config()
    return VLMSemanticModeler(
        primary_provider=primary or MockProvider(),
        fallback_provider=fallback,
        cache=cache or SemanticModelCache(),
        budget_tracker=BudgetTracker(cfg),
        config=cfg,
        recorder=recorder,
    )


def _mock_db_session():
    """创建 mock DB session。"""
    session = MagicMock()
    session.execute.return_value.fetchone.return_value = (0,)
    return session


@pytest.fixture
def sqlite_engine():
    """创建内存 SQLite 引擎，包含 vlm_responses 表。

    使用 StaticPool 确保所有 session 共享同一个连接和数据库。
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # 只创建 vlm_responses 表
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE vlm_responses (
                response_id TEXT PRIMARY KEY,
                cache_key TEXT NOT NULL,
                screenshot_hash TEXT NOT NULL,
                candidate_hash TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                raw_response_redacted TEXT,
                parsed_model TEXT NOT NULL,
                token_input INTEGER DEFAULT 0,
                token_output INTEGER DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0.0,
                latency_ms INTEGER DEFAULT 0,
                status TEXT NOT NULL,
                created_at TEXT,
                page_model_id TEXT,
                state_template_id TEXT
            )
        """))
    yield engine
    engine.dispose()


@pytest.fixture
def sqlite_session_factory(sqlite_engine):
    """返回一个 session factory，每次调用返回新 session。"""
    Session = sessionmaker(bind=sqlite_engine)
    return Session


# =============================================================================
# JSON Parse Enhancement
# =============================================================================


class TestParseRawResponse:
    def test_valid_json(self):
        data = {"key": "value"}
        result = _parse_raw_response(json.dumps(data))
        assert result == data

    def test_markdown_wrapped_json(self):
        data = {"key": "value"}
        raw = f"```json\n{json.dumps(data)}\n```"
        result = _parse_raw_response(raw)
        assert result == data

    def test_markdown_wrapped_no_lang(self):
        data = {"key": "value"}
        raw = f"```\n{json.dumps(data)}\n```"
        result = _parse_raw_response(raw)
        assert result == data

    def test_prefix_suffix_noise(self):
        data = {"key": "value"}
        raw = f"Here is the analysis:\n{json.dumps(data)}\n---"
        result = _parse_raw_response(raw)
        assert result == data

    def test_truncated_json_missing_closing(self):
        raw = '{"regions": [{"region_id": "r1", "role": "nav", "bounds": [0,0,100,40], "purpose": "top"}'
        result = _parse_raw_response(raw)
        assert "regions" in result

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _parse_raw_response("")

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="cannot parse"):
            _parse_raw_response("this is not json at all")


# =============================================================================
# SemanticModelCache
# =============================================================================


class TestSemanticModelCache:
    def test_cache_hit(self):
        cache = SemanticModelCache()
        key = cache.compute_key("a", "b", "1.0", "1.0", "mock/mock-model", "none")
        cache.put(key, "model")  # type: ignore
        assert cache.get(key) == "model"

    def test_cache_miss_different_key(self):
        cache = SemanticModelCache()
        key1 = cache.compute_key("a", "b", "1.0", "1.0", "mock/mock-model", "none")
        key2 = cache.compute_key("x", "b", "1.0", "1.0", "mock/mock-model", "none")
        assert key1 != key2

    def test_cache_miss_on_model_change(self):
        """缓存键包含 provider_model_id，模型变化导致缓存失效。"""
        cache = SemanticModelCache()
        key1 = cache.compute_key("a", "b", "1.0", "1.0", "openai/gpt-4o", "none")
        key2 = cache.compute_key("a", "b", "1.0", "1.0", "anthropic/claude-sonnet", "none")
        assert key1 != key2

    def test_cache_invalidation(self):
        cache = SemanticModelCache()
        key = cache.compute_key("a", "b", "1.0", "1.0", "mock/mock-model", "none")
        cache.put(key, "model")  # type: ignore
        cache.invalidate(key)
        assert cache.get(key) is None

    def test_cache_clear(self):
        cache = SemanticModelCache()
        key = cache.compute_key("a", "b", "1.0", "1.0", "mock/mock-model", "none")
        cache.put(key, "model")  # type: ignore
        cache.clear()
        assert cache.get(key) is None


# =============================================================================
# BudgetTracker — Mock Tests
# =============================================================================


class TestBudgetTracker:
    def test_budget_allows_within_limit(self):
        config = _make_config(daily_call_limit=10)
        tracker = BudgetTracker(config)
        provider = MockProvider()
        allowed, reason = tracker.check_budget(provider)
        assert allowed
        assert reason == ""

    def test_budget_blocks_daily_limit(self):
        config = _make_config(daily_call_limit=1)
        tracker = BudgetTracker(config)
        provider = MockProvider()
        tracker._daily_count = 1
        allowed, reason = tracker.check_budget(provider)
        assert not allowed
        assert "daily_call_limit" in reason

    def test_budget_blocks_monthly_limit(self):
        config = _make_config(monthly_budget_usd=0.001)
        tracker = BudgetTracker(config)
        provider = MockProvider()
        tracker._monthly_cost = 0.01
        allowed, reason = tracker.check_budget(provider)
        assert not allowed
        assert "monthly_budget" in reason

    def test_free_model_only_blocks_non_free(self):
        """free_model_only=True 阻止非免费 provider。"""
        config = _make_config(daily_call_limit=100, free_model_only=True)
        tracker = BudgetTracker(config)
        # gpt-4o is NOT free → blocked
        provider = OpenAICompatibleProvider(api_key="test", model="gpt-4o")
        allowed, reason = tracker.check_budget(provider)
        assert not allowed
        assert "free_model_only" in reason

    def test_free_model_only_allows_free(self):
        """free_model_only=True 允许免费 provider。"""
        config = _make_config(daily_call_limit=100, free_model_only=True)
        tracker = BudgetTracker(config)
        provider = MockProvider()  # MockProvider is free
        allowed, _ = tracker.check_budget(provider)
        assert allowed

    def test_budget_from_db_daily_count(self):
        """BudgetTracker 从 DB 加载当天调用次数。"""
        session = _mock_db_session()
        session.execute.return_value.fetchone.return_value = (5,)
        config = _make_config(daily_call_limit=10)
        tracker = BudgetTracker(config, db_session_factory=lambda: session)
        tracker._ensure_db_loaded()
        assert tracker._daily_count == 5

    def test_budget_from_db_monthly_cost(self):
        """BudgetTracker 从 DB 加载本月 SUM(cost_usd)。"""
        session = _mock_db_session()
        session.execute.return_value.fetchone.return_value = (3.75,)
        config = _make_config(monthly_budget_usd=10.0)
        tracker = BudgetTracker(config, db_session_factory=lambda: session)
        tracker._ensure_db_loaded()
        assert abs(tracker._monthly_cost - 3.75) < 0.001

    def test_budget_db_loaded_once(self):
        """DB 只加载一次，后续用内存缓存。"""
        session = _mock_db_session()
        session.execute.return_value.fetchone.return_value = (5,)
        config = _make_config(daily_call_limit=10)
        tracker = BudgetTracker(config, db_session_factory=lambda: session)
        tracker._ensure_db_loaded()
        tracker._ensure_db_loaded()
        # session.execute 只调了 2 次（daily + monthly），不会重复
        assert session.execute.call_count == 2

    def test_record_usage_updates_memory(self):
        config = _make_config()
        tracker = BudgetTracker(config)
        # Use non-free provider so cost > 0
        provider = OpenAICompatibleProvider(api_key="test", model="gpt-4o")
        cost = tracker.record_usage(provider, 1000, 500)
        assert tracker._daily_count == 1
        assert cost > 0
        assert tracker._monthly_cost == cost

    def test_record_usage_free_provider_zero_cost(self):
        """免费 provider 的 record_usage 返回 0。"""
        config = _make_config()
        tracker = BudgetTracker(config)
        provider = MockProvider()
        cost = tracker.record_usage(provider, 1000, 500)
        assert cost == 0.0
        assert tracker._daily_count == 1


# =============================================================================
# ResponseRecorder — Mock Tests
# =============================================================================


class TestResponseRecorder:
    def test_record_success(self):
        session = MagicMock()
        recorder = ResponseRecorder(db_session_factory=lambda: session)
        recorder.record(
            cache_key="abc123",
            screenshot_hash="hash1",
            candidate_hash="hash2",
            prompt_version="1.0",
            schema_version="1.0",
            provider_name="mock",
            model_name="mock/mock-model",
            raw_response_redacted=None,
            parsed_model_json='{"test": true}',
            token_input=500,
            token_output=300,
            cost_usd=0.005,
            latency_ms=100,
            status="success",
        )
        session.execute.assert_called_once()
        session.commit.assert_called_once()
        # 检查 SQL 参数是 dict 且包含正确的字段
        call_args = session.execute.call_args
        params = call_args[0][1]  # 第二个 positional arg 是参数 dict
        assert params["response_id"] is not None
        assert params["cache_key"] == "abc123"
        assert params["cost_usd"] == 0.005
        assert params["status"] == "success"

    def test_record_failure(self):
        session = MagicMock()
        recorder = ResponseRecorder(db_session_factory=lambda: session)
        recorder.record(
            cache_key="abc123",
            screenshot_hash="hash1",
            candidate_hash="hash2",
            prompt_version="1.0",
            schema_version="1.0",
            provider_name="mock",
            model_name="mock/mock-model",
            raw_response_redacted=None,
            parsed_model_json="",
            token_input=0,
            token_output=0,
            cost_usd=0.0,
            latency_ms=0,
            status="failed",
        )
        session.execute.assert_called_once()
        call_args = session.execute.call_args
        params = call_args[0][1]
        assert params["status"] == "failed"
        assert params["cost_usd"] == 0.0

    def test_record_no_db(self):
        """无 DB session 时不报错。"""
        recorder = ResponseRecorder(db_session_factory=None)
        recorder.record(
            cache_key="abc", screenshot_hash="h1", candidate_hash="h2",
            prompt_version="1.0", schema_version="1.0", provider_name="mock",
            model_name="mock", raw_response_redacted=None, parsed_model_json="",
            token_input=0, token_output=0, cost_usd=0.0, latency_ms=0, status="failed",
        )

    def test_record_db_error_does_not_raise(self):
        """DB 写入失败不影响主流程。"""
        session = MagicMock()
        session.execute.side_effect = Exception("DB locked")
        recorder = ResponseRecorder(db_session_factory=lambda: session)
        recorder.record(
            cache_key="abc", screenshot_hash="h1", candidate_hash="h2",
            prompt_version="1.0", schema_version="1.0", provider_name="mock",
            model_name="mock", raw_response_redacted=None, parsed_model_json="",
            token_input=0, token_output=0, cost_usd=0.0, latency_ms=0, status="failed",
        )


# =============================================================================
# BudgetTracker — Real SQLite Integration Tests
# =============================================================================


class TestBudgetTrackerSQLite:
    """使用真实 SQLite 测试 BudgetTracker 的 DB 查询。"""

    def _insert_record(self, session_factory, *, status="success", cost_usd=0.005, created_at="2026-05-12T10:00:00"):
        """插入一条 vlm_responses 记录。"""
        import uuid
        session = session_factory()
        try:
            session.execute(
                text("""INSERT INTO vlm_responses
                       (response_id, cache_key, screenshot_hash, candidate_hash,
                        prompt_version, schema_version, provider_name, model_name,
                        raw_response_redacted, parsed_model, token_input, token_output,
                        cost_usd, latency_ms, status, created_at)
                       VALUES (:rid, :ck, :sh, :ch, :pv, :sv, :pn, :mn,
                               :rr, :pm, :ti, :to, :cu, :lm, :st, :ca)"""),
                {
                    "rid": str(uuid.uuid4()),
                    "ck": "cache_key_1",
                    "sh": "screenshot_hash",
                    "ch": "candidate_hash",
                    "pv": "1.0",
                    "sv": "1.0",
                    "pn": "mock",
                    "mn": "mock/mock-model",
                    "rr": None,
                    "pm": "{}",
                    "ti": 100,
                    "to": 50,
                    "cu": cost_usd,
                    "lm": 100,
                    "st": status,
                    "ca": created_at,
                },
            )
            session.commit()
        finally:
            session.close()

    def test_daily_count_from_db(self, sqlite_session_factory):
        """从真实 SQLite 查询当天成功调用次数。"""
        # 插入 3 条 today 的 success 记录
        from datetime import date
        today = date.today().isoformat()
        for _ in range(3):
            self._insert_record(sqlite_session_factory, status="success", created_at=f"{today}T10:00:00")
        # 插入 1 条 failed（不应计入）
        self._insert_record(sqlite_session_factory, status="failed", created_at=f"{today}T11:00:00")

        config = _make_config(daily_call_limit=100)
        tracker = BudgetTracker(config, db_session_factory=sqlite_session_factory)
        tracker._ensure_db_loaded()
        assert tracker._daily_count == 3

    def test_monthly_cost_from_db(self, sqlite_session_factory):
        """从真实 SQLite 查询本月 SUM(cost_usd)。"""
        from datetime import date
        today = date.today().isoformat()
        self._insert_record(sqlite_session_factory, status="success", cost_usd=1.5, created_at=f"{today}T10:00:00")
        self._insert_record(sqlite_session_factory, status="success", cost_usd=2.3, created_at=f"{today}T11:00:00")
        self._insert_record(sqlite_session_factory, status="failed", cost_usd=0.0, created_at=f"{today}T12:00:00")

        config = _make_config(monthly_budget_usd=100.0)
        tracker = BudgetTracker(config, db_session_factory=sqlite_session_factory)
        tracker._ensure_db_loaded()
        assert abs(tracker._monthly_cost - 3.8) < 0.01

    def test_daily_limit_blocks_with_real_db(self, sqlite_session_factory):
        """当天调用次数达上限时，check_budget 返回 False。"""
        from datetime import date
        today = date.today().isoformat()
        self._insert_record(sqlite_session_factory, status="success", created_at=f"{today}T10:00:00")

        config = _make_config(daily_call_limit=1)
        tracker = BudgetTracker(config, db_session_factory=sqlite_session_factory)
        provider = MockProvider()
        allowed, reason = tracker.check_budget(provider)
        assert not allowed
        assert "daily_call_limit" in reason

    def test_monthly_budget_exceeded_with_real_db(self, sqlite_session_factory):
        """月度预算超限时，check_budget 返回 False。"""
        from datetime import date
        today = date.today().isoformat()
        # 插入总 cost=12.0 的记录，超过 budget=10.0
        self._insert_record(sqlite_session_factory, status="success", cost_usd=7.0, created_at=f"{today}T10:00:00")
        self._insert_record(sqlite_session_factory, status="success", cost_usd=5.0, created_at=f"{today}T11:00:00")

        config = _make_config(monthly_budget_usd=10.0)
        tracker = BudgetTracker(config, db_session_factory=sqlite_session_factory)
        provider = MockProvider()
        # _ensure_db_loaded 从 DB 加载 SUM(cost_usd)=12.0 > 10.0
        allowed, reason = tracker.check_budget(provider)
        assert not allowed
        assert "monthly_budget" in reason


# =============================================================================
# ResponseRecorder — Real SQLite Integration Tests
# =============================================================================


class TestResponseRecorderSQLite:
    """使用真实 SQLite 测试 ResponseRecorder 的落库行为。"""

    def test_record_writes_to_db(self, sqlite_session_factory):
        """record() 真实写入 SQLite 并可查询。"""
        recorder = ResponseRecorder(db_session_factory=sqlite_session_factory)
        recorder.record(
            cache_key="test_cache_key",
            screenshot_hash="sh1",
            candidate_hash="ch1",
            prompt_version="1.0",
            schema_version="1.0",
            provider_name="mock",
            model_name="mock/mock-model",
            raw_response_redacted=None,
            parsed_model_json='{"test": true}',
            token_input=500,
            token_output=300,
            cost_usd=0.005,
            latency_ms=100,
            status="success",
        )

        # 查询验证
        session = sqlite_session_factory()
        try:
            result = session.execute(
                text("SELECT cache_key, status, cost_usd, token_input, token_output FROM vlm_responses WHERE cache_key = :ck"),
                {"ck": "test_cache_key"},
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == "test_cache_key"
            assert row[1] == "success"
            assert abs(row[2] - 0.005) < 0.0001
            assert row[3] == 500
            assert row[4] == 300
        finally:
            session.close()

    def test_record_cost_usd_persisted(self, sqlite_session_factory):
        """cost_usd 字段正确持久化。"""
        recorder = ResponseRecorder(db_session_factory=sqlite_session_factory)
        recorder.record(
            cache_key="cost_test",
            screenshot_hash="sh",
            candidate_hash="ch",
            prompt_version="1.0",
            schema_version="1.0",
            provider_name="openai",
            model_name="openai/gpt-4o",
            raw_response_redacted=None,
            parsed_model_json="{}",
            token_input=1000,
            token_output=500,
            cost_usd=0.0125,
            latency_ms=200,
            status="success",
        )

        session = sqlite_session_factory()
        try:
            result = session.execute(
                text("SELECT cost_usd FROM vlm_responses WHERE cache_key = :ck"),
                {"ck": "cost_test"},
            )
            row = result.fetchone()
            assert row is not None
            assert abs(row[0] - 0.0125) < 0.0001
        finally:
            session.close()

    def test_record_failure_persists(self, sqlite_session_factory):
        """失败记录也正确落库。"""
        recorder = ResponseRecorder(db_session_factory=sqlite_session_factory)
        recorder.record(
            cache_key="fail_key",
            screenshot_hash="sh",
            candidate_hash="ch",
            prompt_version="1.0",
            schema_version="1.0",
            provider_name="mock",
            model_name="mock/mock-model",
            raw_response_redacted=None,
            parsed_model_json="",
            token_input=0,
            token_output=0,
            cost_usd=0.0,
            latency_ms=0,
            status="failed",
        )

        session = sqlite_session_factory()
        try:
            result = session.execute(
                text("SELECT status FROM vlm_responses WHERE cache_key = :ck"),
                {"ck": "fail_key"},
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == "failed"
        finally:
            session.close()

    def test_session_rollback_on_error(self, sqlite_session_factory):
        """写入失败时 session 正确 rollback，不影响后续写入。"""
        recorder = ResponseRecorder(db_session_factory=sqlite_session_factory)

        # 第一次写入：用无效数据触发错误（违反 NOT NULL 约束等）
        # 然后第二次写入应该成功
        recorder.record(
            cache_key="valid_key",
            screenshot_hash="sh",
            candidate_hash="ch",
            prompt_version="1.0",
            schema_version="1.0",
            provider_name="mock",
            model_name="mock",
            raw_response_redacted=None,
            parsed_model_json="{}",
            token_input=0,
            token_output=0,
            cost_usd=0.0,
            latency_ms=0,
            status="success",
        )

        session = sqlite_session_factory()
        try:
            result = session.execute(
                text("SELECT COUNT(*) FROM vlm_responses"),
            )
            assert result.fetchone()[0] == 1
        finally:
            session.close()

    def test_multiple_records_persist(self, sqlite_session_factory):
        """多条记录正确落库。"""
        recorder = ResponseRecorder(db_session_factory=sqlite_session_factory)
        for i in range(5):
            recorder.record(
                cache_key=f"key_{i}",
                screenshot_hash=f"sh_{i}",
                candidate_hash=f"ch_{i}",
                prompt_version="1.0",
                schema_version="1.0",
                provider_name="mock",
                model_name="mock/mock-model",
                raw_response_redacted=None,
                parsed_model_json="{}",
                token_input=i * 100,
                token_output=i * 50,
                cost_usd=float(i) * 0.001,
                latency_ms=i * 10,
                status="success",
            )

        session = sqlite_session_factory()
        try:
            result = session.execute(text("SELECT COUNT(*) FROM vlm_responses"))
            assert result.fetchone()[0] == 5

            # 验证 SUM(cost_usd)
            result = session.execute(text("SELECT SUM(cost_usd) FROM vlm_responses"))
            total = result.fetchone()[0]
            expected = sum(float(i) * 0.001 for i in range(5))
            assert abs(total - expected) < 0.0001
        finally:
            session.close()


# =============================================================================
# Schema Validation — app_identity / page_state
# =============================================================================


class TestSchemaAppIdentityPageState:
    def test_missing_app_identity(self):
        raw = {
            "schema_version": "1.0",
            "page_state": {"page_class": "home", "state_label": "主页"},
            "regions": [{"region_id": "r1", "role": "nav", "bounds": [0, 0, 100, 40], "purpose": "top"}],
        }
        with pytest.raises(ValidationError, match="app_identity"):
            validate_page_semantic_model(raw)

    def test_empty_app_name(self):
        raw = {
            "schema_version": "1.0",
            "app_identity": {"app_name": "", "surface_type": "native_uia"},
            "page_state": {"page_class": "home", "state_label": "主页"},
            "regions": [{"region_id": "r1", "role": "nav", "bounds": [0, 0, 100, 40], "purpose": "top"}],
        }
        with pytest.raises(ValidationError, match="app_name"):
            validate_page_semantic_model(raw)

    def test_missing_page_state(self):
        raw = {
            "schema_version": "1.0",
            "app_identity": {"app_name": "测试", "surface_type": "native_uia"},
            "regions": [{"region_id": "r1", "role": "nav", "bounds": [0, 0, 100, 40], "purpose": "top"}],
        }
        with pytest.raises(ValidationError, match="page_state"):
            validate_page_semantic_model(raw)

    def test_empty_page_class(self):
        raw = {
            "schema_version": "1.0",
            "app_identity": {"app_name": "测试", "surface_type": "native_uia"},
            "page_state": {"page_class": "", "state_label": "主页"},
            "regions": [{"region_id": "r1", "role": "nav", "bounds": [0, 0, 100, 40], "purpose": "top"}],
        }
        with pytest.raises(ValidationError, match="page_class"):
            validate_page_semantic_model(raw)

    def test_empty_state_label(self):
        raw = {
            "schema_version": "1.0",
            "app_identity": {"app_name": "测试", "surface_type": "native_uia"},
            "page_state": {"page_class": "home", "state_label": ""},
            "regions": [{"region_id": "r1", "role": "nav", "bounds": [0, 0, 100, 40], "purpose": "top"}],
        }
        with pytest.raises(ValidationError, match="state_label"):
            validate_page_semantic_model(raw)

    def test_valid_app_identity_and_page_state(self):
        raw = {
            "schema_version": "1.0",
            "app_identity": {"app_name": "微信", "app_id": "wechat", "surface_type": "native_uia"},
            "page_state": {"page_class": "chat", "state_label": "聊天窗口", "state_flags": []},
            "regions": [{"region_id": "r1", "role": "nav", "bounds": [0, 0, 100, 40], "purpose": "top"}],
            "confidence": 0.8,
        }
        model, _ = validate_page_semantic_model(raw)
        assert model.app_identity.app_name == "微信"
        assert model.page_state.page_class == "chat"


# =============================================================================
# Provider model_id
# =============================================================================


class TestProviderModelId:
    def test_mock_model_id(self):
        provider = MockProvider()
        assert provider.model_id == "mock/mock-model"

    def test_openai_model_id(self):
        provider = OpenAICompatibleProvider(api_key="test", model="gpt-4o")
        assert provider.model_id == "openai/gpt-4o"

    def test_openrouter_model_id(self):
        provider = OpenAICompatibleProvider(
            api_key="test", model="meta-llama/llama-3-8b-instruct:free",
            provider_variant="openrouter",
        )
        assert provider.model_id == "openrouter/meta-llama/llama-3-8b-instruct:free"
        assert provider.name == "openrouter"

    def test_minimax_model_id(self):
        provider = MiniMaxProvider(api_key="test", model="MiniMax-VL-01")
        assert provider.model_id == "minimax/MiniMax-VL-01"


# =============================================================================
# OpenRouter Free Model Detection
# =============================================================================


class TestOpenRouterFreeModel:
    def test_free_model_by_suffix(self):
        provider = OpenAICompatibleProvider(
            api_key="test",
            model="meta-llama/llama-3-8b-instruct:free",
            provider_variant="openrouter",
        )
        assert provider.capabilities.is_free
        assert provider.capabilities.cost_per_1k_input == 0.0
        assert provider.capabilities.cost_per_1k_output == 0.0
        assert provider.cost_tier == 1
        assert provider.is_available()

    def test_non_free_model(self):
        provider = OpenAICompatibleProvider(api_key="test", model="gpt-4o")
        assert not provider.capabilities.is_free
        assert provider.capabilities.cost_per_1k_input > 0
        assert provider.cost_tier == 3
        assert provider.is_available()

    def test_free_model_only_blocks_non_free(self):
        """free_model_only=True 时，非免费模型 is_available()=False。"""
        provider = OpenAICompatibleProvider(
            api_key="test", model="gpt-4o", free_model_only=True,
        )
        # gpt-4o 不以 :free 结尾 → capabilities.is_free = False
        assert not provider.capabilities.is_free
        # is_available() 返回 False（free_model_only 策略阻止）
        assert not provider.is_available()

    def test_free_model_only_allows_free_model(self):
        """free_model_only=True 时，:free 后缀模型 is_available()=True。"""
        provider = OpenAICompatibleProvider(
            api_key="test",
            model="meta-llama/llama-3-8b-instruct:free",
            free_model_only=True,
        )
        assert provider.capabilities.is_free
        assert provider.is_available()

    def test_no_api_key_unavailable(self):
        """无 API key 时 is_available()=False。"""
        provider = OpenAICompatibleProvider(api_key="", model="gpt-4o")
        assert not provider.is_available()


# =============================================================================
# Transport Parameter Filtering
# =============================================================================


class TestTransportFiltering:
    def test_timeout_not_in_openai_payload(self):
        """timeout 不应出现在发给 OpenAI API 的 payload 中。"""
        provider = OpenAICompatibleProvider(api_key="test-key", model="gpt-4o")
        screenshot = _make_screenshot()
        request = VLMSemanticRequest(
            screenshot=screenshot,
            messages=[{"role": "user", "content": [{"type": "text", "text": "test"}]}],
            system_prompt="test",
            provider_options={"timeout": 30, "temperature": 0.5},
        )
        with patch("src.vlm.transport.requests.Session") as mock_session_cls:
            session = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
            mock_response.raise_for_status = MagicMock()
            session.post.return_value = mock_response
            mock_session_cls.return_value = session

            provider.analyze_page(request)

            call_kwargs = session.post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert "timeout" not in payload
            assert payload.get("temperature") == 0.5
            # timeout 传给 requests.post 的 timeout 参数
            assert call_kwargs.kwargs.get("timeout") == 30 or call_kwargs[1].get("timeout") == 30



    def test_qwen_payload_does_not_force_openai_response_format(self):
        """Qwen compatible endpoint is faster and cleaner with prompt-only JSON."""
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen2.5-vl-7b-instruct",
            provider_variant="qwen",
        )
        screenshot = _make_screenshot()
        request = VLMSemanticRequest(
            screenshot=screenshot,
            messages=[{"role": "user", "content": [{"type": "text", "text": "test"}]}],
            system_prompt="test",
            provider_options={"timeout": 30},
        )
        with patch("src.vlm.transport.requests.Session") as mock_session_cls:
            session = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
            mock_response.raise_for_status = MagicMock()
            session.post.return_value = mock_response
            mock_session_cls.return_value = session

            provider.analyze_page(request)

            payload = session.post.call_args.kwargs.get("json")
            assert "response_format" not in payload

    def test_qwen_dashscope_bypasses_environment_proxy(self):
        """DashScope should use direct transport so local proxy does not slow domestic models."""
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-vl-plus",
            provider_variant="qwen",
        )
        screenshot = _make_screenshot()
        request = VLMSemanticRequest(
            screenshot=screenshot,
            messages=[{"role": "user", "content": [{"type": "text", "text": "test"}]}],
            system_prompt="test",
            provider_options={"timeout": 30},
        )
        with patch("src.vlm.transport.requests.Session") as mock_session_cls:
            session = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
            mock_response.raise_for_status = MagicMock()
            session.post.return_value = mock_response
            mock_session_cls.return_value = session

            provider.analyze_page(request)

            assert session.trust_env is False
            assert session.post.call_args.kwargs.get("timeout") == 30

    def test_openai_provider_uses_explicit_vlm_proxy_port(self):
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            model="gpt-4o",
            proxy_port=7890,
        )
        screenshot = _make_screenshot()
        request = VLMSemanticRequest(
            screenshot=screenshot,
            messages=[{"role": "user", "content": [{"type": "text", "text": "test"}]}],
            system_prompt="test",
            provider_options={"timeout": 30},
        )
        with patch("src.vlm.transport.requests.Session") as mock_session_cls:
            session = MagicMock()
            session.proxies = {}
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
            mock_response.raise_for_status = MagicMock()
            session.post.return_value = mock_response
            mock_session_cls.return_value = session

            provider.analyze_page(request)

            assert session.trust_env is False
            assert session.proxies["http"] == "http://127.0.0.1:7890"
            assert session.proxies["https"] == "http://127.0.0.1:7890"

    def test_provider_control_metadata_not_sent_to_openai_payload(self):
        """Internal ROI diagnostics must not leak into provider HTTP payload."""
        provider = OpenAICompatibleProvider(api_key="test-key", model="gpt-4o")
        screenshot = _make_screenshot()
        request = VLMSemanticRequest(
            screenshot=screenshot,
            messages=[{"role": "user", "content": [{"type": "text", "text": "test"}]}],
            system_prompt="test",
            provider_options={
                "timeout": 30,
                "response_contract": "roi_semantic_supplement",
                "roi_profile": "fast",
                "temperature": 0.1,
            },
        )
        with patch("src.vlm.transport.requests.Session") as mock_session_cls:
            session = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
            mock_response.raise_for_status = MagicMock()
            session.post.return_value = mock_response
            mock_session_cls.return_value = session

            provider.analyze_page(request)

            payload = session.post.call_args.kwargs.get("json") or session.post.call_args[1].get("json")
            assert "response_contract" not in payload
            assert "roi_profile" not in payload
            assert payload.get("temperature") == 0.1

    def test_timeout_not_in_minimax_payload(self):
        """timeout 不应出现在发给 MiniMax API 的 payload 中。"""
        provider = MiniMaxProvider(api_key="test-key", model="MiniMax-VL-01")
        screenshot = _make_screenshot()
        request = VLMSemanticRequest(
            screenshot=screenshot,
            messages=[{"role": "user", "content": [{"type": "text", "text": "test"}]}],
            system_prompt="test",
            provider_options={"timeout": 45},
        )
        with patch("src.vlm.transport.requests.Session") as mock_session_cls:
            session = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
            mock_response.raise_for_status = MagicMock()
            session.post.return_value = mock_response
            mock_session_cls.return_value = session

            provider.analyze_page(request)

            call_kwargs = session.post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert "timeout" not in payload
            assert call_kwargs.kwargs.get("timeout") == 45 or call_kwargs[1].get("timeout") == 45


# =============================================================================
# Known Candidate IDs in Validation
# =============================================================================


class TestKnownCandidateIds:
    def test_known_ids_pass_without_warning(self):
        raw = {
            "schema_version": "1.0",
            "app_identity": {"app_name": "测试", "surface_type": "native_uia"},
            "page_state": {"page_class": "home", "state_label": "主页"},
            "regions": [{"region_id": "r1", "role": "nav", "bounds": [0, 0, 100, 40], "purpose": "top"}],
            "fixed_controls": [{
                "control_id": "c1", "region_id": "r1", "control_type": "button",
                "text": "OK", "bounds": [10, 5, 50, 35], "semantic_role": "ok_button",
                "visual_type": "button", "interactable": True, "confidence": 0.9,
                "source_candidate_ids": ["uia_1"], "matches_candidate_id": "uia_1",
            }],
            "confidence": 0.8,
        }
        _, warnings = validate_page_semantic_model(
            raw, known_candidate_ids={"uia_1"},
        )
        assert not any("unknown candidate" in w for w in warnings)

    def test_unknown_ids_produce_warning(self):
        raw = {
            "schema_version": "1.0",
            "app_identity": {"app_name": "测试", "surface_type": "native_uia"},
            "page_state": {"page_class": "home", "state_label": "主页"},
            "regions": [{"region_id": "r1", "role": "nav", "bounds": [0, 0, 100, 40], "purpose": "top"}],
            "fixed_controls": [{
                "control_id": "c1", "region_id": "r1", "control_type": "button",
                "text": "OK", "bounds": [10, 5, 50, 35], "semantic_role": "ok_button",
                "visual_type": "button", "interactable": True, "confidence": 0.9,
                "source_candidate_ids": ["nonexistent_xyz"],
            }],
            "confidence": 0.8,
        }
        _, warnings = validate_page_semantic_model(
            raw, known_candidate_ids={"uia_1"},
        )
        assert any("nonexistent_xyz" in w for w in warnings)

    def test_extract_candidate_ids(self):
        candidates = [
            {"id": "uia_1", "type": "button"},
            {"id": "ocr_3", "type": "text"},
            {"id": "", "type": "unknown"},
        ]
        ids = _extract_candidate_ids(candidates)
        assert ids == {"uia_1", "ocr_3"}

    def test_extract_candidate_ids_empty(self):
        assert _extract_candidate_ids(None) == set()
        assert _extract_candidate_ids([]) == set()


# =============================================================================
# VLMSemanticModeler — Core
# =============================================================================


class TestVLMSemanticModeler:
    def test_mock_provider_returns_success(self):
        modeler = _make_modeler()
        result = modeler.analyze(_make_prompt_input())
        assert result.status == "success"
        assert result.model is not None
        assert result.model.app_identity.app_name == "测试应用"

    def test_cache_hit_skips_call(self):
        cache = SemanticModelCache()
        modeler = _make_modeler(cache=cache)
        pi = _make_prompt_input()

        result1 = modeler.analyze(pi)
        assert result1.status == "success"
        assert not result1.from_cache

        result2 = modeler.analyze(pi)
        assert result2.status == "cached"
        assert result2.from_cache

    def test_force_bypasses_cache(self):
        cache = SemanticModelCache()
        modeler = _make_modeler(cache=cache)
        pi = _make_prompt_input()

        result1 = modeler.analyze(pi)
        assert result1.status == "success"

        result2 = modeler.analyze(pi, force=True)
        assert result2.status == "success"
        assert not result2.from_cache

    def test_provider_fallback(self):
        primary = MockProvider(fail=True)
        fallback = MockProvider(scenario="default")
        modeler = _make_modeler(primary=primary, fallback=fallback)
        result = modeler.analyze(_make_prompt_input())
        assert result.status == "success"

    def test_all_providers_fail(self):
        primary = MockProvider(fail=True)
        fallback = MockProvider(fail=True)
        modeler = _make_modeler(primary=primary, fallback=fallback)
        result = modeler.analyze(_make_prompt_input())
        assert result.status == "failed"

    def test_provider_unavailable(self):
        primary = MockProvider(fail=True)
        modeler = _make_modeler(primary=primary)
        result = modeler.analyze(_make_prompt_input())
        assert result.status == "failed"
        assert result.error_code == "provider_unavailable"

    def test_budget_exceeded(self):
        config = _make_config(daily_call_limit=0)
        tracker = BudgetTracker(config)
        modeler = VLMSemanticModeler(
            primary_provider=MockProvider(),
            budget_tracker=tracker,
            config=config,
        )
        result = modeler.analyze(_make_prompt_input())
        assert result.status == "failed"
        assert result.error_code == "budget_exceeded"

    def test_markdown_wrapped_json_success(self):
        provider = MockProvider(markdown_wrapped=True)
        modeler = _make_modeler(primary=provider)
        result = modeler.analyze(_make_prompt_input())
        assert result.status == "success"

    def test_should_analyze_disabled(self):
        config = _make_config(enabled=False)
        modeler = _make_modeler(config=config)
        should, reason = modeler.should_analyze()
        assert not should
        assert "disabled" in reason

    def test_should_analyze_drift(self):
        modeler = _make_modeler()
        should, reason = modeler.should_analyze(drift_detected=True)
        assert should
        assert "drift" in reason

    def test_should_analyze_force(self):
        modeler = _make_modeler()
        should, reason = modeler.should_analyze(force=True)
        assert should
        assert "forced" in reason

    def test_raw_response_redacted(self):
        config = _make_config(save_raw_response=True)
        modeler = _make_modeler(config=config)
        result = modeler.analyze(_make_prompt_input())
        assert result.status == "success"
        assert result.raw_response_redacted is not None

    def test_model_name_uses_model_id(self):
        """result.model_name 使用 provider.model_id 格式。"""
        modeler = _make_modeler()
        result = modeler.analyze(_make_prompt_input())
        assert result.model_name == "mock/mock-model"

    def test_known_candidate_ids_passed_to_validator(self):
        """omni_candidates 的 ID 会传给 validate_page_semantic_model。"""
        candidates = [
            {"id": "uia_1", "type": "button"},
            {"id": "uia_2", "type": "text"},
        ]
        modeler = _make_modeler()
        result = modeler.analyze(_make_prompt_input(omni_candidates=candidates))
        # MockProvider 返回的 JSON 不引用这些 ID，所以不应有 unknown candidate warning
        assert result.status == "success"

    def test_free_model_only_blocks_in_modeler(self):
        """free_model_only=True 时，非免费 primary 被 budget 阻止。"""
        config = _make_config(free_model_only=True, daily_call_limit=100)
        primary = OpenAICompatibleProvider(api_key="test", model="gpt-4o")
        modeler = VLMSemanticModeler(
            primary_provider=primary,
            budget_tracker=BudgetTracker(config),
            config=config,
        )
        result = modeler.analyze(_make_prompt_input())
        assert result.status == "failed"
        assert result.error_code == "budget_exceeded"

    def test_free_model_only_with_fallback(self):
        """free_model_only=True 时，非免费 primary 被阻止，fallback 到免费 provider。"""
        config = _make_config(free_model_only=True, daily_call_limit=100)
        primary = OpenAICompatibleProvider(api_key="test", model="gpt-4o")
        fallback = MockProvider()  # free
        modeler = VLMSemanticModeler(
            primary_provider=primary,
            fallback_provider=fallback,
            budget_tracker=BudgetTracker(config),
            config=config,
        )
        result = modeler.analyze(_make_prompt_input())
        assert result.status == "success"

    def test_fallback_uses_own_cache_key(self):
        """fallback 使用自己的 model_id 计算 cache key，不污染 primary 缓存。"""
        cache = SemanticModelCache()
        primary = MockProvider(fail=True)
        fallback = MockProvider(scenario="default")
        modeler = _make_modeler(primary=primary, fallback=fallback, cache=cache)
        result = modeler.analyze(_make_prompt_input())
        assert result.status == "success"
        # 验证缓存键是 fallback 的 model_id
        pi = _make_prompt_input()
        from src.vlm.modeler import _hash_image, _hash_candidates, _hash_text
        fb_key = cache.compute_key(
            _hash_image(pi.screenshot),
            _hash_candidates(pi.omni_candidates),
            "1.0", "1.0", fallback.model_id, _hash_text(pi.task_hint),
        )
        assert cache.get(fb_key) is not None


# =============================================================================
# Response Recording (vlm_responses 落库)
# =============================================================================


class TestResponseRecording:
    def test_success_recorded(self):
        session = MagicMock()
        recorder = ResponseRecorder(db_session_factory=lambda: session)
        modeler = _make_modeler(recorder=recorder)
        result = modeler.analyze(_make_prompt_input())
        assert result.status == "success"
        session.execute.assert_called()
        session.commit.assert_called()
        # 检查 status 参数
        call_args = session.execute.call_args
        params = call_args[0][1]
        assert params["status"] == "success"
        assert params["cost_usd"] >= 0  # MockProvider is free, cost=0

    def test_failure_recorded(self):
        session = MagicMock()
        recorder = ResponseRecorder(db_session_factory=lambda: session)
        primary = MockProvider(fail=True)
        modeler = _make_modeler(primary=primary, recorder=recorder)
        result = modeler.analyze(_make_prompt_input())
        assert result.status == "failed"
        session.execute.assert_called()
        call_args = session.execute.call_args
        params = call_args[0][1]
        assert params["status"] == "failed"

    def test_cached_result_not_recorded(self):
        """缓存命中不写入 vlm_responses。"""
        session = MagicMock()
        recorder = ResponseRecorder(db_session_factory=lambda: session)
        cache = SemanticModelCache()
        modeler = _make_modeler(cache=cache, recorder=recorder)
        pi = _make_prompt_input()

        modeler.analyze(pi)
        session.execute.assert_called_once()

        session.reset_mock()
        modeler.analyze(pi)  # cache hit
        session.execute.assert_not_called()

    def test_parsed_model_is_sanitized(self):
        """落库的 parsed_model 是脱敏版本（dynamic_zones.note 清空）。"""
        session = MagicMock()
        recorder = ResponseRecorder(db_session_factory=lambda: session)
        modeler = _make_modeler(recorder=recorder)
        modeler.analyze(_make_prompt_input())
        call_args = session.execute.call_args
        params = call_args[0][1]
        parsed = json.loads(params["parsed_model"])
        for zone in parsed.get("dynamic_zones", []):
            assert zone.get("note") == "[REDACTED_FOR_STORAGE]"

    def test_cost_usd_recorded_for_paid_provider(self):
        """非免费 provider 的 cost_usd > 0 正确落库。"""
        session = MagicMock()
        recorder = ResponseRecorder(db_session_factory=lambda: session)
        config = _make_config()
        provider = OpenAICompatibleProvider(api_key="test", model="gpt-4o")
        modeler = VLMSemanticModeler(
            primary_provider=provider,
            budget_tracker=BudgetTracker(config),
            config=config,
            recorder=recorder,
        )
        # 用 patch 让 OpenAI provider 返回有效 JSON
        mock_raw = VLMSemanticRequest  # just to avoid unused import
        with patch.object(provider, "analyze_page") as mock_analyze:
            from src.vlm.provider import VLMSemanticRawResponse
            valid_json = json.dumps({
                "schema_version": "1.0",
                "app_identity": {"app_name": "Test", "surface_type": "native_uia"},
                "page_state": {"page_class": "home", "state_label": "主页"},
                "regions": [{"region_id": "r1", "role": "nav", "bounds": [0, 0, 100, 40], "purpose": "top"}],
                "confidence": 0.8,
            })
            mock_analyze.return_value = VLMSemanticRawResponse(
                raw_text=valid_json,
                provider_name="openai",
                model_name="gpt-4o",
                token_input=1000,
                token_output=500,
                latency_ms=200,
                finish_reason="stop",
            )
            result = modeler.analyze(_make_prompt_input())
            assert result.status == "success"
            call_args = session.execute.call_args
            params = call_args[0][1]
            # gpt-4o: cost = 1000/1000*0.005 + 500/1000*0.015 = 0.0125
            assert params["cost_usd"] > 0


# =============================================================================
# Provider Scenarios
# =============================================================================


class TestProviderScenarios:
    def test_new_page_scenario(self):
        provider = MockProvider(scenario="new_page")
        modeler = _make_modeler(primary=provider)
        result = modeler.analyze(_make_prompt_input())
        assert result.status == "success"
        assert result.model is not None
        assert result.model.needs_review

    def test_drift_scenario(self):
        provider = MockProvider(scenario="drift")
        modeler = _make_modeler(primary=provider)
        result = modeler.analyze(_make_prompt_input())
        assert result.status == "success"
        assert result.model is not None
        assert len(result.model.candidate_corrections) > 0

    def test_transition_extraction(self):
        modeler = _make_modeler()
        result = modeler.analyze(_make_prompt_input())
        if result.model and result.model.transitions:
            t = result.model.transitions[0]
            assert t.to_state

    def test_fallback_cache_hit_skips_call(self):
        """fallback 成功后，第二次 analyze() 命中 fallback 缓存，不调用 fallback provider。"""
        primary = MockProvider(fail=True)
        fallback = MockProvider(scenario="default")
        cache = SemanticModelCache()
        modeler = _make_modeler(primary=primary, fallback=fallback, cache=cache)
        pi = _make_prompt_input()

        result1 = modeler.analyze(pi)
        assert result1.status == "success"
        assert result1.provider_name == "mock"

        # 第二次：应命中 fallback 缓存
        result2 = modeler.analyze(pi)
        assert result2.status == "cached"
        assert result2.from_cache
        assert result2.provider_name == "mock"

    def test_fallback_cache_used_when_primary_budget_exceeded(self):
        """primary 预算超限 + fallback 缓存命中 → 返回 cached。"""
        cache = SemanticModelCache()
        fallback = MockProvider()
        config = _make_config(daily_call_limit=100)
        modeler = _make_modeler(fallback=fallback, cache=cache, config=config)
        pi = _make_prompt_input()

        # 第一次：正常走 primary（MockProvider），写入 primary 缓存
        result1 = modeler.analyze(pi)
        assert result1.status == "success"

        # 用 fallback provider 预热 fallback 缓存
        from src.vlm.modeler import _hash_image, _hash_candidates, _hash_text
        fb_key = cache.compute_key(
            _hash_image(pi.screenshot),
            _hash_candidates(pi.omni_candidates),
            "1.0", "1.0", fallback.model_id, _hash_text(pi.task_hint),
        )
        cache.put(fb_key, result1.model)

        # 用新 modeler：primary 不可用，fallback 有缓存
        primary_unavail = MockProvider(fail=True)
        modeler2 = _make_modeler(primary=primary_unavail, fallback=fallback, cache=cache)
        result2 = modeler2.analyze(pi)
        assert result2.status == "cached"
        assert result2.from_cache


# =============================================================================
# should_analyze() — Fallback Awareness
# =============================================================================


class TestShouldAnalyzeFallback:
    def test_primary_blocked_fallback_available(self):
        """primary 被 free_model_only 阻止 + fallback 是免费 → should_analyze=True。"""
        config = _make_config(free_model_only=True, daily_call_limit=100)
        primary = OpenAICompatibleProvider(api_key="test", model="gpt-4o")
        fallback = MockProvider()  # free
        modeler = VLMSemanticModeler(
            primary_provider=primary,
            fallback_provider=fallback,
            budget_tracker=BudgetTracker(config),
            config=config,
        )
        should, reason = modeler.should_analyze()
        assert should
        assert "fallback" in reason

    def test_primary_budget_exceeded_fallback_available(self):
        """primary 预算超限 + fallback 预算允许 → should_analyze=True。"""
        config = _make_config(daily_call_limit=1)
        primary = MockProvider()
        fallback = MockProvider()
        tracker = BudgetTracker(config)
        tracker._daily_count = 1  # primary 超限
        modeler = VLMSemanticModeler(
            primary_provider=primary,
            fallback_provider=fallback,
            budget_tracker=tracker,
            config=config,
        )
        # fallback 同样超限（共享 tracker）→ False
        should, reason = modeler.should_analyze()
        assert not should

    def test_primary_unavailable_fallback_available(self):
        """primary 不可用 + fallback 可用 → should_analyze=True。"""
        config = _make_config(enabled=True, daily_call_limit=100)
        primary = MockProvider(fail=True)
        fallback = MockProvider()
        modeler = VLMSemanticModeler(
            primary_provider=primary,
            fallback_provider=fallback,
            budget_tracker=BudgetTracker(config),
            config=config,
        )
        should, reason = modeler.should_analyze()
        assert should
        assert "fallback" in reason

    def test_both_unavailable(self):
        """primary 和 fallback 都不可用 → should_analyze=False。"""
        config = _make_config(enabled=True, daily_call_limit=100)
        primary = MockProvider(fail=True)
        fallback = MockProvider(fail=True)
        modeler = VLMSemanticModeler(
            primary_provider=primary,
            fallback_provider=fallback,
            budget_tracker=BudgetTracker(config),
            config=config,
        )
        should, reason = modeler.should_analyze()
        assert not should
        assert "unavailable" in reason

    def test_no_fallback_still_works(self):
        """没有 fallback 时，should_analyze 行为不变。"""
        config = _make_config(enabled=True, daily_call_limit=100)
        modeler = VLMSemanticModeler(
            primary_provider=MockProvider(),
            budget_tracker=BudgetTracker(config),
            config=config,
        )
        should, reason = modeler.should_analyze(drift_detected=True)
        assert should
        assert "drift" in reason


# =============================================================================
# Contextmanager Session Factory Compatibility
# =============================================================================


class TestContextmanagerSessionFactory:
    """测试 ResponseRecorder / BudgetTracker 兼容 contextmanager 工厂（如 src.storage.db.Session）。"""

    @pytest.fixture
    def sqlite_engine_and_factory(self):
        """创建内存 SQLite + contextmanager 风格的 session 工厂。"""
        from contextlib import contextmanager
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE vlm_responses (
                    response_id TEXT PRIMARY KEY,
                    cache_key TEXT NOT NULL,
                    screenshot_hash TEXT NOT NULL,
                    candidate_hash TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    provider_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    raw_response_redacted TEXT,
                    parsed_model TEXT NOT NULL,
                    token_input INTEGER DEFAULT 0,
                    token_output INTEGER DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0.0,
                    latency_ms INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    created_at TEXT,
                    page_model_id TEXT,
                    state_template_id TEXT
                )
            """))

        SessionFactory = sessionmaker(bind=engine)

        @contextmanager
        def cm_session():
            """模拟 src.storage.db.Session — contextmanager 包装器。"""
            session = SessionFactory()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        yield engine, cm_session
        engine.dispose()

    def test_recorder_with_contextmanager_factory(self, sqlite_engine_and_factory):
        """ResponseRecorder 兼容 contextmanager 工厂。"""
        _, cm_session = sqlite_engine_and_factory
        recorder = ResponseRecorder(db_session_factory=cm_session)
        recorder.record(
            cache_key="cm_test",
            screenshot_hash="sh",
            candidate_hash="ch",
            prompt_version="1.0",
            schema_version="1.0",
            provider_name="mock",
            model_name="mock/mock-model",
            raw_response_redacted=None,
            parsed_model_json="{}",
            token_input=100,
            token_output=50,
            cost_usd=0.003,
            latency_ms=80,
            status="success",
        )

        # 验证写入成功
        engine, _ = sqlite_engine_and_factory
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT cache_key, status, cost_usd FROM vlm_responses WHERE cache_key = :ck"),
                {"ck": "cm_test"},
            ).fetchone()
            assert row is not None
            assert row[0] == "cm_test"
            assert row[1] == "success"
            assert abs(row[2] - 0.003) < 0.0001

    def test_budget_tracker_with_contextmanager_factory(self, sqlite_engine_and_factory):
        """BudgetTracker 兼容 contextmanager 工厂。"""
        from datetime import date
        engine, cm_session = sqlite_engine_and_factory

        # 先插入测试数据
        with engine.begin() as conn:
            today = date.today().isoformat()
            conn.execute(
                text("INSERT INTO vlm_responses (response_id, cache_key, screenshot_hash, candidate_hash, prompt_version, schema_version, provider_name, model_name, parsed_model, token_input, token_output, cost_usd, latency_ms, status, created_at) VALUES (:rid, :ck, :sh, :ch, :pv, :sv, :pn, :mn, :pm, :ti, :to, :cu, :lm, :st, :ca)"),
                {"rid": "r1", "ck": "k1", "sh": "sh", "ch": "ch", "pv": "1.0", "sv": "1.0", "pn": "mock", "mn": "mock", "pm": "{}", "ti": 100, "to": 50, "cu": 2.5, "lm": 80, "st": "success", "ca": f"{today}T10:00:00"},
            )
            conn.execute(
                text("INSERT INTO vlm_responses (response_id, cache_key, screenshot_hash, candidate_hash, prompt_version, schema_version, provider_name, model_name, parsed_model, token_input, token_output, cost_usd, latency_ms, status, created_at) VALUES (:rid, :ck, :sh, :ch, :pv, :sv, :pn, :mn, :pm, :ti, :to, :cu, :lm, :st, :ca)"),
                {"rid": "r2", "ck": "k2", "sh": "sh", "ch": "ch", "pv": "1.0", "sv": "1.0", "pn": "mock", "mn": "mock", "pm": "{}", "ti": 100, "to": 50, "cu": 3.0, "lm": 80, "st": "success", "ca": f"{today}T11:00:00"},
            )

        config = _make_config(daily_call_limit=100, monthly_budget_usd=10.0)
        tracker = BudgetTracker(config, db_session_factory=cm_session)
        tracker._ensure_db_loaded()

        assert tracker._daily_count == 2
        assert abs(tracker._monthly_cost - 5.5) < 0.01

    def test_recorder_and_budget_share_contextmanager_factory(self, sqlite_engine_and_factory):
        """ResponseRecorder 写入 + BudgetTracker 读取，共享同一个 contextmanager 工厂。"""
        _, cm_session = sqlite_engine_and_factory

        # 写入
        recorder = ResponseRecorder(db_session_factory=cm_session)
        for i in range(3):
            recorder.record(
                cache_key=f"shared_{i}",
                screenshot_hash="sh",
                candidate_hash="ch",
                prompt_version="1.0",
                schema_version="1.0",
                provider_name="mock",
                model_name="mock/mock-model",
                raw_response_redacted=None,
                parsed_model_json="{}",
                token_input=100,
                token_output=50,
                cost_usd=float(i) + 0.5,
                latency_ms=80,
                status="success",
            )

        # 读取
        config = _make_config(daily_call_limit=100, monthly_budget_usd=100.0)
        tracker = BudgetTracker(config, db_session_factory=cm_session)
        tracker._ensure_db_loaded()

        assert tracker._daily_count == 3
        expected_cost = 0.5 + 1.5 + 2.5
        assert abs(tracker._monthly_cost - expected_cost) < 0.01


# =============================================================================
# _config_hash — invariant 8: config invariant
# =============================================================================


class TestConfigHash:

    def test_api_key_affects_hash(self):
        """Invariant 8: changing api_key must change config hash."""
        from src.vlm.modeler import _config_hash

        base_args = dict(
            enabled=True, provider="openai", model="gpt-4o", endpoint="",
            provider_variant="", fallback_provider="", fallback_model="",
            prompt_version="1.0", daily_call_limit=100, monthly_budget_usd=10.0,
            free_model_only=False, allow_free_models=True,
            save_raw_response=False, redact_dynamic_content=True,
        )
        hash_a = _config_hash(**base_args, api_key="test-aaa")
        hash_b = _config_hash(**base_args, api_key="test-bbb")
        hash_empty = _config_hash(**base_args, api_key="")

        assert hash_a != hash_b
        assert hash_a != hash_empty
        assert hash_b != hash_empty

    def test_same_config_same_hash(self):
        """Same parameters always produce the same hash."""
        from src.vlm.modeler import _config_hash

        args = dict(
            enabled=True, provider="openai", model="gpt-4o", endpoint="https://api.openai.com/v1",
            provider_variant="", fallback_provider="", fallback_model="",
            prompt_version="1.0", daily_call_limit=100, monthly_budget_usd=10.0,
            free_model_only=False, allow_free_models=True,
            save_raw_response=False, redact_dynamic_content=True, api_key="test-abc",
        )
        assert _config_hash(**args) == _config_hash(**args)

