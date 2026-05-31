"""VLM Semantic Modeler — 核心编排。

缓存检查 → 预算检查 → 输入构造 → 调用 → 校验 → 落库 → 返回。
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Generator

from PIL import Image
from sqlalchemy import text

from src.vlm.privacy import redact_raw_response, sanitize_for_storage
from src.vlm.coordinate_space import (
    normalize_model_to_original,
    prepare_vlm_prompt_input,
)
from src.vlm.prompt_builder import PromptInput, build_page_understanding_prompt
from src.vlm.provider import VLMSemanticProvider, VLMSemanticRawResponse
from src.vlm.schema import (
    CURRENT_SCHEMA_VERSION,
    PageSemanticModel,
    ValidationError,
    VLMSemanticModelerResult,
    validate_page_semantic_model,
)
from src.vlm.trace_artifacts import (
    create_vlm_trace_dir,
    save_image,
    write_json,
    write_text,
)

logger = logging.getLogger(__name__)

PROMPT_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Global Singleton — 避免每次 observe 新建空 cache
# ---------------------------------------------------------------------------

_modeler_instance: VLMSemanticModeler | None = None
_modeler_cache: SemanticModelCache | None = None
_modeler_config_hash: str = ""


def _config_hash(
    enabled: bool,
    provider: str,
    model: str,
    endpoint: str,
    provider_variant: str,
    fallback_provider: str,
    fallback_model: str,
    prompt_version: str,
    daily_call_limit: int,
    monthly_budget_usd: float,
    free_model_only: bool,
    allow_free_models: bool,
    save_raw_response: bool,
    redact_dynamic_content: bool,
    api_key: str = "",
    thinking_mode: str = "auto",
    image_max_width: int = 1280,
) -> str:
    # Hash api_key separately — don't embed raw key in the hash input
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:12] if api_key else ""
    raw = (
        f"{enabled}|{provider}|{model}|{endpoint}|{provider_variant}"
        f"|{fallback_provider}|{fallback_model}|{prompt_version}"
        f"|{daily_call_limit}|{monthly_budget_usd}|{free_model_only}"
        f"|{allow_free_models}|{save_raw_response}|{redact_dynamic_content}"
        f"|{api_key_hash}|{thinking_mode}|{image_max_width}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_modeler_singleton(
    *,
    enabled: bool = False,
    provider: str = "disabled",
    api_key: str = "",
    endpoint: str = "",
    model: str = "",
    provider_variant: str = "",
    free_model_only: bool = False,
    fallback_provider: str = "",
    fallback_model: str = "",
    prompt_version: str = PROMPT_VERSION,
    timeout_seconds: int = 60,
    daily_call_limit: int = 100,
    monthly_budget_usd: float = 10.0,
    save_raw_response: bool = False,
    redact_dynamic_content: bool = True,
    allow_free_models: bool = True,
    thinking_mode: str = "auto",
    image_max_width: int = 1280,
    db_session_factory: Any = None,
) -> VLMSemanticModeler | None:
    """获取全局 singleton modeler。配置变化时重建。

    参数与 create_modeler_provider + SemanticModelerConfig 对齐，
    调用方直接从 config_manager.SemanticModelerConfig 展开传入即可。
    """
    global _modeler_instance, _modeler_cache, _modeler_config_hash

    if not enabled or provider == "disabled":
        _modeler_instance = None
        return None

    cfg_hash = _config_hash(
        enabled, provider, model, endpoint, provider_variant,
        fallback_provider, fallback_model, prompt_version,
        daily_call_limit, monthly_budget_usd, free_model_only,
        allow_free_models, save_raw_response, redact_dynamic_content,
        api_key=api_key, thinking_mode=thinking_mode, image_max_width=image_max_width,
    )
    if _modeler_instance is not None and cfg_hash == _modeler_config_hash:
        return _modeler_instance

    # 配置变化或首次创建 → 重建
    if _modeler_cache is None or cfg_hash != _modeler_config_hash:
        _modeler_cache = SemanticModelCache()
        _warm_cache_from_db(_modeler_cache, db_session_factory)

    from src.vlm.provider import create_modeler_provider as _create_provider

    primary = _create_provider(
        provider=provider,
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        provider_variant=provider_variant,
        free_model_only=free_model_only,
        thinking_mode=thinking_mode,
        image_max_width=image_max_width,
    )
    if primary is None or not primary.is_available():
        return None

    fallback = None
    if fallback_provider:
        fallback = _create_provider(
            provider=fallback_provider,
            api_key=api_key,
            endpoint=endpoint,
            model=fallback_model or model,
            provider_variant=provider_variant,
            free_model_only=free_model_only,
            thinking_mode=thinking_mode,
            image_max_width=image_max_width,
        )

    vlm_config = SemanticModelerConfig(
        enabled=enabled,
        prompt_version=prompt_version,
        timeout_seconds=timeout_seconds,
        daily_call_limit=daily_call_limit,
        monthly_budget_usd=monthly_budget_usd,
        allow_free_models=allow_free_models,
        free_model_only=free_model_only,
        save_raw_response=save_raw_response,
        redact_dynamic_content=redact_dynamic_content,
        image_max_width=image_max_width,
    )

    budget = BudgetTracker(vlm_config, db_session_factory)
    recorder = ResponseRecorder(db_session_factory)

    _modeler_instance = VLMSemanticModeler(
        primary_provider=primary,
        fallback_provider=fallback,
        cache=_modeler_cache,
        budget_tracker=budget,
        config=vlm_config,
        recorder=recorder,
    )
    _modeler_config_hash = cfg_hash
    return _modeler_instance


def _warm_cache_from_db(
    cache: SemanticModelCache,
    db_session_factory: Any = None,
    limit: int = 1000,
) -> None:
    """从 vlm_responses 表加载最近的成功 parsed_model 到内存缓存。"""
    if db_session_factory is None:
        return
    try:
        with _open_session(db_session_factory) as session:
            if session is None:
                return
            result = session.execute(
                text("""SELECT cache_key, parsed_model FROM vlm_responses
                        WHERE status = 'success' AND parsed_model != ''
                        ORDER BY created_at DESC LIMIT :limit"""),
                {"limit": limit},
            )
            rows = result.fetchall()
            loaded = 0
            for row in rows:
                try:
                    model_dict = json.loads(row[1])
                    model = PageSemanticModel.from_dict(model_dict)
                    cache.put(row[0], model)
                    loaded += 1
                except Exception:
                    continue
            if loaded > 0:
                logger.info("Warm cache from DB: loaded %d models", loaded)
    except Exception as exc:
        logger.warning("Warm cache from DB failed: %s", exc)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticModelerConfig:
    """VLM Semantic Modeler 配置（独立于 config_manager，避免循环依赖）。"""
    enabled: bool = False
    prompt_version: str = PROMPT_VERSION
    timeout_seconds: int = 60
    max_retries: int = 2
    # 额度/成本控制
    daily_call_limit: int = 100
    monthly_budget_usd: float = 10.0
    priority: int = 1
    allow_free_models: bool = True
    free_model_only: bool = False
    # 隐私
    save_raw_response: bool = False
    redact_dynamic_content: bool = True
    image_max_width: int = 1280


# ---------------------------------------------------------------------------
# Response Recorder — vlm_responses 落库
# ---------------------------------------------------------------------------


@contextmanager
def _open_session(factory: Any) -> Generator[Any, None, None]:
    """通用 session helper：兼容裸 Session 和 contextmanager 工厂。

    - factory 返回裸 SQLAlchemy Session（或 MagicMock）→ 直接 yield
    - factory 返回 contextmanager（如 src.storage.db.Session）→ __enter__ 获取真正 session
    - factory is None → yield None

    检测策略：duck-type 查 execute 属性。
    裸 Session / sessionmaker() 返回值有 execute；
    _GeneratorContextManager（contextmanager 包装器）没有 execute。
    """
    if factory is None:
        yield None
        return
    raw = factory()
    # 裸 session（有 execute 方法）→ 直接使用
    if hasattr(raw, 'execute'):
        try:
            yield raw
        except Exception:
            if hasattr(raw, 'rollback'):
                raw.rollback()
            raise
        finally:
            if hasattr(raw, 'close'):
                raw.close()
        return
    # contextmanager 包装器（如 _GeneratorContextManager）→ __enter__ 获取真正 session
    if hasattr(raw, '__enter__') and hasattr(raw, '__exit__'):
        session = raw.__enter__()
        exc_info: tuple[Any, Any, Any] = (None, None, None)
        try:
            yield session
        except Exception:
            exc_info = sys.exc_info()
            if hasattr(session, 'rollback'):
                session.rollback()
            raise
        finally:
            raw.__exit__(*exc_info)
        return
    # 不可识别的类型，尝试直接使用
    try:
        yield raw
    except Exception:
        raise
    finally:
        if hasattr(raw, 'close'):
            raw.close()


class ResponseRecorder:
    """将 VLM 调用结果写入 vlm_responses 表。

    db_session_factory 支持两种模式：
    a) 返回裸 SQLAlchemy Session（如 sessionmaker）
    b) 返回 contextmanager（如 src.storage.db.Session）
    """

    def __init__(self, db_session_factory: Any = None) -> None:
        self._db_session_factory = db_session_factory

    def record(
        self,
        *,
        cache_key: str,
        screenshot_hash: str,
        candidate_hash: str,
        prompt_version: str,
        schema_version: str,
        provider_name: str,
        model_name: str,
        raw_response_redacted: str | None,
        parsed_model_json: str,
        token_input: int,
        token_output: int,
        cost_usd: float,
        latency_ms: int,
        status: str,
        page_model_id: str | None = None,
        state_template_id: str | None = None,
    ) -> None:
        """写入一条 vlm_responses 记录。DB 写入失败不影响主流程。"""
        try:
            with _open_session(self._db_session_factory) as session:
                if session is None:
                    return
                response_id = str(uuid.uuid4())
                session.execute(
                    text("""INSERT INTO vlm_responses
                           (response_id, cache_key, screenshot_hash, candidate_hash,
                            prompt_version, schema_version, provider_name, model_name,
                            raw_response_redacted, parsed_model, token_input, token_output,
                            cost_usd, latency_ms, status, created_at, page_model_id, state_template_id)
                           VALUES (:response_id, :cache_key, :screenshot_hash, :candidate_hash,
                            :prompt_version, :schema_version, :provider_name, :model_name,
                            :raw_response_redacted, :parsed_model, :token_input, :token_output,
                            :cost_usd, :latency_ms, :status, :created_at, :page_model_id, :state_template_id)"""),
                    {
                        "response_id": response_id,
                        "cache_key": cache_key,
                        "screenshot_hash": screenshot_hash,
                        "candidate_hash": candidate_hash,
                        "prompt_version": prompt_version,
                        "schema_version": schema_version,
                        "provider_name": provider_name,
                        "model_name": model_name,
                        "raw_response_redacted": raw_response_redacted,
                        "parsed_model": parsed_model_json,
                        "token_input": token_input,
                        "token_output": token_output,
                        "cost_usd": cost_usd,
                        "latency_ms": latency_ms,
                        "status": status,
                        "created_at": datetime.now().isoformat(),
                        "page_model_id": page_model_id,
                        "state_template_id": state_template_id,
                    },
                )
                session.commit()
        except Exception as exc:
            logger.warning("ResponseRecorder: failed to write vlm_response: %s", exc)


# ---------------------------------------------------------------------------
# Budget Tracker
# ---------------------------------------------------------------------------


class BudgetTracker:
    """额度/成本控制。数据来源：vlm_responses 表 + 内存加速缓存。

    db_session_factory 应返回一个 context-managed session。
    """

    def __init__(
        self,
        config: SemanticModelerConfig,
        db_session_factory: Any = None,
    ) -> None:
        self._config = config
        self._db_session_factory = db_session_factory
        # 内存缓存
        self._daily_count: int = 0
        self._monthly_cost: float = 0.0
        self._db_loaded: bool = False

    def check_budget(self, provider: VLMSemanticProvider) -> tuple[bool, str]:
        """检查是否允许调用。

        free_model_only 策略：非免费模型直接不可用，
        不是把付费模型伪装成免费。
        """
        if self._config.free_model_only and not provider.capabilities.is_free:
            return False, "free_model_only: provider is not free"

        if not self._config.allow_free_models and provider.capabilities.is_free:
            return False, "allow_free_models=false: free provider not allowed"

        self._ensure_db_loaded()

        if self._daily_count >= self._config.daily_call_limit:
            return False, f"daily_call_limit reached: {self._daily_count}/{self._config.daily_call_limit}"

        if self._monthly_cost >= self._config.monthly_budget_usd:
            return False, f"monthly_budget_exceeded: ${self._monthly_cost:.4f}/${self._config.monthly_budget_usd:.2f}"

        return True, ""

    def record_usage(
        self,
        provider: VLMSemanticProvider,
        tokens_in: int,
        tokens_out: int,
    ) -> float:
        """记录用量到内存缓存，返回计算出的 cost_usd。"""
        cost = self._compute_cost(provider, tokens_in, tokens_out)
        self._daily_count += 1
        self._monthly_cost += cost
        return cost

    def _compute_cost(
        self,
        provider: VLMSemanticProvider,
        tokens_in: int,
        tokens_out: int,
    ) -> float:
        """按 provider 能力声明计算成本。"""
        caps = provider.capabilities
        return (
            tokens_in / 1000 * caps.cost_per_1k_input
            + tokens_out / 1000 * caps.cost_per_1k_output
        )

    def _ensure_db_loaded(self) -> None:
        """从 DB 加载当天/当月统计（仅首次）。"""
        if self._db_loaded or self._db_session_factory is None:
            return
        self._db_loaded = True
        try:
            self._daily_count = self._query_daily_count()
            self._monthly_cost = self._query_monthly_cost()
        except Exception as exc:
            logger.warning("BudgetTracker DB load failed: %s", exc)

    def _query_daily_count(self) -> int:
        """从 vlm_responses 查询当天成功调用次数。"""
        with _open_session(self._db_session_factory) as session:
            if session is None:
                return 0
            today = date.today().isoformat()
            result = session.execute(
                text("SELECT COUNT(*) FROM vlm_responses WHERE status = 'success' AND created_at >= :today"),
                {"today": today},
            )
            row = result.fetchone()
            return row[0] if row else 0

    def _query_monthly_cost(self) -> float:
        """从 vlm_responses 查询本月累计成本（SUM of cost_usd）。"""
        with _open_session(self._db_session_factory) as session:
            if session is None:
                return 0.0
            month_start = date.today().replace(day=1).isoformat()
            result = session.execute(
                text("SELECT COALESCE(SUM(cost_usd), 0.0) FROM vlm_responses WHERE status = 'success' AND created_at >= :month_start"),
                {"month_start": month_start},
            )
            row = result.fetchone()
            return float(row[0]) if row else 0.0


# ---------------------------------------------------------------------------
# Semantic Model Cache
# ---------------------------------------------------------------------------


class SemanticModelCache:
    """长期缓存，版本变化才失效。"""

    def __init__(self) -> None:
        self._cache: dict[str, PageSemanticModel] = {}

    def compute_key(
        self,
        screenshot_hash: str,
        candidate_hash: str,
        prompt_version: str,
        schema_version: str,
        provider_model_id: str,
        task_hint_hash: str,
    ) -> str:
        raw = f"{screenshot_hash}|{candidate_hash}|{prompt_version}|{schema_version}|{provider_model_id}|{task_hint_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> PageSemanticModel | None:
        return self._cache.get(cache_key)

    def put(self, cache_key: str, model: PageSemanticModel) -> None:
        self._cache[cache_key] = model

    def invalidate(self, cache_key: str) -> None:
        self._cache.pop(cache_key, None)

    def clear(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# JSON Parse Enhancement
# ---------------------------------------------------------------------------


def _parse_raw_response(raw_text: str) -> dict[str, Any]:
    if not raw_text:
        raise ValueError("empty raw_text")

    text_str = raw_text.strip()

    md_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", text_str, re.DOTALL)
    if md_match:
        text_str = md_match.group(1).strip()

    try:
        return json.loads(text_str)
    except json.JSONDecodeError:
        pass

    start = text_str.find("{")
    end = text_str.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text_str[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    if start != -1:
        candidate = text_str[start:]
        open_brackets = candidate.count("{") - candidate.count("}")
        open_squares = candidate.count("[") - candidate.count("]")
        if open_brackets > 0 or open_squares > 0:
            candidate += "]" * open_squares + "}" * open_brackets
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"cannot parse JSON from raw_text (length={len(raw_text)})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_image(img: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _hash_candidates(candidates: list[dict[str, Any]] | None) -> str:
    if not candidates:
        return "none"
    raw = json.dumps(candidates, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _hash_text(text_val: str) -> str:
    if not text_val:
        return "none"
    return hashlib.sha256(text_val.encode("utf-8")).hexdigest()


def _extract_candidate_ids(omni_candidates: list[dict[str, Any]] | None) -> set[str]:
    if not omni_candidates:
        return set()
    return {
        str(c.get("id") or c.get("element_id"))
        for c in omni_candidates
        if c.get("id") or c.get("element_id")
    }


# ---------------------------------------------------------------------------
# Main Class
# ---------------------------------------------------------------------------


class VLMSemanticModeler:
    """VLM Semantic Modeler 核心编排。"""

    def __init__(
        self,
        primary_provider: VLMSemanticProvider,
        fallback_provider: VLMSemanticProvider | None = None,
        cache: SemanticModelCache | None = None,
        budget_tracker: BudgetTracker | None = None,
        config: SemanticModelerConfig | None = None,
        recorder: ResponseRecorder | None = None,
    ) -> None:
        self._primary = primary_provider
        self._fallback = fallback_provider
        self._cache = cache or SemanticModelCache()
        self._config = config or SemanticModelerConfig()
        self._budget = budget_tracker or BudgetTracker(self._config)
        self._recorder = recorder or ResponseRecorder()

    def analyze(
        self,
        prompt_input: PromptInput,
        *,
        force: bool = False,
    ) -> VLMSemanticModelerResult:
        """一次性页面理解请求。"""
        screenshot_hash = _hash_image(prompt_input.screenshot)
        candidate_hash = _hash_candidates(prompt_input.omni_candidates)
        task_hint_hash = _hash_text(prompt_input.task_hint)

        # 检查缓存（用 primary 的 model_id）
        cache_key = self._compute_cache_key(
            self._primary, screenshot_hash, candidate_hash, task_hint_hash,
        )
        if not force:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return VLMSemanticModelerResult(
                    status="cached",
                    model=cached,
                    from_cache=True,
                    provider_name=self._primary.name,
                    model_name=self._primary.model_id,
                )

        # 检查预算
        allowed, reason = self._budget.check_budget(self._primary)
        if not allowed:
            logger.warning("Budget check failed for primary: %s", reason)
            if self._fallback is not None:
                return self._try_fallback(
                    self._fallback, prompt_input, screenshot_hash, candidate_hash, task_hint_hash,
                    error_prefix=f"budget_exceeded: primary={reason}",
                )
            result = VLMSemanticModelerResult(
                status="failed",
                error=f"budget_exceeded: {reason}",
                error_code="budget_exceeded",
            )
            self._record_result(result, cache_key, screenshot_hash, candidate_hash)
            return result

        # 检查 primary 可用性
        if not self._primary.is_available():
            if self._fallback is not None:
                return self._try_fallback(
                    self._fallback, prompt_input, screenshot_hash, candidate_hash, task_hint_hash,
                    error_prefix="primary provider not available",
                )
            result = VLMSemanticModelerResult(
                status="failed",
                error="primary provider not available",
                error_code="provider_unavailable",
            )
            self._record_result(result, cache_key, screenshot_hash, candidate_hash)
            return result

        # 调用 primary
        result = self._call_provider(self._primary, prompt_input, cache_key, screenshot_hash, candidate_hash)
        if result.status == "failed" and self._fallback is not None:
            logger.info("Primary failed, trying fallback")
            return self._try_fallback(
                self._fallback, prompt_input, screenshot_hash, candidate_hash, task_hint_hash,
                error_prefix="primary call failed",
            )

        return result

    def _try_fallback(
        self,
        fallback: VLMSemanticProvider,
        prompt_input: PromptInput,
        screenshot_hash: str,
        candidate_hash: str,
        task_hint_hash: str,
        *,
        error_prefix: str = "",
    ) -> VLMSemanticModelerResult:
        """尝试 fallback：先查缓存，再检查预算，最后调用。"""
        fb_key = self._compute_cache_key(fallback, screenshot_hash, candidate_hash, task_hint_hash)

        # 先查 fallback 缓存
        cached = self._cache.get(fb_key)
        if cached is not None:
            return VLMSemanticModelerResult(
                status="cached",
                model=cached,
                from_cache=True,
                provider_name=fallback.name,
                model_name=fallback.model_id,
            )

        # 检查 fallback 预算
        allowed_fb, reason_fb = self._budget.check_budget(fallback)
        if not allowed_fb:
            result = VLMSemanticModelerResult(
                status="failed",
                error=f"{error_prefix}; fallback budget: {reason_fb}" if error_prefix else f"fallback budget: {reason_fb}",
                error_code="budget_exceeded",
            )
            self._record_result(result, fb_key, screenshot_hash, candidate_hash)
            return result

        return self._call_provider(fallback, prompt_input, fb_key, screenshot_hash, candidate_hash)

    def should_analyze(
        self,
        *,
        force: bool = False,
        drift_detected: bool = False,
        permanence_state: str | None = None,
        observe_count: int = 0,
        fixed_anchor_skip_interval: int = 3,
    ) -> tuple[bool, str]:
        """判断是否值得尝试调用 VLM（粗筛）。

        不做缓存检查（analyze 内部会做）。
        只检查：enabled / provider 可用性 / 预算。
        force=True 或 drift_detected=True 时直接允许。
        fixed_anchor + 无 drift + 非 force → 每 N 次 observe 才调用（N 默认 3）。

        返回 (allowed, reason)。
        """
        if not self._config.enabled:
            return False, "disabled"

        if force or drift_detected:
            return True, "forced" if force else "drift_detected"

        # Phase 6: fixed_anchor reduced frequency
        if permanence_state == "fixed_anchor" and observe_count > 0:
            if observe_count % fixed_anchor_skip_interval != 0:
                return False, "fixed_anchor_reduced_frequency"

        primary_ok = self._primary.is_available()
        primary_budget_ok, _ = self._budget.check_budget(self._primary)

        if not primary_ok or not primary_budget_ok:
            if self._fallback is not None:
                fb_ok = self._fallback.is_available()
                fb_budget_ok, _ = self._budget.check_budget(self._fallback)
                if fb_ok and fb_budget_ok:
                    return True, "fallback_available"
            return False, "provider_unavailable" if not primary_ok else "budget_exceeded"

        return True, "ok"

    def _compute_cache_key(
        self,
        provider: VLMSemanticProvider,
        screenshot_hash: str,
        candidate_hash: str,
        task_hint_hash: str,
    ) -> str:
        return self._cache.compute_key(
            screenshot_hash=screenshot_hash,
            candidate_hash=candidate_hash,
            prompt_version=self._config.prompt_version,
            schema_version=CURRENT_SCHEMA_VERSION,
            provider_model_id=provider.model_id,
            task_hint_hash=task_hint_hash,
        )

    def _call_provider(
        self,
        provider: VLMSemanticProvider,
        prompt_input: PromptInput,
        cache_key: str,
        screenshot_hash: str,
        candidate_hash: str,
    ) -> VLMSemanticModelerResult:
        original_screenshot = prompt_input.screenshot
        prompt_mode = getattr(prompt_input, "task_mode", "full_page_recognition")
        prompt_input, coordinate_meta = prepare_vlm_prompt_input(
            prompt_input,
            max_width=self._config.image_max_width,
            prompt_mode=prompt_mode,
        )
        request = build_page_understanding_prompt(
            prompt_input,
            max_tokens=provider.capabilities.max_tokens_limit,
            provider_options={"timeout": self._config.timeout_seconds},
        )

        trace_dir = create_vlm_trace_dir(provider.name, provider.model_id)

        if trace_dir:
            request.provider_options["_trace_dir"] = str(trace_dir)

            original_image_meta = save_image(
                trace_dir,
                "02_original_screenshot.png",
                original_screenshot,
            )

            sent_image_meta = save_image(
                trace_dir,
                "03_vlm_sent_image.png",
                prompt_input.screenshot,
            )

            meta_json = {
                "provider": provider.name,
                "model": provider.model_id,
                "trace_dir": str(trace_dir),
                "screenshot_size": [original_screenshot.width, original_screenshot.height],
                "original_image": original_image_meta,
                "vlm_sent_image": sent_image_meta,
                "cache_key": cache_key,
                "screenshot_hash": screenshot_hash,
                "candidate_hash": candidate_hash,
                "prompt_version": self._config.prompt_version,
                "schema_version": CURRENT_SCHEMA_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            meta_json.update(coordinate_meta.to_dict())
            write_json(trace_dir, "00_meta.json", meta_json)

            write_json(trace_dir, "01_modeler_request.json", {
                "system_prompt": request.system_prompt,
                "messages": request.messages,
                "max_tokens": request.max_tokens,
                "provider_options": {
                    k: v for k, v in request.provider_options.items()
                    if k != "_trace_dir"
                },
                "screenshot_size": [prompt_input.screenshot.width, prompt_input.screenshot.height],
                "coordinate_space": coordinate_meta.coordinate_space,
                "task_hint": prompt_input.task_hint,
                "omni_candidates": prompt_input.omni_candidates,
                "ocr_texts": prompt_input.ocr_texts,
                "uia_elements": prompt_input.uia_elements,
                "current_page_model": prompt_input.current_page_model,
            })

        # 重试逻辑：超时/连接错误等瞬态故障自动重试
        max_attempts = max(1, self._config.max_retries + 1)
        raw_response = None
        for attempt in range(max_attempts):
            raw_response = provider.analyze_page(request)
            if raw_response.finish_reason != "error":
                break
            # 只对瞬态错误重试（超时、连接错误）
            error_lower = (raw_response.error or "").lower()
            is_transient = any(kw in error_lower for kw in (
                "timeout", "timed out", "connection", "read timed",
                "connect timed", "temporary", "retry",
            ))
            if not is_transient or attempt == max_attempts - 1:
                break
            logger.info(
                "VLM provider transient error (attempt %d/%d): %s",
                attempt + 1, max_attempts, raw_response.error,
            )

        if raw_response.finish_reason == "error":
            if trace_dir:
                write_json(trace_dir, "09_result_summary.json", {
                    "status": "failed",
                    "error_code": "provider_error",
                    "error": raw_response.error or "provider call failed",
                    "provider": provider.name,
                    "model": provider.model_id,
                    "token_input": raw_response.token_input,
                    "token_output": raw_response.token_output,
                    "latency_ms": raw_response.latency_ms,
                })
            result = VLMSemanticModelerResult(
                status="failed",
                error=raw_response.error or "provider call failed",
                error_code="provider_error",
                provider_name=provider.name,
                model_name=provider.model_id,
                token_input=raw_response.token_input,
                token_output=raw_response.token_output,
                latency_ms=raw_response.latency_ms,
            )
            self._record_result(result, cache_key, screenshot_hash, candidate_hash)
            return result

        # 记录用量，获取 cost_usd
        cost_usd = self._budget.record_usage(
            provider, raw_response.token_input, raw_response.token_output,
        )

        # 解析 JSON（先保存原始 copy，再传给 validator）
        try:
            raw_json = _parse_raw_response(raw_response.raw_text)
        except ValueError as exc:
            logger.warning("JSON parse failed: %s", exc)
            if trace_dir:
                write_text(trace_dir, "06_assistant_raw_text.txt", raw_response.raw_text)
                write_json(trace_dir, "09_result_summary.json", {
                    "status": "failed",
                    "error_code": "parse_error",
                    "error": str(exc),
                    "provider": provider.name,
                    "model": provider.model_id,
                    "token_input": raw_response.token_input,
                    "token_output": raw_response.token_output,
                    "latency_ms": raw_response.latency_ms,
                })
            result = VLMSemanticModelerResult(
                status="failed",
                error=f"parse_error: {exc}",
                error_code="parse_error",
                provider_name=provider.name,
                model_name=provider.model_id,
                token_input=raw_response.token_input,
                token_output=raw_response.token_output,
                latency_ms=raw_response.latency_ms,
            )
            self._record_result(result, cache_key, screenshot_hash, candidate_hash, cost_usd=cost_usd)
            return result

        raw_json_original = copy.deepcopy(raw_json)

        if trace_dir:
            write_text(trace_dir, "06_assistant_raw_text.txt", raw_response.raw_text)
            write_json(trace_dir, "07_parsed_raw_original.json", raw_json_original)
            write_json(trace_dir, "07_parsed_raw_original_vlm_coords.json", raw_json_original)

        # 校验 Schema in the coordinate space actually sent to the VLM.
        known_ids = _extract_candidate_ids(prompt_input.omni_candidates)
        try:
            model, warnings = validate_page_semantic_model(
                raw_json,
                image_size=(prompt_input.screenshot.width, prompt_input.screenshot.height),
                known_candidate_ids=known_ids if known_ids else None,
            )
        except ValidationError as exc:
            logger.warning("Schema validation failed: %s", exc)
            if trace_dir:
                write_json(trace_dir, "09_result_summary.json", {
                    "status": "failed",
                    "error_code": "schema_validation_failed",
                    "error": str(exc),
                    "provider": provider.name,
                    "model": provider.model_id,
                    "token_input": raw_response.token_input,
                    "token_output": raw_response.token_output,
                    "latency_ms": raw_response.latency_ms,
                    "parsed_raw_original_file": "07_parsed_raw_original.json",
                })
            result = VLMSemanticModelerResult(
                status="failed",
                error=f"schema_validation_failed: {exc}",
                error_code="schema_validation_failed",
                provider_name=provider.name,
                model_name=provider.model_id,
                token_input=raw_response.token_input,
                token_output=raw_response.token_output,
                latency_ms=raw_response.latency_ms,
            )
            self._record_result(result, cache_key, screenshot_hash, candidate_hash, cost_usd=cost_usd)
            return result

        normalized_model = normalize_model_to_original(model, coordinate_meta)

        # 写入缓存
        self._cache.put(cache_key, normalized_model)

        # 脱敏
        raw_response_redacted = None
        if self._config.save_raw_response:
            raw_response_redacted = redact_raw_response(
                raw_response.raw_text,
                redact_dynamic_content=self._config.redact_dynamic_content,
            )

        result = VLMSemanticModelerResult(
            status="success",
            model=normalized_model,
            from_cache=False,
            provider_name=provider.name,
            model_name=provider.model_id,
            token_input=raw_response.token_input,
            token_output=raw_response.token_output,
            latency_ms=raw_response.latency_ms,
            raw_response_redacted=raw_response_redacted,
            warnings=warnings,
        )

        self._record_result(result, cache_key, screenshot_hash, candidate_hash, cost_usd=cost_usd)

        if trace_dir:
            write_json(trace_dir, "08_validated_or_clamped_model.json", model.to_dict())
            write_json(trace_dir, "08_normalized_original_coords.json", normalized_model.to_dict())
            write_json(trace_dir, "09_result_summary.json", {
                "status": "success",
                "provider": provider.name,
                "model": provider.model_id,
                "token_input": raw_response.token_input,
                "token_output": raw_response.token_output,
                "latency_ms": raw_response.latency_ms,
                "warnings": warnings,
                "image_size": [prompt_input.screenshot.width, prompt_input.screenshot.height],
                "original_size": list(coordinate_meta.original_size),
                "vlm_size": list(coordinate_meta.vlm_size),
                "coordinate_space": coordinate_meta.coordinate_space,
                "canonical_coordinate_space": coordinate_meta.canonical_coordinate_space,
                "raw_text_file": "06_assistant_raw_text.txt",
                "parsed_raw_original_file": "07_parsed_raw_original.json",
                "parsed_raw_vlm_coords_file": "07_parsed_raw_original_vlm_coords.json",
                "validated_model_file": "08_validated_or_clamped_model.json",
                "normalized_model_file": "08_normalized_original_coords.json",
            })
        return result

    def _record_result(
        self,
        result: VLMSemanticModelerResult,
        cache_key: str,
        screenshot_hash: str,
        candidate_hash: str,
        cost_usd: float = 0.0,
    ) -> None:
        parsed_model_json = ""
        if result.model:
            parsed_model_json = json.dumps(
                sanitize_for_storage(result.model), ensure_ascii=False
            )

        self._recorder.record(
            cache_key=cache_key,
            screenshot_hash=screenshot_hash,
            candidate_hash=candidate_hash,
            prompt_version=self._config.prompt_version,
            schema_version=CURRENT_SCHEMA_VERSION,
            provider_name=result.provider_name,
            model_name=result.model_name,
            raw_response_redacted=result.raw_response_redacted,
            parsed_model_json=parsed_model_json,
            token_input=result.token_input,
            token_output=result.token_output,
            cost_usd=cost_usd,
            latency_ms=result.latency_ms,
            status=result.status,
        )
