"""
数据库 Schema 定义
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON,
    Index, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import relationship

from src.storage.db import Base


class App(Base):
    """软件表"""
    __tablename__ = "apps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    canonical_name = Column(String(255), nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    publisher = Column(String(255), nullable=True)
    install_location = Column(String(512), nullable=True)
    icon_path = Column(String(512), nullable=True)
    version = Column(String(64), nullable=True)
    is_uwp = Column(Boolean, default=False)
    confidence = Column(Float, default=0.5)
    last_seen_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # 关系
    aliases = relationship("AppAlias", back_populates="app", cascade="all, delete-orphan")
    launch_targets = relationship("LaunchTarget", back_populates="app", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<App(id={self.id}, name={self.display_name}, canonical={self.canonical_name})>"


class AppAlias(Base):
    """软件别名表"""
    __tablename__ = "app_aliases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    app_id = Column(Integer, ForeignKey("apps.id"), nullable=False)
    alias = Column(String(255), nullable=False, index=True)
    source = Column(String(32), default="common")  # common, user
    weight = Column(Float, default=1.0)

    # 关系
    app = relationship("App", back_populates="aliases")

    __table_args__ = (
        Index("idx_alias_app", "alias", "app_id"),
    )

    def __repr__(self):
        return f"<AppAlias(id={self.id}, alias={self.alias}, app_id={self.app_id})>"


class LaunchTarget(Base):
    """启动入口表"""
    __tablename__ = "app_launch_targets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    app_id = Column(Integer, ForeignKey("apps.id"), nullable=False)
    path = Column(String(1024), nullable=False)
    args = Column(String(512), nullable=True)
    target_type = Column(String(32), default="exe")  # exe, uwp, shortcut, url
    score = Column(Float, default=0.0)
    source = Column(String(32), nullable=False)  # registry, startmenu, desktop, filesystem

    # 关系
    app = relationship("App", back_populates="launch_targets")

    __table_args__ = (
        Index("idx_launch_app", "app_id"),
        Index("idx_launch_path", "path"),
    )

    def __repr__(self):
        return f"<LaunchTarget(id={self.id}, path={self.path}, score={self.score})>"


class PageTemplate(Base):
    """页面模板表"""
    __tablename__ = "page_templates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    app_id = Column(String(128), nullable=False, index=True)
    page_type = Column(String(64), nullable=False)
    fingerprint_hash = Column(String(64), nullable=True)
    screenshot_path = Column(String(1024), nullable=True)
    structure_json = Column(JSON, nullable=True)
    confidence = Column(Float, default=0.5)
    version = Column(String(32), default="1.0")
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_template_app_page", "app_id", "page_type"),
    )

    def __repr__(self):
        return f"<PageTemplate(id={self.id}, app={self.app_id}, type={self.page_type})>"


class Candidate(Base):
    """页面元素表"""
    __tablename__ = "page_elements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    template_id = Column(Integer, ForeignKey("page_templates.id"), nullable=False)
    element_key = Column(String(128), nullable=False)
    role = Column(String(64), nullable=False)
    rect_json = Column(String(256), nullable=True)  # {"left":0,"top":0,"right":100,"bottom":50}
    locator_json = Column(JSON, nullable=True)
    confidence = Column(Float, default=0.5)

    __table_args__ = (
        Index("idx_element_template", "template_id"),
    )

    def __repr__(self):
        return f"<Candidate(id={self.id}, key={self.element_key}, role={self.role})>"


class ActionRecipe(Base):
    """动作配方表"""
    __tablename__ = "action_recipes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    app_id = Column(String(128), nullable=False, index=True)
    page_type = Column(String(64), nullable=False)
    action_name = Column(String(128), nullable=False)
    steps_json = Column(JSON, nullable=False)
    success_rule_json = Column(JSON, nullable=True)
    fallback_rule_json = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_recipe_app_action", "app_id", "action_name"),
    )

    def __repr__(self):
        return f"<ActionRecipe(id={self.id}, app={self.app_id}, action={self.action_name})>"


class ExecutionLog(Base):
    """执行日志表"""
    __tablename__ = "execution_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), nullable=True)
    app_id = Column(String(128), nullable=True)
    page_type = Column(String(64), nullable=True)
    action_name = Column(String(128), nullable=True)
    status = Column(String(32), nullable=False)  # success, failed, cancelled
    screenshot_before = Column(String(1024), nullable=True)
    screenshot_after = Column(String(1024), nullable=True)
    detail_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_log_task", "task_id"),
        Index("idx_log_app", "app_id"),
        Index("idx_log_created", "created_at"),
    )

    def __repr__(self):
        return f"<ExecutionLog(id={self.id}, action={self.action_name}, status={self.status})>"


class DriftEvent(Base):
    """漂移事件表"""
    __tablename__ = "drift_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    template_id = Column(Integer, ForeignKey("page_templates.id"), nullable=False)
    old_fingerprint = Column(String(64), nullable=True)
    new_fingerprint = Column(String(64), nullable=True)
    resolution = Column(String(32), nullable=True)  # auto_fixed, manual_fixed, ignored
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<DriftEvent(id={self.id}, template_id={self.template_id}, resolution={self.resolution})>"


class InteractionCanvasRecord(Base):
    """画布缓存表"""
    __tablename__ = "interaction_canvases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    canvas_id = Column(String(64), unique=True, nullable=False, index=True)
    hwnd = Column(Integer, nullable=True)
    app_id = Column(String(128), nullable=True, index=True)
    page_class = Column(String(256), nullable=True)
    surface_type = Column(String(64), nullable=True)
    stable = Column(Boolean, default=True)
    partial = Column(Boolean, default=False)
    confidence = Column(Float, default=0.0)
    canvas_json = Column(Text, nullable=False)
    captured_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_canvas_app", "app_id"),
        Index("idx_canvas_captured", "captured_at"),
    )

    def __repr__(self):
        return f"<InteractionCanvasRecord(id={self.id}, canvas_id={self.canvas_id})>"


class CandidateHistory(Base):
    """候选历史表"""
    __tablename__ = "candidate_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    canvas_id = Column(String(64), nullable=True)
    candidate_key = Column(String(256), nullable=False, index=True)
    semantic_role = Column(String(64), nullable=True)
    confidence = Column(Float, default=0.0)
    source_json = Column(JSON, nullable=True)
    last_seen_at = Column(DateTime, nullable=True)
    use_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)

    __table_args__ = (
        UniqueConstraint("candidate_key", name="uq_candidate_key"),
        Index("idx_candidate_role", "semantic_role"),
    )

    def __repr__(self):
        return f"<CandidateHistory(id={self.id}, key={self.candidate_key})>"


class TransitionEdge(Base):
    """历史状态变化图表"""
    __tablename__ = "transition_edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    from_page_class = Column(String(256), nullable=False)
    to_page_class = Column(String(256), nullable=False)
    trigger_action = Column(String(128), nullable=True)
    trigger_candidate_key = Column(String(256), nullable=True)
    observe_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    last_observed_at = Column(DateTime, nullable=True)
    drifted = Column(Boolean, default=False)
    drift_detected_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("from_page_class", "to_page_class", "trigger_action", name="uq_transition"),
        Index("idx_transition_from", "from_page_class"),
    )

    def __repr__(self):
        return f"<TransitionEdge(id={self.id}, {self.from_page_class}->{self.to_page_class})>"


class ControlStateTransition(Base):
    """控件-状态跳转关联表。

    与 transition_edges 的页面级边不同，这张表明确记录“哪个候选控件”
    通过哪种动作把页面从 A 状态带到 B 状态。
    """
    __tablename__ = "control_state_transitions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    candidate_key = Column(String(256), nullable=False, index=True)
    stable_key_id = Column(String(36), nullable=True, index=True)
    from_page_class = Column(String(256), nullable=False, index=True)
    to_page_class = Column(String(256), nullable=False, index=True)
    action_type = Column(String(64), nullable=True)
    canvas_id_before = Column(String(64), nullable=True)
    canvas_id_after = Column(String(64), nullable=True)
    observe_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    last_observed_at = Column(DateTime, nullable=True)
    metadata_json = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("candidate_key", "from_page_class", "to_page_class", "action_type", name="uq_control_transition"),
        Index("idx_control_transition_from", "from_page_class", "candidate_key"),
    )

    def __repr__(self):
        return f"<ControlStateTransition(key={self.candidate_key}, {self.from_page_class}->{self.to_page_class})>"


class FeedbackRecord(Base):
    """Agent 反馈表"""
    __tablename__ = "feedback_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    canvas_id = Column(String(64), nullable=True)
    candidate_key = Column(String(256), nullable=True, index=True)
    feedback_type = Column(String(32), nullable=False)  # success / failure / unclear
    detail_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_feedback_candidate", "candidate_key"),
        Index("idx_feedback_created", "created_at"),
    )

    def __repr__(self):
        return f"<FeedbackRecord(id={self.id}, type={self.feedback_type})>"


class StableCandidateKey(Base):
    """持久候选身份表"""
    __tablename__ = "stable_candidate_keys"

    key_id = Column(String(36), primary_key=True)  # UUID
    app_id = Column(String(128), nullable=False, index=True)
    page_class = Column(String(256), nullable=False)
    canonical_region = Column(String(64), nullable=False)
    canonical_text = Column(String(512), nullable=False)  # 固定控件存原文，动态内容存 text_hash
    canonical_role = Column(String(64), nullable=False)
    created_at = Column(String(32), nullable=False)
    last_seen_at = Column(String(32), nullable=False)
    verify_count = Column(Integer, default=1)
    latest_bounds = Column(Text, nullable=True)  # JSON
    latest_crop_hash = Column(String(64), nullable=True)
    # Phase 2 预留
    layout_signature = Column(String(128), nullable=True)
    state_signature = Column(String(128), nullable=True)
    template_scope = Column(String(16), default="local")  # local / global
    # Phase 6: permanence state machine
    permanence_state = Column(String(16), default="new")
    state_changed_at = Column(String(32), nullable=True)
    fail_count = Column(Integer, default=0)
    coordinate_drift = Column(Float, default=0.0)
    latest_profile_id = Column(String(36), nullable=True)

    __table_args__ = (
        Index("idx_sck_app_page", "app_id", "page_class"),
    )

    def __repr__(self):
        return f"<StableCandidateKey(key_id={self.key_id}, app={self.app_id})>"


class CandidateTemplate(Base):
    """候选模板表 — 每个 StableCandidateKey 的观测快照"""
    __tablename__ = "candidate_templates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key_id = Column(String(36), ForeignKey("stable_candidate_keys.key_id"), nullable=False, index=True)
    app_id = Column(String(128), nullable=False)
    page_class = Column(String(256), nullable=False)
    region_id = Column(String(64), nullable=False)
    region_role = Column(String(64), nullable=False)
    relative_bounds = Column(Text, nullable=False)  # JSON float 4-tuple
    size_ratio = Column(Float, nullable=False)
    text_normalized = Column(String(512), nullable=False)  # 固定控件存原文，动态内容存 text_hash
    text_hash = Column(String(64), nullable=True)  # 动态内容的 SHA-256 hash
    visual_type = Column(String(32), nullable=False)
    role_label = Column(String(128), nullable=True)
    semantic_tags = Column(Text, nullable=True)  # JSON array
    crop_hash = Column(String(64), nullable=True)
    provider_sources = Column(Text, nullable=True)  # JSON array
    confidence = Column(Float, nullable=False)
    is_fixed_control = Column(Integer, nullable=False)  # 0=动态, 1=固定
    updated_at = Column(String(32), nullable=False)

    __table_args__ = (
        Index("idx_ct_app_page", "app_id", "page_class"),
        Index("idx_ct_key", "key_id"),
    )

    def __repr__(self):
        return f"<CandidateTemplate(id={self.id}, key_id={self.key_id})>"


# =============================================================================
# E Phase 2: 软件虚拟模型
# =============================================================================


class PageModelRecord(Base):
    """页面模型表 — 识别同一软件的同一逻辑页面"""
    __tablename__ = "page_models"

    page_model_id = Column(String(36), primary_key=True)  # UUID
    app_id = Column(String(128), nullable=False, index=True)
    page_class_prefix = Column(String(256), nullable=False, index=True)
    display_name = Column(String(256), nullable=True)
    surface_type = Column(String(64), nullable=True)
    layout_signature = Column(String(128), nullable=True)
    created_at = Column(String(32), nullable=False)
    last_seen_at = Column(String(32), nullable=False)
    observe_count = Column(Integer, default=1)
    state_count = Column(Integer, default=0)

    def __repr__(self):
        return f"<PageModel(id={self.page_model_id}, app={self.app_id})>"


class StateTemplateRecord(Base):
    """页面状态模板表 — 同一页面的同一可操作状态"""
    __tablename__ = "state_templates"

    state_template_id = Column(String(36), primary_key=True)  # UUID
    page_model_id = Column(String(36), ForeignKey("page_models.page_model_id"), nullable=False, index=True)
    app_id = Column(String(128), nullable=False, index=True)
    page_class = Column(String(256), nullable=False)
    state_signature = Column(String(64), nullable=False, index=True)
    layout_fingerprint = Column(Text, nullable=False)  # JSON
    state_label = Column(String(128), nullable=True)
    fixed_element_count = Column(Integer, default=0)
    total_element_count = Column(Integer, default=0)
    created_at = Column(String(32), nullable=False)
    last_seen_at = Column(String(32), nullable=False)
    verify_count = Column(Integer, default=1)
    snapshot_count = Column(Integer, default=0)
    max_snapshots = Column(Integer, default=10)

    def __repr__(self):
        return f"<StateTemplate(id={self.state_template_id}, pm={self.page_model_id})>"


class CanvasSnapshotRecord(Base):
    """画布快照归属表 — 每次 observe 归属到 StateTemplate"""
    __tablename__ = "canvas_snapshots"

    snapshot_id = Column(String(36), primary_key=True)  # UUID
    canvas_id = Column(String(64), nullable=False, unique=True, index=True)
    page_model_id = Column(String(36), ForeignKey("page_models.page_model_id"), nullable=True, index=True)
    state_template_id = Column(String(36), ForeignKey("state_templates.state_template_id"), nullable=True, index=True)
    captured_at = Column(String(32), nullable=False)
    element_count = Column(Integer, default=0)
    has_screenshot = Column(Integer, default=0)
    canvas_json = Column(Text, nullable=True)

    def __repr__(self):
        return f"<CanvasSnapshot(id={self.snapshot_id}, canvas={self.canvas_id})>"


class CanvasProcessingRecord(Base):
    """Canvas 后台增强状态表 — 支持服务重启后恢复 processing/job 状态。"""
    __tablename__ = "canvas_processing_states"

    canvas_id = Column(String(64), primary_key=True)
    processing_state = Column(String(32), default="local_ready", nullable=False)
    jobs_json = Column(Text, nullable=True)  # JSON array of job_id
    last_error = Column(Text, nullable=True)
    updated_at = Column(String(32), nullable=False)

    __table_args__ = (
        Index("idx_canvas_processing_state", "processing_state", "updated_at"),
    )

    def __repr__(self):
        return f"<CanvasProcessing(canvas={self.canvas_id}, state={self.processing_state})>"


class ProcessingJobRecord(Base):
    """后台任务状态表 — Phase 1 使用内存执行器，状态持久化到 SQLite。"""
    __tablename__ = "processing_jobs"

    job_id = Column(String(36), primary_key=True)
    canvas_id = Column(String(64), nullable=False, index=True)
    job_type = Column(String(64), nullable=False)
    state = Column(String(32), default="queued", nullable=False)
    error = Column(Text, nullable=True)
    started_at = Column(String(32), nullable=False)
    updated_at = Column(String(32), nullable=False)

    __table_args__ = (
        Index("idx_processing_job_canvas_type", "canvas_id", "job_type", "state"),
        Index("idx_processing_job_state", "state", "updated_at"),
    )

    def __repr__(self):
        return f"<ProcessingJob(id={self.job_id}, canvas={self.canvas_id}, state={self.state})>"


class AppShellTemplateRecord(Base):
    """应用外壳模板快照 — 由当前虚拟模型摘要派生，可后续升级为人工维护。"""
    __tablename__ = "app_shell_templates"

    shell_id = Column(String(36), primary_key=True)
    app_id = Column(String(128), nullable=False, index=True)
    page_model_id = Column(String(36), nullable=True, index=True)
    state_template_id = Column(String(36), nullable=True, index=True)
    display_name = Column(String(256), nullable=True)
    surface_type = Column(String(64), nullable=True)
    page_class = Column(String(256), nullable=True)
    state_label = Column(String(128), nullable=True)
    shell_json = Column(Text, nullable=False)
    source = Column(String(32), default="derived")
    updated_at = Column(String(32), nullable=False)

    __table_args__ = (
        UniqueConstraint("app_id", "state_template_id", name="uq_app_shell_state"),
    )


class RegionTemplateRecord(Base):
    """区域模板快照。"""
    __tablename__ = "region_templates"

    region_template_id = Column(String(36), primary_key=True)
    app_id = Column(String(128), nullable=False, index=True)
    page_model_id = Column(String(36), nullable=True, index=True)
    state_template_id = Column(String(36), nullable=True, index=True)
    region_id = Column(String(128), nullable=False)
    role = Column(String(128), nullable=True)
    purpose = Column(Text, nullable=True)
    bounds_json = Column(Text, nullable=True)
    candidate_count = Column(Integer, default=0)
    source = Column(String(32), default="derived")
    template_json = Column(Text, nullable=False)
    updated_at = Column(String(32), nullable=False)

    __table_args__ = (
        UniqueConstraint("state_template_id", "region_id", name="uq_region_template_state_region"),
        Index("idx_region_template_role", "app_id", "role"),
    )


class ElementTemplateRecord(Base):
    """元素模板快照。"""
    __tablename__ = "element_templates"

    element_template_id = Column(String(36), primary_key=True)
    app_id = Column(String(128), nullable=False, index=True)
    page_model_id = Column(String(36), nullable=True, index=True)
    state_template_id = Column(String(36), nullable=True, index=True)
    key_id = Column(String(256), nullable=False, index=True)
    role = Column(String(128), nullable=True)
    label = Column(String(512), nullable=True)
    region_id = Column(String(128), nullable=True)
    kind = Column(String(32), nullable=True)
    actionability = Column(String(32), default="review")
    confidence = Column(Float, default=0.0)
    source = Column(String(32), default="derived")
    template_json = Column(Text, nullable=False)
    updated_at = Column(String(32), nullable=False)

    __table_args__ = (
        UniqueConstraint("state_template_id", "key_id", name="uq_element_template_state_key"),
        Index("idx_element_template_region", "state_template_id", "region_id"),
    )


class CandidateStateRecord(Base):
    """候选-状态关联表 — 记录候选在哪些 StateTemplate 下出现过"""
    __tablename__ = "candidate_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key_id = Column(String(36), ForeignKey("stable_candidate_keys.key_id"), nullable=False, index=True)
    state_template_id = Column(String(36), ForeignKey("state_templates.state_template_id"), nullable=False, index=True)
    page_model_id = Column(String(36), ForeignKey("page_models.page_model_id"), nullable=False)
    first_seen_at = Column(String(32), nullable=False)
    last_seen_at = Column(String(32), nullable=False)
    seen_count = Column(Integer, default=1)

    __table_args__ = (
        UniqueConstraint("key_id", "state_template_id", name="uq_candidate_state"),
    )

    def __repr__(self):
        return f"<CandidateState(key={self.key_id}, st={self.state_template_id})>"


class CandidateOverrideRecord(Base):
    """候选人工修正表 — 只存有效覆盖层，不改原始识别结果"""
    __tablename__ = "candidate_overrides"

    override_id = Column(String(36), primary_key=True)  # UUID
    scope_key = Column(String(256), nullable=False, unique=True, index=True)
    app_id = Column(String(128), nullable=True, index=True)
    page_model_id = Column(String(36), ForeignKey("page_models.page_model_id"), nullable=True, index=True)
    state_template_id = Column(String(36), ForeignKey("state_templates.state_template_id"), nullable=True, index=True)
    canvas_id = Column(String(64), nullable=True, index=True)
    element_id = Column(String(128), nullable=True)
    stable_key_id = Column(String(36), ForeignKey("stable_candidate_keys.key_id"), nullable=True, index=True)
    label = Column(String(512), nullable=True)
    semantic_role = Column(String(128), nullable=True)
    visual_type = Column(String(64), nullable=True)
    region_id = Column(String(128), nullable=True)
    kind = Column(String(32), nullable=True)  # fixed / dynamic / container / ignored
    relative_bounds = Column(Text, nullable=True)  # JSON float 4-tuple
    absolute_bounds = Column(Text, nullable=True)  # JSON int 4-tuple
    source = Column(String(32), default="manual")
    status = Column(String(32), default="active")  # active / suppressed
    created_at = Column(String(32), nullable=False)
    updated_at = Column(String(32), nullable=False)

    __table_args__ = (
        Index("idx_override_context", "state_template_id", "canvas_id"),
    )

    def __repr__(self):
        return f"<CandidateOverride(scope={self.scope_key}, status={self.status})>"


# =============================================================================
# VLM Semantic Modeler
# =============================================================================


class VLMResponseRecord(Base):
    """VLM 响应记录表 — 缓存和预算追踪"""
    __tablename__ = "vlm_responses"

    response_id = Column(String(36), primary_key=True)  # UUID
    cache_key = Column(String(64), nullable=False, index=True)  # SHA-256
    screenshot_hash = Column(String(64), nullable=False)
    candidate_hash = Column(String(64), nullable=False)
    prompt_version = Column(String(16), nullable=False)
    schema_version = Column(String(16), nullable=False)
    provider_name = Column(String(32), nullable=False)
    model_name = Column(String(64), nullable=False)
    raw_response_redacted = Column(Text, nullable=True)
    parsed_model = Column(Text, nullable=False)  # JSON
    token_input = Column(Integer, default=0)
    token_output = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0, nullable=False)
    latency_ms = Column(Integer, default=0)
    status = Column(String(16), nullable=False)  # success / failed
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    page_model_id = Column(String(36), nullable=True)
    state_template_id = Column(String(36), nullable=True)

    __table_args__ = (
        Index("idx_vlm_cache", "cache_key"),
        Index("idx_vlm_status_date", "status", "created_at"),
    )

    def __repr__(self):
        return f"<VLMResponse(id={self.response_id}, provider={self.provider_name}, status={self.status})>"


# =============================================================================
# 多源证据融合 + 视觉资产持久化 + fixed_anchor 状态机
# =============================================================================


class VisualAssetRecord(Base):
    """视觉资产表 — 文件系统图片的元数据索引（不存 BLOB，只存 path + hash）"""
    __tablename__ = "visual_assets"

    asset_id = Column(String(36), primary_key=True)  # UUID
    asset_type = Column(String(32), nullable=False)  # page_keyframe | region_crop | control_crop | warm_snapshot
    app_id = Column(String(128), nullable=True)
    page_model_id = Column(String(36), nullable=True)
    state_template_id = Column(String(36), nullable=True)
    stable_key_id = Column(String(36), nullable=True, index=True)
    canvas_id = Column(String(64), nullable=True)
    region_id = Column(String(64), nullable=True)
    path = Column(String(1024), nullable=False)  # 相对路径
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    format = Column(String(8), nullable=False)  # jpeg | png | webp
    quality = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)
    bounds_json = Column(Text, nullable=True)  # [l,t,r,b] 窗口局部坐标
    relative_bounds_json = Column(Text, nullable=True)  # [x1,y1,x2,y2]
    window_size_json = Column(Text, nullable=True)  # {"w":1920,"h":1080}
    dpi_scale = Column(Float, nullable=True)
    screenshot_hash = Column(String(64), nullable=True)
    hash_version = Column(String(32), default="dhash_v1")
    dhash = Column(String(16), nullable=True)  # 64-bit hex (16 chars)
    phash = Column(String(16), nullable=True)
    source_provider = Column(String(32), nullable=True)
    parent_asset_id = Column(String(36), nullable=True)
    created_at = Column(String(32), nullable=False)
    last_seen_at = Column(String(32), nullable=True)
    version = Column(Integer, default=1)
    status = Column(String(16), default="active")  # active | stale | archived
    extra_metadata = Column(Text, nullable=True)  # JSON

    __table_args__ = (
        Index("idx_va_key", "stable_key_id", "asset_type"),
        Index("idx_va_page", "page_model_id", "state_template_id"),
        Index("idx_va_dhash", "dhash"),
        Index("idx_va_status", "status", "created_at"),
        Index("idx_va_region", "region_id"),
    )

    def __repr__(self):
        return f"<VisualAsset(id={self.asset_id}, type={self.asset_type}, status={self.status})>"


class VisualObservationRecord(Base):
    """视觉观察记录表 — 跨截图视觉对比结果"""
    __tablename__ = "visual_observations"

    observation_id = Column(String(36), primary_key=True)  # UUID
    asset_id_a = Column(String(36), nullable=False)  # 基准帧
    asset_id_b = Column(String(36), nullable=True)  # 当前帧（可 NULL）
    stable_key_id = Column(String(36), nullable=True)
    canvas_id_a = Column(String(64), nullable=True)
    canvas_id_b = Column(String(64), nullable=True)
    observed_canvas_id = Column(String(64), nullable=True)
    observed_bounds_json = Column(Text, nullable=True)  # [l,t,r,b]
    observed_crop_path = Column(String(1024), nullable=True)  # 临时 crop 路径
    dhash_distance = Column(Integer, nullable=True)
    ssim_score = Column(Float, nullable=True)
    template_score = Column(Float, nullable=True)
    coordinate_drift = Column(Float, nullable=True)
    match_status = Column(String(32), nullable=False)  # exact_match|strong_match|weak_match|no_match|drift_detected
    roi_bounds_json = Column(Text, nullable=True)
    compared_at = Column(String(32), nullable=False)
    method = Column(String(32), nullable=False)
    extra_metadata = Column(Text, nullable=True)  # JSON

    __table_args__ = (
        Index("idx_vo_stable", "stable_key_id", "compared_at"),
        Index("idx_vo_asset", "asset_id_a", "asset_id_b"),
        Index("idx_vo_status", "match_status"),
    )

    def __repr__(self):
        return f"<VisualObservation(id={self.observation_id}, status={self.match_status})>"


class CandidateEvidenceRecord(Base):
    """候选证据表 — 每次 source 对一个候选的观察证据（只追加，不修改）"""
    __tablename__ = "candidate_evidence"

    evidence_id = Column(String(36), primary_key=True)  # UUID
    canvas_id = Column(String(64), nullable=True)
    element_id = Column(String(128), nullable=True)
    stable_key_id = Column(String(36), nullable=True)
    provider = Column(String(32), nullable=False)  # uia|vlm|omni|ocr|dom|template|memory|feedback|manual|agent
    provider_element_id = Column(String(128), nullable=True)
    evidence_scope = Column(String(16), nullable=False)  # candidate|region|page|action
    evidence_event = Column(String(32), nullable=False)  # observe|feedback|override|vlm_semantic|memory_replay
    region_id = Column(String(64), nullable=True)
    page_model_id = Column(String(36), nullable=True)
    state_template_id = Column(String(36), nullable=True)
    bounds_json = Column(Text, nullable=True)
    relative_bounds_json = Column(Text, nullable=True)
    text = Column(Text, nullable=True)
    semantic_role = Column(String(64), nullable=True)
    control_type = Column(String(64), nullable=True)
    raw_confidence = Column(Float, nullable=True)
    spatial_score = Column(Float, nullable=True)
    semantic_score = Column(Float, nullable=True)
    text_score = Column(Float, nullable=True)
    structure_score = Column(Float, nullable=True)
    visual_score = Column(Float, nullable=True)
    action_score = Column(Float, nullable=True)
    memory_score = Column(Float, nullable=True)
    feedback_outcome = Column(String(16), nullable=True)  # success|fail|partial
    action_type = Column(String(32), nullable=True)  # click|type_text|focus|scroll
    error_code = Column(String(64), nullable=True)
    match_method = Column(String(32), nullable=True)
    match_confidence = Column(Float, nullable=True)
    raw_data_ref = Column(String(512), nullable=True)
    created_at = Column(String(32), nullable=False)
    extra_metadata = Column(Text, nullable=True)  # JSON

    __table_args__ = (
        Index("idx_ce_key", "stable_key_id"),
        Index("idx_ce_canvas", "canvas_id", "element_id"),
        Index("idx_ce_provider", "provider", "created_at"),
        Index("idx_ce_scope", "evidence_scope", "evidence_event"),
        Index("idx_ce_page", "page_model_id", "state_template_id"),
    )

    def __repr__(self):
        return f"<CandidateEvidence(id={self.evidence_id}, provider={self.provider}, scope={self.evidence_scope})>"


class CandidateConfidenceProfileRecord(Base):
    """候选置信度快照表 — 每次融合计算的不可变快照"""
    __tablename__ = "candidate_confidence_profiles"

    profile_id = Column(String(36), primary_key=True)  # UUID
    stable_key_id = Column(String(36), nullable=False)
    canvas_id = Column(String(64), nullable=True)
    page_model_id = Column(String(36), nullable=True)
    state_template_id = Column(String(36), nullable=True)
    spatial_confidence = Column(Float, default=0.0)
    semantic_confidence = Column(Float, default=0.0)
    text_confidence = Column(Float, default=0.0)
    structure_confidence = Column(Float, default=0.0)
    visual_anchor_confidence = Column(Float, default=0.0)
    memory_confidence = Column(Float, default=0.0)
    action_confidence = Column(Float, default=0.0)
    conflict_penalty = Column(Float, default=0.0)
    fused_confidence = Column(Float, nullable=False)
    formula_version = Column(String(32), nullable=False)
    source_count = Column(Integer, default=0)
    source_diversity = Column(Integer, default=0)
    evidence_ids = Column(Text, nullable=True)  # JSON array
    computed_at = Column(String(32), nullable=False)
    extra_metadata = Column(Text, nullable=True)  # JSON

    __table_args__ = (
        Index("idx_ccp_key", "stable_key_id", "computed_at"),
        Index("idx_ccp_canvas", "canvas_id"),
    )

    def __repr__(self):
        return f"<ConfidenceProfile(id={self.profile_id}, fused={self.fused_confidence})>"
