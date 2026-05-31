"""
统一记忆服务

整合页面模板、漂移检测、动作配方、过渡图、反馈管理、候选身份、页面模型的统一接口。
"""
from typing import Any

from src.memory.candidate_identity import FIXED_CONTROL_ROLES, build_signature
from src.memory.candidate_key_store import CandidateKeyStore
from src.memory.page_identity import (
    build_layout_fingerprint,
    build_layout_signature,
    classify_state_label,
    fingerprint_to_signature,
    normalize_page_class_for_memory,
)
from src.memory.page_model_store import PageModelStore
from src.memory.page_template_manager import PageTemplateManager
from src.memory.drift_detector import DriftDetector, DriftReport
from src.memory.action_recipe_manager import ActionRecipeManager, ActionRecipeData
from src.memory.transition_graph import TransitionGraphManager, TransitionInfo
from src.memory.feedback_manager import FeedbackManager, FeedbackStats


class MemoryService:
    """
    统一记忆服务

    使用流程：
    1. perceive_and_match() — 感知当前页面并匹配模板
    2. 如果漂移：relearn() — 更新模板
    3. 如果有配方：execute_recipe() — 执行保存的动作配方
    4. resolve_candidate_keys() — 为画布候选解析持久身份
    """

    def __init__(self) -> None:
        self._template_manager = PageTemplateManager()
        self._drift_detector = DriftDetector()
        self._recipe_manager = ActionRecipeManager()
        self._key_store = CandidateKeyStore()
        self._page_model_store = PageModelStore()

    def perceive_and_match(
        self,
        app_id: str,
        zone_structure: Any,
    ) -> tuple[str, Any | None, DriftReport | None]:
        """
        感知并匹配模板

        Args:
            app_id: 应用标识
            zone_structure: 当前感知结果

        Returns:
            (page_type, matched_template_or_none, drift_report_or_none)
        """
        # 尝试匹配已有模板
        template, confidence = self._template_manager.match(app_id, zone_structure)

        if template:
            # 检测漂移
            drift_report = self._drift_detector.detect(template, zone_structure)
            return template.page_type, template, drift_report

        return "", None, None

    def learn_page(
        self,
        app_id: str,
        page_type: str,
        zone_structure: Any,
        screenshot_path: str | None = None,
    ) -> Any:
        """
        学习新页面（保存模板）

        Args:
            app_id: 应用标识
            page_type: 页面类型
            zone_structure: 感知结果
            screenshot_path: 截图路径

        Returns:
            PageTemplate
        """
        return self._template_manager.save(
            app_id, page_type, zone_structure, screenshot_path, overwrite=True
        )

    def execute_recipe(
        self,
        app_id: str,
        page_type: str,
        action_name: str,
        executor: Any,
    ) -> tuple[bool, str]:
        """
        执行动作配方

        Args:
            app_id: 应用标识
            page_type: 页面类型
            action_name: 动作名称
            executor: ActionExecutor 实例

        Returns:
            (success, message)
        """
        recipe = self._recipe_manager.load(app_id, page_type, action_name)
        if not recipe:
            return False, f"配方不存在: {app_id}/{page_type}/{action_name}"

        for step in recipe.steps:
            if step.action == "click":
                # step.target 格式: "zone:type:name" 或 "x,y"
                if "," in step.target:
                    x, y = map(int, step.target.split(","))
                    result = executor.click(0, x, y)  # hwnd=0 使用当前窗口
                else:
                    # 元素级点击（待实现：需要从 zone_structure 查找元素坐标）
                    return False, f"元素级点击待实现: {step.target}"
                if not result.success:
                    return False, f"步骤失败: {step.action} {step.target} — {result.error}"
            elif step.action == "send_text":
                result = executor.send_text(0, step.value or "")
                if not result.success:
                    return False, f"步骤失败: {step.action} {step.target} — {result.error}"
            else:
                return False, f"未知动作类型: {step.action}"

        return True, f"配方执行成功: {action_name}"

    def save_recipe(self, recipe: ActionRecipeData) -> ActionRecipeData:
        """保存动作配方"""
        saved = self._recipe_manager.save(recipe)
        return recipe

    def list_recipes(self, app_id: str) -> list[ActionRecipeData]:
        """列出应用的所有配方"""
        return self._recipe_manager.list_recipes(app_id)

    # --- Transition Graph ---

    def record_transition(
        self,
        from_class: str,
        to_class: str,
        action: str | None = None,
        candidate_key: str | None = None,
        success: bool = True,
    ) -> None:
        """记录一次页面状态转移"""
        # Uses a new session for each call
        from src.storage.db import Session
        with Session() as session:
            manager = TransitionGraphManager(session)
            manager.record_transition(from_class, to_class, action, candidate_key, success)

    def get_transitions(self, page_class: str) -> list[TransitionInfo]:
        """查询从某个状态出发的所有历史转移"""
        from src.storage.db import Session
        with Session() as session:
            manager = TransitionGraphManager(session)
            return manager.get_transitions(page_class)

    # --- Feedback ---

    def record_feedback(
        self,
        canvas_id: str,
        candidate_key: str,
        feedback_type: str,
        detail: dict | None = None,
    ) -> None:
        """记录 Agent 反馈"""
        from src.storage.db import Session
        with Session() as session:
            manager = FeedbackManager(session)
            manager.record_feedback(canvas_id, candidate_key, feedback_type, detail)

    def get_feedback_stats(self, candidate_key: str) -> FeedbackStats:
        """查询候选的反馈统计"""
        from src.storage.db import Session
        with Session() as session:
            manager = FeedbackManager(session)
            return manager.get_stats(candidate_key)

    # --- Page Model ---

    def resolve_page_model(
        self,
        canvas: Any,
        has_screenshot: bool = False,
        *,
        vlm_app_name: str | None = None,
        vlm_display_name: str | None = None,
        vlm_state_label: str | None = None,
        vlm_state_flags: list[str] | None = None,
    ) -> tuple[str, str, str, str]:
        """解析页面模型，返回 (page_model_id, state_template_id, page_status, state_status)。

        page_status: "reused" | "new"
        state_status: "reused" | "new"

        集成链路：
        1. extract_page_class_prefix → page_class_prefix
        2. build_layout_signature → layout_signature
        3. page_model_store.find_or_create_page_model
        4. build_layout_fingerprint → LayoutFingerprint
        5. classify_state_label → state_label
        6. page_model_store.find_or_create_state_template
        7. page_model_store.record_canvas_snapshot
        """
        from src.storage.db import Session

        app_id = canvas.app.app_id or "unknown"
        page_class = canvas.page_class or ""
        process_name = canvas.app.process_name if canvas.app else None
        surface_type = canvas.surface.surface_type.value if canvas.surface else None
        window_title = canvas.window.title if canvas.window else None
        artifacts = canvas.artifacts if isinstance(getattr(canvas, "artifacts", None), dict) else {}
        visual_pattern = artifacts.get("visual_pattern") if isinstance(artifacts.get("visual_pattern"), dict) else {}
        page_class = normalize_page_class_for_memory(
            page_class,
            process_name=process_name,
            visual_mode=str(visual_pattern.get("mode") or ""),
        )
        if getattr(canvas, "page", None) is not None:
            canvas.page.page_class = page_class

        layout_sig = build_layout_signature(canvas)
        fingerprint = build_layout_fingerprint(canvas)
        state_label = classify_state_label(canvas)

        fixed_count = sum(
            1 for e in canvas.elements
            if (e.semantic_role.value if hasattr(e.semantic_role, "value") else str(e.semantic_role))
            in FIXED_CONTROL_ROLES
        )

        with Session() as session:
            pm, page_status = self._page_model_store.find_or_create_page_model(
                session, app_id, page_class, process_name, surface_type, layout_sig,
                window_title=window_title,
                vlm_app_name=vlm_app_name,
                vlm_display_name=vlm_display_name,
            )
            st, state_status = self._page_model_store.find_or_create_state_template(
                session, pm.page_model_id, app_id, page_class,
                fingerprint, len(canvas.elements), fixed_count, state_label,
                vlm_state_label=vlm_state_label,
                vlm_state_flags=vlm_state_flags,
            )
            self._page_model_store.record_canvas_snapshot(
                session, canvas.canvas_id, pm.page_model_id,
                st.state_template_id, len(canvas.elements), has_screenshot,
            )

            return pm.page_model_id, st.state_template_id, page_status, state_status

    # --- Candidate Identity ---

    def resolve_candidate_keys(
        self,
        canvas: Any,
        page_model_id: str | None = None,
        state_template_id: str | None = None,
        force_transient: bool = False,
    ) -> int:
        """为画布中的候选解析持久身份（stable_key_id）。

        遍历 canvas.candidates，为每个候选构建签名，
        在 key_store 中查找或创建 StableCandidateKey，
        并回写 candidate.stable_key_id。
        如果提供了 page_model_id 和 state_template_id，还会记录候选-状态关联。

        Args:
            canvas: InteractionCanvas
            page_model_id: 页面模型 ID（可选）
            state_template_id: 状态模板 ID（可选）
            force_transient: True 时强制持久化所有候选（含 transient）

        Returns:
            被分配 stable_key_id 的候选数量
        """
        from src.storage.db import Session

        app_id = canvas.app.app_id or "unknown"
        page_class = canvas.page_class or ""

        assigned = 0
        with Session() as session:
            for candidate in canvas.elements:
                sig = build_signature(candidate, canvas)
                key, status = self._key_store.find_or_create(
                    session, app_id, page_class, sig, force=force_transient,
                )
                if key is not None:
                    candidate.stable_key_id = key.key_id
                    assigned += 1

                    # 记录候选-状态关联
                    if state_template_id and page_model_id:
                        self._page_model_store.record_candidate_state(
                            session, key.key_id, state_template_id, page_model_id,
                        )

        return assigned

    @property
    def key_store(self) -> CandidateKeyStore:
        """暴露 key_store 给 diff_engine 使用。"""
        return self._key_store

    @property
    def page_model_store(self) -> PageModelStore:
        """暴露 page_model_store 给 API 使用。"""
        return self._page_model_store
