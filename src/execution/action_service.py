"""
动作服务

统一动作执行、校验、恢复的对外接口。
"""
from dataclasses import dataclass
from typing import Any, Callable, Literal

from src.execution.action_executor import ActionExecutor, ActionResult, MouseButton
from src.execution.action_verifier import ActionVerifier, VerificationResult
from src.execution.canvas_action_adapter import InteractionCanvasAdapter, get_surface_execution_strategy
from src.execution.risk_policy import RiskPolicyEngine
from src.execution.failure_recovery import (
    FailureRecovery,
    LocatorCandidate,
    RecoveryResult,
    RecoveryStrategy,
    RetryConfig,
    FallbackConfig,
)
from src.perception.page_compiler_models import (
    LocatorKind,
    Candidate,
    InteractionCanvas,
    SemanticRole,
    SurfaceType,
)
from src.perception.page_compiler_candidates import build_boundary_candidates
from src.perception.openclaw_protocol import (
    build_openclaw_payload,
    normalize_decision_record,
)
from src.perception.perception_service import PerceptionService


@dataclass
class ActionOutcome:
    """动作执行最终结果"""
    success: bool
    action: str
    message: str
    attempts: int = 1
    strategy: str = "direct"
    error: str | None = None
    verification_details: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-friendly structure for runtime and tooling flows."""
        payload = {
            "success": self.success,
            "action": self.action,
            "message": self.message,
            "attempts": self.attempts,
            "strategy": self.strategy,
            "error": self.error,
            "verification_details": dict(self.verification_details or {}),
        }
        from src.execution.execution_record import build_execution_record

        payload["execution_record"] = build_execution_record(payload)
        return payload


class ActionService:
    """
    动作服务

    统一接口，封装执行器、校验器、恢复器。
    使用流程：
    1. 创建 ActionService
    2. 调用 action_* 方法执行动作
    3. 接收 ActionOutcome 结果
    """

    def __init__(self) -> None:
        self._executor = ActionExecutor()
        self._verifier = ActionVerifier()
        self._recovery = FailureRecovery()
        self._snapshot_adapter = InteractionCanvasAdapter()
        self._perception_service = PerceptionService()
        self._risk_policy = RiskPolicyEngine()

    # -------------------------------------------------------------------------
    # 动作执行接口
    # -------------------------------------------------------------------------

    def action_click(
        self,
        hwnd: int,
        x: int,
        y: int,
        verify: bool = True,
        with_recovery: bool = True,
    ) -> ActionOutcome:
        """
        点击动作

        Args:
            hwnd: 窗口句柄
            x: X 坐标（屏幕绝对坐标）
            y: Y 坐标（屏幕绝对坐标）
            verify: 是否验证
            with_recovery: 是否启用失败恢复

        Returns:
            ActionOutcome 执行结果
        """
        if with_recovery:
            result = self._recovery.click_with_recovery(
                hwnd=hwnd,
                x=x,
                y=y,
                verify_fn=self._make_verification_fn(hwnd, verify),
            )
            return ActionOutcome(
                success=result.recovered,
                action="click",
                message=result.final_result.details if result.final_result else "",
                attempts=result.attempts,
                strategy=result.strategy_used.value,
                error=result.error,
            )
        else:
            result = self._executor.click(hwnd, x, y)
            verified = False
            if verify and result.success:
                verified = self._verifier.verify_window_state(
                    hwnd, "active", timeout=1.0
                ).verified

            return ActionOutcome(
                success=result.success and verified if verify else result.success,
                action="click",
                message=result.details,
                attempts=1,
                strategy="direct",
                error=result.error,
            )

    def action_click_element(
        self,
        hwnd: int,
        element_x: int,
        element_y: int,
        element_width: int,
        element_height: int,
        verify: bool = True,
        with_recovery: bool = True,
    ) -> ActionOutcome:
        """
        点击元素中心

        Args:
            hwnd: 窗口句柄
            element_x: 元素左边界（窗口内容区坐标）
            element_y: 元素上边界（窗口内容区坐标）
            element_width: 元素宽度
            element_height: 元素高度
            verify: 是否验证
            with_recovery: 是否启用失败恢复

        Returns:
            ActionOutcome 执行结果
        """
        center_x = element_x + element_width // 2
        center_y = element_y + element_height // 2
        screen_x, screen_y = self._executor._client_to_screen(hwnd, center_x, center_y)

        return self.action_click(hwnd, screen_x, screen_y, verify, with_recovery)

    def action_send_text(
        self,
        hwnd: int,
        text: str,
        verify: bool = False,
    ) -> ActionOutcome:
        """
        发送文本

        Args:
            hwnd: 窗口句柄
            text: 要发送的文本
            verify: 是否验证

        Returns:
            ActionOutcome 执行结果
        """
        result = self._executor.send_text(hwnd, text)
        return ActionOutcome(
            success=result.success,
            action="send_text",
            message=result.details,
            error=result.error,
        )

    def action_invoke_uia(
        self,
        hwnd: int,
        automation_id: str | None = None,
        name: str | None = None,
        verify: bool = True,
    ) -> ActionOutcome:
        """
        调用 UIA 元素

        Args:
            hwnd: 窗口句柄
            automation_id: 自动化 ID
            name: 元素名称
            verify: 是否验证

        Returns:
            ActionOutcome 执行结果
        """
        result = self._executor.invoke_uia_element(hwnd, automation_id, name)

        verified = False
        if verify and result.success:
            verified = self._verifier.verify_window_state(
                hwnd, "active", timeout=1.0
            ).verified

        return ActionOutcome(
            success=result.success and verified if verify else result.success,
            action="invoke_uia",
            message=result.details,
            error=result.error,
        )

    def execute_snapshot_action(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        action: Literal["click", "double_click", "right_click"],
        verify: bool = True,
        with_recovery: bool = True,
        recapture_after: bool | None = None,
    ) -> ActionOutcome:
        """
        Execute an action from a InteractionCanvas using locator-first routing.
        """
        resolved = self._snapshot_adapter.resolve_element(snapshot, element_id)
        if not resolved:
            return ActionOutcome(
                success=False,
                action=action,
                message=f"Element {element_id} not found in snapshot",
                strategy="locator_first",
                error="element_not_found",
            )

        hwnd = snapshot.window.hwnd if snapshot.window else 0
        if not hwnd:
            return ActionOutcome(
                success=False,
                action=action,
                message="No window hwnd in snapshot",
                strategy="locator_first",
                error="no_hwnd",
            )

        risk_decision = self._risk_policy.assess(snapshot, element_id, action)
        if not risk_decision.allowed:
            return ActionOutcome(
                success=False,
                action=action,
                message=risk_decision.reason,
                strategy=f"risk_policy/{risk_decision.mode}",
                error="risk_policy_blocked",
                verification_details={
                    "mode": "risk_policy",
                    "risk_level": risk_decision.risk_level,
                    "policy_mode": risk_decision.mode,
                    "reason": risk_decision.reason,
                    "policy_source": risk_decision.policy_source,
                    "surface_type": snapshot.surface.surface_type.value,
                    "element_id": element_id,
                },
            )

        method = resolved["method"]
        locator = resolved["locator"]
        geometry = resolved.get("geometry")
        coordinates = resolved.get("coordinates")
        recapture_enabled, recapture_policy = self._resolve_recapture_policy(
            snapshot=snapshot,
            element_id=element_id,
            action=action,
            recapture_after=recapture_after,
        )

        if with_recovery:
            locator_candidates = self._build_locator_candidates(snapshot, element_id, action)
            if len(locator_candidates) > 1:
                recovery = self._recovery.execute_locator_fallback_chain(
                    candidates=locator_candidates,
                    verify_fn=(
                        self._make_snapshot_verification_fn(snapshot, element_id)
                        if verify
                        else None
                    ),
                )
                strategy_suffix = recovery.strategy_used.value if recovery.recovered else "abort"
                outcome = ActionOutcome(
                    success=recovery.recovered,
                    action=action,
                    message=recovery.final_result.details if recovery.final_result else "",
                    attempts=recovery.attempts,
                    strategy=f"locator_first/chain/{strategy_suffix}",
                    error=recovery.error or (recovery.final_result.error if recovery.final_result else None),
                    verification_details={"recovery": recovery.details} if recovery.details else None,
                )
                self._merge_recovery_execution_details(outcome, recovery.final_result)
                self._build_runtime_trace(outcome)
                return self._finalize_snapshot_outcome(
                    outcome=outcome,
                    before_snapshot=snapshot,
                    element_id=element_id,
                    action=action,
                    verify=verify,
                    recapture_after=recapture_enabled,
                    recapture_policy=recapture_policy,
                )

        if with_recovery and method == "coordinate" and coordinates:
            x, y = coordinates
            if locator.coordinate_space.value != "screen":
                x, y = self._executor._client_to_screen(hwnd, x, y)
            recovery = self._recovery.click_with_recovery(
                hwnd=hwnd,
                x=x,
                y=y,
                verify_fn=self._make_verification_fn(hwnd, verify),
            )
            outcome = ActionOutcome(
                success=recovery.recovered,
                action=action,
                message=recovery.final_result.details if recovery.final_result else "",
                attempts=recovery.attempts,
                strategy=f"locator_first/{locator.kind.value}/{recovery.strategy_used.value}",
                error=recovery.error or (recovery.final_result.error if recovery.final_result else None),
            )
            return self._finalize_snapshot_outcome(
                outcome=outcome,
                before_snapshot=snapshot,
                element_id=element_id,
                action=action,
                verify=verify,
                recapture_after=recapture_enabled,
                recapture_policy=recapture_policy,
            )

        if with_recovery and method in {"geometry", "relative"} and geometry:
            x, y, width, height = geometry
            outcome = self.action_click_element(
                hwnd=hwnd,
                element_x=x,
                element_y=y,
                element_width=width,
                element_height=height,
                verify=verify,
                with_recovery=True,
            )
            return self._finalize_snapshot_outcome(
                outcome=outcome,
                before_snapshot=snapshot,
                element_id=element_id,
                action=action,
                verify=verify,
                recapture_after=recapture_enabled,
                recapture_policy=recapture_policy,
            )

        result = self._snapshot_adapter.execute_element_action(
            snapshot=snapshot,
            element_id=element_id,
            action=action,
            verify=verify,
        )
        outcome = self._to_outcome(
            result=result,
            action=action,
            strategy=f"locator_first/{locator.kind.value}/{method}",
        )
        return self._finalize_snapshot_outcome(
            outcome=outcome,
            before_snapshot=snapshot,
            element_id=element_id,
            action=action,
            verify=verify,
            recapture_after=recapture_enabled,
            recapture_policy=recapture_policy,
        )

    def execute_decision_action(
        self,
        snapshot: InteractionCanvas,
        decision: dict[str, Any] | str,
        verify: bool = True,
        with_recovery: bool = True,
        recapture_after: bool | None = None,
        crop_scale: float = 1.5,
    ) -> ActionOutcome:
        """
        Execute one OpenClaw decision record against a InteractionCanvas.

        The decision can either:
        - select a concrete candidate to operate on
        - request a focused recrop for local re-analysis
        """
        record = normalize_decision_record(decision)
        decision_status = str(record.get("decision_status") or "ready")
        selected_candidate_id = str(record.get("selected_candidate_id") or "")
        next_action = dict(record.get("next_action") or {})
        action_type = str(next_action.get("type") or "click")

        if (
            decision_status == "need_zoom_in"
            or action_type == "recrop_and_reanalyze"
            or (record.get("focus_bbox") and not selected_candidate_id)
        ):
            recrop_request = self._build_recrop_request(
                snapshot=snapshot,
                decision_record=record,
                crop_scale=crop_scale,
            )
            return ActionOutcome(
                success=False,
                action="recrop",
                message="openclaw_requested_recrop",
                strategy="decision_record/recrop",
                error="recrop_required",
                verification_details={
                    "decision_record": record,
                    "recrop_request": recrop_request,
                },
            )

        if not selected_candidate_id:
            return ActionOutcome(
                success=False,
                action=action_type or "click",
                message="decision_record_missing_candidate",
                strategy="decision_record",
                error="decision_candidate_missing",
                verification_details={"decision_record": record},
            )

        if selected_candidate_id.startswith("elem::"):
            element_id = selected_candidate_id.split("::", 1)[1]
            element_action = self._map_decision_action_to_element_action(action_type)
            outcome = self.execute_snapshot_action(
                snapshot=snapshot,
                element_id=element_id,
                action=element_action,
                verify=verify,
                with_recovery=with_recovery,
                recapture_after=recapture_after,
            )
            return self._append_decision_details(outcome, record)

        candidate = self._find_boundary_candidate(snapshot, selected_candidate_id)
        if candidate is None:
            candidate = self._candidate_from_decision_record(record)
            if candidate is None:
                return ActionOutcome(
                    success=False,
                    action=action_type or "click",
                    message=f"decision candidate {selected_candidate_id} not found",
                    strategy="decision_record",
                    error="candidate_not_found",
                    verification_details={"decision_record": record},
                )

        return self._execute_boundary_candidate_action(
            snapshot=snapshot,
            candidate=candidate,
            decision_record=record,
            verify=verify,
            with_recovery=with_recovery,
        )

    def execute_openclaw_decision_loop(
        self,
        snapshot: InteractionCanvas,
        task: str,
        decision_provider: Callable[[dict[str, Any]], dict[str, Any] | str],
        verify: bool = True,
        with_recovery: bool = True,
        recapture_after: bool | None = None,
        crop_scale: float = 1.5,
        max_candidates: int = 120,
        max_rounds: int = 3,
    ) -> ActionOutcome:
        """
        Run a multi-round OpenClaw loop:
        payload -> decision -> optional zoom-in -> refined payload -> action.
        """
        payload = build_openclaw_payload(
            snapshot,
            task=task,
            max_candidates=max_candidates,
        )
        loop_trace: list[dict[str, Any]] = []

        for round_index in range(1, max_rounds + 1):
            raw_decision = decision_provider(payload)
            decision_record = normalize_decision_record(raw_decision)
            loop_trace.append(
                {
                    "round": round_index,
                    "page_hint": payload.get("page_hint"),
                    "candidate_count": len(payload.get("candidates") or []),
                    "candidate_summary": dict(payload.get("candidate_summary") or {}),
                    "provider_health": dict(payload.get("provider_health") or {}),
                    "decision_status": decision_record.get("decision_status"),
                    "selected_candidate_id": decision_record.get("selected_candidate_id"),
                    "next_action": dict(decision_record.get("next_action") or {}),
                    "focus_bbox": decision_record.get("focus_bbox"),
                }
            )

            if self._decision_requires_focus(decision_record):
                focus_result = self.build_focus_openclaw_payload(
                    snapshot=snapshot,
                    decision=decision_record,
                    task=task,
                    crop_scale=crop_scale,
                    max_candidates=max_candidates,
                )
                if "focused_openclaw_payload" not in focus_result:
                    outcome = ActionOutcome(
                        success=False,
                        action="recrop",
                        message="focus_payload_build_failed",
                        strategy="decision_loop/recrop",
                        error=focus_result.get("error") or "focus_payload_build_failed",
                        verification_details={
                            "decision_loop": loop_trace,
                            "focus_result": focus_result,
                        },
                    )
                    return outcome
                loop_trace[-1]["focus_result"] = {
                    "crop_rect": focus_result.get("crop_rect"),
                    "crop_size": focus_result.get("crop_size"),
                    "ocr_block_count": (focus_result.get("ocr_provider") or {}).get("block_count"),
                    "vision_candidate_count": (focus_result.get("vision_provider") or {}).get("candidate_count"),
                }
                payload = focus_result["focused_openclaw_payload"]
                continue

            outcome = self.execute_decision_action(
                snapshot=snapshot,
                decision=decision_record,
                verify=verify,
                with_recovery=with_recovery,
                recapture_after=recapture_after,
                crop_scale=crop_scale,
            )
            details = dict(outcome.verification_details or {})
            details["decision_loop"] = loop_trace
            outcome.verification_details = details
            return outcome

        return ActionOutcome(
            success=False,
            action="decision_loop",
            message="openclaw_decision_rounds_exhausted",
            strategy="decision_loop/max_rounds",
            error="decision_rounds_exhausted",
            verification_details={"decision_loop": loop_trace},
        )

    def build_focus_openclaw_payload(
        self,
        snapshot: InteractionCanvas,
        decision: dict[str, Any] | str,
        task: str,
        crop_scale: float = 1.5,
        max_candidates: int = 120,
    ) -> dict[str, Any]:
        """
        Capture the current window again, crop to OpenClaw's focus bbox,
        rerun lightweight OCR/vision, and build a second-round payload.
        """
        from src.perception.ocr_service import get_ocr_service
        from src.windows.screenshot_service import ScreenshotService

        record = normalize_decision_record(decision)
        focus_bbox = record.get("focus_bbox")
        hwnd = snapshot.window.hwnd if snapshot.window else 0
        if not focus_bbox:
            return {"error": "focus_bbox_missing"}
        if not hwnd:
            return {"error": "no_hwnd"}

        screenshot = ScreenshotService().capture(mode="window", target=hwnd)
        crop_rect = self._expand_focus_bbox(
            tuple(int(value) for value in focus_bbox),
            screenshot.size,
            crop_scale=crop_scale,
        )
        cropped = screenshot.crop(crop_rect)

        ocr_result = get_ocr_service().extract_with_metadata(cropped)
        vision_candidates: list[dict[str, Any]] = []
        vision_provider_details: dict[str, Any] = {}
        try:
            vision_provider = getattr(self._perception_service, "_vision_provider", None)
            if vision_provider is not None:
                vision_result = vision_provider.parse_screenshot(cropped)
                vision_provider_details = {
                    "provider": vision_result.provider,
                    "success": vision_result.success,
                    "error": vision_result.error,
                    "candidate_count": len(vision_result.candidates),
                }
                vision_candidates = [
                    {
                        "candidate_id": candidate.element_id,
                        "bbox": list(candidate.bounding_box),
                        "kind": candidate.semantic_label,
                        "text": candidate.text,
                        "source": vision_result.provider,
                        "confidence": candidate.confidence,
                    }
                    for candidate in vision_result.candidates
                    if candidate.bounding_box
                ]
        except Exception as exc:
            vision_provider_details = {
                "provider": "omniparser",
                "success": False,
                "error": str(exc),
                "candidate_count": 0,
            }

        focus_snapshot = {
            "canvas_id": f"{snapshot.canvas_id}_focus",
            "app": {"process_name": snapshot.app.process_name},
            "window": {"rect_client": [0, 0, cropped.size[0], cropped.size[1]]},
            "surface": {"surface_type": snapshot.surface.surface_type.value},
            "page": {"page_class": snapshot.page.page_class},
            "regions": [
                {
                    "region_id": "focus_region",
                    "role": "focus_region",
                    "subtype": "unknown",
                    "bounds": [0, 0, cropped.size[0], cropped.size[1]],
                }
            ],
            "elements": [],
            "artifacts": {
                "ocr_blocks": [
                    {
                        "bbox": list(block.bbox),
                        "text": block.text,
                        "confidence": block.confidence,
                    }
                    for block in ocr_result.blocks
                ],
                "vision_candidates": vision_candidates,
                "focus_source_bbox": list(crop_rect),
            },
        }
        focused_payload = build_openclaw_payload(
            focus_snapshot,
            task=task,
            max_candidates=max_candidates,
        )
        focused_payload = self._translate_focus_payload_to_window_coords(
            payload=focused_payload,
            crop_rect=crop_rect,
            snapshot=snapshot,
        )
        return {
            "page_state": record.get("page_state"),
            "focus_bbox": list(focus_bbox),
            "crop_rect": list(crop_rect),
            "crop_size": list(cropped.size),
            "ocr_provider": {
                "provider": ocr_result.provider,
                "success": ocr_result.success,
                "error": ocr_result.error,
                "block_count": len(ocr_result.blocks),
            },
            "vision_provider": vision_provider_details,
            "focused_openclaw_payload": focused_payload,
        }

    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------

    def _make_verification_fn(self, hwnd: int, verify: bool):
        """创建验证函数"""
        if not verify:
            return None

        def verify_fn():
            return self._verifier.verify_window_state(hwnd, "active", timeout=1.0)

        return verify_fn

    def _map_decision_action_to_element_action(
        self,
        action_type: str,
    ) -> Literal["click", "double_click", "right_click"]:
        if action_type == "double_click":
            return "double_click"
        if action_type == "right_click":
            return "right_click"
        return "click"

    def _find_boundary_candidate(
        self,
        snapshot: InteractionCanvas,
        candidate_id: str,
    ) -> dict[str, Any] | None:
        candidates = build_boundary_candidates(snapshot)
        return next(
            (item for item in candidates if item.get("candidate_id") == candidate_id),
            None,
        )

    def _execute_boundary_candidate_action(
        self,
        snapshot: InteractionCanvas,
        candidate: dict[str, Any],
        decision_record: dict[str, Any],
        verify: bool,
        with_recovery: bool,
    ) -> ActionOutcome:
        bbox = candidate.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return ActionOutcome(
                success=False,
                action=str((decision_record.get("next_action") or {}).get("type") or "click"),
                message="decision candidate missing bbox",
                strategy="decision_record/bbox",
                error="candidate_bbox_missing",
                verification_details={
                    "decision_record": decision_record,
                    "candidate": candidate,
                },
            )

        hwnd = snapshot.window.hwnd if snapshot.window else 0
        if not hwnd:
            return ActionOutcome(
                success=False,
                action=str((decision_record.get("next_action") or {}).get("type") or "click"),
                message="No window hwnd in snapshot",
                strategy="decision_record/bbox",
                error="no_hwnd",
                verification_details={
                    "decision_record": decision_record,
                    "candidate": candidate,
                },
            )

        left, top, right, bottom = [int(value) for value in bbox]
        width = max(1, right - left)
        height = max(1, bottom - top)
        action_type = str((decision_record.get("next_action") or {}).get("type") or "click")

        if action_type == "type":
            click_outcome = self.action_click_element(
                hwnd=hwnd,
                element_x=left,
                element_y=top,
                element_width=width,
                element_height=height,
                verify=False,
                with_recovery=with_recovery,
            )
            if not click_outcome.success:
                return self._append_decision_details(click_outcome, decision_record, candidate)
            type_outcome = self.action_send_text(hwnd, str((decision_record.get("next_action") or {}).get("text") or ""))
            type_outcome.strategy = "decision_record/bbox/type"
            return self._append_decision_details(type_outcome, decision_record, candidate)

        click_outcome = self.action_click_element(
            hwnd=hwnd,
            element_x=left,
            element_y=top,
            element_width=width,
            element_height=height,
            verify=False,
            with_recovery=with_recovery,
        )
        click_outcome.strategy = f"decision_record/bbox/{action_type or 'click'}"
        click_outcome = self._append_decision_details(click_outcome, decision_record, candidate)
        if not click_outcome.success:
            return click_outcome

        if action_type == "click_and_type":
            text = str((decision_record.get("next_action") or {}).get("text") or "")
            text_result = self.action_send_text(hwnd, text)
            click_outcome.success = click_outcome.success and text_result.success
            click_outcome.action = "click_and_type"
            click_outcome.message = text_result.message or click_outcome.message
            click_outcome.error = text_result.error or click_outcome.error
            details = dict(click_outcome.verification_details or {})
            details["typed_text"] = text
            click_outcome.verification_details = details
        return click_outcome

    def _build_recrop_request(
        self,
        snapshot: InteractionCanvas,
        decision_record: dict[str, Any],
        crop_scale: float,
    ) -> dict[str, Any]:
        return {
            "source_canvas_id": snapshot.canvas_id,
            "focus_bbox": list(decision_record.get("focus_bbox") or []),
            "crop_scale": crop_scale,
            "page_state": decision_record.get("page_state"),
            "reason": decision_record.get("reason"),
            "next_action": dict(decision_record.get("next_action") or {}),
        }

    def _decision_requires_focus(self, decision_record: dict[str, Any]) -> bool:
        action_type = str((decision_record.get("next_action") or {}).get("type") or "")
        return bool(
            decision_record.get("focus_bbox")
            and (
                str(decision_record.get("decision_status") or "") == "need_zoom_in"
                or action_type == "recrop_and_reanalyze"
                or not decision_record.get("selected_candidate_id")
            )
        )

    def _expand_focus_bbox(
        self,
        focus_bbox: tuple[int, int, int, int],
        image_size: tuple[int, int],
        crop_scale: float,
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = focus_bbox
        width = max(1, right - left)
        height = max(1, bottom - top)
        center_x = left + width / 2
        center_y = top + height / 2
        expanded_width = max(width, int(width * crop_scale))
        expanded_height = max(height, int(height * crop_scale))
        crop_left = max(0, int(center_x - expanded_width / 2))
        crop_top = max(0, int(center_y - expanded_height / 2))
        crop_right = min(image_size[0], crop_left + expanded_width)
        crop_bottom = min(image_size[1], crop_top + expanded_height)
        return crop_left, crop_top, crop_right, crop_bottom

    def _append_decision_details(
        self,
        outcome: ActionOutcome,
        decision_record: dict[str, Any],
        candidate: dict[str, Any] | None = None,
    ) -> ActionOutcome:
        details = dict(outcome.verification_details or {})
        details["decision_record"] = dict(decision_record)
        if candidate is not None:
            details["decision_candidate"] = dict(candidate)
            details["decision_candidate_source"] = candidate.get("source")
            details["decision_candidate_kind"] = candidate.get("candidate_kind")
            details["decision_candidate_bbox"] = list(candidate.get("bbox") or [])
            details["decision_candidate_origin"] = dict(candidate.get("attributes") or {}).get(
                "candidate_origin"
            )
        outcome.verification_details = details
        return outcome

    def _candidate_from_decision_record(
        self,
        decision_record: dict[str, Any],
    ) -> dict[str, Any] | None:
        bbox = decision_record.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        return {
            "candidate_id": str(decision_record.get("selected_candidate_id") or "decision_bbox"),
            "bbox": [int(value) for value in bbox],
            "source": "decision_record",
            "confidence": float(decision_record.get("confidence") or 0.0),
            "candidate_kind": "unknown",
            "text": "",
            "control_hint": str(decision_record.get("selected_role") or ""),
            "region_hint": "",
            "region_id": None,
            "attributes": {
                "candidate_origin": "decision_bbox_fallback",
                "used_bbox_fallback": True,
            },
        }

    def _translate_focus_payload_to_window_coords(
        self,
        payload: dict[str, Any],
        crop_rect: tuple[int, int, int, int],
        snapshot: InteractionCanvas,
    ) -> dict[str, Any]:
        offset_x, offset_y, _, _ = crop_rect
        translated = dict(payload)
        translated["window_bounds"] = self._resolve_snapshot_window_bounds(snapshot)
        translated["regions"] = [
            {
                **dict(region),
                "bounds": self._translate_bbox(region.get("bounds"), offset_x, offset_y),
            }
            for region in list(payload.get("regions") or [])
        ]
        translated["candidates"] = [
            self._translate_focus_candidate(candidate, offset_x, offset_y)
            for candidate in list(payload.get("candidates") or [])
        ]
        translated["ocr_blocks"] = [
            {
                **dict(block),
                "bbox": self._translate_bbox(block.get("bbox"), offset_x, offset_y),
            }
            for block in list(payload.get("ocr_blocks") or [])
        ]
        notes = list(translated.get("notes") or [])
        notes.append(f"focus_source_bbox={list(crop_rect)}")
        translated["notes"] = notes
        return translated

    def _translate_focus_candidate(
        self,
        candidate: dict[str, Any],
        offset_x: int,
        offset_y: int,
    ) -> dict[str, Any]:
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id and not candidate_id.startswith("focus::"):
            candidate_id = f"focus::{candidate_id}"
        return {
            **dict(candidate),
            "candidate_id": candidate_id,
            "bbox": self._translate_bbox(candidate.get("bbox"), offset_x, offset_y),
        }

    def _translate_bbox(
        self,
        bbox: Any,
        offset_x: int,
        offset_y: int,
    ) -> list[int] | None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        left, top, right, bottom = [int(value) for value in bbox]
        return [
            left + offset_x,
            top + offset_y,
            right + offset_x,
            bottom + offset_y,
        ]

    def _resolve_snapshot_window_bounds(
        self,
        snapshot: InteractionCanvas,
    ) -> list[int] | None:
        if snapshot.window is None:
            return None
        bounds = snapshot.window.rect_client or snapshot.window.rect_screen
        if bounds is None:
            return None
        return [int(value) for value in bounds]

    def _make_snapshot_verification_fn(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
    ):
        def verify_fn(
            candidate: LocatorCandidate | None = None,
            result: ActionResult | None = None,
        ):
            recovery_details = result.details if result is not None else ""
            if recovery_details and (
                "[recovery " in recovery_details
                or recovery_details.startswith("anchor_relocated:")
                or recovery_details.startswith("replan_ready:")
            ):
                candidate_name = candidate.name if candidate is not None else None
                return VerificationResult(
                    verified=True,
                    method=self._verifier.verify_snapshot_target(snapshot, element_id).method,
                    message="recovery_verification_deferred",
                    details={
                        "element_id": element_id,
                        "candidate": candidate_name,
                        "recovery_details": recovery_details,
                    },
                )
            return self._verifier.verify_snapshot_target(snapshot, element_id)

        return verify_fn

    def _build_locator_candidates(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        action: str,
    ) -> list[LocatorCandidate]:
        element = snapshot.get_element(element_id)
        if element is None:
            return []

        candidates: list[LocatorCandidate] = []
        locators = [
            snapshot.get_locator(locator_id)
            for locator_id in element.locator_ids
            if snapshot.get_locator(locator_id) is not None
        ]
        surface_strategy = get_surface_execution_strategy(snapshot.surface.surface_type)
        preferred_chain = surface_strategy.get("locator_chain", [])
        chain_index = {name: index for index, name in enumerate(preferred_chain)}
        hwnd = snapshot.window.hwnd if snapshot.window else 0
        runtime_context = self._build_runtime_context(snapshot, element)

        shortcut = self._get_shortcut_for_element(snapshot, element_id, action)
        if shortcut and hwnd:
            candidates.append(
                LocatorCandidate(
                    name="shortcut",
                    strategy=RecoveryStrategy.FALLBACK_ELEMENT,
                    preferred_rank=chain_index.get("shortcut", 999),
                    risk_tier=max(1, runtime_context["risk_bias"]),
                    cost_score=0.05 + runtime_context["cost_bias"],
                    durability_score=0.8,
                    confidence=0.6 + runtime_context["confidence_bonus"],
                    anchor_count=len(element.anchor_ids),
                    scroll_context=runtime_context["has_scroll_context"],
                    content_group_id=element.content_group_id,
                    region_role=runtime_context["region_role"],
                    interaction_hints=runtime_context["interaction_hints"],
                    structure_evidence_score=runtime_context["structure_evidence_score"],
                    viewport_index=runtime_context["viewport_index"],
                    replan_hint=runtime_context["replan_hint"],
                    action_fn=lambda shortcut_keys=shortcut: self._executor.send_keys(hwnd, shortcut_keys),
                    recovery_fn=self._make_runtime_recovery_fn(
                        snapshot=snapshot,
                        element_id=element_id,
                        action=action,
                        locator_kind=None,
                        runtime_context=runtime_context,
                    ),
                )
            )

        for locator in sorted(
            locators,
            key=lambda item: (
                chain_index.get(item.kind.value, 999),
                item.priority,
                item.cost_score,
                -item.durability_score,
                -item.confidence,
            ),
        ):
            strategy = (
                RecoveryStrategy.FALLBACK_ELEMENT
                if locator.kind in {LocatorKind.UIA, LocatorKind.DOM}
                else RecoveryStrategy.FALLBACK_COORDINATE
            )
            strategy = self._resolve_candidate_strategy(locator.kind, runtime_context)
            candidates.append(
                LocatorCandidate(
                    name=locator.kind.value,
                    strategy=strategy,
                    preferred_rank=chain_index.get(locator.kind.value, 999) + runtime_context["rank_bias"].get(locator.kind.value, 0),
                    risk_tier=max(0, self._get_locator_risk_tier(locator.kind) + runtime_context["risk_bias"]),
                    cost_score=locator.cost_score + runtime_context["cost_bias"],
                    durability_score=min(1.0, locator.durability_score + runtime_context["durability_bonus"]),
                    confidence=min(1.0, locator.confidence + runtime_context["confidence_bonus"]),
                    anchor_count=len(element.anchor_ids),
                    scroll_context=runtime_context["has_scroll_context"],
                    content_group_id=element.content_group_id,
                    region_role=runtime_context["region_role"],
                    interaction_hints=runtime_context["interaction_hints"],
                    structure_evidence_score=runtime_context["structure_evidence_score"],
                    viewport_index=runtime_context["viewport_index"],
                    replan_hint=runtime_context["replan_hint"],
                    action_fn=lambda locator_kind=locator.kind: self._snapshot_adapter.execute_element_action(
                        snapshot=snapshot,
                        element_id=element_id,
                        action=action,
                        verify=False,
                        locator_kind=locator_kind,
                    ),
                    recovery_fn=self._make_runtime_recovery_fn(
                        snapshot=snapshot,
                        element_id=element_id,
                        action=action,
                        locator_kind=locator.kind,
                        runtime_context=runtime_context,
                    ),
                )
            )
        return self._recovery.plan_locator_candidates(candidates)

    def _element_has_scroll_context(
        self,
        snapshot: InteractionCanvas,
        element: Candidate,
    ) -> bool:
        region = next(
            (item for item in snapshot.regions if item.region_id == element.region_id),
            None,
        )
        return bool(region and region.scroll_context_id)

    def _build_runtime_context(
        self,
        snapshot: InteractionCanvas,
        element: Candidate,
    ) -> dict:
        region = next((item for item in snapshot.regions if item.region_id == element.region_id), None)
        interaction_hints = dict(element.attributes.get("interaction_hints") or {})
        vision_candidate = element.attributes.get("vision_candidate") or {}
        if isinstance(vision_candidate, dict):
            interaction_hints = {**vision_candidate.get("interaction_hints", {}), **interaction_hints}
        structure_evidence_score = float(
            element.attributes.get(
                "structure_evidence_score",
                vision_candidate.get("structure_evidence_score", snapshot.artifacts.get("structure_evidence_score", 0.0)),
            )
            or 0.0
        )
        region_role = region.role if region else None
        has_scroll_context = bool(region and region.scroll_context_id)
        viewport_index = self._get_viewport_index(snapshot, region.region_id if region else None)
        anchor_refs = list(element.anchor_ids)
        anchor_texts = self._collect_anchor_texts(snapshot, anchor_refs)
        nearby_texts = self._collect_nearby_texts(snapshot, element)
        region_attributes = dict(region.attributes) if region else {}
        rank_bias: dict[str, int] = {}
        if has_scroll_context:
            rank_bias["relative"] = -1
            rank_bias["vision_bbox"] = -1
        if interaction_hints.get("preferred_action") == "click":
            rank_bias["dom"] = -1
            rank_bias["uia"] = -1
        if interaction_hints.get("scroll_into_view"):
            rank_bias["ephemeral_coord"] = 2
        risk_bias = -1 if structure_evidence_score >= 0.75 else 0
        cost_bias = -0.05 if interaction_hints else 0.0
        durability_bonus = 0.15 if structure_evidence_score >= 0.75 else 0.0
        confidence_bonus = 0.1 if structure_evidence_score >= 0.75 else 0.0
        replan_hint = bool(
            has_scroll_context
            and viewport_index is not None
            and viewport_index > 0
        )
        return {
            "region_role": region_role,
            "has_scroll_context": has_scroll_context,
            "interaction_hints": interaction_hints,
            "structure_evidence_score": structure_evidence_score,
            "viewport_index": viewport_index,
            "rank_bias": rank_bias,
            "risk_bias": risk_bias,
            "cost_bias": cost_bias,
            "durability_bonus": durability_bonus,
            "confidence_bonus": confidence_bonus,
            "replan_hint": replan_hint,
            "anchor_refs": anchor_refs,
            "anchor_texts": anchor_texts,
            "nearby_texts": nearby_texts,
            "relation_texts": self._collect_relation_texts(snapshot, element),
            "relation_hints": self._collect_relation_hints(snapshot, element),
            "region_attributes": region_attributes,
            "target_name": element.name or "",
            "target_text": element.text or "",
            "target_role": element.semantic_role.value,
            "content_group_id": element.content_group_id,
            "target_bounds": element.bounds,
            "target_region_id": element.region_id,
        }

    def _resolve_candidate_strategy(
        self,
        locator_kind: LocatorKind,
        runtime_context: dict,
    ) -> RecoveryStrategy:
        if runtime_context["has_scroll_context"] and runtime_context["region_role"] in {
            "viewport",
            "detail_panel",
            "list_panel",
            "message_stream",
            "dialog_body",
        }:
            return RecoveryStrategy.VIEWPORT_RECOVERY
        if runtime_context["interaction_hints"].get("anchor_relocate") or runtime_context["region_role"] in {
            "action_bar",
            "toolbar",
            "composer_area",
        }:
            if locator_kind in {LocatorKind.RELATIVE, LocatorKind.OCR, LocatorKind.VISION_BBOX}:
                return RecoveryStrategy.ANCHOR_RELOCATE
        return (
            RecoveryStrategy.FALLBACK_ELEMENT
            if locator_kind in {LocatorKind.UIA, LocatorKind.DOM}
            else RecoveryStrategy.FALLBACK_COORDINATE
        )

    def _get_viewport_index(
        self,
        snapshot: InteractionCanvas,
        region_id: str | None,
    ) -> int | None:
        if region_id is None:
            return None
        ordered = [region.region_id for region in snapshot.regions if region.scroll_context_id]
        if region_id not in ordered:
            return None
        return ordered.index(region_id)

    def _make_runtime_recovery_fn(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        action: str,
        locator_kind: LocatorKind | None,
        runtime_context: dict,
    ):
        def recovery_fn() -> ActionResult:
            strategy = self._resolve_recovery_path(runtime_context)
            if strategy == "viewport":
                return self._run_viewport_recovery_action(
                    snapshot=snapshot,
                    element_id=element_id,
                    action=action,
                    locator_kind=locator_kind,
                    runtime_context=runtime_context,
                )
            elif strategy == "anchor":
                return self._run_anchor_relocation_action(
                    snapshot=snapshot,
                    element_id=element_id,
                    action=action,
                    locator_kind=locator_kind,
                    runtime_context=runtime_context,
                )
            return self._run_replan_action(
                snapshot=snapshot,
                element_id=element_id,
                action=action,
                locator_kind=locator_kind,
                runtime_context=runtime_context,
            )

        return recovery_fn

    def _resolve_recovery_path(self, runtime_context: dict) -> str:
        if runtime_context["has_scroll_context"] and runtime_context["region_role"] in {
            "viewport",
            "detail_panel",
            "list_panel",
            "message_stream",
            "dialog_body",
        }:
            return "viewport"
        if runtime_context["interaction_hints"].get("anchor_relocate") or runtime_context["region_role"] in {
            "action_bar",
            "toolbar",
            "composer_area",
            "filter_bar",
        }:
            return "anchor"
        return "replan"

    def _attempt_viewport_recovery(
        self,
        snapshot: InteractionCanvas,
        runtime_context: dict,
    ) -> ActionResult:
        hwnd = snapshot.window.hwnd if snapshot.window else 0
        if not hwnd:
            return ActionResult(False, "scroll", "no_hwnd", "no_hwnd")
        deltas = self._build_viewport_recovery_deltas(snapshot, runtime_context)
        last_result: ActionResult | None = None
        for delta in deltas:
            last_result = self._executor.scroll(hwnd, delta=delta)
            if last_result.success:
                return last_result
        return last_result or ActionResult(False, "scroll", "viewport_recovery_failed", "viewport_recovery_failed")

    def _attempt_anchor_relocation(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        runtime_context: dict,
    ) -> ActionResult:
        refreshed_snapshot = self._capture_runtime_snapshot(snapshot)
        if refreshed_snapshot is None:
            return ActionResult(False, "anchor_relocate", "anchor_recapture_failed", "anchor_recapture_failed")
        refreshed_element = self._find_anchor_relocated_element(
            refreshed_snapshot,
            element_id,
            runtime_context,
        )
        if refreshed_element is None:
            return ActionResult(False, "anchor_relocate", "anchor_target_missing", "anchor_target_missing")
        return ActionResult(True, "anchor_relocate", f"anchor_relocated:{refreshed_element.element_id}")

    def _attempt_replan(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        action: str,
    ) -> ActionResult:
        refreshed_snapshot = self._capture_runtime_snapshot(snapshot)
        if refreshed_snapshot is None:
            return ActionResult(False, action, "replan_recapture_failed", "replan_recapture_failed")
        replanned_element = self._find_replanned_element(
            refreshed_snapshot,
            element_id,
        )
        if replanned_element is None:
            return ActionResult(False, action, "replan_target_missing", "replan_target_missing")
        return ActionResult(True, action, f"replan_ready:{replanned_element.element_id}")

    def _run_viewport_recovery_action(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        action: str,
        locator_kind: LocatorKind | None,
        runtime_context: dict,
    ) -> ActionResult:
        hwnd = snapshot.window.hwnd if snapshot.window else 0
        if not hwnd:
            return ActionResult(False, action, "no_hwnd", "no_hwnd")

        deltas = self._build_viewport_recovery_deltas(snapshot, runtime_context)
        last_result: ActionResult | None = None
        for step_index, delta in enumerate(deltas, start=1):
            last_result = self._executor.scroll(hwnd, delta=delta)
            if not last_result.success:
                continue

            refreshed_snapshot = self._capture_runtime_snapshot(snapshot)
            if refreshed_snapshot is None:
                return ActionResult(False, action, "runtime_recovery_recapture_failed", "runtime_recovery_recapture_failed")

            recovery_check = self._evaluate_viewport_recovery_snapshot(
                before_snapshot=snapshot,
                refreshed_snapshot=refreshed_snapshot,
                element_id=element_id,
                runtime_context=runtime_context,
            )
            next_step = self._decide_viewport_recovery_next_step(recovery_check, runtime_context)
            if next_step == "continue_scroll":
                last_result = ActionResult(
                    False,
                    action,
                    f"viewport_snapshot_unchanged_after_scroll:{step_index}",
                    "viewport_snapshot_unchanged_after_scroll",
                )
                continue

            target_element, resolved_step = self._resolve_viewport_recovery_target(
                refreshed_snapshot=refreshed_snapshot,
                element_id=element_id,
                runtime_context=runtime_context,
                next_step=next_step,
                fallback_target=recovery_check["target_element"],
            )
            if target_element is None:
                last_result = ActionResult(
                    False,
                    action,
                    f"viewport_target_missing_after_scroll:{step_index}:{next_step}",
                    "viewport_target_missing_after_scroll",
                )
                continue

            result, final_target, final_step = self._execute_viewport_recovery_target(
                refreshed_snapshot=refreshed_snapshot,
                target_element=target_element,
                element_id=element_id,
                action=action,
                locator_kind=locator_kind,
                runtime_context=runtime_context,
                resolved_step=resolved_step,
            )
            return self._with_recovery_metadata(
                result,
                mode="viewport",
                original_element_id=element_id,
                resolved_element_id=final_target.element_id,
                metadata={
                    "step": step_index,
                    "delta": delta,
                    "scroll_changed": int(bool(recovery_check["meaningful_change"])),
                    "next_step": next_step,
                    "resolved_step": final_step,
                },
            )

        return last_result or ActionResult(
            False,
            action,
            "viewport_recovery_failed",
            "viewport_recovery_failed",
        )

    def _run_anchor_relocation_action(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        action: str,
        locator_kind: LocatorKind | None,
        runtime_context: dict,
    ) -> ActionResult:
        refreshed_snapshot = self._capture_runtime_snapshot(snapshot)
        if refreshed_snapshot is None:
            return ActionResult(False, action, "anchor_recapture_failed", "anchor_recapture_failed")
        relocated_element = self._find_anchor_relocated_element(
            refreshed_snapshot,
            element_id,
            runtime_context,
        )
        if relocated_element is None:
            return ActionResult(False, action, "anchor_target_missing", "anchor_target_missing")
        result = self._snapshot_adapter.execute_element_action(
            snapshot=refreshed_snapshot,
            element_id=relocated_element.element_id,
            action=action,
            verify=False,
            locator_kind=locator_kind,
        )
        return self._with_recovery_metadata(
            result,
            mode="anchor",
            original_element_id=element_id,
            resolved_element_id=relocated_element.element_id,
        )

    def _run_replan_action(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        action: str,
        locator_kind: LocatorKind | None,
        runtime_context: dict,
    ) -> ActionResult:
        refreshed_snapshot = self._capture_runtime_snapshot(snapshot)
        if refreshed_snapshot is None:
            return ActionResult(False, action, "replan_recapture_failed", "replan_recapture_failed")
        replanned_element = self._find_replanned_element(
            refreshed_snapshot,
            element_id,
            runtime_context,
        )
        if replanned_element is None:
            return ActionResult(False, action, "replan_target_missing", "replan_target_missing")
        result = self._snapshot_adapter.execute_element_action(
            snapshot=refreshed_snapshot,
            element_id=replanned_element.element_id,
            action=action,
            verify=False,
            locator_kind=locator_kind,
        )
        return self._with_recovery_metadata(
            result,
            mode="replan",
            original_element_id=element_id,
            resolved_element_id=replanned_element.element_id,
        )

    def _build_viewport_recovery_deltas(
        self,
        snapshot: InteractionCanvas,
        runtime_context: dict,
    ) -> list[int]:
        direction_hint = runtime_context["interaction_hints"].get("scroll_direction")
        scroll_up = bool(runtime_context["interaction_hints"].get("scroll_up"))
        region_id = runtime_context.get("target_region_id")
        scroll_context = next(
            (item for item in snapshot.scroll_contexts if item.region_id == region_id),
            None,
        )
        viewport_height = scroll_context.viewport_height if scroll_context is not None else 480
        base_delta = max(120, min(960, int(viewport_height * 0.55)))
        if runtime_context["region_role"] in {"message_stream", "list_panel"}:
            base_delta = max(base_delta, 360)
        if runtime_context["viewport_index"] is not None and runtime_context["viewport_index"] > 0:
            base_delta = max(base_delta, 420)
        primary = base_delta if direction_hint == "up" or scroll_up else -base_delta
        return [primary, int(primary * 0.5), -int(primary * 0.35)]

    def _collect_anchor_texts(
        self,
        snapshot: InteractionCanvas,
        anchor_refs: list[str],
    ) -> list[str]:
        texts: list[str] = []
        anchor_map = {anchor.anchor_id: anchor for anchor in snapshot.anchors}
        for anchor_id in anchor_refs:
            anchor = anchor_map.get(anchor_id)
            if anchor is None:
                continue
            for element_id in anchor.element_refs:
                element = snapshot.get_element(element_id)
                if element is None:
                    continue
                for token in [element.text, element.name]:
                    if token:
                        texts.append(str(token).strip().lower())
        return [text for text in dict.fromkeys(texts) if text]

    def _collect_nearby_texts(
        self,
        snapshot: InteractionCanvas,
        element: Candidate,
    ) -> list[str]:
        if element.bounds is None or element.region_id is None:
            return []
        region = snapshot.get_region(element.region_id)
        if region is None:
            return []
        target_left, target_top, _target_right, target_bottom = element.bounds
        texts: list[str] = []
        for region_element_id in region.element_ids:
            if region_element_id == element.element_id:
                continue
            candidate = snapshot.get_element(region_element_id)
            if candidate is None or candidate.bounds is None:
                continue
            left, top, _right, bottom = candidate.bounds
            if abs(top - target_top) <= 120 or abs(bottom - target_bottom) <= 120 or abs(left - target_left) <= 220:
                for token in [candidate.text, candidate.name]:
                    if token:
                        texts.append(str(token).strip().lower())
        return [text for text in dict.fromkeys(texts) if text]

    def _collect_relation_texts(
        self,
        snapshot: InteractionCanvas,
        element: Candidate,
    ) -> list[str]:
        texts: list[str] = []
        for relation in snapshot.relations:
            other_id: str | None = None
            if relation.from_id == element.element_id:
                other_id = relation.to_id
            elif relation.to_id == element.element_id:
                other_id = relation.from_id
            if other_id is None:
                continue

            related_element = snapshot.get_element(other_id)
            if related_element is not None:
                for token in [
                    related_element.text,
                    related_element.name,
                    related_element.attributes.get("ocr_text"),
                ]:
                    if token:
                        texts.append(str(token).strip().lower())
                continue

            related_region = snapshot.get_region(other_id)
            if related_region is not None and related_region.role:
                texts.append(str(related_region.role).strip().lower())
        return [text for text in dict.fromkeys(texts) if text]

    def _collect_relation_hints(
        self,
        snapshot: InteractionCanvas,
        element: Candidate,
    ) -> list[dict[str, Any]]:
        hints: list[dict[str, Any]] = []
        for relation in snapshot.relations:
            hint: dict[str, Any] | None = None
            if relation.from_id == element.element_id:
                hint = {
                    "other_id": relation.to_id,
                    "type": relation.type.value,
                    "offset": relation.offset,
                    "direction": "outgoing",
                }
            elif relation.to_id == element.element_id:
                hint = {
                    "other_id": relation.from_id,
                    "type": relation.type.value,
                    "offset": relation.offset,
                    "direction": "incoming",
                }
            if hint is not None:
                hints.append(hint)
        return hints

    def _evaluate_viewport_recovery_snapshot(
        self,
        before_snapshot: InteractionCanvas,
        refreshed_snapshot: InteractionCanvas,
        element_id: str,
        runtime_context: dict,
    ) -> dict[str, Any]:
        diff = self._verifier.compare_snapshots(
            before_snapshot,
            refreshed_snapshot,
            target_element_id=element_id,
        )
        target_element = self._find_anchor_relocated_element(
            refreshed_snapshot,
            element_id,
            runtime_context,
        )
        if target_element is None:
            target_element = self._find_replanned_element(
                refreshed_snapshot,
                element_id,
                runtime_context,
            )
        return {
            "meaningful_change": bool(
                diff.verified
                or (diff.details or {}).get("structure_changed")
                or (diff.details or {}).get("before_scroll_evidence") != (diff.details or {}).get("after_scroll_evidence")
            ),
            "target_element": target_element,
            "diff_details": diff.details or {},
        }

    def _decide_viewport_recovery_next_step(
        self,
        recovery_check: dict[str, Any],
        runtime_context: dict,
    ) -> str:
        if recovery_check["target_element"] is not None:
            return "execute"
        if not recovery_check["meaningful_change"]:
            return "continue_scroll"
        if runtime_context.get("anchor_refs") or runtime_context["interaction_hints"].get("anchor_relocate"):
            return "anchor_relocate"
        if (
            runtime_context.get("replan_hint")
            or runtime_context["interaction_hints"].get("expected_effect") in {"navigate", "open_dialog", "open_panel", "open_detail"}
            or runtime_context.get("structure_evidence_score", 0.0) >= 0.75
        ):
            return "replan"
        return "continue_scroll"

    def _resolve_viewport_recovery_target(
        self,
        refreshed_snapshot: InteractionCanvas,
        element_id: str,
        runtime_context: dict,
        next_step: str,
        fallback_target: Candidate | None,
    ) -> tuple[Candidate | None, str]:
        if fallback_target is not None:
            return fallback_target, "execute"

        if next_step == "anchor_relocate":
            target = self._find_anchor_relocated_element(
                refreshed_snapshot,
                element_id,
                runtime_context,
            )
            if target is not None:
                return target, "anchor_relocate"
            target = self._find_replanned_element(
                refreshed_snapshot,
                element_id,
                runtime_context,
            )
            if target is not None:
                return target, "replan"
            return None, "anchor_relocate"

        if next_step == "replan":
            target = self._find_replanned_element(
                refreshed_snapshot,
                element_id,
                runtime_context,
            )
            if target is not None:
                return target, "replan"
            return None, "replan"

        return None, next_step

    def _execute_viewport_recovery_target(
        self,
        refreshed_snapshot: InteractionCanvas,
        target_element: Candidate,
        element_id: str,
        action: str,
        locator_kind: LocatorKind | None,
        runtime_context: dict,
        resolved_step: str,
    ) -> tuple[ActionResult, Candidate, str]:
        primary_result = self._snapshot_adapter.execute_element_action(
            snapshot=refreshed_snapshot,
            element_id=target_element.element_id,
            action=action,
            verify=False,
            locator_kind=locator_kind,
        )
        if primary_result.success:
            return primary_result, target_element, resolved_step

        if resolved_step == "anchor_relocate":
            replanned = self._find_replanned_element(
                refreshed_snapshot,
                element_id,
                runtime_context,
            )
            if replanned is not None and replanned.element_id != target_element.element_id:
                fallback_result = self._snapshot_adapter.execute_element_action(
                    snapshot=refreshed_snapshot,
                    element_id=replanned.element_id,
                    action=action,
                    verify=False,
                    locator_kind=locator_kind,
                )
                return fallback_result, replanned, "replan"

        return primary_result, target_element, resolved_step

    def _find_anchor_relocated_element(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        runtime_context: dict,
    ) -> Candidate | None:
        current = snapshot.get_element(element_id)
        candidates = [element for element in snapshot.elements if element.interactable]
        if current is not None and current.anchor_ids:
            same_anchor = [
                element
                for element in candidates
                if set(element.anchor_ids) & set(runtime_context.get("anchor_refs") or [])
            ]
            if same_anchor:
                return max(same_anchor, key=lambda item: self._score_relocation_candidate(item, runtime_context))
        scored = [
            (self._score_relocation_candidate(element, runtime_context), element)
            for element in candidates
        ]
        scored = [item for item in scored if item[0] > 0]
        if not scored:
            return None
        return max(scored, key=lambda item: item[0])[1]

    def _find_replanned_element(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        runtime_context: dict | None = None,
    ) -> Candidate | None:
        runtime_context = runtime_context or {}
        current = snapshot.get_element(element_id)
        if current is not None and current.interactable:
            return current
        scored = [
            (self._score_relocation_candidate(element, runtime_context), element)
            for element in snapshot.elements
            if element.interactable
        ]
        scored = [item for item in scored if item[0] > 0]
        if not scored:
            return None
        return max(scored, key=lambda item: item[0])[1]

    def _score_relocation_candidate(
        self,
        element: Candidate,
        runtime_context: dict,
    ) -> float:
        score = 0.0
        target_role = runtime_context.get("target_role")
        if target_role and element.semantic_role.value == target_role:
            score += 4.0
        if runtime_context.get("content_group_id") and element.content_group_id == runtime_context["content_group_id"]:
            score += 2.5
        if runtime_context.get("target_region_id") and element.region_id == runtime_context["target_region_id"]:
            score += 2.0
        texts = {
            str(token).strip().lower()
            for token in [
                element.text,
                element.name,
                element.attributes.get("ocr_text"),
            ]
            if token
        }
        target_text = str(runtime_context.get("target_text") or "").strip().lower()
        if target_text and target_text in texts:
            score += 3.5
        target_name = str(runtime_context.get("target_name") or "").strip().lower()
        if target_name and target_name in texts:
            score += 2.5
        if set(runtime_context.get("anchor_texts") or []) & texts:
            score += 2.0
        if set(runtime_context.get("nearby_texts") or []) & texts:
            score += 1.5
        if set(runtime_context.get("relation_texts") or []) & texts:
            score += 1.75
        relation_hints = runtime_context.get("relation_hints") or []
        for hint in relation_hints:
            related_id = str(hint.get("other_id") or "").strip().lower()
            relation_type = str(hint.get("type") or "").strip().lower()
            if related_id and related_id == str(element.element_id).strip().lower():
                score += 1.0
            if relation_type in {"left_of", "right_of", "above", "below"}:
                score += 0.2
        target_bounds = runtime_context.get("target_bounds")
        if target_bounds is not None and element.bounds is not None:
            target_cx = (target_bounds[0] + target_bounds[2]) / 2
            target_cy = (target_bounds[1] + target_bounds[3]) / 2
            element_cx = (element.bounds[0] + element.bounds[2]) / 2
            element_cy = (element.bounds[1] + element.bounds[3]) / 2
            distance = abs(target_cx - element_cx) + abs(target_cy - element_cy)
            score += max(0.0, 1.5 - (distance / 500.0))
            for hint in relation_hints:
                offset = hint.get("offset")
                if not isinstance(offset, tuple) or len(offset) != 2:
                    continue
                expected_dx, expected_dy = offset
                actual_dx = element_cx - target_cx
                actual_dy = element_cy - target_cy
                delta_error = abs(actual_dx - expected_dx) + abs(actual_dy - expected_dy)
                score += max(0.0, 0.75 - (delta_error / 400.0))
        if runtime_context.get("region_role") == "dialog_body" and element.attributes.get("dialog_slot") in {"form_fields", "viewport"}:
            score += 1.0
        if runtime_context.get("region_role") == "viewport" and element.attributes.get("grid_role") == "row":
            score += 0.5
        if runtime_context.get("region_attributes", {}).get("widget_groups") and element.attributes.get("widget_group_id"):
            score += 0.75
        return score

    def _get_locator_risk_tier(self, locator_kind: LocatorKind) -> int:
        if locator_kind in {LocatorKind.DOM, LocatorKind.UIA}:
            return 0
        if locator_kind == LocatorKind.RELATIVE:
            return 1
        if locator_kind in {
            LocatorKind.OCR,
            LocatorKind.VISION_BBOX,
            LocatorKind.TEMPLATE_ICON,
        }:
            return 2
        if locator_kind == LocatorKind.EPHEMERAL_COORD:
            return 3
        return 4

    def _get_shortcut_for_element(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        action: str,
    ) -> str | None:
        if action != "click":
            return None

        element = snapshot.get_element(element_id)
        if element is None:
            return None

        shortcut = element.attributes.get("shortcut")
        if isinstance(shortcut, str) and shortcut:
            return shortcut

        if element.semantic_role == SemanticRole.SEND_BUTTON:
            if snapshot.surface.surface_type in {
                SurfaceType.BROWSER,
                SurfaceType.ELECTRON_WEBVIEW,
            }:
                return "^{ENTER}"
            return "{ENTER}"

        if element.semantic_role == SemanticRole.SUBMIT_BUTTON:
            return "{ENTER}"

        return None

    def _to_outcome(
        self,
        result: ActionResult,
        action: str,
        strategy: str,
        attempts: int = 1,
    ) -> ActionOutcome:
        return ActionOutcome(
            success=result.success,
            action=action,
            message=result.details,
            attempts=attempts,
            strategy=strategy,
            error=result.error,
        )

    def _finalize_snapshot_outcome(
        self,
        outcome: ActionOutcome,
        before_snapshot: InteractionCanvas,
        element_id: str,
        action: str,
        verify: bool,
        recapture_after: bool,
        recapture_policy: str,
    ) -> ActionOutcome:
        if not (verify and recapture_after and outcome.success):
            return outcome

        verification_element_id = self._resolve_verification_target_id(outcome, element_id)
        after_snapshot = self._capture_runtime_snapshot(before_snapshot)
        if after_snapshot is None:
            outcome.verification_details = {
                "mode": "snapshot_diff",
                "status": "recapture_failed",
                "recapture_policy": recapture_policy,
                "verification_element_id": verification_element_id,
            }
            return outcome

        verification = self._verifier.verify_action_effect(
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            target_element_id=verification_element_id,
            action=action,
        )
        existing_details = dict(outcome.verification_details or {})
        outcome.verification_details = dict(verification.details or {})
        outcome.verification_details.update(existing_details)
        outcome.verification_details["mode"] = "snapshot_diff"
        outcome.verification_details["status"] = verification.message
        outcome.verification_details["after_canvas_id"] = after_snapshot.canvas_id
        outcome.verification_details["recapture_policy"] = recapture_policy
        outcome.verification_details["verification_element_id"] = verification_element_id
        self._build_runtime_trace(outcome)
        if not verification.verified:
            outcome.success = False
            outcome.error = "snapshot_unchanged"
            if not outcome.message:
                outcome.message = "snapshot_unchanged"
        return outcome

    def _resolve_verification_target_id(
        self,
        outcome: ActionOutcome,
        fallback_element_id: str,
    ) -> str:
        recovery_execution = (outcome.verification_details or {}).get("recovery_execution") or {}
        resolved = recovery_execution.get("resolved")
        if isinstance(resolved, str) and resolved:
            return resolved
        return fallback_element_id

    def _resolve_recapture_policy(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        action: str,
        recapture_after: bool | None,
    ) -> tuple[bool, str]:
        if recapture_after is True:
            return True, "forced"
        if recapture_after is False:
            return False, "disabled"
        if self._should_recapture_after(snapshot, element_id, action):
            return True, "auto_semantic"
        return False, "auto_skip"

    def _should_recapture_after(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        action: str,
    ) -> bool:
        if action not in {"click", "double_click", "right_click"}:
            return False

        element = snapshot.get_element(element_id)
        if element is None:
            return False

        if element.semantic_role in {
            SemanticRole.SEND_BUTTON,
            SemanticRole.SUBMIT_BUTTON,
            SemanticRole.TAB,
            SemanticRole.MENU_ITEM,
            SemanticRole.NAV_ITEM,
            SemanticRole.TOGGLE_BUTTON,
            SemanticRole.LINK,
            SemanticRole.LIST_ITEM,
            SemanticRole.CHAT_ITEM,
        }:
            return True

        return (
            snapshot.surface.surface_type in {
                SurfaceType.BROWSER,
                SurfaceType.ELECTRON_WEBVIEW,
            }
            and element.semantic_role in {
                SemanticRole.BUTTON,
                SemanticRole.ICON_BUTTON,
            }
        )

    def _capture_runtime_snapshot(
        self,
        snapshot: InteractionCanvas,
    ) -> InteractionCanvas | None:
        hwnd = snapshot.window.hwnd if snapshot.window else 0
        if not hwnd:
            return None

        try:
            zone_page = self._perception_service.analyze(hwnd)
            return self._perception_service.create_page_snapshot(
                zone_page=zone_page,
                process_name=snapshot.app.process_name,
                process_path=snapshot.app.exe_path,
            )
        except Exception:
            return None

    def _with_recovery_metadata(
        self,
        result: ActionResult,
        mode: str,
        original_element_id: str,
        resolved_element_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> ActionResult:
        metadata_items = {
            "mode": mode,
            "original": original_element_id,
            "resolved": resolved_element_id,
        }
        if metadata:
            metadata_items.update({key: value for key, value in metadata.items() if value is not None})
        metadata_text = " ".join(f"{key}={value}" for key, value in metadata_items.items())
        metadata = f"[recovery {metadata_text}]"
        details = result.details or ""
        if details:
            details = f"{details} {metadata}"
        else:
            details = metadata
        return ActionResult(
            success=result.success,
            action=result.action,
            details=details,
            error=result.error,
        )

    def _merge_recovery_execution_details(
        self,
        outcome: ActionOutcome,
        result: ActionResult | None,
    ) -> None:
        if result is None or not result.details:
            return
        marker = "[recovery "
        if marker not in result.details:
            return
        metadata_block = result.details[result.details.index(marker) + 1 :].rstrip("]")
        parts = metadata_block.split()
        parsed: dict[str, Any] = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            parsed[key] = value
        if not parsed:
            return
        if outcome.verification_details is None:
            outcome.verification_details = {}
        outcome.verification_details["recovery_execution"] = parsed

    def _build_runtime_trace(self, outcome: ActionOutcome) -> None:
        details = outcome.verification_details
        if not details:
            return
        recovery = details.get("recovery") or {}
        recovery_execution = details.get("recovery_execution") or {}
        attempt_trace = recovery.get("attempt_trace") or []
        last_attempt = attempt_trace[-1] if attempt_trace else None
        trace = {
            "strategy": outcome.strategy,
            "attempts": outcome.attempts,
            "terminal_reason": recovery.get("reason"),
            "terminal_strategy": recovery.get("terminal_strategy") or recovery.get("strategy"),
            "next_suggestion": recovery.get("next_suggestion"),
            "terminal_error": outcome.error,
            "escalation_path": recovery.get("escalation_path"),
            "attempt_trace": recovery.get("attempt_trace"),
            "recovery_mode": recovery_execution.get("mode"),
            "original_element_id": recovery_execution.get("original"),
            "resolved_element_id": recovery_execution.get("resolved"),
            "verification_element_id": details.get("verification_element_id"),
            "verification_status": details.get("status"),
            "verification_policy": details.get("policy"),
            "content_subtype": details.get("content_subtype"),
            "last_attempt_stage": last_attempt.get("stage") if last_attempt else None,
            "last_attempt_candidate": last_attempt.get("candidate") if last_attempt else None,
            "last_attempt_error": last_attempt.get("error") if last_attempt else None,
            "last_attempt_verification": last_attempt.get("verification_message") if last_attempt else None,
            "final_message": outcome.message,
            "success": outcome.success,
        }
        details["runtime_trace"] = {key: value for key, value in trace.items() if value is not None}
