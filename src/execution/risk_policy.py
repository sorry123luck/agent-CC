"""Minimal runtime risk policy for browser and webview automation."""

from __future__ import annotations

from dataclasses import dataclass

from src.perception.page_compiler_models import (
    Candidate,
    InteractionCanvas,
    RiskLevel,
    RiskTag,
    SemanticRole,
    SurfaceType,
)


@dataclass
class RiskPolicyDecision:
    allowed: bool
    risk_level: str
    mode: str
    reason: str
    policy_source: str = "default"


class RiskPolicyEngine:
    """Apply minimal guardrails before runtime execution."""

    def assess(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        action: str,
    ) -> RiskPolicyDecision:
        surface_type = snapshot.surface.surface_type
        if surface_type not in {SurfaceType.BROWSER, SurfaceType.ELECTRON_WEBVIEW}:
            return RiskPolicyDecision(
                allowed=True,
                risk_level="low",
                mode="native_safe",
                reason="non_browser_surface",
            )

        element = snapshot.get_element(element_id)
        if element is None:
            return RiskPolicyDecision(
                allowed=False,
                risk_level="unknown",
                mode="blocked",
                reason="element_not_found",
                policy_source="element_lookup",
            )

        mode, policy_source, policy = self._resolve_mode(snapshot, element_id)
        risk_level = self._classify_risk(snapshot, element_id, action)
        if element.attributes.get("risk_approved") is True:
            return RiskPolicyDecision(
                allowed=True,
                risk_level=risk_level,
                mode=mode,
                reason="risk_approved",
                policy_source=policy_source,
            )

        if policy.get("api_available") is True or mode == "api_first":
            return RiskPolicyDecision(
                allowed=False,
                risk_level=risk_level,
                mode="api_first",
                reason="browser_api_first",
                policy_source=policy_source,
            )

        if policy.get("require_confirmation") is True:
            return RiskPolicyDecision(
                allowed=False,
                risk_level=risk_level,
                mode=mode,
                reason="browser_confirmation_required",
                policy_source=policy_source,
            )

        if not self._mode_allowed(mode, policy):
            return RiskPolicyDecision(
                allowed=False,
                risk_level=risk_level,
                mode=mode,
                reason="browser_mode_not_allowed",
                policy_source=policy_source,
            )

        if not self._risk_within_limit(risk_level, policy.get("max_auto_risk_level")):
            return RiskPolicyDecision(
                allowed=False,
                risk_level=risk_level,
                mode=mode,
                reason="browser_risk_above_policy_limit",
                policy_source=policy_source,
            )

        if mode == "read_only" and risk_level != "low":
            return RiskPolicyDecision(
                allowed=False,
                risk_level=risk_level,
                mode=mode,
                reason="browser_read_only_block",
                policy_source=policy_source,
            )

        if mode in {"assist_fill", "confirm_then_execute", "guarded_automation"} and risk_level in {
            "high",
            "critical",
        }:
            return RiskPolicyDecision(
                allowed=False,
                risk_level=risk_level,
                mode=mode,
                reason="browser_high_risk_block",
                policy_source=policy_source,
            )

        return RiskPolicyDecision(
            allowed=True,
            risk_level=risk_level,
            mode=mode,
            reason="browser_policy_allow",
            policy_source=policy_source,
        )

    def _resolve_mode(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
    ) -> tuple[str, str, dict]:
        element = snapshot.get_element(element_id)
        if element is None:
            return "guarded_automation", "default", {}

        element_policy = element.attributes.get("workflow_policy")
        if isinstance(element_policy, dict):
            mode = self._policy_mode(element_policy, default="guarded_automation")
            return mode, "element.workflow_policy", element_policy

        workflow_policy = snapshot.artifacts.get("workflow_policy")
        if isinstance(workflow_policy, dict):
            mode = self._policy_mode(workflow_policy, default="guarded_automation")
            return mode, "snapshot.workflow_policy", workflow_policy

        site_policy = snapshot.artifacts.get("site_policy")
        if isinstance(site_policy, dict):
            mode = self._policy_mode(site_policy, default="guarded_automation")
            return mode, "snapshot.site_policy", site_policy

        element_mode = element.attributes.get("browser_risk_mode")
        if isinstance(element_mode, str) and element_mode:
            return element_mode, "element.browser_risk_mode", {}

        artifact_mode = snapshot.artifacts.get("browser_risk_mode")
        if isinstance(artifact_mode, str) and artifact_mode:
            return artifact_mode, "snapshot.browser_risk_mode", {}

        return "guarded_automation", "default", {}

    def _policy_mode(self, policy: dict, default: str) -> str:
        mode = policy.get("mode")
        if isinstance(mode, str) and mode:
            return mode
        return default

    def _mode_allowed(self, mode: str, policy: dict) -> bool:
        allowed_modes = policy.get("allowed_modes")
        if not isinstance(allowed_modes, list) or not allowed_modes:
            return True
        return mode in allowed_modes

    def _risk_within_limit(
        self,
        risk_level: str,
        max_auto_risk_level,
    ) -> bool:
        if not isinstance(max_auto_risk_level, str) or not max_auto_risk_level:
            return True
        order = {
            "low": 0,
            "medium": 1,
            "high": 2,
            "critical": 3,
            "unknown": 4,
        }
        return order.get(risk_level, 4) <= order.get(max_auto_risk_level, 4)

    def _classify_risk(
        self,
        snapshot: InteractionCanvas,
        element_id: str,
        action: str,
    ) -> str:
        element = snapshot.get_element(element_id)
        if element is None:
            return "unknown"

        if action not in {"click", "double_click", "right_click"}:
            return "medium"

        text_blobs = [
            element.text or "",
            element.name or "",
            str(element.attributes.get("label", "")),
        ]
        combined_text = " ".join(text_blobs).lower()
        critical_terms = [
            "delete",
            "publish",
            "pay",
            "buy",
            "submit order",
            "删除",
            "发布",
            "支付",
            "付款",
            "购买",
            "下单",
        ]
        if element.attributes.get("sensitive") is True or any(term in combined_text for term in critical_terms):
            return "critical"

        if element.semantic_role in {
            SemanticRole.SEND_BUTTON,
            SemanticRole.SUBMIT_BUTTON,
        }:
            return "high"

        if element.semantic_role in {
            SemanticRole.BUTTON,
            SemanticRole.ICON_BUTTON,
            SemanticRole.TOGGLE_BUTTON,
            SemanticRole.MENU_ITEM,
        }:
            return "medium"

        if element.semantic_role in {
            SemanticRole.LINK,
            SemanticRole.TAB,
            SemanticRole.LIST_ITEM,
            SemanticRole.CHAT_ITEM,
            SemanticRole.NAV_ITEM,
        }:
            return "low"

        return "medium"

    def classify_risk_tags(self, candidate: Candidate) -> list[str]:
        """根据候选的文字和语义角色自动打风险标签。"""
        tags: list[str] = []
        text = " ".join([
            candidate.text or "",
            candidate.name or "",
            str(candidate.attributes.get("label", "")),
        ]).lower()

        # Keyword-based tag detection
        tag_keywords: list[tuple[str, list[str]]] = [
            (RiskTag.SEND.value, ["发送", "send", "提交", "submit"]),
            (RiskTag.DELETE.value, ["删除", "delete", "移除", "remove"]),
            (RiskTag.PAYMENT.value, ["支付", "付款", "pay", "purchase", "转账", "transfer"]),
            (RiskTag.SUBMIT.value, ["提交", "submit", "确认", "confirm"]),
            (RiskTag.PUBLISH.value, ["发布", "publish", "公开", "public"]),
            (RiskTag.LOGIN.value, ["登录", "login", "signin", "sign in"]),
            (RiskTag.CONFIG_CHANGE.value, ["设置", "settings", "配置", "config", "preferences"]),
        ]
        for tag, keywords in tag_keywords:
            if any(kw in text for kw in keywords):
                tags.append(tag)

        # Semantic role-based tag detection
        role_tags: dict[str, list[str]] = {
            SemanticRole.SEND_BUTTON.value: [RiskTag.SEND.value],
            SemanticRole.SUBMIT_BUTTON.value: [RiskTag.SUBMIT.value],
        }
        role_tag = role_tags.get(candidate.semantic_role.value if hasattr(candidate.semantic_role, 'value') else str(candidate.semantic_role), [])
        for t in role_tag:
            if t not in tags:
                tags.append(t)

        return tags

    def classify_risk_level(self, candidate: Candidate) -> str:
        """根据风险标签和语义角色判断风险等级。"""
        tags = set(candidate.risk_tags) if candidate.risk_tags else set(self.classify_risk_tags(candidate))

        if tags & {RiskTag.PAYMENT.value, RiskTag.PUBLISH.value}:
            return RiskLevel.L3.value
        if tags & {RiskTag.SEND.value, RiskTag.DELETE.value, RiskTag.LOGIN.value}:
            return RiskLevel.L2.value
        if tags & {RiskTag.SUBMIT.value, RiskTag.CONFIG_CHANGE.value}:
            return RiskLevel.L2.value

        # Fallback to semantic role
        role = candidate.semantic_role
        if role in {SemanticRole.SEND_BUTTON, SemanticRole.SUBMIT_BUTTON}:
            return RiskLevel.L2.value
        if role in {SemanticRole.BUTTON, SemanticRole.ICON_BUTTON, SemanticRole.MENU_ITEM}:
            return RiskLevel.L1.value

        return RiskLevel.L0.value
