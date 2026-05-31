"""
Page Class Classifier — P3-B

page_class 命名格式：<app>/<workflow>/<state>/<variant>

粒度原则：可执行动作集合变化时才切新 class。
输入：InteractionCanvas（含 app / surface / regions / elements）
输出：page_class 字符串 + class_confidence
"""

from dataclasses import dataclass, field
from typing import Any

from src.perception.page_compiler_models import (
    InteractionCanvas,
    SurfaceType,
    ContentAreaSubtype,
    SemanticRole,
)


@dataclass
class PageClassEvidence:
    """Page class 证据"""
    feature: str
    weight: float
    value: bool | float
    description: str


@dataclass
class PageClassResult:
    """Page class 分类结果"""
    page_class: str                    # e.g. "wechat/chat/main/default"
    class_confidence: float
    evidence: list[PageClassEvidence] = field(default_factory=list)
    fingerprint: dict[str, str] = field(default_factory=dict)  # layout_hash, anchor_hash, text_hash

    @property
    def app(self) -> str:
        return self.page_class.split("/")[0] if "/" in self.page_class else "unknown"

    @property
    def workflow(self) -> str:
        return self.page_class.split("/")[1] if self.page_class.count("/") >= 1 else "unknown"

    @property
    def state(self) -> str:
        return self.page_class.split("/")[2] if self.page_class.count("/") >= 2 else "unknown"

    @property
    def variant(self) -> str:
        return self.page_class.split("/")[3] if self.page_class.count("/") >= 3 else "default"


class PageClassClassifier:
    """
    Page Class 分类器

    命名格式：<app>/<workflow>/<state>/<variant>
    - app: 软件标识（进程名映射）
    - workflow: 工作流阶段（如 chat / settings / profile）
    - state: 页面状态（如 main / empty / loading / dialog）
    - variant: 布局变体（如 default / compact / wide）

    判断逻辑：
    1. 从 app 字段获取或从进程名推断
    2. 从 surface_type + content_area_subtype 推断 workflow
    3. 从元素分布和状态标志推断 state
    4. 从布局特征推断 variant
    """

    DEFAULT_CONFIDENCE = 0.30

    # 进程名到 app ID 的映射
    APP_ID_MAP: dict[str, str] = {
        "wechat.exe": "wechat",
        "weixin.exe": "wechat",
        "dingtalk.exe": "dingtalk",
        "taobao.exe": "taobao",
        "lark.exe": "lark",
        "feishu.exe": "lark",
        "wework.exe": "wework",
        "vscode.exe": "vscode",
        "code.exe": "vscode",
        "chrome.exe": "chrome",
        "msedge.exe": "edge",
        "firefox.exe": "firefox",
        "notion.exe": "notion",
        "obsidian.exe": "obsidian",
        "slack.exe": "slack",
        "teams.exe": "teams",
        "discord.exe": "discord",
    }

    # Surface + ContentAreaSubtype 到 workflow 的映射
    WORKFLOW_MAP: dict[tuple[SurfaceType, ContentAreaSubtype], str] = {
        (SurfaceType.NATIVE_UIA, ContentAreaSubtype.CHAT): "chat",
        (SurfaceType.NATIVE_UIA, ContentAreaSubtype.FORM): "form",
        (SurfaceType.NATIVE_UIA, ContentAreaSubtype.LIST_DETAIL): "list_detail",
        (SurfaceType.NATIVE_UIA, ContentAreaSubtype.EDITOR): "editor",
        (SurfaceType.NATIVE_UIA, ContentAreaSubtype.DASHBOARD): "dashboard",
        (SurfaceType.NATIVE_UIA, ContentAreaSubtype.CANVAS_DOC_VIEWER): "document",
        (SurfaceType.BROWSER, ContentAreaSubtype.CHAT): "web_chat",
        (SurfaceType.BROWSER, ContentAreaSubtype.FORM): "web_form",
        (SurfaceType.BROWSER, ContentAreaSubtype.LIST_DETAIL): "web_list",
        (SurfaceType.BROWSER, ContentAreaSubtype.EDITOR): "web_editor",
        (SurfaceType.ELECTRON_WEBVIEW, ContentAreaSubtype.CHAT): "app_chat",
        (SurfaceType.ELECTRON_WEBVIEW, ContentAreaSubtype.FORM): "app_form",
        (SurfaceType.ELECTRON_WEBVIEW, ContentAreaSubtype.LIST_DETAIL): "app_list",
        (SurfaceType.CANVAS_SELF_DRAWN, ContentAreaSubtype.CANVAS_DOC_VIEWER): "viewer",
    }

    def classify(self, snapshot: InteractionCanvas) -> PageClassResult:
        """
        分类入口

        Args:
            snapshot: InteractionCanvas

        Returns:
            PageClassResult 包含 page_class + confidence + evidence
        """
        evidence: list[PageClassEvidence] = []

        # === 1. 确定 app ===
        app = self._resolve_app(snapshot, evidence)

        # === 2. 确定 workflow ===
        workflow = self._resolve_workflow(snapshot, evidence)

        # === 3. 确定 state ===
        state = self._resolve_state(snapshot, evidence)

        # === 4. 确定 variant ===
        variant = self._resolve_variant(snapshot, evidence)

        # === 5. 组合 page_class ===
        page_class = f"{app}/{workflow}/{state}/{variant}"
        confidence = self._compute_confidence(evidence)

        # === 6. 生成指纹 ===
        fingerprint = self._generate_fingerprint(snapshot)

        return PageClassResult(
            page_class=page_class,
            class_confidence=round(confidence, 3),
            evidence=evidence,
            fingerprint=fingerprint,
        )

    def _resolve_app(self, snapshot: InteractionCanvas, evidence: list[PageClassEvidence]) -> str:
        """从 app 字段或进程名推断 app"""
        # 优先用 snapshot.app.app_id
        if snapshot.app and snapshot.app.app_id:
            evidence.append(PageClassEvidence(
                feature="app_id_field",
                weight=0.40,
                value=True,
                description=f"使用 app_id 字段: {snapshot.app.app_id}"
            ))
            return snapshot.app.app_id

        # 降级：用进程名映射
        if snapshot.app and snapshot.app.process_name:
            pn = snapshot.app.process_name.lower()
            if pn in self.APP_ID_MAP:
                app_id = self.APP_ID_MAP[pn]
                evidence.append(PageClassEvidence(
                    feature="process_name_mapped",
                    weight=0.30,
                    value=True,
                    description=f"从进程名映射到 app_id: {app_id}"
                ))
                return app_id
            # 去掉 .exe 作为 app_id
            app_id = pn.replace(".exe", "")
            evidence.append(PageClassEvidence(
                feature="process_name_fallback",
                weight=0.20,
                value=True,
                description=f"从进程名推断 app_id: {app_id}"
            ))
            return app_id

        evidence.append(PageClassEvidence(
            feature="no_app_info",
            weight=0.10,
            value=False,
            description="无 app 信息，使用 unknown"
        ))
        return "unknown"

    def _resolve_workflow(
        self, snapshot: InteractionCanvas, evidence: list[PageClassEvidence]
    ) -> str:
        """从 surface_type + content_area_subtype 推断 workflow"""
        surface_type = snapshot.surface.surface_type

        # 尝试从 content_area 获取 subtype
        content_area = self._find_content_area(snapshot)
        subtype = ContentAreaSubtype.UNKNOWN
        if content_area:
            subtype = content_area.get("subtype", ContentAreaSubtype.UNKNOWN)

        key = (surface_type, subtype)
        if key in self.WORKFLOW_MAP:
            workflow = self.WORKFLOW_MAP[key]
            evidence.append(PageClassEvidence(
                feature="workflow_from_surface_subtype",
                weight=0.35,
                value=True,
                description=f"从 (surface, subtype) 推断 workflow: {workflow}"
            ))
            return workflow

        # 降级：从 surface_type 推断
        workflow_fallback = {
            SurfaceType.NATIVE_UIA: "gui",
            SurfaceType.BROWSER: "web",
            SurfaceType.ELECTRON_WEBVIEW: "app",
            SurfaceType.CANVAS_SELF_DRAWN: "canvas",
            SurfaceType.UNKNOWN: "unknown",
        }
        workflow = workflow_fallback.get(surface_type, "unknown")
        evidence.append(PageClassEvidence(
            feature="workflow_from_surface",
            weight=0.25,
            value=True,
            description=f"从 surface_type 推断 workflow: {workflow}"
        ))
        return workflow

    def _resolve_state(
        self, snapshot: InteractionCanvas, evidence: list[PageClassEvidence]
    ) -> str:
        """从元素分布和状态标志推断 state"""
        # 优先用 state_flags
        if snapshot.page and snapshot.page.state_flags:
            if snapshot.page.state_flags.get("editable"):
                evidence.append(PageClassEvidence(
                    feature="state_from_flags_editable",
                    weight=0.30,
                    value=True,
                    description="state_flags 指示 editable → main"
                ))
                return "main"
            if snapshot.page.state_flags.get("dialog_open"):
                evidence.append(PageClassEvidence(
                    feature="state_from_flags_dialog",
                    weight=0.35,
                    value=True,
                    description="state_flags 指示 dialog_open → dialog"
                ))
                return "dialog"
            if snapshot.page.state_flags.get("loading"):
                evidence.append(PageClassEvidence(
                    feature="state_from_flags_loading",
                    weight=0.30,
                    value=True,
                    description="state_flags 指示 loading → loading"
                ))
                return "loading"

        # 降级：从元素数量推断
        element_count = len(snapshot.elements)
        if element_count == 0:
            evidence.append(PageClassEvidence(
                feature="state_from_element_count",
                weight=0.15,
                value=True,
                description="无元素 → empty"
            ))
            return "empty"
        elif element_count < 5:
            evidence.append(PageClassEvidence(
                feature="state_from_element_count",
                weight=0.15,
                value=True,
                description=f"元素极少({element_count}) → sparse"
            ))
            return "sparse"

        evidence.append(PageClassEvidence(
            feature="state_default",
            weight=0.10,
            value=True,
            description="默认 state: main"
        ))
        return "main"

    def _resolve_variant(
        self, snapshot: InteractionCanvas, evidence: list[PageClassEvidence]
    ) -> str:
        """从布局特征推断 variant"""
        content_area = self._find_content_area(snapshot)
        if not content_area:
            return "default"

        # 检查 split layout
        bounds = content_area.get("bounds")
        if bounds:
            l, t, r, b = bounds
            width = r - l
            height = b - t
            if width > 800:
                evidence.append(PageClassEvidence(
                    feature="variant_wide",
                    weight=0.15,
                    value=True,
                    description=f"内容区宽度 {width} > 800 → wide"
                ))
                return "wide"
            elif width < 400:
                evidence.append(PageClassEvidence(
                    feature="variant_compact",
                    weight=0.15,
                    value=True,
                    description=f"内容区宽度 {width} < 400 → compact"
                ))
                return "compact"

        return "default"

    def _find_content_area(self, snapshot: InteractionCanvas) -> dict[str, Any] | None:
        """从 regions 中找到 content_area"""
        for region in snapshot.regions:
            if region.role == "content_area":
                return {
                    "region_id": region.region_id,
                    "subtype": region.subtype,
                    "bounds": region.bounds,
                    "element_count": len(region.element_ids),
                }
        return None

    def _compute_confidence(self, evidence: list[PageClassEvidence]) -> float:
        """根据证据计算置信度"""
        if not evidence:
            return self.DEFAULT_CONFIDENCE

        # 取最高权重证据的加权和
        total_weight = sum(e.weight for e in evidence)
        if total_weight == 0:
            return self.DEFAULT_CONFIDENCE

        weighted = sum(e.weight * (1.0 if e.value is True else float(e.value)) for e in evidence)
        return min(weighted / total_weight * 1.2, 1.0)

    def _generate_fingerprint(self, snapshot: InteractionCanvas) -> dict[str, str]:
        """生成页面指纹"""
        import hashlib

        # Layout hash: 基于 region 结构
        region_signatures = []
        for r in sorted(snapshot.regions, key=lambda x: x.region_id):
            region_signatures.append(f"{r.role}:{r.subtype.value if r.subtype else 'none'}")
        layout_str = "|".join(region_signatures)
        layout_hash = hashlib.md5(layout_str.encode()).hexdigest()[:12]

        # Element count hash
        elem_count = len(snapshot.elements)
        count_hash = hashlib.md5(str(elem_count).encode()).hexdigest()[:8]

        # Surface type hash
        surface_str = snapshot.surface.surface_type.value
        surface_hash = hashlib.md5(surface_str.encode()).hexdigest()[:8]

        return {
            "layout_hash": layout_hash,
            "element_count_hash": count_hash,
            "surface_hash": surface_hash,
        }
