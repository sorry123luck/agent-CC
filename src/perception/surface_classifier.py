"""
Surface Type Classifier — P3-C

四层证据打分判定当前窗口的 surface_type：
1. 进程签名层：进程名、模块名判断
2. UIA 树模式层：控件树特征判断
3. DOM 可用性层：DOM bridge 是否存在
4. 视觉特征层：截图视觉特征辅助判断

输出：SurfaceType + confidence + evidence列表

按 surface_type 的 Locator 优先级：
- native_uia:     uia > relative > ocr > template_icon > vision_bbox > ephemeral_coord
- browser:       dom > relative > vision_bbox > ocr > template_icon > ephemeral_coord
- electron_webview: 外壳 uia，内核 dom 或 vision_bbox
- canvas_self_drawn: vision_bbox > ocr > template_icon > relative > ephemeral_coord
"""

from dataclasses import dataclass

from src.perception.page_compiler_models import SurfaceType, SurfaceEvidence


# =============================================================================
# 进程签名数据库（可扩展）
# =============================================================================

KNOWN_BROWSER_PROCESSES: set[str] = {
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "brave.exe",
    "opera.exe",
    "chromium.exe",
}

KNOWN_ELECTRON_PROCESS_PATTERNS: set[str] = {
    # Electron 应用常见进程名模式
    "electron.exe",
    # 已知的 Electron 应用
    "teams.exe",       # Microsoft Teams
    "slack.exe",        # Slack
    "discord.exe",      # Discord
    "notion.exe",      # Notion
    "figma.exe",        # Figma
    "vscode.exe",       # VS Code (Electron shell)
    "code.exe",         # VS Code (alternate process name)
    "obsidian.exe",     # Obsidian
    "obsidian",         # Obsidian (alternate)
}

# WebView2 内核特征进程（Chromium 内核）
KNOWN_WEBVIEW2_APPS: set[str] = {
    "wechat.exe",       # 微信 Windows 版使用 WebView2
    "taobao.exe",       # 淘宝
    "dingtalk.exe",     # 钉钉
    "wework.exe",       # 企业微信
    "lark.exe",         # 飞书
    "weixin.exe",       # 微信（备选进程名）
}

# Canvas / 自绘应用特征进程
KNOWN_CANVAS_SELF_DRAWN_PATTERNS: set[str] = {
    "paint.exe",        # Windows 画图
    "photoshop.exe",    # Photoshop
    "gimp.exe",         # GIMP
    "blender.exe",      # Blender
    "unity.exe",        # Unity Editor
    "unreal engine.exe", # Unreal Editor
    "idea.exe",         # IntelliJ IDEA (部分 UI 自绘)
}

# UIA 树模式特征
UIA_RICH_CONTROL_PATTERNS: set[str] = {
    # 丰富的标准控件树特征，表示标准 Windows 控件
    "MenuBar",
    "ToolBar",
    "StatusBar",
    "Tree",
    "DataGrid",
    "Grid",
    "List",
    "Document",
    "Edit",
    "Button",
}

UIA_SPARSSE_PATTERNS: set[str] = {
    # 稀疏控件树特征，表示可能是 WebView / 自绘
    "WebView",
    "WebView2",
    "InternetExplorer",
    "Chrome_RenderWidgetHostHWND",
    "MozillaWindowClass",
    "Chrome_WidgetWin_0",
    "Chrome_WidgetWin_1",
}


# =============================================================================
# Surface Classification Result
# =============================================================================

@dataclass
class SurfaceClassification:
    """surface_type 分类结果"""
    surface_type: SurfaceType
    confidence: float
    evidence: list[SurfaceEvidence]
    locator_priority: list[str]  # 该 surface_type 对应的 locator 优先级

    @property
    def evidence_summary(self) -> list[str]:
        return [e.description for e in self.evidence]


# =============================================================================
# Surface Classifier
# =============================================================================

class SurfaceClassifier:
    """
    四层证据打分 surface_type 分类器

    使用流程：
    1. 对每个候选 surface_type 收集 4 层证据
    2. 加权计算每种 surface_type 的置信度
    3. 选择置信度最高的作为输出
    """

    # 进程层权重
    PROCESS_WEIGHT = 0.30

    # UIA 树模式权重
    UIA_PATTERN_WEIGHT = 0.30

    # DOM 可用性权重
    DOM_WEIGHT = 0.20

    # 视觉特征权重（最低，作为兜底）
    VISUAL_WEIGHT = 0.10

    # 无法判断时的默认
    DEFAULT_CONFIDENCE = 0.30

    def classify(
        self,
        process_name: str | None,
        process_path: str | None = None,
        uia_element_count: int = 0,
        uia_control_types: list[str] | None = None,
        has_dom_bridge: bool = False,
        dom_ready: bool = False,
        visual_score: float | None = None,
    ) -> SurfaceClassification:
        """
        分类入口

        Args:
            process_name: 进程名（e.g. "WeChat.exe"）
            process_path: 进程路径（可选，更精确）
            uia_element_count: UIA 元素数量
            uia_control_types: UIA 控件类型列表
            has_dom_bridge: 是否有 DOM bridge（e.g. Chromium IPC）
            dom_ready: DOM 是否可访问
            visual_score: 视觉分析分数（0-1，越高越像标准 GUI）

        Returns:
            SurfaceClassification 包含 surface_type + confidence + evidence
        """
        uia_control_types = uia_control_types or []
        evidence: list[SurfaceEvidence] = []

        # === 层1：进程签名 ===
        process_score = self._score_process_layer(process_name, process_path, evidence)
        is_known_electron = process_score["electron_webview"] >= 0.80
        is_known_webview2 = process_score["electron_webview"] >= 0.80
        is_known_browser = process_score["browser"] >= 0.80

        # === 层2：UIA 树模式 ===
        uia_score = self._score_uia_pattern_layer(
            uia_element_count, uia_control_types, evidence,
            is_known_electron=is_known_electron,
            is_known_browser=is_known_browser,
        )

        # === 层3：DOM 可用性 ===
        dom_score = self._score_dom_layer(has_dom_bridge, dom_ready, evidence)

        # === 层4：视觉特征 ===
        visual_s = self._score_visual_layer(visual_score, evidence)

        # === 综合打分 ===
        scores = {
            SurfaceType.NATIVE_UIA: (
                process_score["native_uia"] * self.PROCESS_WEIGHT
                + uia_score["native_uia"] * self.UIA_PATTERN_WEIGHT
                + dom_score["native_uia"] * self.DOM_WEIGHT
                + visual_s["native_uia"] * self.VISUAL_WEIGHT
            ),
            SurfaceType.BROWSER: (
                process_score["browser"] * self.PROCESS_WEIGHT
                + uia_score["browser"] * self.UIA_PATTERN_WEIGHT
                + dom_score["browser"] * self.DOM_WEIGHT
                + visual_s["browser"] * self.VISUAL_WEIGHT
            ),
            SurfaceType.ELECTRON_WEBVIEW: (
                process_score["electron_webview"] * self.PROCESS_WEIGHT
                + uia_score["electron_webview"] * self.UIA_PATTERN_WEIGHT
                + dom_score["electron_webview"] * self.DOM_WEIGHT
                + visual_s["electron_webview"] * self.VISUAL_WEIGHT
            ),
            SurfaceType.CANVAS_SELF_DRAWN: (
                process_score["canvas_self_drawn"] * self.PROCESS_WEIGHT
                + uia_score["canvas_self_drawn"] * self.UIA_PATTERN_WEIGHT
                + dom_score["canvas_self_drawn"] * self.DOM_WEIGHT
                + visual_s["canvas_self_drawn"] * self.VISUAL_WEIGHT
            ),
        }

        # 选择置信度最高的
        best_type = max(scores, key=scores.get)
        best_confidence = scores[best_type]

        # 归一化：如果最高分低于阈值，降为 UNKNOWN
        if best_confidence < 0.20:
            best_type = SurfaceType.UNKNOWN
            best_confidence = self.DEFAULT_CONFIDENCE

        return SurfaceClassification(
            surface_type=best_type,
            confidence=round(best_confidence, 3),
            evidence=evidence,
            locator_priority=self._get_locator_priority(best_type),
        )

    def _score_process_layer(
        self,
        process_name: str | None,
        process_path: str | None,
        evidence: list[SurfaceEvidence],
    ) -> dict[str, float]:
        """层1：进程签名打分"""
        scores = {
            "native_uia": 0.0,
            "browser": 0.0,
            "electron_webview": 0.0,
            "canvas_self_drawn": 0.0,
        }
        if not process_name:
            evidence.append(SurfaceEvidence(
                layer="process", feature="process_name", weight=0.0,
                value=False, description="进程名未知"
            ))
            return scores

        pn_lower = process_name.lower()

        # 浏览器
        if pn_lower in KNOWN_BROWSER_PROCESSES:
            scores["browser"] = 1.0
            evidence.append(SurfaceEvidence(
                layer="process", feature="known_browser",
                weight=self.PROCESS_WEIGHT, value=True,
                description=f"已知浏览器进程: {process_name}"
            ))
        # Electron
        elif pn_lower in KNOWN_ELECTRON_PROCESS_PATTERNS or any(
            p in pn_lower for p in KNOWN_ELECTRON_PROCESS_PATTERNS
        ):
            scores["electron_webview"] = 1.0
            evidence.append(SurfaceEvidence(
                layer="process", feature="known_electron",
                weight=self.PROCESS_WEIGHT, value=True,
                description=f"已知 Electron 应用: {process_name}"
            ))
        # WebView2 内核应用
        elif pn_lower in KNOWN_WEBVIEW2_APPS:
            scores["electron_webview"] = 0.85
            evidence.append(SurfaceEvidence(
                layer="process", feature="known_webview2_app",
                weight=self.PROCESS_WEIGHT, value=True,
                description=f"已知 WebView2 应用（Electron 等效）: {process_name}"
            ))
        # Canvas 自绘
        elif any(p in pn_lower for p in KNOWN_CANVAS_SELF_DRAWN_PATTERNS):
            scores["canvas_self_drawn"] = 1.0
            evidence.append(SurfaceEvidence(
                layer="process", feature="known_canvas_app",
                weight=self.PROCESS_WEIGHT, value=True,
                description=f"已知自绘应用: {process_name}"
            ))
        else:
            # 未知进程，默认原生 GUI
            scores["native_uia"] = 0.5
            evidence.append(SurfaceEvidence(
                layer="process", feature="unknown_process",
                weight=self.PROCESS_WEIGHT, value=0.5,
                description=f"未知进程，默认原生 GUI: {process_name}"
            ))

        return scores

    def _score_uia_pattern_layer(
        self,
        element_count: int,
        control_types: list[str],
        evidence: list[SurfaceEvidence],
        is_known_electron: bool = False,
        is_known_browser: bool = False,
    ) -> dict[str, float]:
        """层2：UIA 树模式打分"""
        scores = {
            "native_uia": 0.0,
            "browser": 0.0,
            "electron_webview": 0.0,
            "canvas_self_drawn": 0.0,
        }

        if not control_types:
            evidence.append(SurfaceEvidence(
                layer="uia_pattern", feature="no_uia_data",
                weight=0.0, value=False, description="无 UIA 控件数据"
            ))
            return scores

        ct_lower = [ct.lower() for ct in control_types]
        ct_set = set(ct_lower)

        # 检查 WebView 特征
        has_webview = any(p in ct for ct in ct_lower for p in [
            "webview", "webview2", "internetexplorer"
        ])
        has_chrome_widget = any("chrome_widgetwin" in ct or "chrome_renderer" in ct
                                  for ct in ct_lower)
        has_mozilla = any("mozillawindowclass" in ct or "firefox" in ct
                           for ct in ct_lower)

        # 检查标准控件特征
        has_standard_controls = sum(1 for p in UIA_RICH_CONTROL_PATTERNS if p.lower() in ct_set)
        has_toolbar = any("toolbar" in ct for ct in ct_lower)
        has_menubar = any("menubar" in ct for ct in ct_lower)
        has_statusbar = any("statusbar" in ct for ct in ct_lower)
        has_datagrid = any("datagrid" in ct or "grid" in ct for ct in ct_lower)

        # 元素数量特征
        is_ultra_sparse = element_count < 30
        is_sparse = element_count < 50
        is_rich_tree = element_count > 200 and has_standard_controls >= 3

        if has_webview or has_chrome_widget or has_mozilla:
            # WebView / Chromium 特征 → electron 或 browser
            if has_standard_controls >= 2 and (has_toolbar or has_menubar):
                # 有标准 chrome 控件 → Electron 外壳
                scores["electron_webview"] = 0.80
                evidence.append(SurfaceEvidence(
                    layer="uia_pattern", feature="electron_shell",
                    weight=self.UIA_PATTERN_WEIGHT, value=0.80,
                    description="UIA 树含 WebView + 标准 chrome 控件 → Electron 外壳"
                ))
            else:
                # 纯 WebView 内容 → browser
                scores["browser"] = 0.90
                evidence.append(SurfaceEvidence(
                    layer="uia_pattern", feature="browser_webview",
                    weight=self.UIA_PATTERN_WEIGHT, value=0.90,
                    description="UIA 树含 WebView 特征 → 浏览器"
                ))
        elif is_ultra_sparse and not is_known_electron and not is_known_browser:
            # 极稀疏树 + 非已知浏览器进程 → Canvas 自绘（无需 WebView 指标）
            scores["canvas_self_drawn"] = 0.65
            evidence.append(SurfaceEvidence(
                layer="uia_pattern", feature="ultra_sparse_no_standard_controls",
                weight=self.UIA_PATTERN_WEIGHT, value=0.65,
                description=f"UIA 树极稀疏({element_count}元素) + 无标准控件 + 非已知Browser进程 → Canvas 自绘"
            ))
        elif is_sparse and has_webview or has_chrome_widget:
            # 稀疏树 + 有 WebView/Chrome 指标 → 可能是 Canvas 或 Browser
            if is_known_electron or is_known_browser:
                scores["browser"] = 0.50
                evidence.append(SurfaceEvidence(
                    layer="uia_pattern", feature="sparse_webview_known_process",
                    weight=self.UIA_PATTERN_WEIGHT, value=0.50,
                    description=f"UIA 树稀疏({element_count}元素) + WebView指标 + 已知Browser进程 → Browser"
                ))
            else:
                scores["canvas_self_drawn"] = 0.55
                evidence.append(SurfaceEvidence(
                    layer="uia_pattern", feature="sparse_webview_unknown_process",
                    weight=self.UIA_PATTERN_WEIGHT, value=0.55,
                    description=f"UIA 树稀疏({element_count}元素) + WebView指标 → Canvas 自绘"
                ))
        elif is_rich_tree:
            # 丰富控件树：如果是已知 Electron 进程 → Electron 外壳；否则 → 原生 GUI
            if is_known_electron:
                scores["electron_webview"] = 0.75
                scores["native_uia"] = 0.20
                evidence.append(SurfaceEvidence(
                    layer="uia_pattern", feature="rich_standard_controls_known_electron",
                    weight=self.UIA_PATTERN_WEIGHT, value=0.75,
                    description=f"UIA 树丰富({element_count}元素) + 已知Electron进程 → Electron外壳"
                ))
            elif is_known_browser:
                scores["browser"] = 0.75
                scores["native_uia"] = 0.20
                evidence.append(SurfaceEvidence(
                    layer="uia_pattern", feature="rich_standard_controls_known_browser",
                    weight=self.UIA_PATTERN_WEIGHT, value=0.75,
                    description=f"UIA 树丰富({element_count}元素) + 已知Browser进程 → Browser"
                ))
            else:
                scores["native_uia"] = 0.90
                evidence.append(SurfaceEvidence(
                    layer="uia_pattern", feature="rich_standard_controls",
                    weight=self.UIA_PATTERN_WEIGHT, value=0.90,
                    description=f"UIA 树丰富({element_count}元素, {has_standard_controls}种标准控件) → 原生 GUI"
                ))
        elif has_datagrid:
            # DataGrid 特征 → 可能是原生复杂 GUI
            scores["native_uia"] = 0.70
            evidence.append(SurfaceEvidence(
                layer="uia_pattern", feature="has_datagrid",
                weight=self.UIA_PATTERN_WEIGHT, value=0.70,
                description="UIA 树含 DataGrid → 原生 GUI"
            ))
        else:
            # 中等复杂度：如果进程已知为 Electron/WebView2，偏向 Electron；否则默认原生 GUI
            if is_known_electron:
                scores["electron_webview"] = 0.55
                evidence.append(SurfaceEvidence(
                    layer="uia_pattern", feature="medium_complexity_known_electron",
                    weight=self.UIA_PATTERN_WEIGHT, value=0.55,
                    description=f"UIA 树中等复杂度({element_count}元素) + 已知Electron进程 → Electron外壳"
                ))
            elif is_known_browser:
                scores["browser"] = 0.55
                evidence.append(SurfaceEvidence(
                    layer="uia_pattern", feature="medium_complexity_known_browser",
                    weight=self.UIA_PATTERN_WEIGHT, value=0.55,
                    description=f"UIA 树中等复杂度({element_count}元素) + 已知Browser进程 → Browser"
                ))
            else:
                scores["native_uia"] = 0.50
                evidence.append(SurfaceEvidence(
                    layer="uia_pattern", feature="medium_complexity",
                    weight=self.UIA_PATTERN_WEIGHT, value=0.50,
                    description=f"UIA 树中等复杂度({element_count}元素) → 默认原生 GUI"
                ))

        return scores

    def _score_dom_layer(
        self,
        has_dom_bridge: bool,
        dom_ready: bool,
        evidence: list[SurfaceEvidence],
    ) -> dict[str, float]:
        """层3：DOM 可用性打分"""
        scores = {
            "native_uia": 0.0,
            "browser": 0.0,
            "electron_webview": 0.0,
            "canvas_self_drawn": 0.0,
        }

        if dom_ready and has_dom_bridge:
            # DOM 可用 → 浏览器或 Electron 内核
            scores["browser"] = 1.0
            scores["electron_webview"] = 0.80
            evidence.append(SurfaceEvidence(
                layer="dom", feature="dom_available",
                weight=self.DOM_WEIGHT, value=True,
                description="DOM 可访问 → 浏览器/Electron"
            ))
        elif has_dom_bridge and not dom_ready:
            # 有 bridge 但 DOM 未就绪
            scores["browser"] = 0.40
            scores["electron_webview"] = 0.30
            evidence.append(SurfaceEvidence(
                layer="dom", feature="dom_bridge_present_not_ready",
                weight=self.DOM_WEIGHT, value=0.30,
                description="DOM bridge 存在但未就绪 → 浏览器（低置信）"
            ))
        else:
            # 无 DOM → 原生 GUI 或 Canvas
            scores["native_uia"] = 0.60
            scores["canvas_self_drawn"] = 0.40
            evidence.append(SurfaceEvidence(
                layer="dom", feature="no_dom",
                weight=self.DOM_WEIGHT, value=False,
                description="无 DOM bridge → 原生 GUI / Canvas"
            ))

        return scores

    def _score_visual_layer(
        self,
        visual_score: float | None,
        evidence: list[SurfaceEvidence],
    ) -> dict[str, float]:
        """层4：视觉特征打分"""
        scores = {
            "native_uia": 0.5,
            "browser": 0.5,
            "electron_webview": 0.5,
            "canvas_self_drawn": 0.5,
        }

        if visual_score is None:
            evidence.append(SurfaceEvidence(
                layer="visual", feature="no_visual_data",
                weight=0.0, value=False, description="无视觉分析数据（使用默认）"
            ))
            return scores

        # 视觉分数 > 0.8：很像标准 Windows GUI
        if visual_score > 0.8:
            scores["native_uia"] = 1.0
            evidence.append(SurfaceEvidence(
                layer="visual", feature="high_gui_fidelity",
                weight=self.VISUAL_WEIGHT, value=visual_score,
                description=f"视觉分数高({visual_score:.2f}) → 标准 Windows GUI"
            ))
        # 视觉分数 < 0.3：可能是 WebView 或 Canvas
        elif visual_score < 0.3:
            scores["browser"] = 0.60
            scores["canvas_self_drawn"] = 0.60
            evidence.append(SurfaceEvidence(
                layer="visual", feature="low_gui_fidelity",
                weight=self.VISUAL_WEIGHT, value=visual_score,
                description=f"视觉分数低({visual_score:.2f}) → WebView / Canvas"
            ))
        else:
            evidence.append(SurfaceEvidence(
                layer="visual", feature="medium_visual_score",
                weight=self.VISUAL_WEIGHT, value=visual_score,
                description=f"视觉分数中等({visual_score:.2f}) → 无偏好"
            ))

        return scores

    def _get_locator_priority(self, surface_type: SurfaceType) -> list[str]:
        """返回对应 surface_type 的 locator 优先级"""
        priorities = {
            SurfaceType.NATIVE_UIA: [
                "uia", "relative", "ocr", "template_icon", "vision_bbox", "ephemeral_coord"
            ],
            SurfaceType.BROWSER: [
                "dom", "relative", "vision_bbox", "ocr", "template_icon", "ephemeral_coord"
            ],
            SurfaceType.ELECTRON_WEBVIEW: [
                "uia", "dom", "vision_bbox", "relative", "ocr", "template_icon", "ephemeral_coord"
            ],
            SurfaceType.CANVAS_SELF_DRAWN: [
                "vision_bbox", "ocr", "template_icon", "relative", "ephemeral_coord"
            ],
            SurfaceType.UNKNOWN: [
                "uia", "relative", "ocr", "vision_bbox", "ephemeral_coord"
            ],
        }
        return priorities.get(surface_type, priorities[SurfaceType.UNKNOWN])
