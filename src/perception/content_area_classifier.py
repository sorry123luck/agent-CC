"""
Content Area Subtype Classifier — P3-A

对已划分的 content_area 进行子类型判定：
- CHAT         — 聊天面板：消息流、输入框、发送按钮
- FORM         — 表单：label-field 配对、提交按钮
- LIST_DETAIL  — 列表详情：split layout、列表项+详情区
- EDITOR       — 编辑器：tab 栏、工具栏、文本视口
- DASHBOARD    — 仪表盘：控件、表格、图表区
- CANVAS_DOC_VIEWER — 文档查看器：翻页控件、视口

输入：content_area 区域的 elements 列表
输出：ContentAreaSubtype + confidence + evidence
"""

from dataclasses import dataclass, field
from typing import Any

from src.perception.page_compiler_models import ContentAreaSubtype


# =============================================================================
# Evidence
# =============================================================================

@dataclass
class ContentAreaEvidence:
    """Content area 子类型的证据"""
    feature: str          # 特征名
    weight: float         # 权重（0-1）
    value: bool | float  # 证据值
    description: str      # 证据描述


@dataclass
class ContentAreaClassification:
    """Content area 子类型分类结果"""
    subtype: ContentAreaSubtype
    confidence: float
    evidence: list[ContentAreaEvidence] = field(default_factory=list)

    @property
    def evidence_summary(self) -> list[str]:
        return [e.description for e in self.evidence]


# =============================================================================
# Content Area Subtype Classifier
# =============================================================================

class ContentAreaClassifier:
    """
    Content Area 子类型分类器

    启发式判断：
    1. 统计 content_area 内各控件类型的数量和分布
    2. 分析布局结构（水平/垂直分割、输入区位置）
    3. 检查关键特征（输入框+按钮、标签配对、列表+详情等）
    """

    # 各 subtype 的默认置信度
    DEFAULT_CONFIDENCE = 0.30

    # 表单识别：Label 与 Input 的邻近距离阈值（Y方向像素）
    LABEL_INPUT_Y_THRESHOLD = 30
    # 表单识别：Label 与 Input 的 X方向重叠阈值
    LABEL_INPUT_X_OVERLAP_THRESHOLD = 0.3

    def classify(
        self,
        elements: list[dict[str, Any]],
    ) -> ContentAreaClassification:
        """
        分类入口

        Args:
            elements: content_area 区域的元素列表
                      每个元素为 dict，包含：
                      - control_type: str
                      - name: str | None
                      - value: str | None
                      - text: str | None
                      - bounding_rect: tuple[int, int, int, int] | None
                      - left, top, right, bottom: int | None

        Returns:
            ContentAreaClassification 包含 subtype + confidence + evidence
        """
        evidence: list[ContentAreaEvidence] = []

        if not elements:
            return ContentAreaClassification(
                subtype=ContentAreaSubtype.UNKNOWN,
                confidence=self.DEFAULT_CONFIDENCE,
                evidence=[ContentAreaEvidence(
                    feature="no_elements",
                    weight=1.0,
                    value=False,
                    description="content_area 无元素"
                )]
            )

        # 统计控件类型
        control_types = self._count_control_types(elements)
        # 分析布局特征
        layout = self._analyze_layout(elements)
        # 检测关键特征
        features = self._detect_features(elements, control_types, layout)

        # 收集证据
        for f in features:
            evidence.append(f)

        # 判分子类型
        subtype, confidence = self._decide_subtype(control_types, layout, features)

        return ContentAreaClassification(
            subtype=subtype,
            confidence=round(confidence, 3),
            evidence=evidence,
        )

    def _count_control_types(
        self, elements: list[dict[str, Any]]
    ) -> dict[str, int]:
        """统计各控件类型的数量"""
        counts: dict[str, int] = {}
        for elem in elements:
            ct = (elem.get("control_type") or "").lower()
            if not ct:
                continue
            # 归一化：保留主要类型
            # 注意：text (静态文本) 不应归为 edit (输入框)
            if "edit" in ct:
                ct = "edit"
            elif "button" in ct:
                ct = "button"
            elif "list" in ct or "listitem" in ct:
                ct = "list"
            elif "tree" in ct or "treeitem" in ct:
                ct = "tree"
            elif "tab" in ct or "tabitem" in ct:
                ct = "tab"
            elif "toolbar" in ct:
                ct = "toolbar"
            elif "menu" in ct or "menuitem" in ct:
                ct = "menu"
            elif "scroll" in ct:
                ct = "scroll"
            elif "pane" in ct or "panel" in ct:
                ct = "pane"
            elif "document" in ct:
                ct = "document"
            elif "image" in ct:
                ct = "image"
            elif "custom" in ct:
                ct = "custom"
            elif "data" in ct or "grid" in ct:
                ct = "datagrid"
            else:
                ct = "other"
            counts[ct] = counts.get(ct, 0) + 1
        return counts

    def _analyze_layout(
        self, elements: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """分析布局结构"""
        layout: dict[str, Any] = {
            "has_split_vertical": False,
            "has_split_horizontal": False,
            "input_zone_bottom": False,
            "input_zone_top": False,
            "total_area": 0,
            "vertical_distribution": {"top": 0, "middle": 0, "bottom": 0},
        }

        if not elements:
            return layout

        # 计算总体边界
        all_rects = [
            (e.get("left", 0), e.get("top", 0),
             e.get("right", 0), e.get("bottom", 0))
            for e in elements
            if e.get("bounding_rect")
        ]
        if not all_rects:
            return layout

        min_x = min(r[0] for r in all_rects)
        max_x = max(r[2] for r in all_rects)
        min_y = min(r[1] for r in all_rects)
        max_y = max(r[3] for r in all_rects)
        total_area = (max_x - min_x) * (max_y - min_y)
        layout["total_area"] = total_area

        window_height = max_y - min_y if max_y > min_y else 1
        window_width = max_x - min_x if max_x > min_x else 1

        # 分析垂直分布
        for elem in elements:
            rect = elem.get("bounding_rect")
            if not rect:
                continue
            l, t, r, b = rect
            rel_top = (t - min_y) / window_height if window_height else 0
            if rel_top < 0.33:
                layout["vertical_distribution"]["top"] += 1
            elif rel_top < 0.66:
                layout["vertical_distribution"]["middle"] += 1
            else:
                layout["vertical_distribution"]["bottom"] += 1

        # 检测垂直分割（左右结构）
        # 从 bounding_rect 提取 left/right 坐标
        def _get_left(e) -> int:
            r = e.get("bounding_rect")
            return r[0] if r else (e.get("left") or 0)
        def _get_right(e) -> int:
            r = e.get("bounding_rect")
            return r[2] if r else (e.get("right") or 0)

        left_elements = [e for e in elements if _get_left(e) < min_x + window_width * 0.4]
        right_elements = [e for e in elements if _get_right(e) > min_x + window_width * 0.6]
        if len(left_elements) >= 3 and len(right_elements) >= 3:
            # 检查是否有明显分界线（中间空白区域）
            left_max_x = max((_get_right(e) for e in left_elements), default=0)
            right_min_x = min((_get_left(e) for e in right_elements), default=0)
            if left_max_x < right_min_x and (right_min_x - left_max_x) > window_width * 0.05:
                layout["has_split_vertical"] = True

        # 检测输入区位置（底部有输入框+按钮 → 典型聊天/表单特征）
        # 从 bounding_rect 提取 top 坐标
        def _get_top(e) -> int:
            r = e.get("bounding_rect")
            return r[1] if r else (e.get("top") or 0)

        bottom_elements = [
            e for e in elements
            if e.get("bounding_rect") and _get_top(e) > min_y + window_height * 0.7
        ]
        bottom_control_types = [e.get("control_type", "").lower() for e in bottom_elements]
        has_bottom_edit = any("edit" in ct or "text" in ct for ct in bottom_control_types)
        has_bottom_button = any("button" in ct for ct in bottom_control_types)
        if has_bottom_edit and has_bottom_button:
            layout["input_zone_bottom"] = True

        return layout

    def _detect_features(
        self,
        elements: list[dict[str, Any]],
        control_types: dict[str, int],
        layout: dict[str, Any],
    ) -> list[ContentAreaEvidence]:
        """检测关键特征"""
        features: list[ContentAreaEvidence] = []
        total = len(elements)
        if total == 0:
            return features

        # === 消息输入特征（CHAT）===
        has_input_zone = layout.get("input_zone_bottom", False)
        has_edit = control_types.get("edit", 0) > 0
        has_button = control_types.get("button", 0) > 0
        has_scroll = control_types.get("scroll", 0) > 0 or control_types.get("list", 0) > 0

        if has_input_zone and has_edit and has_button:
            features.append(ContentAreaEvidence(
                feature="chat_input_zone",
                weight=0.30,
                value=True,
                description="检测到底部输入区（输入框+按钮）→ CHAT 特征"
            ))
        elif has_edit and has_button and not layout.get("has_split_vertical", False):
            features.append(ContentAreaEvidence(
                feature="input_with_button_no_split",
                weight=0.15,
                value=True,
                description="有输入框和按钮但无垂直分割 → FORM 特征"
            ))

        # === 表单特征（FORM）===
        # 表单：多个 Edit + 可能的 Label 配对
        edit_count = control_types.get("edit", 0)
        button_count = control_types.get("button", 0)
        if edit_count >= 2 and button_count >= 1:
            features.append(ContentAreaEvidence(
                feature="multi_edit_fields",
                weight=0.25,
                value=True,
                description=f"多输入框({edit_count}个) + 按钮({button_count}个) → FORM 特征"
            ))

        # 检查 label-edit 配对（通过邻近关系）
        label_edit_pairs = self._find_label_edit_pairs(elements)
        if len(label_edit_pairs) >= 2:
            features.append(ContentAreaEvidence(
                feature="label_edit_pairs",
                weight=0.30,
                value=float(min(len(label_edit_pairs), 5)),
                description=f"检测到 {len(label_edit_pairs)} 个 label-edit 配对 → FORM 特征"
            ))

        # === 列表详情特征（LIST_DETAIL）===
        if layout.get("has_split_vertical", False):
            features.append(ContentAreaEvidence(
                feature="split_layout_vertical",
                weight=0.35,
                value=True,
                description="检测到垂直分割布局（左右结构）→ LIST_DETAIL 特征"
            ))

        has_tree = control_types.get("tree", 0) > 0
        has_list = control_types.get("list", 0) > 0
        if (has_tree or has_list) and edit_count >= 1:
            features.append(ContentAreaEvidence(
                feature="list_with_detail",
                weight=0.25,
                value=True,
                description="列表/树 + 输入框/详情 → LIST_DETAIL 特征"
            ))

        # === 编辑器特征（EDITOR）===
        has_tab = control_types.get("tab", 0) > 0
        has_toolbar = control_types.get("toolbar", 0) > 0
        has_document = control_types.get("document", 0) > 0
        edit_area_ratio = control_types.get("edit", 0) / total if total > 0 else 0

        if has_tab and has_toolbar:
            features.append(ContentAreaEvidence(
                feature="tab_with_toolbar",
                weight=0.30,
                value=True,
                description="Tab + ToolBar → EDITOR 特征"
            ))
        if has_document:
            features.append(ContentAreaEvidence(
                feature="document_area",
                weight=0.25,
                value=True,
                description="Document 控件 → EDITOR 特征"
            ))

        # === 仪表盘特征（DASHBOARD）===
        unique_types = len([ct for ct, c in control_types.items() if c > 0 and ct not in ("other", "pane", "custom")])
        pane_ratio = control_types.get("pane", 0) / total if total > 0 else 0

        if unique_types >= 4 and pane_ratio > 0.2:
            features.append(ContentAreaEvidence(
                feature="multi_widget_diversity",
                weight=0.25,
                value=float(unique_types),
                description=f"多种控件类型({unique_types}种) + 多个 Pane → DASHBOARD 特征"
            ))

        # === 文档查看器特征（CANVAS_DOC_VIEWER）===
        pane_count = control_types.get("pane", 0)
        pane_ratio = pane_count / total if total > 0 else 0
        # 检查 Pane 是否主导：要么元素数量多，要么覆盖面积大
        pane_area_dominant = False
        for elem in elements:
            rect = elem.get("bounding_rect")
            if rect and elem.get("control_type", "").lower() in ("pane", "panel"):
                l, t, r, b = rect
                elem_area = (r - l) * (b - t)
                # 如果 Pane 覆盖面积超过总面积的 60%，视为面积主导
                if layout.get("total_area", 0) > 0 and elem_area / layout["total_area"] > 0.6:
                    pane_area_dominant = True
                    break
        # 条件：Pane 数量主导(>25%) 或面积主导，且无 Edit 输入框
        if (pane_ratio > 0.25 or pane_area_dominant) and edit_count == 0:
            features.append(ContentAreaEvidence(
                feature="pane_dominant_no_input",
                weight=0.25,
                value=True,
                description=f"Pane 主导(数量比{pane_ratio:.0%} 或面积主导) + 无 Edit 输入框 → CANVAS_DOC_VIEWER 特征"
            ))

        return features

    def _find_label_edit_pairs(
        self, elements: list[dict[str, Any]]
    ) -> list[tuple[Any, Any]]:
        """
        查找 label-edit 配对

        启发式：Label 在 Edit 上方或左方，且距离在阈值内
        """
        pairs: list[tuple[Any, Any]] = []

        labels = [e for e in elements
                  if "label" in (e.get("control_type") or "").lower()
                  or "text" in (e.get("control_type") or "").lower()
                  and e.get("name") and not e.get("value")]
        edits = [e for e in elements
                 if "edit" in (e.get("control_type") or "").lower()]

        for label in labels:
            l_rect = label.get("bounding_rect")
            if not l_rect:
                continue
            l_l, l_t, l_r, l_b = l_rect

            for edit in edits:
                e_rect = edit.get("bounding_rect")
                if not e_rect:
                    continue
                e_l, e_t, e_r, e_b = e_rect

                # Label 在 Edit 上方（Y方向）
                if l_b <= e_t and (e_t - l_b) < self.LABEL_INPUT_Y_THRESHOLD:
                    pairs.append((label, edit))
                    continue

                # Label 在 Edit 左侧（X方向重叠）
                x_overlap = min(l_r, e_r) - max(l_l, e_l)
                if x_overlap > 0 and l_r <= e_l and (e_l - l_r) < self.LABEL_INPUT_Y_THRESHOLD:
                    pairs.append((label, edit))

        return pairs

    def _decide_subtype(
        self,
        control_types: dict[str, int],
        layout: dict[str, Any],
        features: list[ContentAreaEvidence],
    ) -> tuple[ContentAreaSubtype, float]:
        """根据特征决定子类型"""
        feature_weights: dict[str, float] = {
            "chat_input_zone": 0.80,
            "multi_edit_fields": 0.60,
            "label_edit_pairs": 0.75,
            "split_layout_vertical": 0.70,
            "list_with_detail": 0.65,
            "tab_with_toolbar": 0.70,
            "document_area": 0.65,
            "multi_widget_diversity": 0.60,
            "pane_dominant_no_input": 0.50,
            "input_with_button_no_split": 0.40,
        }

        scores: dict[ContentAreaSubtype, float] = {
            ContentAreaSubtype.CHAT: 0.0,
            ContentAreaSubtype.FORM: 0.0,
            ContentAreaSubtype.LIST_DETAIL: 0.0,
            ContentAreaSubtype.EDITOR: 0.0,
            ContentAreaSubtype.DASHBOARD: 0.0,
            ContentAreaSubtype.CANVAS_DOC_VIEWER: 0.0,
            ContentAreaSubtype.UNKNOWN: self.DEFAULT_CONFIDENCE,
        }

        for feat in features:
            w = feature_weights.get(feat.feature, 0.1)
            value = 1.0 if feat.value is True else float(feat.value)

            if feat.feature == "chat_input_zone":
                scores[ContentAreaSubtype.CHAT] += w * value
            elif feat.feature in ("multi_edit_fields", "label_edit_pairs", "input_with_button_no_split"):
                scores[ContentAreaSubtype.FORM] += w * value
            elif feat.feature in ("split_layout_vertical", "list_with_detail"):
                scores[ContentAreaSubtype.LIST_DETAIL] += w * value
            elif feat.feature in ("tab_with_toolbar", "document_area"):
                scores[ContentAreaSubtype.EDITOR] += w * value
            elif feat.feature == "multi_widget_diversity":
                scores[ContentAreaSubtype.DASHBOARD] += w * value
            elif feat.feature == "pane_dominant_no_input":
                scores[ContentAreaSubtype.CANVAS_DOC_VIEWER] += w * value

        # 附加启发式规则
        total = sum(control_types.values()) if control_types else 1
        edit_ratio = control_types.get("edit", 0) / total
        button_ratio = control_types.get("button", 0) / total
        pane_ratio = control_types.get("pane", 0) / total
        has_split = layout.get("has_split_vertical", False)

        # 如果有 split 且 pane 多 → LIST_DETAIL 增强
        if has_split and pane_ratio > 0.2:
            scores[ContentAreaSubtype.LIST_DETAIL] += 0.25

        # 如果底部输入区 + scroll → 强化 CHAT
        if layout.get("input_zone_bottom", False) and control_types.get("scroll", 0) > 0:
            scores[ContentAreaSubtype.CHAT] += 0.20

        # 如果多种控件类型混杂 → DASHBOARD
        unique_types = len([ct for ct, c in control_types.items() if c > 0 and ct not in ("other", "pane")])
        if unique_types >= 5:
            scores[ContentAreaSubtype.DASHBOARD] += 0.15

        # 如果 editor/document 控件 → EDITOR 增强
        if control_types.get("document", 0) > 0:
            scores[ContentAreaSubtype.EDITOR] += 0.30

        # 找最高分
        best_subtype = max(scores, key=scores.get)
        best_score = scores[best_subtype]

        # 阈值过滤
        if best_score < 0.20:
            return ContentAreaSubtype.UNKNOWN, self.DEFAULT_CONFIDENCE

        return best_subtype, min(best_score, 1.0)
