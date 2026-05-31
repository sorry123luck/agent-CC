"""
InteractionCanvas 数据模型

基于 Pro 版 P3 最终设计基线定义的标准数据结构。
所有模型均对标 docs/pro/pro_final_design_baseline.md 和 docs/pro/pro_data_model_and_storage_baseline.md。

本文件包含：
- InteractionCanvas — 运行时交互画布（核心输出）
- Candidate        — 可操作候选点位
- Locator          — 元素多路定位器
- Anchor           — 漂移修复锚点
- Region           — 页面区域
- Relation         — 元素布局语义关系
- SurfaceInfo      — Surface 类型与证据
- PageInfo         — Page 分类信息

注：PageTemplate 属于 P5 记忆层，在 src/memory/ 模块中定义，不在此文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol


# =============================================================================
# Enums
# =============================================================================

class SurfaceType(Enum):
    """
    界面表面类型 — 执行层(parser/executor)选路依据

    四层证据打分：进程签名 / UIA 树模式 / DOM 可用性 / 视觉特征
    """
    NATIVE_UIA = "native_uia"          # 原生 Windows GUI，UIA 主链
    BROWSER = "browser"                # 网页，DOM/Playwright 主链
    ELECTRON_WEBVIEW = "electron_webview"  # Electron 外壳 UIA + 内核 DOM/视觉
    CANVAS_SELF_DRAWN = "canvas_self_drawn"  # 自绘界面，视觉主链
    UNKNOWN = "unknown"


class ContentAreaSubtype(Enum):
    """
    content_area 内容区子类型

    先分型，再进专项 parser。
    """
    CHAT = "chat"              # 聊天面板：消息流、输入框、发送按钮
    FORM = "form"              # 表单：label-field 配对、提交按钮
    LIST_DETAIL = "list_detail"  # 列表详情：split layout、列表项+详情区
    EDITOR = "editor"         # 编辑器：tab 栏、工具栏、文本视口
    DASHBOARD = "dashboard"    # 仪表盘：控件、表格、图表区
    CANVAS_DOC_VIEWER = "canvas_doc_viewer"  # 文档查看器：翻页控件、视口
    GRID_TABLE = "grid_table"  # 网格表格（预留）
    TIMELINE_MEDIA = "timeline_media"  # 时间线/媒体（预留）
    UNKNOWN = "unknown"


class SemanticRole(Enum):
    """
    元素语义角色 — 描述元素在页面中的功能
    """
    # 通用
    UNKNOWN = "unknown"
    CONTAINER = "container"
    LAYOUT = "layout"

    # 输入类
    MESSAGE_INPUT = "message_input"
    TEXT_INPUT = "text_input"
    SEARCH_INPUT = "search_input"
    PASSWORD_INPUT = "password_input"
    FILE_INPUT = "file_input"

    # 按钮类
    BUTTON = "button"
    SEND_BUTTON = "send_button"
    SUBMIT_BUTTON = "submit_button"
    CANCEL_BUTTON = "cancel_button"
    ICON_BUTTON = "icon_button"
    TOGGLE_BUTTON = "toggle_button"

    # 列表类
    LIST_ITEM = "list_item"
    CHAT_ITEM = "chat_item"
    TREE_ITEM = "tree_item"
    MENU_ITEM = "menu_item"

    # 内容类
    TEXT = "text"
    IMAGE = "image"
    LINK = "link"
    TAB = "tab"

    # 导航类
    NAV_ITEM = "nav_item"
    SIDEBAR = "sidebar"
    TOOLBAR = "toolbar"
    MENU_BAR = "menu_bar"
    TITLE_BAR = "title_bar"
    STATUS_BAR = "status_bar"


class LocatorKind(Enum):
    """
    定位器类型 — 7 种定位方案

    ephemeral_coord 只允许运行时临时使用，不允许写入模板主链。
    """
    DOM = "dom"                    # DOM 元素定位
    UIA = "uia"                   # UIA 自动化属性定位
    OCR = "ocr"                   # OCR 文本匹配定位
    VISION_BBOX = "vision_bbox"  # 视觉边界框定位
    RELATIVE = "relative"         # 相对定位（相对锚点）
    TEMPLATE_ICON = "template_icon"  # 图标模板匹配定位
    EPHEMERAL_COORD = "ephemeral_coord"  # 临时绝对坐标（只用于运行时兜底）


class LocatorStatus(Enum):
    """定位器状态"""
    CANDIDATE = "candidate"   # 待验证
    ACTIVE = "active"         # 已验证可用
    DRIFTED = "drifted"       # 已漂移
    REPAIRED = "repaired"     # 已修复
    SUPERSEDED = "superseded"  # 已废弃
    RETIRED = "retired"       # 已退役


class CoordinateSpace(Enum):
    """坐标空间"""
    SCREEN = "screen"       # 屏幕坐标
    WINDOW = "window"       # 窗口客户区坐标
    REGION = "region"       # 区域局部坐标
    RELATIVE = "relative"   # 相对坐标


class RelationType(Enum):
    """元素布局关系类型"""
    RIGHT_OF = "right_of"
    LEFT_OF = "left_of"
    ABOVE = "above"
    BELOW = "below"
    INSIDE = "inside"
    CONTAINS = "contains"
    BELONGS_TO = "belongs_to"


class AnchorKind(Enum):
    """锚点类型"""
    SINGLE = "single"           # 单元素锚点
    COMPOUND = "compound"       # 复合锚点（多元素组合）
    TEXT = "text"               # 文本锚点
    ICON = "icon"               # 图标锚点
    GEOMETRIC = "geometric"     # 几何特征锚点


class CandidateKind(Enum):
    """弱语义候选框类型。"""
    UNKNOWN = "unknown"
    TEXT = "text_candidate"
    INPUT = "input_candidate"
    BUTTON = "button_candidate"
    ICON = "icon_candidate"
    LIST_ITEM = "list_item_candidate"
    PANEL = "panel_candidate"
    HEADER = "header_candidate"
    IMAGE = "image_candidate"


class CaptureReason(Enum):
    """快照采集原因"""
    SCHEDULED = "scheduled"           # 定时采集
    ACTION_COMPLETED = "action_completed"  # 动作完成后采集
    DRIFT_DETECTED = "drift_detected"  # 检测到漂移
    USER_REQUEST = "user_request"    # 用户请求
    INITIAL_CAPTURE = "initial_capture"  # 首次采集


class ConfidenceLevel(str, Enum):
    """候选可信度等级"""
    HIGH = "high"      # 多源一致，或单源高置信 + 记忆确认
    MEDIUM = "medium"  # 单源中置信，或有多源但有轻微冲突
    LOW = "low"        # 单源低置信，或多源冲突明显


class RiskTag(str, Enum):
    """风险标签 — 标记候选的潜在风险类型"""
    SEND = "send"              # 发送消息/邮件
    DELETE = "delete"          # 删除内容
    PAYMENT = "payment"        # 支付/转账
    SUBMIT = "submit"          # 提交表单
    PUBLISH = "publish"        # 发布内容
    LOGIN = "login"            # 登录/切换账号
    CONFIG_CHANGE = "config_change"  # 修改配置


class RiskLevel(str, Enum):
    """风险等级"""
    L0 = "L0"  # 读取、普通查看
    L1 = "L1"  # 普通操作（表单填写、筛选、页面跳转）
    L2 = "L2"  # 高风险（登录、修改配置、批量操作）
    L3 = "L3"  # 极高风险（下单、发布、支付）


# =============================================================================
# Nested Data Classes
# =============================================================================

@dataclass(frozen=True)
class Rect:
    """矩形区域"""
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    @classmethod
    def from_tuple(cls, t: tuple[int, int, int, int]) -> Rect:
        return cls(left=t[0], top=t[1], right=t[2], bottom=t[3])


@dataclass
class ElementState:
    """元素运行时状态"""
    visible: bool = True
    enabled: bool = True
    focused: bool = False
    selected: bool = False
    checked: bool = False
    expanded: bool = False


@dataclass
class LocatorStats:
    """定位器统计信息"""
    use_count: int = 0
    success_count: int = 0
    drift_count: int = 0
    last_used_at: datetime | None = None
    last_success_at: datetime | None = None


@dataclass
class AppInfo:
    """
    App 信息

    对应 InteractionCanvas.app 字段
    """
    app_id: str | None = None               # 应用标识（e.g. "wechat"）
    app_profile_id: str | None = None       # 应用配置档案 ID
    process_name: str | None = None         # 进程名（e.g. "WeChat.exe"）
    process_id: int | None = None           # 进程 ID
    exe_path: str | None = None             # 可执行文件路径


@dataclass
class WindowInfoSnapshot:
    """
    Window 信息（快照版）

    对应 InteractionCanvas.window 字段
    """
    hwnd: int
    title: str = ""
    class_name: str | None = None
    rect_screen: tuple[int, int, int, int] | None = None  # 屏幕坐标
    rect_client: tuple[int, int, int, int] | None = None  # 客户区坐标
    dpi_scale: float = 1.0
    is_foreground: bool = False
    is_minimized: bool = False
    occluded: bool = False


@dataclass
class SurfaceEvidence:
    """
    单条 Surface 证据 — 四层证据打分统一结构

    供 SurfaceClassifier -> SurfaceInfo 传递证据用
    """
    layer: str           # 证据来源层：process / uia_pattern / dom / visual
    feature: str          # 具体特征名
    weight: float         # 权重（0-1）
    value: bool | float  # 证据值
    description: str      # 证据描述


@dataclass
class SurfaceInfo:
    """
    Surface 信息 — 决定后续 parser/executor 选路

    对应 InteractionCanvas.surface 字段
    """
    surface_type: SurfaceType = SurfaceType.UNKNOWN
    confidence: float = 0.0
    evidence: list[SurfaceEvidence] = field(default_factory=list)  # 证据列表


@dataclass
class PageInfo:
    """
    Page 信息 — route 和 memory 的主索引

    对应 InteractionCanvas.page 字段
    """
    page_class: str = "unknown/unknown/unknown"  # 格式：<app>/<workflow>/<state>/<variant>
    class_confidence: float = 0.0
    page_variant: str = "default"
    fingerprint: dict[str, str] = field(default_factory=dict)  # layout_hash, anchor_hash, text_hash
    state_flags: dict[str, bool] = field(default_factory=dict)  # editable, scrollable, dialog_open...


@dataclass
class ScrollContext:
    """
    滚动上下文 — 解决滚动/虚拟视口问题

    对应 InteractionCanvas.scroll_contexts 字段
    """
    scroll_context_id: str
    region_id: str
    scroll_type: str = "vertical"  # vertical / horizontal / both
    viewport_height: int = 0
    viewport_width: int = 0
    scroll_offset: int = 0
    total_content_height: int | None = None
    is_virtual: bool = False  # 是否虚拟列表


@dataclass
class ProviderTrace:
    """
    Provider 追溯信息

    对应 InteractionCanvas.provider_trace 字段
    记录本次由哪些 provider 拼装得到
    """
    uia_used: bool = False
    ocr_used: bool = False
    vision_used: bool = False
    vlm_used: bool = False
    dom_used: bool = False
    opencv_used: bool = False
    template_used: bool = False
    provider_details: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Locator
# =============================================================================

@dataclass
class Locator:
    """
    定位器 — element 的多路定位方案

    对应 InteractionCanvas.locators 字段
    每个 element 可以有多个 locator，按 kind + priority 排序尝试
    """
    locator_id: str
    element_ref: str                    # 对应 element_id

    # 基础属性
    kind: LocatorKind
    priority: int                      # 越小越优先
    status: LocatorStatus = LocatorStatus.CANDIDATE

    # 作用域
    coordinate_space: CoordinateSpace = CoordinateSpace.WINDOW
    scope_region_id: str | None = None  # 定位时限定在哪个 region 内

    # 选择器
    selector: dict[str, Any] = field(default_factory=dict)  # kind-specific selector
    expected: dict[str, Any] = field(default_factory=dict)  # 期望命中时的属性

    # 锚点引用
    anchor_refs: list[str] = field(default_factory=list)  # 依赖的 anchor_id 列表

    # 约束条件
    preconditions: dict[str, Any] = field(default_factory=dict)  # 前置条件
    verification_hints: dict[str, Any] = field(default_factory=dict)  # 校验提示

    # 质量指标
    confidence: float = 0.0      # 该 locator 在当前 snapshot 的置信度
    durability_score: float = 0.0  # 持久性评分（跨版本稳定性）
    cost_score: float = 0.0      # 执行成本（速度/资源）

    # 统计
    stats: LocatorStats = field(default_factory=LocatorStats)

    # 生命周期
    lifecycle: str = "candidate"  # candidate → active → drifted → repaired / superseded → retired

    # 备注
    notes: str = ""


# =============================================================================
# Anchor
# =============================================================================

@dataclass
class Anchor:
    """
    锚点 — 漂移修复核心

    对应 InteractionCanvas.anchors 字段
    锚点是页面中稳定可识别元素的组合签名，用于检测和修复漂移
    """
    anchor_id: str
    kind: AnchorKind = AnchorKind.SINGLE

    # 组成元素
    element_refs: list[str] = field(default_factory=list)  # 组成该锚点的 element_id 列表

    # 签名特征
    signature: dict[str, Any] = field(default_factory=dict)  # relative_geometry, text_hint, icon_hash...
    stability_score: float = 0.0  # 稳定性评分（0-1）
    drift_threshold: float = 0.15  # 漂移容忍阈值

    # 关联
    region_id: str | None = None  # 所属 region
    notes: str = ""


# =============================================================================
# Element
# =============================================================================

@dataclass
class Candidate:
    """
    候选点位 — 实际可操作对象

    对应 InteractionCanvas.elements 字段
    """
    element_id: str
    region_id: str | None = None

    # 语义
    semantic_role: SemanticRole = SemanticRole.UNKNOWN
    control_type: str = ""         # 原始 ControlType（e.g. "Edit", "Button"）

    # 几何
    bounds: tuple[int, int, int, int] | None = None  # left, top, right, bottom

    # 内容
    text: str = ""
    name: str | None = None        # UIA Name 属性
    value: str | None = None       # UIA Value 属性
    placeholder: str | None = None

    # 状态
    interactable: bool = True
    state: ElementState = field(default_factory=ElementState)

    # 来源
    provider_sources: list[str] = field(default_factory=list)  # ["uia"], ["uia", "ocr"]

    # 定位与锚点
    locator_ids: list[str] = field(default_factory=list)
    anchor_ids: list[str] = field(default_factory=list)

    # 布局语义
    content_group_id: str | None = None  # 所属内容组（e.g. "cg_composer"）
    tab_group_id: str | None = None      # 所属标签组

    # 扩展
    attributes: dict[str, Any] = field(default_factory=dict)

    # 数值型可信度（0.0-1.0），来自 provider 原始评分
    confidence: float = 0.0

    # --- Phase 1 新增字段 ---
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM  # 可信度等级
    risk_tags: list[str] = field(default_factory=list)           # 风险标签
    risk_level: RiskLevel = RiskLevel.L0                         # 风险等级
    from_memory: bool = False                                    # 是否来自记忆
    suggest_confirm: bool = False                                # 是否建议确认
    click_point: tuple[int, int] | None = None                   # 推荐点击坐标（窗口局部坐标）
    scroll_context: ScrollContext | None = None                  # 滚动信息

    # --- Refine: 三层可扩展语义复核（不覆盖原始字段） ---
    # 第一层：固定通用视觉类型（少量跨软件通用集合）
    visual_type: str = ""          # icon / text / button / input / list_item / image / avatar / panel / toolbar / menu / checkbox / unknown
    # 第二层：可扩展语义标签（字符串，不写死枚举）
    semantic_tags: list[str] = field(default_factory=list)  # e.g. ["wechat.emoji_button", "messaging.emoji"]
    role_label: str | None = None  # 人类可读角色标签 e.g. "表情按钮"
    role_confidence: float = 0.0   # 语义判断可信度 (0.0-1.0)
    role_source: str = ""          # heuristic / vlm / agent / memory / profile
    role_evidence: list[str] = field(default_factory=list)  # 判断依据列表
    refine_status: str = "unreviewed"  # unreviewed / refined / uncertain
    stable_key_id: str | None = None  # StableCandidateKey.key_id（持久身份）

    @property
    def rect(self) -> Rect | None:
        return Rect.from_tuple(self.bounds) if self.bounds else None


# =============================================================================
# VisualRegion — compatibility wrapper for phased switch
# =============================================================================

@dataclass
class VisualRegion:
    """Legacy visual layout region. Kept for parallel output during phased switch."""
    region_type: str = "unknown"
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0
    actionability: float = 0.0
    source: str = "visual_layout"


# =============================================================================
# Region
# =============================================================================

@dataclass
class Region:
    """
    页面区域 — 执行层第一作用域

    对应 InteractionCanvas.regions 字段
    """
    region_id: str
    role: str = "unknown"          # title_bar / menu_bar / tool_bar / sidebar / content_area / status_bar
    subtype: ContentAreaSubtype = ContentAreaSubtype.UNKNOWN  # 仅 content_area 有效

    # 层级
    parent_region_id: str | None = None

    # 几何
    bounds: tuple[int, int, int, int] | None = None  # left, top, right, bottom
    geometric_source_id: str | None = None  # 追溯几何来源（GeometricRegion.region_id）

    # 滚动上下文
    scroll_context_id: str | None = None

    # 子元素与锚点
    element_ids: list[str] = field(default_factory=list)
    anchor_ids: list[str] = field(default_factory=list)

    # 子 regions（嵌套区域）
    child_region_ids: list[str] = field(default_factory=list)

    # 扩展属性
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def rect(self) -> Rect | None:
        return Rect.from_tuple(self.bounds) if self.bounds else None


# =============================================================================
# Relation
# =============================================================================

@dataclass
class ElementRelation:
    """
    元素布局语义关系

    对应 InteractionCanvas.relations 字段
    """
    relation_id: str
    from_id: str         # 源 element_id 或 region_id
    to_id: str           # 目标 element_id 或 region_id
    type: RelationType
    strength: float = 1.0  # 关系强度（0-1）
    offset: tuple[int, int] | None = None  # 相对偏移量


@dataclass
class BoundaryCandidate:
    """
    弱语义候选框。

    给 DeskCanvas 的输入只保留坐标、文本、来源、弱 hint，
    不要求本地先做最终语义解释。
    """
    candidate_id: str
    bbox: tuple[int, int, int, int]
    source: str = "unknown"
    confidence: float = 0.0
    candidate_kind: CandidateKind = CandidateKind.UNKNOWN
    text: str = ""
    control_hint: str = ""
    region_hint: str = ""
    region_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionAlternative:
    """DeskCanvas 排除的候选。"""
    candidate_id: str
    selected_role: str = ""
    reject_reason: str = ""


@dataclass
class DecisionAction:
    """DeskCanvas 返回的下一步动作建议。"""
    type: str = ""
    text: str = ""


@dataclass
class DecisionRecord:
    """
    DeskCanvas 的结构化决策回执。
    """
    page_state: str = ""
    task_understanding: str = ""
    selected_candidate_id: str = ""
    selected_role: str = ""
    bbox: tuple[int, int, int, int] | None = None
    reason: str = ""
    alternatives: list[DecisionAlternative] = field(default_factory=list)
    next_action: DecisionAction = field(default_factory=DecisionAction)
    confidence: float = 0.0
    decision_status: str = "ready"
    focus_bbox: tuple[int, int, int, int] | None = None


# =============================================================================
# InteractionCanvas
# =============================================================================

@dataclass
class InteractionCanvas:
    """
    交互画布 — 每次感知后的标准输出

    对应 Pro 版 P3 最终输出目标
    直接给 P4 执行层消费

    本质：
    - 真实（真实采集）
    - 当前（当前时刻）
    - 完整（完整结构）
    - 可变（每次采集可能不同）
    - 用于执行（直接驱动动作）
    """
    # 顶层元信息
    schema_version: str = "2.0"
    canvas_id: str = ""                    # 本次画布唯一 ID（原 snapshot_id）
    captured_at: datetime = field(default_factory=datetime.now)
    capture_reason: CaptureReason = CaptureReason.SCHEDULED

    # 上下文
    app: AppInfo = field(default_factory=AppInfo)
    session_id: str | None = None
    window: WindowInfoSnapshot | None = None

    # 核心分类
    surface: SurfaceInfo = field(default_factory=SurfaceInfo)
    page: PageInfo = field(default_factory=PageInfo)

    # 滚动上下文
    scroll_contexts: list[ScrollContext] = field(default_factory=list)

    # 结构化页面对象
    regions: list[Region] = field(default_factory=list)
    elements: list[Candidate] = field(default_factory=list)
    locators: list[Locator] = field(default_factory=list)
    anchors: list[Anchor] = field(default_factory=list)
    relations: list[ElementRelation] = field(default_factory=list)

    # 追溯与调试
    provider_trace: ProviderTrace = field(default_factory=ProviderTrace)
    artifacts: dict[str, Any] = field(default_factory=dict)  # 调试证据、原始截图路径等

    # --- Phase 1 新增字段 ---
    monitor_id: int = 0                          # 显示器编号
    stable: bool = True                          # 页面是否稳定
    loading: bool = False                        # 是否在加载中
    animation_detected: bool = False             # 是否检测到动画
    partial: bool = False                        # 画布是否不完整（部分源失败）
    providers_used: list[str] = field(default_factory=list)   # 成功的感知源
    providers_failed: list[str] = field(default_factory=list)  # 失败的感知源
    canvas_schema_version: str = "1.0"           # 画布 schema 版本

    # --- E Phase 2: 软件虚拟模型 ---
    page_model_id: str | None = None             # 页面模型 ID
    state_template_id: str | None = None         # 页面状态模板 ID

    # 快照指纹（用于模板匹配）
    @property
    def page_class(self) -> str:
        return self.page.page_class

    @property
    def surface_type(self) -> SurfaceType:
        return self.surface.surface_type

    def get_element(self, element_id: str) -> Candidate | None:
        for e in self.elements:
            if e.element_id == element_id:
                return e
        return None

    def get_region(self, region_id: str) -> Region | None:
        for r in self.regions:
            if r.region_id == region_id:
                return r
        return None

    def get_locator(self, locator_id: str) -> Locator | None:
        for l in self.locators:
            if l.locator_id == locator_id:
                return l
        return None

    def get_anchors_for_element(self, element_id: str) -> list[Anchor]:
        return [a for a in self.anchors if element_id in a.element_refs]


# =============================================================================
# Provider 抽象基类（供后续 P2 Provider 层使用）
# =============================================================================

class PerceptionProvider(Protocol):
    """
    感知 Provider 协议

    定义 UIA / OCR / Vision / DOM / Template 等 provider 的统一接口。
    供 P2 Provider 层使用。
    """

    def name(self) -> str:
        """Provider 名称"""
        ...

    def supports(self, surface_type: SurfaceType) -> bool:
        """是否支持给定 surface_type"""
        ...

    def extract(
        self,
        window_info: WindowInfoSnapshot,
        existing_structure: InteractionCanvas | None = None,
    ) -> InteractionCanvas:
        """从窗口提取页面结构"""
        ...
