"""
Pydantic 数据模型定义
"""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


# ===== 软件相关模型 =====

class App(BaseModel):
    """软件记录"""
    id: int | None = None
    canonical_name: str = Field(..., description="规范名称，如 WeChat")
    display_name: str = Field(..., description="显示名称，如 微信")
    publisher: str | None = None
    install_location: str | None = None
    launch_path: str | None = None
    launch_args: str | None = None
    is_uwp: bool = False
    icon_path: str | None = None
    version: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    last_seen_at: datetime | None = None

    model_config = {"from_attributes": True}


class AppAlias(BaseModel):
    """软件别名"""
    id: int | None = None
    app_id: int
    alias: str
    source: str = "common"  # common 或 user
    weight: float = 1.0

    model_config = {"from_attributes": True}


class LaunchTarget(BaseModel):
    """启动入口"""
    id: int | None = None
    app_id: int
    path: str = Field(..., description="启动路径")
    args: str | None = None
    target_type: str = "exe"  # exe, uwp, shortcut, url
    score: float = Field(default=0.0, ge=0.0, le=100.0, description="置信度分数")
    source: str = Field(..., description="来源，如 registry, startmenu")

    model_config = {"from_attributes": True}


class AliasMatchResult(BaseModel):
    """别名匹配结果"""
    canonical_name: str
    display_name: str
    score: float = Field(ge=0.0, le=1.0)
    matched_alias: str


class LaunchResult(BaseModel):
    """启动结果"""
    success: bool
    app_name: str
    message: str
    hwnd: int | None = None


# ===== 窗口相关模型 =====

class WindowInfo(BaseModel):
    """窗口信息"""
    hwnd: int
    title: str
    class_name: str | None = None
    process_name: str | None = None
    process_id: int | None = None
    rect: tuple[int, int, int, int] | None = None  # left, top, right, bottom
    is_visible: bool = True
    is_enabled: bool = True


class Screenshot(BaseModel):
    """截图"""
    path: str
    width: int
    height: int
    timestamp: datetime
    hwnd: int | None = None


# ===== 页面元素模型 =====

class Rect(BaseModel):
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


class Candidate(BaseModel):
    """页面元素"""
    element_id: str
    role: str
    rect: Rect
    text: str | None = None
    name: str | None = None
    source: list[str] = Field(default_factory=list)  # uia, ocr, vision
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    locator: dict[str, Any] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)


class PageModel(BaseModel):
    """统一页面模型"""
    app_id: str
    page_type: str
    window: WindowInfo | None = None
    fingerprint: dict[str, Any] = Field(default_factory=dict)
    elements: list[Candidate] = Field(default_factory=list)
    screenshot_path: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    timestamp: datetime | None = None


# ===== 动作相关模型 =====

class ActionStep(BaseModel):
    """动作步骤"""
    type: str  # click, type_text, hotkey, etc.
    target: str | None = None  # element_id 或坐标
    value: Any = None  # 输入文本等
    params: dict[str, Any] = Field(default_factory=dict)


class ActionResult(BaseModel):
    """动作执行结果"""
    success: bool
    action: ActionStep
    message: str
    screenshot_before: str | None = None
    screenshot_after: str | None = None
    verify_result: str | None = None  # pass, fail, uncertain


# ===== 任务相关模型 =====

class TaskRequest(BaseModel):
    """任务请求"""
    app_name: str
    goal: str
    params: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    risk_level: str = "medium"


class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str
    status: str
    message: str
    data: dict[str, Any] | None = None


# ===== 操作包模型 =====

class OperationPackManifest(BaseModel):
    """操作包清单"""
    schema_version: str = "1.0.0"
    pack_version: str = "1.0.0"
    engine_compat: str = ">=1.0.0,<2.0.0"
    app_id: str
    pack_name: str
    display_name: str
    summary: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
