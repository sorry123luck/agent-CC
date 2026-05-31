"""
Page Identity — 页面模型与状态模板的核心逻辑

E Phase 2 核心模块：识别"同一个页面"和"同一个状态"。
- PageModel：同一软件的同一逻辑页面
- StateTemplate：同一页面的同一可操作状态
- LayoutFingerprint：结构化布局指纹（用于状态匹配）

状态设计原则：
  StateTemplate 关注"页面可操作状态"，不是动态内容本身。
  - 输入框空→有内容：不同 StateTemplate
  - 输入框内容 123→456：同一 StateTemplate（都是 has_text_input）
  - 菜单关闭→展开：不同 StateTemplate
  - 弹窗关闭→打开：不同 StateTemplate
  - 聊天消息新增：同一 StateTemplate（只是 CanvasSnapshot 变化）
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.perception.page_compiler_models import InteractionCanvas


# =============================================================================
# Known Process → App Mapping
# =============================================================================

KNOWN_PROCESSES: dict[str, tuple[str, str]] = {
    'wechat.exe': ('wechat', '微信'),
    'weixin.exe': ('wechat', '微信'),
    'notepad.exe': ('notepad', '记事本'),
    'notepad++.exe': ('notepadpp', 'Notepad++'),
    'chrome.exe': ('chrome', 'Chrome'),
    'msedge.exe': ('edge', 'Edge'),
    'code.exe': ('vscode', 'VS Code'),
    'explorer.exe': ('explorer', '资源管理器'),
    'cmd.exe': ('cmd', '命令提示符'),
    'powershell.exe': ('powershell', 'PowerShell'),
    'windowsterminal.exe': ('terminal', '终端'),
    'feishu.exe': ('feishu', '飞书'),
    'dingtalk.exe': ('dingtalk', '钉钉'),
    'qq.exe': ('qq', 'QQ'),
    'telegram.exe': ('telegram', 'Telegram'),
    'slack.exe': ('slack', 'Slack'),
    'discord.exe': ('discord', 'Discord'),
    'photoshop.exe': ('photoshop', 'Photoshop'),
    'figma.exe': ('figma', 'Figma'),
}

WORKFLOW_DISPLAY: dict[str, str] = {
    'chat': '聊天页面',
    'contacts': '联系人页面',
    'editor': '文档编辑页',
    'main': '主界面',
    'settings': '设置页面',
    'search': '搜索页面',
    'feed': '信息流页面',
    'detail': '详情页面',
    'gui': '主界面',
    'web': '网页',
    'app': '应用界面',
    'document': '文档页面',
    'form': '表单页面',
    'dashboard': '仪表盘',
}

# Derived: app_id → display_name (from KNOWN_PROCESSES)
_APP_DISPLAY_NAMES: dict[str, str] = {
    app_id: display_name
    for _, (app_id, display_name) in KNOWN_PROCESSES.items()
}


def infer_app_from_process(process_name: str | None) -> tuple[str, str] | None:
    """从进程名推断 (app_id, display_name)。

    优先查 KNOWN_PROCESSES，未命中时从进程名推导：
    xxx.exe → (xxx, xxx)

    Returns:
        (app_id, display_name) 或 None（process_name 为空）
    """
    if not process_name:
        return None
    key = process_name.lower()
    if key in KNOWN_PROCESSES:
        return KNOWN_PROCESSES[key]
    # 未知软件：去掉 .exe 后缀作为 app_id 和 display_name
    stem = key
    if stem.endswith(".exe"):
        stem = stem[:-4]
    return (stem, stem)


def get_app_display_name(
    app_id: str,
    process_name: str | None = None,
) -> str | None:
    """获取 app 的中文显示名。

    优先从 process_name 推断（KNOWN_PROCESSES 有中文名），再从 app_id 查找。
    都没有时，用 process_name（去 .exe）或 app_id 兜底。
    """
    if process_name:
        key = process_name.lower()
        if key in KNOWN_PROCESSES:
            return KNOWN_PROCESSES[key][1]
    # app_id 在已知表中
    if app_id in _APP_DISPLAY_NAMES:
        return _APP_DISPLAY_NAMES[app_id]
    # 兜底：用 process_name（去 .exe）或 app_id 本身
    if process_name:
        stem = process_name.lower()
        if stem.endswith(".exe"):
            stem = stem[:-4]
        return stem
    return app_id


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class LayoutFingerprint:
    """结构化布局指纹 — 用于 StateTemplate 的相似度匹配。

    设计原则：只关注可操作状态，不关注动态内容。
    - fixed_roles：存在哪些固定控件类型
    - region_structure：区域分布（哪个区域有几个元素）
    - state_flags：可操作状态标志
    - input_state：输入框是空还是有内容（不关注具体内容）
    - has_dialog / has_menu：是否有弹窗/菜单展开
    """
    fixed_roles: tuple[str, ...]           # 存在的固定控件角色（排序去重）
    region_structure: tuple[tuple[str, int], ...]  # (region_role, element_count) 排序
    state_flags: tuple[tuple[str, bool], ...]      # (flag_name, value) 排序
    element_count_bucket: str              # "0-5" / "5-20" / "20-50" / "50+"
    input_state: str                       # "empty" / "filled" / "unknown"
    has_dialog: bool
    has_menu: bool


@dataclass(frozen=True)
class PageModelData:
    """页面模型 — 跨状态的持久页面身份。"""
    page_model_id: str
    app_id: str
    page_class_prefix: str
    display_name: str | None
    surface_type: str | None
    layout_signature: str | None
    created_at: str
    last_seen_at: str
    observe_count: int
    state_count: int


@dataclass(frozen=True)
class StateTemplateData:
    """页面状态模板 — 同一页面的同一可操作状态。"""
    state_template_id: str
    page_model_id: str
    app_id: str
    page_class: str
    state_signature: str
    layout_fingerprint: str  # JSON
    state_label: str | None
    fixed_element_count: int
    total_element_count: int
    created_at: str
    last_seen_at: str
    verify_count: int
    snapshot_count: int
    max_snapshots: int


@dataclass(frozen=True)
class CanvasSnapshotRef:
    """画布快照归属。"""
    snapshot_id: str
    canvas_id: str
    page_model_id: str
    state_template_id: str
    captured_at: str
    element_count: int
    has_screenshot: bool


# =============================================================================
# Page Class Prefix Extraction
# =============================================================================


def extract_page_class_prefix(
    page_class: str,
    process_name: str | None = None,
    surface_type: str | None = None,
) -> str:
    """从 page_class 提取页面模型前缀。

    正常：wechat/chat/main/default → wechat/chat
    unknown：unknown/chat/main/wide + wechat.exe → wechat/chat
    unknown+无进程：unknown/document/main/wide → unknown/gui
    """
    parts = page_class.split("/") if page_class else []

    if len(parts) >= 2 and parts[0] != "unknown":
        return f"{parts[0]}/{parts[1]}"

    # app 从进程名推断
    app = _app_from_process(process_name)

    # workflow 优先从 page_class 第二段取（如果不是 unknown），否则从 surface_type 推断
    if len(parts) >= 2 and parts[1] != "unknown":
        workflow = parts[1]
    else:
        workflow = _workflow_from_surface(surface_type)

    return f"{app}/{workflow}"


def normalize_page_class_for_memory(
    page_class: str,
    process_name: str | None = None,
    visual_mode: str | None = None,
) -> str:
    """Normalize noisy page_class classifier output before PageModel matching.

    The generic page classifier can call the same chat surface "viewer",
    "dashboard", or "chat" depending on transient message content. Visual
    pattern mode is a stronger signal for these app families, so memory uses a
    stable workflow class while StateTemplate still separates the active state.
    """
    process = (process_name or "").lower()
    mode = visual_mode or ""
    if process == "qq.exe" and mode in {"chat_workspace", "chat_search_results"}:
        return "qq/chat/main/wide"
    if process in {"feishu.exe", "lark.exe"} and mode.startswith("collaboration_"):
        return f"{_app_from_process(process)}/collaboration/main/wide"
    return page_class


def _app_from_process(process_name: str | None) -> str:
    """从进程名提取 app 标识。

    优先使用 KNOWN_PROCESSES 映射，兜底去掉 .exe 后缀。
    """
    if not process_name:
        return "unknown"
    key = process_name.lower()
    if key in KNOWN_PROCESSES:
        return KNOWN_PROCESSES[key][0]
    return key.replace(".exe", "")


def _workflow_from_surface(surface_type: str | None) -> str:
    """从 surface_type 推断 workflow。"""
    mapping = {
        "native_uia": "gui",
        "browser": "web",
        "electron_webview": "app",
        "canvas_self_drawn": "canvas",
    }
    return mapping.get(surface_type or "", "gui")


# =============================================================================
# Layout Fingerprint
# =============================================================================


def build_layout_fingerprint(canvas: InteractionCanvas) -> LayoutFingerprint:
    """从 canvas 构建布局指纹。

    只关注可操作状态，不关注动态内容。
    """
    from src.memory.candidate_identity import FIXED_CONTROL_ROLES

    fixed_element_ids: set[str] = set()
    for element in canvas.elements:
        role = element.semantic_role.value if hasattr(element.semantic_role, "value") else str(element.semantic_role)
        if role in FIXED_CONTROL_ROLES:
            fixed_element_ids.add(element.element_id)

    # 1. 固定控件角色（排序去重）
    fixed_roles = sorted(set(
        e.semantic_role.value if hasattr(e.semantic_role, "value") else str(e.semantic_role)
        for e in canvas.elements
        if e.element_id in fixed_element_ids
    ))

    # 2. 区域结构（region_role → fixed_element_count）
    region_counts: dict[str, int] = {}
    for region in canvas.regions:
        role = region.role or "unknown"
        fixed_count = sum(1 for element_id in region.element_ids if element_id in fixed_element_ids)
        region_counts[role] = region_counts.get(role, 0) + fixed_count
    region_structure = tuple(sorted(region_counts.items()))

    # 3. 状态标志
    state_flags_dict = {}
    if canvas.page and canvas.page.state_flags:
        for k, v in sorted(canvas.page.state_flags.items()):
            if isinstance(v, bool):
                state_flags_dict[k] = v
    send_enabled = _detect_primary_send_enabled(canvas)
    if send_enabled is not None:
        state_flags_dict["send_enabled"] = send_enabled
    state_flags = tuple(sorted(state_flags_dict.items()))

    # 4. 固定元素数量区间。动态消息、通知、正文文本不应创建新 StateTemplate。
    total = len(fixed_element_ids)
    if total <= 5:
        bucket = "0-5"
    elif total <= 20:
        bucket = "5-20"
    elif total <= 50:
        bucket = "20-50"
    else:
        bucket = "50+"

    # 5. 输入框状态（只关注空/有内容，不关注具体内容）
    input_state = _detect_input_state(canvas)

    # 6. 弹窗/菜单
    has_dialog = state_flags_dict.get("dialog_open", False)
    has_menu = state_flags_dict.get("menu_open", False)

    return LayoutFingerprint(
        fixed_roles=tuple(fixed_roles),
        region_structure=region_structure,
        state_flags=state_flags,
        element_count_bucket=bucket,
        input_state=input_state,
        has_dialog=has_dialog,
        has_menu=has_menu,
    )


def _detect_input_state(canvas: InteractionCanvas) -> str:
    """检测输入框状态：empty / filled / unknown。

    只关注是否有输入框以及它是否为空，不关注具体内容。
    """
    input_roles = {"text_input", "message_input", "search_input", "password_input", "file_input"}
    for e in canvas.elements:
        role = e.semantic_role.value if hasattr(e.semantic_role, "value") else str(e.semantic_role)
        if role in input_roles:
            if _input_candidate_has_text(e):
                return "filled"
            if _input_candidate_has_overlapping_ocr_text(e, canvas):
                return "filled"
            return "empty"
    return "unknown"


_INPUT_PLACEHOLDER_TEXTS = {
    "search",
    "搜索",
    "搜索(ctrl+k)",
    "搜索 (ctrl+k)",
    "请输入",
    "输入",
    "发送给",
    "发消息",
    "说点什么",
    "message",
}

_INPUT_CONTROL_TEXTS = {
    "发送",
    "send",
    "语音",
    "表情",
    "文件",
    "截图",
    "图片",
    "更多",
    "附件",
    "剪刀",
}


def _input_candidate_has_text(candidate) -> bool:
    attrs = getattr(candidate, "attributes", None) or {}
    values = [
        getattr(candidate, "text", "") or "",
        attrs.get("input_text") or "",
        attrs.get("ocr_text") or "",
        attrs.get("value") or "",
    ]
    return any(_is_meaningful_input_text(str(value)) for value in values)


def _input_candidate_has_overlapping_ocr_text(candidate, canvas: InteractionCanvas) -> bool:
    bounds = _bounds_list(getattr(candidate, "bounds", None))
    if not bounds:
        return False
    artifacts = getattr(canvas, "artifacts", None) or {}
    for block in list(artifacts.get("ocr_blocks") or []):
        if not isinstance(block, dict):
            continue
        text = str(block.get("text") or "")
        if not _is_meaningful_input_text(text):
            continue
        block_bounds = _bounds_list(block.get("bbox") or block.get("bounds"))
        if not block_bounds:
            continue
        if _point_in_bounds(_bounds_center(block_bounds), bounds):
            return True
    return False


def _is_meaningful_input_text(text: str) -> bool:
    compact = "".join(str(text or "").split())
    if not compact:
        return False
    lowered = compact.lower()
    if lowered in _INPUT_PLACEHOLDER_TEXTS or lowered in _INPUT_CONTROL_TEXTS:
        return False
    if all(not ch.isalnum() for ch in compact):
        return False
    return True


def _bounds_list(value) -> list[int]:
    items = list(value or [])
    if len(items) != 4:
        return []
    try:
        left, top, right, bottom = [int(item) for item in items]
    except (TypeError, ValueError):
        return []
    if right <= left or bottom <= top:
        return []
    return [left, top, right, bottom]


def _bounds_center(bounds: list[int]) -> tuple[int, int]:
    left, top, right, bottom = bounds
    return (left + right) // 2, (top + bottom) // 2


def _point_in_bounds(point: tuple[int, int], bounds: list[int]) -> bool:
    x, y = point
    left, top, right, bottom = bounds
    return left <= x <= right and top <= y <= bottom


def _detect_primary_send_enabled(canvas: InteractionCanvas) -> bool | None:
    send_candidates = [element for element in canvas.elements if _is_send_role(element)]
    if not send_candidates:
        return None
    # Prefer the largest visible send-like control. Smaller child icons often
    # coexist with a parent button in chat composers.
    primary = max(send_candidates, key=lambda element: _bounds_area(_bounds_list(getattr(element, "bounds", None))))
    return _candidate_enabled(primary)


def _is_send_role(candidate) -> bool:
    role = candidate.semantic_role.value if hasattr(candidate.semantic_role, "value") else str(candidate.semantic_role)
    risk_tags = {str(item).lower() for item in list(getattr(candidate, "risk_tags", None) or [])}
    label = str(getattr(candidate, "role_label", "") or "").lower()
    text = str(getattr(candidate, "text", "") or "").lower()
    name = str(getattr(candidate, "name", "") or "").lower()
    return (
        role == "send_button"
        or "send" in risk_tags
        or text == "send"
        or "发送" in text
        or name == "send"
        or "send" in label
        or "发送" in label
    )


def _candidate_enabled(candidate) -> bool:
    attrs = getattr(candidate, "attributes", None) or {}
    for key in ("enabled", "is_enabled", "send_enabled"):
        if isinstance(attrs.get(key), bool):
            return bool(attrs[key])
    state = getattr(candidate, "state", None)
    if state is not None and hasattr(state, "enabled"):
        return bool(getattr(state, "enabled"))
    return True


def _bounds_area(bounds: list[int]) -> int:
    if not bounds:
        return 0
    left, top, right, bottom = bounds
    return max(0, right - left) * max(0, bottom - top)


def fingerprint_to_signature(fp: LayoutFingerprint) -> str:
    """将布局指纹转换为确定性 hash（用于 exact match 快速查找）。"""
    # 确定性序列化
    data = {
        "fixed_roles": list(fp.fixed_roles),
        "region_structure": [list(r) for r in fp.region_structure],
        "state_flags": [list(f) for f in fp.state_flags],
        "element_count_bucket": fp.element_count_bucket,
        "input_state": fp.input_state,
        "has_dialog": fp.has_dialog,
        "has_menu": fp.has_menu,
    }
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def compute_fingerprint_similarity(a: LayoutFingerprint, b: LayoutFingerprint) -> float:
    """计算两个布局指纹的结构化相似度（0.0-1.0）。

    权重：
    - 固定控件角色 Jaccard：0.40
    - 区域结构余弦：0.30
    - 状态标志匹配：0.20
    - 元素数量区间匹配：0.05
    - 输入框状态匹配：0.05
    """
    # 1. 固定控件角色 Jaccard
    set_a = set(a.fixed_roles)
    set_b = set(b.fixed_roles)
    if set_a or set_b:
        role_sim = len(set_a & set_b) / max(len(set_a | set_b), 1)
    else:
        role_sim = 1.0

    # 2. 区域结构余弦相似度
    dict_a = dict(a.region_structure)
    dict_b = dict(b.region_structure)
    all_regions = set(dict_a.keys()) | set(dict_b.keys())
    if all_regions:
        dot = sum(dict_a.get(r, 0) * dict_b.get(r, 0) for r in all_regions)
        norm_a = sum(v ** 2 for v in dict_a.values()) ** 0.5
        norm_b = sum(v ** 2 for v in dict_b.values()) ** 0.5
        region_sim = dot / max(norm_a * norm_b, 1e-6)
    else:
        region_sim = 1.0

    # 3. 状态标志匹配（包含 has_dialog / has_menu）
    flags_a = dict(a.state_flags)
    flags_b = dict(b.state_flags)
    # 将 has_dialog / has_menu 合并到 state_flags 统一计算
    flags_a["has_dialog"] = a.has_dialog
    flags_b["has_dialog"] = b.has_dialog
    flags_a["has_menu"] = a.has_menu
    flags_b["has_menu"] = b.has_menu
    all_flags = set(flags_a.keys()) | set(flags_b.keys())
    if all_flags:
        flag_matches = sum(1 for f in all_flags if flags_a.get(f) == flags_b.get(f))
        flag_sim = flag_matches / len(all_flags)
    else:
        flag_sim = 1.0

    # 4. 元素数量区间匹配
    bucket_sim = 1.0 if a.element_count_bucket == b.element_count_bucket else 0.0

    # 5. 输入框状态匹配
    input_sim = 1.0 if a.input_state == b.input_state else 0.0

    weighted_sim = (
        role_sim * 0.25 +
        region_sim * 0.15 +
        flag_sim * 0.35 +
        bucket_sim * 0.05 +
        input_sim * 0.20
    )

    # 状态相关字段（input_state, has_dialog, has_menu）是"硬"差异：
    # 如果这些字段不同，说明页面可操作状态不同，应该创建不同的 StateTemplate。
    # 将相似度上限限制在 0.70 以下（低于 0.80 的复用阈值）。
    if input_sim < 1.0 or a.has_dialog != b.has_dialog or a.has_menu != b.has_menu:
        weighted_sim = min(weighted_sim, 0.70)

    return weighted_sim


# =============================================================================
# State Label Classification
# =============================================================================


def classify_state_label(canvas: InteractionCanvas) -> str:
    """从 canvas 推断中文状态标签。

    基于可操作状态，不基于动态内容。
    """
    state_flags = {}
    if canvas.page and canvas.page.state_flags:
        state_flags = canvas.page.state_flags

    # 弹窗优先
    if state_flags.get("dialog_open"):
        return "弹窗打开"

    # 菜单展开
    if state_flags.get("menu_open"):
        return "菜单展开"

    # 加载中
    if state_flags.get("loading"):
        return "加载中"

    # 输入框状态
    input_state = _detect_input_state(canvas)
    if input_state == "empty":
        return "输入框为空"
    elif input_state == "filled":
        return "输入框有内容"

    # 空页面
    if len(canvas.elements) <= 3:
        return "空页面"

    return "主页面"


# =============================================================================
# Layout Signature (for PageModel matching)
# =============================================================================


def build_layout_signature(canvas: InteractionCanvas) -> str:
    """构建页面布局签名（用于 PageModel 兜底匹配）。

    与 fingerprint 不同：layout_signature 关注页面整体结构，
    不关注状态细节（如弹窗、菜单、输入框内容）。
    """
    from src.memory.candidate_identity import FIXED_CONTROL_ROLES

    # 固定控件角色集合
    roles = sorted(set(
        e.semantic_role.value if hasattr(e.semantic_role, "value") else str(e.semantic_role)
        for e in canvas.elements
        if (e.semantic_role.value if hasattr(e.semantic_role, "value") else str(e.semantic_role)) in FIXED_CONTROL_ROLES
    ))

    # 区域角色集合
    region_roles = sorted(set(r.role for r in canvas.regions if r.role))

    # surface_type
    surface = canvas.surface.surface_type.value if canvas.surface else "unknown"

    data = {
        "roles": roles,
        "region_roles": region_roles,
        "surface": surface,
    }
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
