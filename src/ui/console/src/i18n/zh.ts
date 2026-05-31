/**
 * 中文字段映射 — E Phase 2 控制台中文化
 */

// 页面模型相关
export const pageModelLabels: Record<string, string> = {
  page_model: '页面模型',
  state_template: '页面状态',
  canvas_snapshot: '本次截图',
  stable_key_id: '稳定标识',
};

// 元素相关
export const elementLabels: Record<string, string> = {
  semantic_role: '语义角色',
  confidence: '可信度',
  provider_sources: '感知来源',
  surface_type: '窗口类型',
  page_class: '页面类型',
  element_id: '元素ID',
  control_type: '控件类型',
  risk_level: '风险等级',
  refine_status: '复核状态',
  visual_type: '视觉类型',
};

// 通用翻译
export const commonLabels: Record<string, string> = {
  unknown: '未知',
  temporary: '临时',
};

// 语义角色翻译
export const semanticRoleLabels: Record<string, string> = {
  // 通用
  unknown: '未知',
  container: '容器',
  layout: '布局',

  // 输入类
  message_input: '消息输入框',
  text_input: '文本输入框',
  search_input: '搜索框',
  password_input: '密码框',
  file_input: '文件选择',

  // 按钮类
  button: '按钮',
  send_button: '发送按钮',
  submit_button: '提交按钮',
  cancel_button: '取消按钮',
  icon_button: '图标按钮',
  toggle_button: '切换按钮',

  // 列表类
  list_item: '列表项',
  chat_item: '聊天项',
  tree_item: '树节点',
  menu_item: '菜单项',

  // 内容类
  text: '文本',
  image: '图像',
  link: '链接',
  tab: '标签页',

  // 导航类
  nav_item: '导航项',
  sidebar: '侧边栏',
  toolbar: '工具栏',
  menu_bar: '菜单栏',
  title_bar: '标题栏',
  status_bar: '状态栏',
};

// 窗口类型翻译
export const surfaceTypeLabels: Record<string, string> = {
  native_uia: '原生控件',
  browser: '浏览器',
  electron_webview: 'Electron',
  canvas_self_drawn: '自绘界面',
  unknown: '未知',
};

// 状态标签翻译
export const stateLabels: Record<string, string> = {
  // 页面状态
  '输入框为空': '输入框为空',
  '输入框有内容': '输入框有内容',
  '菜单展开': '菜单展开',
  '弹窗打开': '弹窗打开',
  '加载中': '加载中',
  '空页面': '空页面',
  '主页面': '主页面',

  // 元素状态
  stable: '稳定',
  loading: '加载中',
  partial: '部分',
  refined: '已复核',
  uncertain: '待确认',
  unreviewed: '未复核',
};

// 风险等级翻译
export const riskLevelLabels: Record<string, string> = {
  L0: '低风险',
  L1: '普通',
  L2: '高风险',
  L3: '极高风险',
};

// 可信度等级翻译
export const confidenceLevelLabels: Record<string, string> = {
  high: '高',
  medium: '中',
  low: '低',
};

// 感知源标签
export const providerLabels: Record<string, string> = {
  uia: '原生控件',
  ocr: '文字识别',
  vision: '视觉候选',
  vlm: '视觉模型',
  dom: '网页结构',
  memory: '记忆',
  omniparser: '视觉解析',
  heuristic: '规则推断',
};

// 动作标签
export const actionLabels: Record<string, string> = {
  observe: '观察',
  query: '查询',
  diff: '对比',
  remember: '记忆',
  feedback: '反馈',
  refine: '复核',
};

// 模型匹配状态标签
export const modelMatchStatusLabels: Record<string, string> = {
  reused: '模型复用',
  new_page: '新页面',
  new_state: '新状态',
};

// 动态内容标签
export const dynamicContentLabels: Record<string, string> = {
  dynamic_content: '动态内容',
  chat: '聊天',
  editor: '编辑器',
  canvas_doc_viewer: '文档查看',
  timeline_media: '时间线',
};

// UI 标签
export const uiLabels: Record<string, string> = {
  providers: '感知源',
  'heuristic_refine': '规则复核',
  'refining...': '复核中...',
  'observe': '观察',
  'canvas_cache': '画布缓存',
  'element_list': '元素列表',
  'no_elements': '无匹配元素',
  'search_hint': '搜索文本/ID/角色...',
  'all': '全部',
  'low_confidence': '低置信度',
  'high_risk': '高风险',
  'interactable': '可交互',
  'from_memory': '来自记忆',
  'refined': '已复核',
  'uncertain': '待确认',
  'group_by_region': '按区域',
  'group_by_role': '按角色',
  'group_by_provider': '按来源',
  'unassigned_region': '未分配区域',
};

/**
 * 获取翻译后的标签。
 *
 * 搜索顺序（首个匹配生效）：
 * 1. pageModelLabels — 页面模型字段
 * 2. elementLabels — 元素字段
 * 3. semanticRoleLabels — 语义角色
 * 4. surfaceTypeLabels — 窗口类型
 * 5. stateLabels — 状态标签
 * 6. riskLevelLabels — 风险等级
 * 7. confidenceLevelLabels — 可信度等级
 * 8. commonLabels — 通用翻译
 * 9. providerLabels — 感知源
 * 10. actionLabels — 操作名称
 * 11. uiLabels — UI 文本
 *
 * 注意：同名 key 会按上述优先级被先匹配的 map 吞掉。
 * 新增 key 时请确认不会与已有 map 中的 key 冲突。
 */
export function t(key: string, fallback?: string): string {
  return (
    pageModelLabels[key] ??
    elementLabels[key] ??
    semanticRoleLabels[key] ??
    surfaceTypeLabels[key] ??
    stateLabels[key] ??
    riskLevelLabels[key] ??
    confidenceLevelLabels[key] ??
    commonLabels[key] ??
    providerLabels[key] ??
    actionLabels[key] ??
    modelMatchStatusLabels[key] ??
    dynamicContentLabels[key] ??
    uiLabels[key] ??
    fallback ??
    key
  );
}

/**
 * 格式化稳定标识（只显示前 8 位）
 */
export function formatStableKeyId(stableKeyId: string | null | undefined): string {
  if (!stableKeyId) return commonLabels.temporary;
  return stableKeyId.slice(0, 8);
}
