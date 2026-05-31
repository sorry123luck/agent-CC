# IMPLEMENT.md

> 本文件是公开版实施规则。内部 agent 运行记录不属于公开仓库内容。

---

## 当前实施口径

DeskCanvas 是一个**桌面感知服务**，核心是把真实窗口转换成 InteractionCanvas。

- 软件负责：感知、画布生成、候选查询、差异比较、记忆、反馈
- Agent 负责：决策、业务流程
- act 是可选执行适配层，不是核心协议

---

## 当前数据模型

### InteractionCanvas

核心输出。1:1 对应真实窗口的结构化地图。

### Candidate

每个可操作元素是一个候选点位，包含语义角色、文字、位置、可信度、风险标签、定位器、证据。

### Observed Transition Graph

页面状态之间的历史转移记录，不是自动执行流程。

---

## 当前 API 端点

### 核心协议（5 个）

| 端点 | 说明 |
|------|------|
| `POST /api/v1/observe` | 生成 InteractionCanvas 并缓存 |
| `POST /api/v1/query` | 基于 canvas_id 查询候选点位 |
| `POST /api/v1/diff` | 对比前后两个画布 |
| `POST /api/v1/remember` | 将画布写入记忆系统 |
| `POST /api/v1/feedback` | 记录 Agent 反馈 |

### 可选执行适配层

| 端点 | 说明 |
|------|------|
| `POST /api/v1/act` | 执行 Agent 明确指定的候选动作，不做决策 |

### 辅助端点

| 端点 | 说明 |
|------|------|
| `GET /api/v1/capabilities` | 查询软件支持哪些能力 |
| `GET /api/v1/apps/search` | 搜索本机软件 |
| `POST /api/v1/apps/launch` | 启动软件 |

---

## 当前代码主链

### 感知

- `src/perception/perception_service.py` — 多源感知入口
- `src/perception/page_compiler.py` — InteractionCanvasEngine
- `src/perception/page_compiler_models.py` — InteractionCanvas / Candidate 模型

### 画布引擎

- `src/canvas/query_engine.py` — CanvasQueryEngine
- `src/canvas/diff_engine.py` — CanvasDiffEngine
- `src/canvas/canvas_cache.py` — 进程内 LRU 缓存

### 记忆

- `src/memory/feedback_manager.py` — Agent 反馈管理
- `src/memory/transition_graph.py` — Observed Transition Graph
- `src/memory/page_template_manager.py` — 模板管理

### 集成

- `src/integration/api_server.py` — API 服务器（所有端点）
- `src/integration/api_models.py` — API 请求/响应模型

### 通用

- `src/common/privacy.py` — 字段级脱敏
- `src/common/config_manager.py` — 配置管理

---

## 感知源定位

| 源 | 定位 |
|---|---|
| UIA | 原生 Win32/WPF 应用首选 |
| PaddleOCR | 文本提取、标签补充 |
| OmniParser | 感知增强 Provider，不是唯一真相源 |
| VLM | 感知增强 Provider，不暴露为独立 API |
| Memory | 历史模板和经验 |
| DOM | browser surface_type 主链 |

---

## 施工规则

1. 先计划，后实现
2. 每个工作包有明确的"完成定义"和"验证方法"
3. 验证通过才算完成，没通过必须先修
4. 严格控制 scope，不做工作包外的事情
5. 用 STATUS.md 持续记录进展
6. 每个工作包开始前确认边界：改哪些文件、输出什么、验收标准
7. 开发 Agent、审查 Agent、测试 Agent 分开
8. 不允许只说"测试通过"，必须说明用户能看到什么、Agent 能调用什么

---

## 测试基线

```bash
pytest tests/unit/ -q
```

当前结果：**347 passed, 13 warnings**

---

## 入口文件（开工必读）

1. `CLAUDE.md` — 项目总线约束（最高优先级）
2. `CURRENT_MILESTONE.md` — 当前施工主线
3. `STATUS.md` — 模块状态总览
4. `PLANS.md` — 工作包计划
5. `docs/current/refactored-sprouting-dijkstra.md` — 最终方案
