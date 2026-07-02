# Agent 架构设计文档

> **项目目标**：从零构建一个模块化 AI Agent，五大模块通过明确定义的接口协作，各模块可独立开发、测试、替换。

---

## 1. 整体架构

```
                        ┌──────────────────────────────────────┐
                        │           记忆模块 (Memory)            │
                        │   短期记忆 │ 长期记忆 │ 工作记忆        │
                        └──────┬────────────────────▲──────────┘
                               │ 写入               │ 检索上下文
                               ▼                    │
  ┌──────────┐    ┌──────────┐    ┌───────────────┐    ┌──────────┐
  │ 感知模块  │───▶│ 规划模块  │───▶│   LLM 模块     │◀──▶│ 工具模块  │
  │Perception│    │Planning  │    │  (内部闭环)    │    │  Tools   │
  └──────────┘    └──────────┘    │                │    └──────────┘
       │                          │ ┌────────────┐ │
       │                          │ │ Tool Loop  │ │
       │                          │ │ 自动管理   │ │
       │                          │ └────────────┘ │
       │                          └───────┬────────┘
       │                                  │
  原始输入                            最终输出
  (文本/文件)                      (给用户的回复，不含中间工具调用细节)
```

**核心数据流**：

```
原始输入 → 感知模块(标准化) → 规划模块(拆解任务) → LLM模块(推理 ↔ 工具调用 闭环)
                                    ↑                      │
                                    │    ┌─────────────────┘
                                    │    ▼
                              记忆模块(上下文)    工具模块(执行)
```

**LLM 模块内部闭环逻辑**：

```
messages ──→ [调用 qwen3.7-plus] ──→ finish_reason?
              ↑                      ├─ "stop"      → 返回最终文本 □
              │                      └─ "tool_calls" → 执行工具 →
              │                                        结果追加到消息列表 → 循环
              └──────────────────────────────────────────────┘
              （内部自动循环，对外只暴露最终结果）
```

---

## 2. 模块规格

### 2.1 感知模块 (Perception) —— ✅ 已完成

| 维度 | 说明 |
|------|------|
| **职责** | 接收多源输入，清洗、校验、标准化后输出统一 `Message` |
| **输入** | 文本字符串 / 文件路径（支持 7 种格式） |
| **输出** | `Message` 对象（role, content, source_type, metadata, is_rejected, reject_reason, attachments） |
| **内部结构** | InputRouter → TextProcessor/FileProcessor(7 Readers) → PreprocessorPipeline(4 Filters) → OutputAdapter |
| **对外接口** | `PerceptionModule.process(raw_input) -> Message` |

**待定**：
- [ ] 是否扩展支持流式输入（WebSocket / API）？
- [ ] 敏感词库是否需要热更新？

---

### 2.2 LLM 模块 —— 🔲 待开发

| 维度 | 说明 |
|------|------|
| **职责** | 封装 LLM 调用，管理 Function Calling 闭环，对外只暴露"输入消息 → 最终回复" |
| **模型** | **qwen3.7-plus**（DashScope，OpenAI 兼容协议） |
| **协议** | **OpenAI Function Calling 格式** |
| **输入** | 对话历史 + 工具定义列表 + 记忆上下文 |
| **输出** | `LLMResponse`（final_content, tool_calls_log, token_usage） |

**LLM 模块内部闭环流程**：

```
                    ┌─────────────────────────────────────────┐
                    │           LLM 模块 （内部闭环）           │
                    │                                          │
  messages ────────▶│  ① 组装 Prompt（系统提示 + 上下文 +      │
  tools ───────────▶│     历史消息 + 工具 schema）              │
                    │      │                                   │
                    │      ▼                                   │
                    │  ② 调用 qwen3.7-plus API                 │
                    │     （DashScope OpenAI 兼容接口）          │
                    │      │                                   │
                    │      ▼                                   │
                    │  ③ 判断 finish_reason                    │
                    │     ┌──────────┬──────────────┐          │
                    │     │ "stop"   │ "tool_calls" │          │
                    │     ▼          ▼               │          │
                    │  ④ 返回    ⑤ 执行工具 ────────┘          │
                    │  最终回复    结果追加到消息列表           │
                    │             loop 回到 ②                  │
                    │                                          │
                    └─────────────────────────────────────────┘
```

**核心组件**：

```
LLM 模块
├── QwenClient             # DashScope API 封装（base_url + api_key + 模型名）
├── PromptBuilder          # Message列表 + Context + Tool Schema → OpenAI 格式 messages
├── ToolLoopManager        # 管理 Function Calling 循环（最大轮数、超时、死循环检测）
├── StreamHandler          # 流式输出 → 按 chunk 回调，最终返回完整内容
└── RetryHandler           # 网络重试 + API 错误降级
```

**对外接口**：

```python
@dataclass
class LLMResponse:
    """LLM 模块对外唯一返回值 —— 已经过完整处理流程的最终回复"""
    content: str                        # 最终回复文本（给用户的）
    tool_calls_log: list[dict]          # 工具调用记录（调试/审计用）
    finish_reason: str                  # "stop" / "tool_calls" / "error"
    token_usage: dict                   # {prompt_tokens, completion_tokens, total_tokens}

class LLMModule:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3.7-plus",
        max_tool_rounds: int = 10,      # 最大工具调用轮数，防止死循环
        system_prompt: str | None = None,
    ): ...

    def chat(
        self,
        messages: list[Message],         # 对话历史
        tools: list[ToolDef] | None,     # 可用工具（由 ToolModule.get_schemas() 提供）
        context: str | None = None,      # 记忆模块注入的检索上下文
    ) -> LLMResponse: ...
```

**内部 Function Calling 循环伪代码**：

```python
def _run_tool_loop(self, messages: list[dict], tool_schemas: list[dict]) -> LLMResponse:
    """工具调用闭环 —— LLM 模块内部管理，对外透明"""
    tool_log = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for _ in range(self.max_tool_rounds):
        # 调用 API
        response = self._call_api(messages, tool_schemas)
        total_usage = _accumulate_usage(total_usage, response.usage)

        # 正常结束 → 返回文本
        if response.choices[0].finish_reason == "stop":
            return LLMResponse(
                content=response.choices[0].message.content,
                tool_calls_log=tool_log,
                finish_reason="stop",
                token_usage=total_usage,
            )

        # 工具调用 → 执行 → 结果回填
        if response.choices[0].finish_reason == "tool_calls":
            for tc in response.choices[0].message.tool_calls:
                result = self._tool_executor(tc.function.name, tc.function.arguments)
                tool_log.append({"name": tc.function.name, "args": tc.function.arguments, "result": result})
                # 追加 assistant(tool_call) + tool(result) 到消息列表
                messages.append(_make_assistant_tool_call_msg(tc))
                messages.append(_make_tool_result_msg(tc.id, result))
            continue  # 继续下一轮

        # 其他情况（内容过滤、长度限制等）
        return LLMResponse(content="", finish_reason="error", ...)

    # 超过最大轮数 → 强制结束
    return LLMResponse(content="任务执行轮数超限，已强制终止。", ...)
```

**关键设计要点**：

- **工具执行回调注入**：LLM 模块不持有 ToolsModule，而是通过构造函数接收一个 `tool_executor: Callable[[str, dict], ToolResult]`，保持模块解耦
- **最大轮数限制**：`max_tool_rounds` 默认 10，防止模型在工具调用中死循环
- **Token 累计**：多轮调用累加 token 用量，最终统一返回
- **流式输出**：当 stream=True 时，最终轮文本通过回调实时输出，中间轮的工具调用不对外暴露流

---

### 2.3 记忆模块 (Memory) —— ✅ 已完成

| 维度 | 说明 |
|------|------|
| **职责** | 管理对话历史、长期知识、当前任务上下文，为 LLM 提供精准的检索增强 |
| **输入** | 对话轮次（dict 列表）、用户偏好、实体信息 |
| **输出** | 检索到的相关上下文（Context 字符串，可直接注入 LLMModule.chat()） |

**三层记忆架构**：

```
MemoryModule
├── MemoryManager（策略层 —— 协调三层记忆的读写压缩遗忘）
│   ├── WorkingMemory（短期记忆 / 滑动窗口）
│   │   ├── deque 消息队列 + tiktoken 精确计数
│   │   ├── Token 预算控制 → 超预算自动截断旧消息
│   │   └── 摘要压缩：旧消息 → LLM 摘要 → 存入长期记忆
│   │
│   ├── LongTermMemory（长期记忆）
│   │   ├── FAISSVectorStore（FAISS IndexFlatIP + L2 归一化）
│   │   │   └── 持久化：faiss.write_index + pickle metadata
│   │   ├── RelStore（SQLite，零外部依赖）
│   │   │   ├── 用户偏好表（key-value）
│   │   │   ├── 实体表（name-type-properties）
│   │   │   └── 关系表（source-relation-target 三元组）
│   │   └── Embedder 抽象层
│   │       ├── DashScopeEmbedder（text-embedding-v3, dim=1024）
│   │       └── LocalEmbedder（sentence-transformers, 可选依赖）
│   │
│   └── SessionStore（会话 KV 存储）
│       ├── 内存 dict 实现 + TTL 过期机制
│       └── export_summary() → 注入 LLM 上下文
│
├── 记忆写入策略：自动提取偏好/实体（规则匹配 + 后续可接入 LLM）
├── 记忆检索策略：语义搜索（FAISS）+ 关键词（SQLite）+ 类型过滤
└── 记忆遗忘策略：时间衰减 + 重要性阈值 + 容量上限
```

**对外接口**：

```python
class MemoryModule:
    def __init__(
        self,
        embedder: Embedder | None = None,
        embedder_provider: str = "dashscope",
        persist_dir: str | None = None,
        max_working_tokens: int = 8000,
        system_prompt: str | None = None,
        api_key: str | None = None,
    ): ...

    def retrieve(self, query: str, top_k: int = 5) -> str: ...
    def remember(self, messages: list[dict]) -> None: ...
    def remember_item(self, content: str, memory_type: str, importance: float) -> str: ...
    def compress(self) -> str | None: ...
    def add_preference(self, key: str, value: str) -> None: ...
    def get_preference(self, key: str) -> str | None: ...
    def set_session(self, key: str, value, ttl: float = None) -> None: ...
    def get_session(self, key: str, default=None) -> Any: ...
    def run_forgetting(self) -> dict: ...
    def save(self) -> None: ...
    def clear_session(self) -> None: ...
```

**与 LLM 模块集成**：

```python
# MemoryModule.retrieve() 的返回值直接传给 LLMModule
context = memory.retrieve("用户最近在聊什么？")
response = llm.chat(messages=[...], context=context)

# 对话结束后保存
memory.remember([
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
])
```

---

### 2.4 规划模块 (Planning) —— 🔲 待开发

| 维度 | 说明 |
|------|------|
| **职责** | 将用户意图分解为可执行的步骤序列，管理任务依赖和分支逻辑 |
| **输入** | 用户 Message + 记忆模块上下文 + 可用工具列表 |
| **输出** | `TaskPlan`（步骤列表，每步包含 action/args/depends_on） |

**规划模式（按复杂度递进）**：

```
规划模块
├── ReAct 模式          # Thought → Action → Observation 循环（开箱即用）
├── Plan-and-Execute    # 先出完整计划 → 逐步执行（适合复杂任务）
├── 动态重规划          # 执行中遇错/新信息 → 调整后续步骤
└── 子任务委派          # 将子任务委派给子 Agent
```

**对外接口（草案）**：

```python
class PlanningModule:
    def create_plan(
        self,
        intent: str,
        context: str,
        available_tools: list[ToolDef],
        history: list[Message],
    ) -> TaskPlan: ...

    def revise_plan(
        self,
        current_plan: TaskPlan,
        observation: str,          # 上一步执行结果
        step_index: int,           # 当前步骤索引
    ) -> TaskPlan: ...             # 返回调整后的计划
```

**TaskPlan 数据结构**：

```python
@dataclass
class TaskStep:
    id: str                  # 步骤唯一 ID
    description: str         # 步骤描述（给用户看的）
    action: str              # 动作类型: "think" / "tool_call" / "respond"
    tool_name: str | None    # 要调用的工具名
    tool_args: dict | None   # 工具参数
    depends_on: list[str]    # 依赖的前置步骤 ID

@dataclass
class TaskPlan:
    goal: str                # 任务目标
    steps: list[TaskStep]    # 步骤列表
    status: str              # "pending" / "running" / "completed" / "failed"
```

**关键设计决策（用户决定）**：

1. **默认规划模式？**
   - [ ] ReAct（简单、稳定、适合大多数场景）
   - [ ] Plan-and-Execute（更结构化，适合多步任务）
   - [ ] 自适应（简单任务 ReAct，复杂任务 Plan-and-Execute）

2. **规划由谁执行？**
   - [ ] 由 LLM 模块生成计划（Planning 模块只是数据结构 + 状态管理）
   - [ ] Planning 模块内有独立推理逻辑（规则 + 模板）

---

### 2.5 工具模块 (Tools) —— 🔲 待开发

| 维度 | 说明 |
|------|------|
| **职责** | 工具注册、Schema 管理、调用执行、结果解析 |
| **协议** | **OpenAI Function Calling 格式**（与 LLM 模块统一） |
| **输入** | 工具调用请求 `{name, arguments}` |
| **输出** | `ToolResult`（success, content, error, metadata） |

**核心架构**：

```
工具模块
├── Tool Registry        # 工具注册中心
│   ├── 工具注册：按名称注册，包含 schema + handler
│   ├── Schema 导出：get_schemas() → OpenAI Function Calling 格式
│   │   # 输出示例: [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
│   └── 工具发现：按标签过滤
│
├── Tool Executor        # 工具执行器
│   ├── 参数校验：根据 parameters JSON Schema 校验 + 类型强制转换
│   ├── 超时控制：每个工具独立超时（默认 30s）
│   ├── 异常捕获：工具抛异常不中断主流程，包装为 ToolResult(success=False)
│   └── 结果序列化：确保返回给 LLM 的是纯文本字符串
│
└── Built-in Tools       # 内置工具集（待定，见下方决策）
```

**ToolDef 数据结构**：

```python
@dataclass
class ToolDef:
    """工具定义 —— 注册时填写"""
    name: str                           # 工具名称（唯一标识）
    description: str                    # 用途描述（给 LLM 看的，要写得清楚）
    parameters: dict                    # JSON Schema 参数定义
    handler: Callable[..., str]         # 执行函数，返回结果字符串
    require_confirm: bool = False       # 是否需要用户确认
    timeout: float = 30.0               # 超时秒数
    tags: list[str] = field(default_factory=list)  # 分类: ["network", "file", "dangerous", "readonly"]

@dataclass
class ToolResult:
    """工具执行结果 —— 统一返回格式"""
    tool_name: str
    success: bool
    content: str                        # 返回给 LLM 的结果文本
    error: str | None = None
    metadata: dict = field(default_factory=dict)  # elapsed_ms 等
```

**Schema 导出格式**（OpenAI Function Calling 标准）：

```python
def get_schemas(self) -> list[dict]:
    """导出工具列表，格式对齐 OpenAI Function Calling"""
    # 返回值示例:
    # [
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "search_web",
    #             "description": "搜索互联网获取最新信息",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {"query": {"type": "string", "description": "搜索关键词"}},
    #                 "required": ["query"]
    #             }
    #         }
    #     }
    # ]
```

**对外接口**：

```python
class ToolsModule:
    def register(self, tool: ToolDef) -> None: ...
    def unregister(self, name: str) -> None: ...
    def get_schemas(self) -> list[dict]: ...         # 导出给 LLM 模块
    def execute(self, name: str, arguments: dict) -> ToolResult: ...
    def get_executor(self) -> Callable: ...           # 返回 execute 的可调用引用，注入 LLM 模块
    def list_tools(self, tag: str | None = None) -> list[str]: ...
```

**关键设计决策（用户决定）**：

1. **工具执行安全策略？**
   - [ ] 所有工具都需要用户确认
   - [ ] 按标签区分：`dangerous` 标签需确认，其余自动执行
   - [ ] 信任模式：高风险操作（删文件/网络请求）才需要确认

2. **首批内置工具？**
   - [ ] 仅框架，工具由用户自由注册
   - [ ] 提供基础工具集（搜索、文件、代码执行）
   - [ ] 对接现有工具生态（如 LangChain Tools）

---

## 3. 模块间通信协议

所有模块通过 **Agent 编排器 (AgentOrchestrator)** 协调，模块之间不直接耦合：

```python
class AgentOrchestrator:
    """Agent 核心编排器 —— 管理五大模块的协作流程"""

    def __init__(self):
        self.perception = PerceptionModule()
        self.memory = MemoryModule()
        self.planning = PlanningModule()
        self.llm = LLMModule()
        self.tools = ToolsModule()

    def run(self, raw_input) -> str:
        """
        主执行循环：
        1. 感知输入 → Message
        2. 记忆检索 → Context
        3. 任务规划 → TaskPlan
        4. 逐步执行：LLM 推理 ↔ 工具调用
        5. 记忆写入 → 保存本轮对话
        6. 返回最终回复
        """
        ...
```

**模块依赖关系**（编译期依赖，非运行时耦合）：

```
Orchestrator
  ├── PerceptionModule    (无依赖，独立)
  ├── MemoryModule        (依赖 Message 类型)
  ├── PlanningModule      (依赖 Message, ToolDef 类型)
  ├── LLMModule            (依赖 Message, ToolDef 类型)
  └── ToolsModule          (依赖 ToolDef, ToolResult 类型)
```

---

## 4. 实现路线图

```
Phase 1: 感知模块 ✅ 已完成
Phase 2: LLM 模块    ← 当前阶段
Phase 3: 工具模块     (与 LLM 紧密配合，建议紧随其后)
Phase 4: 记忆模块     (LLM + 工具稳定后再接入长期记忆)
Phase 5: 规划模块     (复杂任务编排，放最后以利用各模块的成熟能力)
Phase 6: 编排器 + 端到端集成测试
```

**建议顺序的理由**：LLM 和工具是 Agent 的核心循环（思考 ↔ 行动），应先打通；记忆为 LLM 提供上下文增强；规划在最上层编排复杂流程。

---

## 5. 设计原则

1. **接口隔离**：模块只依赖抽象类型（Message, ToolDef, TaskPlan），不依赖具体实现
2. **可插拔**：每个模块的核心组件（如 LLM Provider、向量数据库）可替换
3. **渐进增强**：先跑通最简链路（感知 → LLM → 回复），再逐步接入工具、记忆、规划
4. **测试友好**：每个模块可独立单测，不依赖其他模块的完整实现
5. **中文优先**：代码用英文命名，关键逻辑加中文注释；面向中文场景设计

---

## 6. 决策记录

| # | 决策项 | 所属模块 | 状态 | 决定 |
|---|--------|---------|:---:|------|
| 1 | LLM Provider | LLM | ✅ | **qwen3.7-plus**（DashScope，OpenAI 兼容协议） |
| 2 | 工具调用协议 | LLM + Tools | ✅ | **OpenAI Function Calling 格式** |
| 3 | Func Call 循环管理 | LLM | ✅ | **LLM 模块内部闭环**（对外透明） |
| 4 | 向量数据库选型 | Memory | ✅ | **FAISS**（IndexFlatIP，轻量本地索引） |
| 5 | Embedding 模型选型 | Memory | ✅ | **抽象接口 + 双实现**（DashScope API + 本地 BGE） |
| 6 | 长期记忆范围 | Memory | ✅ | **全部**（对话摘要 + 用户偏好 + 实体关系 + 知识片段） |
| 7 | 会话存储实现 | Memory | ✅ | **内存 dict**（接口化，后续可换 Redis） |
| 8 | 默认规划模式 | Planning | 🟡 | 待定：ReAct / Plan-and-Execute |
| 9 | 工具安全确认策略 | Tools | ✅ | **回调注入**（require_confirm + confirm_callback） |
| 10 | 首批内置工具 | Tools | ✅ | **仅框架**，工具由用户自由注册 |
| 11 | LLM API Key 配置方式 | LLM | ✅ | **环境变量 + 显式传入**（两种方式） |
