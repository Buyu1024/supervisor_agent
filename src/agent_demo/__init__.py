"""AgentDemo —— 模块化 AI Agent 框架

五大模块 + 中央编排器:
    - PerceptionModule  — 感知模块（多格式输入 → 标准化 Message）
    - MemoryModule       — 记忆模块（三层记忆：工作 + 长期 + 会话）
    - PlanningModule     — 规划模块（纯状态机 Plan-and-Execute）
    - LLMModule          — LLM 模块（qwen3.7-plus + Function Calling 闭环）
    - ToolsModule        — 工具模块（注册、Schema 导出、执行）
    - AgentOrchestrator  — 编排器（五大模块的中央协调者）

快速开始:
    from agent_demo import AgentOrchestrator

    agent = AgentOrchestrator()
    result = agent.run("你好！")
    print(result.content)
"""

from .perception import PerceptionModule, Message, RejectException
from .memory import MemoryModule, MemoryItem, MemorySearchResult
from .planning import PlanningModule, TaskPlan, TaskStep, StepResult
from .llm import LLMModule, LLMResponse
from .tools import ToolsModule, ToolDef, ToolResult
from .orchestrator import AgentOrchestrator, OrchestratorConfig, OrchestratorResult

__all__ = [
    # 编排器
    "AgentOrchestrator",
    "OrchestratorConfig",
    "OrchestratorResult",
    # 五大模块
    "PerceptionModule",
    "MemoryModule",
    "PlanningModule",
    "LLMModule",
    "ToolsModule",
    # 核心类型
    "Message",
    "RejectException",
    "MemoryItem",
    "MemorySearchResult",
    "TaskPlan",
    "TaskStep",
    "StepResult",
    "LLMResponse",
    "ToolDef",
    "ToolResult",
]
