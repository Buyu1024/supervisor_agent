"""规划模块 —— Agent 的任务规划与执行编排

职责:
    - 计划生成: 构建 prompt，让 LLM 将用户意图分解为步骤序列
    - 状态管理: 追踪每一步的执行状态（pending/running/completed/failed）
    - 依赖解析: 按拓扑顺序执行步骤，检查前置依赖
    - 重规划: 步骤失败时重新规划后续步骤

设计理念（借鉴 Claude Code + Codex + Anthropic 指导）:
    - 纯状态机: 不持有 LLM/Tools/Memory，由 Orchestrator 驱动
    - Plan-and-Execute: 先生成完整计划，再逐步执行
    - 计划质量取决于 prompt 工程
    - 计划是活文档（可序列化、可恢复、可修订）
    - 一次只做一件事（one feature per step）
"""

from .module import PlanningModule
from .types import TaskPlan, TaskStep, StepResult
from .executor import PlanExecutor
from .prompt_builder import PlanPromptBuilder

__all__ = [
    "PlanningModule",
    "TaskPlan",
    "TaskStep",
    "StepResult",
    "PlanExecutor",
    "PlanPromptBuilder",
]
