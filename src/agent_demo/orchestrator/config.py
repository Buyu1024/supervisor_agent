"""编排器配置与结果类型"""

from dataclasses import dataclass, field

from ..llm.client import DASHSCOPE_BASE_URL, DEFAULT_MODEL
from ..planning.types import TaskPlan


@dataclass
class OrchestratorConfig:
    """一站式配置 —— Orchestrator 负责分发给各子模块

    使用示例:
        # 全默认（API Key 从环境变量读取）
        config = OrchestratorConfig()

        # 自定义
        config = OrchestratorConfig(
            api_key="sk-xxx",
            model="qwen3.7-plus",
            memory_persist_dir="data/memory",
            max_working_tokens=4000,
        )
        agent = AgentOrchestrator(config)
    """

    # ---- LLM ----
    api_key: str | None = None          # None → 从环境变量 DASHSCOPE_API_KEY 读取
    base_url: str = DASHSCOPE_BASE_URL
    model: str = DEFAULT_MODEL
    max_tool_rounds: int = 10           # 单步最大工具调用轮数

    # ---- Memory ----
    memory_persist_dir: str | None = None  # None → 纯内存模式
    max_working_tokens: int = 8000
    embedder_provider: str = "dashscope"   # "dashscope" | "local"

    # ---- Planning ----
    max_retries_per_step: int = 2
    max_revisions: int = 3

    # ---- Perception ----
    max_input_length: int = 4000

    # ---- System ----
    system_prompt: str | None = None    # None → 使用内置默认提示词


@dataclass
class OrchestratorResult:
    """编排器统一返回 —— 包含执行全过程的信息

    不论简单对话还是复杂任务，都通过此结构返回。
    """

    content: str                                    # 最终回复文本
    success: bool                                   # 整体是否成功
    plan: TaskPlan | None = None                    # 执行的计划（调试/审计用）
    total_tool_calls: int = 0                       # 总工具调用次数
    total_tokens: dict = field(default_factory=dict)  # {prompt_tokens, completion_tokens, total_tokens}
    error: str | None = None                        # 失败原因

    def __repr__(self) -> str:
        steps = len(self.plan.steps) if self.plan else 0
        return (
            f"<OrchestratorResult success={self.success} "
            f"steps={steps} tools={self.total_tool_calls} "
            f"tokens={self.total_tokens.get('total_tokens', 0)} "
            f"len={len(self.content)}>"
        )
