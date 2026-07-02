"""编排器模块 —— 五大模块的中央协调者

AgentOrchestrator 是 Agent 的唯一对外入口。用户只需创建编排器并调用 run()，
内部自动完成感知 → 记忆 → 规划 → 执行 → 收尾的完整流程。

使用示例:
    from agent_demo.orchestrator import AgentOrchestrator

    agent = AgentOrchestrator()
    result = agent.run("帮我搜索 Python 和 Rust 的性能对比")
    print(result.content)
    print(f"工具调用: {result.total_tool_calls} 次")
"""

from .orchestrator import AgentOrchestrator
from .config import OrchestratorConfig, OrchestratorResult

__all__ = [
    "AgentOrchestrator",
    "OrchestratorConfig",
    "OrchestratorResult",
]
