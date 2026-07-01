"""LLM 模块 —— 数据结构定义"""

from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    """LLM 模块对外唯一返回值 —— 已经过工具调用闭环的最终结果"""

    content: str = ""                               # 最终回复文本（给用户的）
    finish_reason: str = ""                         # "stop" / "tool_calls" / "error" / "max_rounds"
    token_usage: dict = field(default_factory=dict) # {prompt_tokens, completion_tokens, total_tokens}
    tool_calls_log: list[dict] = field(default_factory=list)
    # tool_calls_log 每项: {"round": int, "name": str, "arguments": dict, "result": str}

    def __repr__(self) -> str:
        rounds = len(self.tool_calls_log)
        return (
            f"<LLMResponse finish={self.finish_reason} "
            f"len={len(self.content)} tool_rounds={rounds} "
            f"tokens={self.token_usage.get('total_tokens', 0)}>"
        )
