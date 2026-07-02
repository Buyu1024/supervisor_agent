"""工具模块 —— 数据结构定义"""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolDef:
    """工具定义 —— 注册时填写"""

    name: str                                   # 工具名称（唯一标识）
    description: str                            # 用途描述（给 LLM 看的）
    parameters: dict                            # JSON Schema 参数定义
    handler: Callable[..., str]                 # 执行函数，返回结果字符串
    require_confirm: bool = False               # 执行前是否需要用户确认
    timeout: float = 30.0                       # 超时秒数
    tags: list[str] = field(default_factory=list)  # 分类标签: ["network", "file", "dangerous", "readonly"]

    def __repr__(self) -> str:
        return f"<ToolDef name={self.name} confirm={self.require_confirm} tags={self.tags}>"


@dataclass
class ToolResult:
    """工具执行结果 —— 统一返回格式"""

    tool_name: str
    success: bool
    content: str                                # 返回给 LLM 的结果文本
    error: str | None = None
    metadata: dict = field(default_factory=dict)  # elapsed_ms, confirmed 等

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return (
            f"<ToolResult {self.tool_name} {status} "
            f"len={len(self.content)}>"
        )
