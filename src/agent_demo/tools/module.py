"""ToolsModule —— 工具模块主入口，组装 ToolRegistry + ToolExecutor"""

import logging
from typing import Callable
from .types import ToolDef, ToolResult
from .registry import ToolRegistry
from .executor import ToolExecutor

logger = logging.getLogger(__name__)


class ToolsModule:
    """
    工具模块 —— 管理工具的注册、Schema 导出与执行

    使用示例:
        # 初始化（confirm_callback 可选，不传则所有工具自动执行）
        tools = ToolsModule(confirm_callback=my_confirm_fn)

        # 方式一：装饰器注册
        @tools.register(description="搜索网页", parameters={...})
        def search_web(query: str) -> str:
            return f"搜索结果: {query}"

        # 方式二：ToolDef 对象注册
        tools.register(ToolDef(name="calc", description="...", parameters={...}, handler=calc_fn))

        # 导出给 LLM 模块
        llm = LLMModule(tool_executor=tools.get_executor())
        response = llm.chat(messages=[...], tools=tools.get_schemas())
    """

    def __init__(self, confirm_callback: Callable | None = None):
        """
        Args:
            confirm_callback: (tool_name, arguments) -> bool
                              True = 允许执行, False = 拦截
                              None = 所有工具自动放行（默认）
        """
        self._registry = ToolRegistry()
        self._executor = ToolExecutor(confirm_callback=confirm_callback)

    # ---- 注册 / 注销 ----

    def register(self, tool: ToolDef | None = None, **kwargs):
        """
        注册工具 —— 两种方式:

        # 方式一：直接传入 ToolDef 对象
        tools.register(ToolDef(name="foo", description="...", parameters={...}, handler=fn))

        # 方式二：装饰器（kwargs 传入 name/description/parameters 等）
        @tools.register(description="搜索网页", parameters={...})
        def search_web(query: str) -> str: ...

        当 tool 为 None 时进入装饰器模式，否则直接注册 ToolDef。
        """
        # 方式一：ToolDef 对象直接注册
        if tool is not None:
            self._registry.register(tool)
            return

        # 方式二：装饰器模式
        return self._registry.register_decorator(**kwargs)

    def unregister(self, name: str) -> None:
        """注销工具"""
        self._registry.unregister(name)

    # ---- 查询 ----

    def list_tools(self, tag: str | None = None) -> list[str]:
        """列出已注册工具名，可按标签过滤"""
        return self._registry.list_names(tag)

    # ---- Schema 导出 ----

    def get_schemas(self) -> list[dict]:
        """导出 OpenAI Function Calling 格式的工具定义列表"""
        return self._registry.get_schemas()

    # ---- 执行 ----

    def execute(self, name: str, arguments: dict) -> ToolResult:
        """
        按名称执行工具

        Args:
            name: 工具名称
            arguments: LLM 传入的参数字典

        Returns:
            ToolResult
        """
        tool = self._registry.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name,
                success=False,
                content=f"错误：未找到工具 '{name}'，可用工具: {self._registry.list_names()}",
                error="tool_not_found",
            )
        return self._executor.execute(tool, arguments)

    def get_executor(self) -> Callable[[str, dict], ToolResult]:
        """
        返回工具执行回调，供 LLM 模块注入

        返回的 Callable 签名: (name: str, arguments: dict) -> ToolResult
        """
        return self.execute
