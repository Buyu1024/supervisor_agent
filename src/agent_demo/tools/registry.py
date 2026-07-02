"""ToolRegistry —— 工具注册中心

职责：
    1. 工具注册 / 注销（支持 ToolDef 对象和装饰器两种方式）
    2. 按名称查找工具
    3. 导出 OpenAI Function Calling 格式的 schema 列表
    4. 按标签筛选工具
"""

import logging
from .types import ToolDef

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册中心 —— 管理所有已注册的工具定义"""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    # ---- 注册 / 注销 ----

    def register(self, tool: ToolDef) -> None:
        """注册工具（ToolDef 对象方式）"""
        if tool.name in self._tools:
            logger.warning(f"工具 '{tool.name}' 已存在，将被覆盖")
        self._tools[tool.name] = tool
        logger.info(f"工具已注册: {tool.name} (tags={tool.tags})")

    def register_decorator(
        self,
        name: str | None = None,
        description: str = "",
        parameters: dict | None = None,
        require_confirm: bool = False,
        timeout: float = 30.0,
        tags: list[str] | None = None,
    ):
        """
        装饰器注册方式 —— 将普通函数包装为工具并注册

        使用示例:
            @tools.register_decorator(
                description="搜索网页",
                parameters={"type": "object", "properties": {...}, "required": [...]}
            )
            def search_web(query: str) -> str:
                ...

        Args:
            name: 工具名，None 则使用函数名
            description: 给 LLM 看的用途描述
            parameters: JSON Schema 参数定义
            require_confirm: 是否需要用户确认
            timeout: 超时秒数
            tags: 分类标签
        """
        def decorator(func):
            tool_name = name or func.__name__
            tool = ToolDef(
                name=tool_name,
                description=description or func.__doc__ or "",
                parameters=parameters or {"type": "object", "properties": {}, "required": []},
                handler=func,
                require_confirm=require_confirm,
                timeout=timeout,
                tags=tags or [],
            )
            self.register(tool)
            return func  # 返回原函数，不影响其独立使用

        return decorator

    def unregister(self, name: str) -> None:
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"工具已注销: {name}")

    # ---- 查询 ----

    def get(self, name: str) -> ToolDef | None:
        """按名称获取工具定义"""
        return self._tools.get(name)

    def list_names(self, tag: str | None = None) -> list[str]:
        """列出所有工具名，可按标签过滤"""
        if tag:
            return [name for name, t in self._tools.items() if tag in t.tags]
        return list(self._tools.keys())

    # ---- Schema 导出 ----

    def get_schemas(self) -> list[dict]:
        """
        导出为 OpenAI Function Calling 格式的工具定义列表

        返回值示例:
            [
                {
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "搜索网页获取最新信息",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string", "description": "搜索关键词"}},
                            "required": ["query"]
                        }
                    }
                }
            ]
        """
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return schemas
