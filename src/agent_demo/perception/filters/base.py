"""过滤器抽象基类"""

from abc import ABC, abstractmethod


class BaseFilter(ABC):
    """预处理过滤器基类 —— 每个过滤器对 data dict 做一种转换"""

    # 过滤器名称，用于调试和日志
    name: str = "base"

    @abstractmethod
    def process(self, data: dict) -> dict:
        """
        处理输入数据

        Args:
            data: {"content": str, "metadata": dict}

        Returns:
            处理后的 data dict（字段可能被修改或新增 metadata 条目）

        Raises:
            RejectException: 内容应被拦截时抛出
        """
        ...
