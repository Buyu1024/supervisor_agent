"""预处理管道 —— 责任链模式串联各过滤器"""

from .filters.base import BaseFilter
from .message import RejectException


class PreprocessorPipeline:
    """
    预处理管道：
    - 按注册顺序依次执行过滤器
    - 任何一个过滤器抛出 RejectException 时，管道返回 None
    """

    def __init__(self):
        self._filters: list[BaseFilter] = []

    def add_filter(self, filter_obj: BaseFilter) -> None:
        """向管道末尾追加一个过滤器"""
        self._filters.append(filter_obj)

    def run(self, data: dict) -> dict | None:
        """
        按序执行所有过滤器

        Args:
            data: {"content": str, "metadata": dict}

        Returns:
            处理后的 data dict；被 RejectException 拦截时返回 None
        """
        try:
            for f in self._filters:
                data = f.process(data)
            return data
        except RejectException:
            return None
