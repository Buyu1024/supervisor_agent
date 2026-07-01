"""输入路由器 —— 根据输入类型分发给对应处理器"""

from .processors.base import BaseProcessor
from .processors.text import TextProcessor
from .processors.file import FileProcessor


class InputRouter:
    """输入路由器：遍历已注册的处理器，返回第一个能处理该输入的处理器"""

    def __init__(self):
        # 存放已注册的处理器（按注册顺序匹配）
        self._processors: list[BaseProcessor] = []
        # 注册内置处理器（File 优先：先检查是否为有效文件路径，否则当作文本）
        self.register(FileProcessor())
        self.register(TextProcessor())

    def register(self, processor: BaseProcessor) -> None:
        """注册一个新的输入处理器（支持后期扩展，如 API、WebSocket 等）"""
        self._processors.append(processor)

    def dispatch(self, raw_input) -> BaseProcessor:
        """遍历处理器列表，返回第一个 can_handle 为 True 的处理器"""
        for processor in self._processors:
            if processor.can_handle(raw_input):
                return processor
        raise ValueError(
            f"无法识别的输入类型: {type(raw_input).__name__}。"
            f"支持的输入: 文本字符串 或 文件路径"
        )
