"""纯文本输入处理器"""

from .base import BaseProcessor


class TextProcessor(BaseProcessor):
    """处理用户直接输入的文本字符串"""

    source_type = "text"

    def can_handle(self, raw_input) -> bool:
        # 纯文本的前提：是字符串且不是文件路径
        return isinstance(raw_input, str)

    def read(self, raw_input: str) -> str:
        return raw_input
