"""输入处理器包 - 不同类型输入的读取器"""

from .base import BaseProcessor
from .text import TextProcessor
from .file import FileProcessor

__all__ = ["BaseProcessor", "TextProcessor", "FileProcessor"]
