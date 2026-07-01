"""处理器抽象基类"""

from abc import ABC, abstractmethod
from typing import Any


class BaseProcessor(ABC):
    """输入处理器基类 —— 定义处理器接口，子类实现具体输入类型的读取逻辑"""

    # 输入来源标识，子类必须覆盖（如 "text"、"file"）
    source_type: str = ""

    @abstractmethod
    def can_handle(self, raw_input: Any) -> bool:
        """判断能否处理该输入"""
        ...

    @abstractmethod
    def read(self, raw_input: Any) -> str:
        """读取原始输入，返回纯文本内容"""
        ...
