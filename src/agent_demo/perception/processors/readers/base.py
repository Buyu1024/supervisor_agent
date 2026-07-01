"""文件格式读取器抽象基类"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseFileReader(ABC):
    """文件格式读取器基类 —— 每种文件格式一个子类"""

    # 子类必须覆盖：该 Reader 支持的文件后缀集合
    suffixes: set[str] = set()

    @abstractmethod
    def read(self, path: Path) -> str:
        """读取文件，返回提取到的纯文本内容"""
        ...
