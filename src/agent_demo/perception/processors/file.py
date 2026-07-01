"""文件输入处理器 —— 根据后缀委托给对应的 Reader 读取"""

from pathlib import Path
from .base import BaseProcessor
from .readers.base import BaseFileReader
from .readers.text_reader import TextFileReader
from .readers.markdown_reader import MarkdownReader
from .readers.docx_reader import DocxReader
from .readers.pptx_reader import PptxReader
from .readers.pdf_reader import PdfReader
from .readers.image_reader import ImageReader


class FileProcessor(BaseProcessor):
    """处理文件路径输入，根据文件后缀委托给对应的格式 Reader 读取"""

    source_type = "file"

    def __init__(self):
        # 注册所有格式 Reader（按优先级排列，TextFileReader 兜底放最后）
        self._readers: list[BaseFileReader] = [
            DocxReader(),
            PptxReader(),
            PdfReader(),
            ImageReader(),
            MarkdownReader(),   # Markdown 优先匹配 → 可处理图片引用
            TextFileReader(),   # 纯文本格式，兜底
        ]
        # 汇总所有支持的后缀
        self._suffixes: set[str] = set()
        for reader in self._readers:
            self._suffixes.update(reader.suffixes)

    @property
    def supported_suffixes(self) -> set[str]:
        """返回所有支持的文件后缀"""
        return self._suffixes.copy()

    def register_reader(self, reader: BaseFileReader) -> None:
        """注册新 Reader（支持后期扩展）"""
        self._readers.insert(0, reader)  # 新 Reader 优先级高于文本
        self._suffixes.update(reader.suffixes)

    def can_handle(self, raw_input) -> bool:
        """仅当输入为存在的文件路径且后缀在支持列表中时才处理"""
        if not isinstance(raw_input, (str, Path)):
            return False
        path = Path(raw_input)
        return (
            path.exists()
            and path.is_file()
            and path.suffix in self._suffixes
        )

    def read(self, raw_input) -> str:
        """遍历 Reader 列表，找到第一个能处理该后缀的 Reader"""
        path = Path(raw_input)
        for reader in self._readers:
            if path.suffix in reader.suffixes:
                return reader.read(path)
        # 正常情况下不会走到这里（can_handle 已过滤）
        raise ValueError(f"不支持的文件格式: {path.suffix}")
