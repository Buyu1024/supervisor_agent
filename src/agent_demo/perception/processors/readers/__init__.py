"""文件格式读取器 —— 每种文件格式一个独立的 Reader"""

from .base import BaseFileReader
from .text_reader import TextFileReader
from .markdown_reader import MarkdownReader
from .docx_reader import DocxReader
from .pptx_reader import PptxReader
from .pdf_reader import PdfReader
from .image_reader import ImageReader

__all__ = [
    "BaseFileReader",
    "TextFileReader",
    "MarkdownReader",
    "DocxReader",
    "PptxReader",
    "PdfReader",
    "ImageReader",
]
