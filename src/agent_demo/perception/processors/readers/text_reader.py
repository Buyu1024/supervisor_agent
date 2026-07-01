"""纯文本文件读取器 —— 处理 .txt .md .json 等文本类文件"""

from pathlib import Path
from .base import BaseFileReader


class TextFileReader(BaseFileReader):
    """读取编码文本文件（UTF-8 优先，失败回退 GBK）"""

    suffixes = {
        '.txt', '.json', '.csv', '.py', '.yaml', '.yml',
        '.log', '.xml', '.html', '.toml', '.cfg', '.ini', '.env'
    }
    # 注: .md / .markdown 由 MarkdownReader 专门处理

    def read(self, path: Path) -> str:
        try:
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return path.read_text(encoding='gbk')
