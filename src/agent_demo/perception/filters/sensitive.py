"""敏感词过滤器 —— 命中敏感词则抛出 RejectException，整条内容被拒绝"""

from pathlib import Path
from .base import BaseFilter
from ..message import RejectException


class SensitiveFilter(BaseFilter):
    """
    敏感词过滤：
    - 从外部文件加载词库（每行一个词，# 开头为注释）
    - 任意一个敏感词命中，立即抛出 RejectException
    """

    name = "sensitive"

    def __init__(self, word_file_path: str | Path | None = None):
        """
        Args:
            word_file_path: 敏感词文件路径，None 时使用空词库（不过滤）
        """
        self._words: set[str] = set()
        if word_file_path:
            self._load_words(Path(word_file_path))

    def _load_words(self, path: Path) -> None:
        """从文件逐行加载敏感词，自动跳过空行和注释行"""
        if not path.exists():
            raise FileNotFoundError(f"敏感词文件不存在: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith('#'):
                    self._words.add(word)

    @property
    def word_count(self) -> int:
        """已加载的敏感词数量"""
        return len(self._words)

    def add_word(self, word: str) -> None:
        """运行时动态追加敏感词"""
        self._words.add(word)

    def process(self, data: dict) -> dict:
        """遍历敏感词，命中任何一个则抛出 RejectException"""
        if not self._words:
            return data

        text = data["content"]
        for word in self._words:
            if word in text:
                raise RejectException(f"内容包含敏感词")

        return data
