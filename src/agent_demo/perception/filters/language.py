"""语言检测过滤器 —— 识别文本语言，仅记录到 metadata"""

from langdetect import detect, DetectorFactory
from .base import BaseFilter

# 固定随机种子，确保检测结果可复现
DetectorFactory.seed = 0


class LanguageFilter(BaseFilter):
    """
    语言检测：
    - 通过 langdetect 库识别文本语种
    - 结果写入 metadata["language"]，不做额外干预
    """

    name = "language"

    def process(self, data: dict) -> dict:
        text = data["content"]

        if not text.strip():
            data["metadata"]["language"] = "unknown"
            return data

        try:
            data["metadata"]["language"] = detect(text)
        except Exception:
            # 文本过短或无法识别时标记为 unknown
            data["metadata"]["language"] = "unknown"

        return data
