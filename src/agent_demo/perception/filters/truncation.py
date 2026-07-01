"""长度截断过滤器 —— 超过上限时截断，并记录 metadata"""

from .base import BaseFilter


class TruncationFilter(BaseFilter):
    """
    长度截断：
    - content 超过 max_length 时，保留前 max_length 个字符
    - metadata 中记录截断标记和原始长度
    """

    name = "truncation"

    def __init__(self, max_length: int = 4000):
        self.max_length = max_length

    def process(self, data: dict) -> dict:
        text = data["content"]
        original_length = len(text)

        data["metadata"]["original_length"] = original_length

        if original_length > self.max_length:
            data["content"] = text[:self.max_length]
            data["metadata"]["truncated"] = True
            data["metadata"]["truncated_length"] = self.max_length
        else:
            data["metadata"]["truncated"] = False

        return data
