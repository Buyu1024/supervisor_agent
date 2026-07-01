"""去噪过滤器 —— 清洗文本格式"""

import re
from .base import BaseFilter


class DenoiseFilter(BaseFilter):
    """
    文本去噪处理：
    1. 移除控制字符（保留 \\n \\t）
    2. 统一换行符
    3. 压缩连续空行（保留段落分隔）
    4. 去除首尾空白，行内压缩多余空格
    """

    name = "denoise"

    def process(self, data: dict) -> dict:
        text = data["content"]

        # 移除控制字符（保留 \n \t）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 连续空行压缩为单个空行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 去除首尾空白
        text = text.strip()

        # 行尾去空白 + 行内多空格压缩
        lines = []
        for line in text.split('\n'):
            stripped = line.rstrip()
            stripped = re.sub(r'[ \t]+', ' ', stripped)
            lines.append(stripped)
        text = '\n'.join(lines)

        data["content"] = text
        return data
