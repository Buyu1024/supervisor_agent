"""LLM 模块 —— Agent 的大语言模型调用层

职责：封装 qwen3.7-plus 调用 + Function Calling 闭环管理
"""

from .module import LLMModule
from .types import LLMResponse
from .client import QwenClient

__all__ = ["LLMModule", "LLMResponse", "QwenClient"]
