"""QwenClient —— 封装 DashScope OpenAI 兼容接口"""

import os
import time
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# DashScope 默认端点
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-plus"


class QwenClient:
    """
    Qwen 模型客户端，基于 openai SDK 调用 DashScope API

    使用方式:
        # 方式一：环境变量 DASHSCOPE_API_KEY
        client = QwenClient()

        # 方式二：显式传入
        client = QwenClient(api_key="sk-xxx")

        response = client.chat_completion(messages=[...], tools=[...])
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DASHSCOPE_BASE_URL,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            api_key: DashScope API Key，None 则从环境变量 DASHSCOPE_API_KEY 读取
            base_url: API 端点地址
            model: 模型名称
            max_retries: 网络错误最大重试次数
            retry_delay: 重试间隔秒数（指数退避基数）
        """
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DashScope API Key 未设置。请设置环境变量 DASHSCOPE_API_KEY "
                "或通过 QwenClient(api_key='sk-xxx') 传入"
            )

        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 初始化 OpenAI 兼容客户端
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ):
        """
        调用 Chat Completion API（OpenAI 兼容格式）

        Args:
            messages: 标准 OpenAI messages 格式列表
            tools: 工具定义列表（OpenAI Function Calling 格式），None 表示纯对话
            temperature: 采样温度

        Returns:
            openai.types.chat.ChatCompletion 对象

        Raises:
            RuntimeError: 重试耗尽后仍失败
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if tools:
                    kwargs["tools"] = tools

                return self._client.chat.completions.create(**kwargs)

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"API 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}，"
                        f"{wait:.1f}s 后重试..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"API 调用失败，已达最大重试次数: {e}")

        raise RuntimeError(
            f"DashScope API 调用失败，已重试 {self.max_retries} 次。"
            f"最后错误: {last_error}"
        )
