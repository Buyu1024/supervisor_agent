"""ToolLoopManager —— Function Calling 闭环管理"""

import json
import logging
from .types import LLMResponse
from .client import QwenClient

logger = logging.getLogger(__name__)

# 工具执行器回调类型: (tool_name, arguments) -> ToolResult
# ToolResult 包含 content 字段，作为工具返回给 LLM 的结果文本


class ToolLoopManager:
    """
    Function Calling 闭环管理器

    职责:
        1. 调用 LLM API
        2. 判断 finish_reason：
           - "stop" → 提取文本内容，返回 LLMResponse
           - "tool_calls" → 调用 tool_executor → 结果回填消息列表 → 循环
        3. 累计 token 用量
        4. 超过 max_rounds 强制终止

    不持有 ToolsModule，通过构造函数注入 tool_executor 回调保持解耦。
    """

    def __init__(
        self,
        client: QwenClient,
        tool_executor,
        max_rounds: int = 10,
    ):
        """
        Args:
            client: QwenClient 实例
            tool_executor: Callable[[str, dict], ToolResult]
                           工具执行回调，由 ToolsModule.get_executor() 提供
            max_rounds: 最大工具调用轮数，防止死循环
        """
        self._client = client
        self._tool_executor = tool_executor  # (name, arguments) -> ToolResult
        self._max_rounds = max_rounds

    def run(
        self,
        messages: list[dict],
        tool_schemas: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> tuple[LLMResponse, list[dict]]:
        """
        执行工具调用闭环

        Args:
            messages: 初始消息列表（OpenAI 格式，含最新用户消息）
            tool_schemas: 可用工具列表，None 或空列表 = 纯对话模式
            temperature: 采样温度

        Returns:
            (LLMResponse, messages) —— response 是最终结果，messages 是追加了
            工具调用中间消息后的完整列表（供调用方同步回对话历史）
        """
        tool_log = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # 没有工具 schema → 单次调用直接返回
        if not tool_schemas:
            response = self._client.chat_completion(
                messages, tools=None, temperature=temperature
            )
            return (
                self._build_response(
                    response=response,
                    tool_log=[],
                    token_usage=self._extract_usage(response),
                ),
                messages,
            )

        # ---- 工具调用闭环 ----
        for round_num in range(1, self._max_rounds + 1):
            # 调用 API
            response = self._client.chat_completion(
                messages, tools=tool_schemas, temperature=temperature
            )
            usage = self._extract_usage(response)
            total_usage = self._accumulate_usage(total_usage, usage)

            choice = response.choices[0]
            finish_reason = choice.finish_reason

            # 正常结束 → 返回文本
            if finish_reason == "stop":
                content = choice.message.content or ""
                return (
                    LLMResponse(
                        content=content,
                        finish_reason="stop",
                        token_usage=total_usage,
                        tool_calls_log=tool_log,
                    ),
                    messages,
                )

            # 工具调用 → 执行 → 结果回填
            if finish_reason == "tool_calls":
                tool_calls = choice.message.tool_calls
                if not tool_calls:
                    # 空 tool_calls，当 stop 处理
                    content = choice.message.content or ""
                    return (
                        LLMResponse(
                            content=content,
                            finish_reason="stop",
                            token_usage=total_usage,
                            tool_calls_log=tool_log,
                        ),
                        messages,
                    )

                # 追加 assistant 消息（含 tool_calls）
                messages.append(self._make_assistant_tool_call_msg(tool_calls))

                # 逐个执行工具
                for tc in tool_calls:
                    tool_name = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    # 调用工具执行器
                    if self._tool_executor:
                        result = self._tool_executor(tool_name, arguments)
                        result_text = result.content if hasattr(result, 'content') else str(result)
                    else:
                        result_text = f"错误：未配置工具执行器，无法执行工具 {tool_name}"

                    # 记录日志
                    tool_log.append({
                        "round": round_num,
                        "name": tool_name,
                        "arguments": arguments,
                        "result": result_text[:500],  # 日志截断
                    })

                    # 追加 tool 结果消息
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })

                continue  # 继续下一轮

            # 其他 finish_reason（content_filter, length 等）
            content = choice.message.content or ""
            return (
                LLMResponse(
                    content=content or f"对话异常终止，原因: {finish_reason}",
                    finish_reason=finish_reason or "error",
                    token_usage=total_usage,
                    tool_calls_log=tool_log,
                ),
                messages,
            )

        # 超过最大轮数
        logger.warning(f"工具调用已达最大轮数限制 ({self._max_rounds})，强制终止")
        return (
            LLMResponse(
                content="任务执行步数超过限制，已强制终止。请简化任务后重试。",
                finish_reason="max_rounds",
                token_usage=total_usage,
                tool_calls_log=tool_log,
            ),
            messages,
        )

    # ---- 辅助方法 ----

    def _build_response(
        self,
        response,
        tool_log: list[dict],
        token_usage: dict,
    ) -> LLMResponse:
        """从 API 返回构造 LLMResponse"""
        choice = response.choices[0]
        finish_reason = choice.finish_reason
        content = choice.message.content or ""

        if finish_reason == "tool_calls":
            # tool_calls 但没有 tool_executor 或空 tools → 返回提示
            content = content or "（模型请求调用工具，但当前未配置可用工具）"

        return LLMResponse(
            content=content,
            finish_reason=finish_reason or "stop",
            token_usage=token_usage,
            tool_calls_log=tool_log,
        )

    @staticmethod
    def _extract_usage(response) -> dict:
        """从 API 响应中提取 token 用量"""
        if hasattr(response, "usage") and response.usage:
            return {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @staticmethod
    def _accumulate_usage(total: dict, usage: dict) -> dict:
        """累加 token 用量"""
        return {
            "prompt_tokens": total["prompt_tokens"] + usage["prompt_tokens"],
            "completion_tokens": total["completion_tokens"] + usage["completion_tokens"],
            "total_tokens": total["total_tokens"] + usage["total_tokens"],
        }

    @staticmethod
    def _make_assistant_tool_call_msg(tool_calls) -> dict:
        """构造 assistant 消息（含 tool_calls）—— OpenAI 格式"""
        tc_list = []
        for tc in tool_calls:
            tc_list.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
        return {"role": "assistant", "tool_calls": tc_list}
