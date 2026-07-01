"""LLMModule —— LLM 模块主入口，组装 QwenClient + PromptBuilder + ToolLoop"""

import logging
from typing import Callable

from .types import LLMResponse
from .client import QwenClient, DASHSCOPE_BASE_URL, DEFAULT_MODEL
from .prompt_builder import PromptBuilder
from .tool_loop import ToolLoopManager

logger = logging.getLogger(__name__)


class LLMModule:
    """
    LLM 模块 —— 封装 qwen3.7-plus 调用 + Function Calling 闭环

    使用示例:
        # 纯对话（无工具）
        llm = LLMModule()
        response = llm.chat(messages=[user_msg])

        # 带工具（注入 ToolsModule 的 executor）
        from agent_demo.tools import ToolsModule
        tools_mod = ToolsModule()
        tools_mod.register(...)

        llm = LLMModule(tool_executor=tools_mod.get_executor())
        response = llm.chat(
            messages=[user_msg],
            tools=tools_mod.get_schemas(),
        )
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DASHSCOPE_BASE_URL,
        model: str = DEFAULT_MODEL,
        system_prompt: str | None = None,
        tool_executor: Callable | None = None,
        max_tool_rounds: int = 10,
        temperature: float = 0.7,
    ):
        """
        Args:
            api_key: DashScope API Key。
                     None → 从环境变量 DASHSCOPE_API_KEY 读取
                     str  → 显式传入，优先级高于环境变量
            base_url: API 端点，默认 DashScope 兼容接口
            model: 模型名称，默认 qwen3.7-plus
            system_prompt: 全局系统提示词
            tool_executor: 工具执行回调 (name, arguments) -> ToolResult
                           由 ToolsModule.get_executor() 提供
            max_tool_rounds: 单次 chat 最大工具调用轮数
            temperature: 采样温度，默认 0.7
        """
        # 初始化 QwenClient（API Key: 显式传参 > 环境变量）
        self._client = QwenClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

        # 初始化 PromptBuilder
        self._prompt_builder = PromptBuilder(system_prompt=system_prompt)

        # 初始化 ToolLoopManager
        self._tool_loop = ToolLoopManager(
            client=self._client,
            tool_executor=tool_executor,
            max_rounds=max_tool_rounds,
        )

        self._temperature = temperature

        # 内部对话历史（OpenAI 格式）
        self._history: list[dict] = []

    # ---- 对外接口 ----

    def chat(
        self,
        messages: list,
        tools: list[dict] | None = None,
        context: str | None = None,
    ) -> LLMResponse:
        """
        发送消息并获取最终回复（自动处理工具调用闭环）

        Args:
            messages: 对话消息列表，每项可以是:
                      - agent_demo.perception.Message 对象
                      - dict: {"role": "user/assistant/tool", "content": "..."}
            tools: 工具 schema 列表（OpenAI Function Calling 格式）
                   None = 纯对话模式，不触发工具调用
            context: 记忆模块注入的检索上下文

        Returns:
            LLMResponse（已完成全部工具调用的最终结果）
        """
        # 1. 将新消息转为 OpenAI 格式
        new_messages = self._prompt_builder.build(
            messages=messages,
            context=context,
        )

        # 2. 合并历史
        if self._history:
            # 提取新消息中的 system 消息（可能含更新的 context）
            new_system = next((m for m in new_messages if m["role"] == "system"), None)
            non_system = [m for m in new_messages if m["role"] != "system"]

            # 更新 system 消息：用新的替换旧的，保证 context 始终是最新的
            if new_system:
                if self._history[0]["role"] == "system":
                    self._history[0] = new_system
                else:
                    self._history.insert(0, new_system)

            self._history.extend(non_system)
            api_messages = self._history  # 直接用引用，让 tool_loop 追加工具消息
        else:
            self._history.extend(new_messages)
            api_messages = self._history

        # 3. 执行工具调用闭环（直接操作 _history，工具消息自动同步）
        response, updated_messages = self._tool_loop.run(
            messages=api_messages,
            tool_schemas=tools,
            temperature=self._temperature,
        )
        self._history = updated_messages  # 同步工具调用产生的中间消息

        # 4. 将 AI 回复追加到内部历史
        if response.content:
            self._history.append({
                "role": "assistant",
                "content": response.content,
            })

        return response

    # ---- 会话管理 ----

    def get_history(self) -> list[dict]:
        """获取当前会话的完整对话历史（OpenAI 格式）"""
        return self._history.copy()

    def clear_history(self) -> None:
        """清空对话历史"""
        self._history.clear()
        logger.info("对话历史已清空")

    def set_system_prompt(self, system_prompt: str) -> None:
        """动态更新系统提示词"""
        self._prompt_builder.system_prompt = system_prompt

    # ---- 属性 ----

    @property
    def model(self) -> str:
        return self._client.model
