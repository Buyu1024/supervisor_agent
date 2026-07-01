"""PromptBuilder —— 将内部格式转为 OpenAI 标准 messages 列表"""

from agent_demo.perception.message import Message


class PromptBuilder:
    """
    消息组装器：将对话历史、系统上下文、工具 schema 组装为 OpenAI messages 格式

    支持两种输入格式:
        - agent_demo.perception.Message 对象（感知模块输出）
        - 纯 dict: {"role": "user/assistant/system/tool", "content": "..."}
    """

    def __init__(self, system_prompt: str | None = None):
        """
        Args:
            system_prompt: 全局系统提示词，会在每条消息列表最前面插入
        """
        self.system_prompt = system_prompt

    def build(
        self,
        messages: list[Message | dict],
        context: str | None = None,
    ) -> list[dict]:
        """
        组装最终发送给 LLM 的 messages 列表

        Args:
            messages: 对话历史（从旧到新）
            context: 记忆模块注入的检索上下文，会拼入系统提示

        Returns:
            标准 OpenAI messages 格式列表:
            [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        """
        result = []

        # ---- 1. 系统提示词（全局 + 记忆上下文） ----
        system_content = self._build_system_content(context)
        if system_content:
            result.append({"role": "system", "content": system_content})

        # ---- 2. 对话历史 ----
        for msg in messages:
            normalized = self._normalize_message(msg)
            if normalized:
                result.append(normalized)

        # ---- 3. 如果没有任何消息，插入空提示 ----
        if not any(m["role"] == "user" for m in result):
            # 如果 system 后有 assistant/tool 但没有 user，补一个空的
            pass  # 上层保证至少有一条用户消息

        return result

    def _build_system_content(self, context: str | None) -> str | None:
        """拼接系统提示词：全局提示 + 检索上下文"""
        parts = []

        if self.system_prompt:
            parts.append(self.system_prompt)

        if context:
            parts.append(f"\n--- 相关上下文（来自记忆模块） ---\n{context}")

        return "\n".join(parts) if parts else None

    def _normalize_message(self, msg: Message | dict) -> dict | None:
        """将单个消息规范化为 OpenAI 标准格式 {"role": ..., "content": ...}"""
        # 兼容 dict 格式
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # 处理 tool 消息的特殊字段
            if role == "tool":
                return {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": str(content),
                }
            return {"role": role, "content": str(content)}

        # 处理感知模块 Message 对象
        if isinstance(msg, Message):
            return {
                "role": msg.role,      # 默认 "user"
                "content": msg.content,
            }

        return None
