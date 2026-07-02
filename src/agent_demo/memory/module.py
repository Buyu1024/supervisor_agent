"""MemoryModule —— 记忆模块主入口

组装三层记忆系统（工作记忆 + 长期记忆 + 会话存储），对外提供统一接口。

使用示例:
    from agent_demo.memory import MemoryModule

    # 初始化（默认 DashScope embedding）
    memory = MemoryModule(
        api_key="sk-xxx",
        persist_dir="data/memory",
        system_prompt="你是助手",
    )

    # 检索上下文 → 注入 LLM
    context = memory.retrieve("用户最近在聊什么？")
    llm.chat(messages=[...], context=context)

    # 保存本轮对话
    memory.remember([
        {"role": "user", "content": "北京天气怎么样？"},
        {"role": "assistant", "content": "北京今天晴，18-26°C。"},
    ])

    # 管理偏好和会话状态
    memory.add_preference("language", "中文")
    memory.set_session("current_task", "规划旅行路线")
"""

import logging
from typing import Callable, Optional

from .embedder import create_embedder, Embedder
from .long_term import LongTermMemory
from .manager import MemoryManager
from .session_store import SessionStore

logger = logging.getLogger(__name__)


class MemoryModule:
    """
    记忆模块 —— Agent 三层记忆系统的统一入口

    架构:
        MemoryModule
        └── MemoryManager（策略层）
            ├── WorkingMemory   （短期 / 滑动窗口）
            ├── LongTermMemory  （长期 / 向量+结构化）
            └── SessionStore    （会话 / KV）
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        embedder_provider: str = "dashscope",
        embedder_kwargs: dict | None = None,
        persist_dir: str | None = None,
        max_working_tokens: int = 8000,
        system_prompt: str | None = None,
        api_key: str | None = None,
    ):
        """
        Args:
            embedder: Embedder 实例（优先级最高，传入后忽略 embedder_provider）
            embedder_provider: Embedding 提供商 "dashscope" | "local"
            embedder_kwargs: 传递给 create_embedder 的额外参数
            persist_dir: 长期记忆持久化目录，None 则纯内存模式
            max_working_tokens: 工作记忆最大 token 预算
            system_prompt: 系统提示词（会注入到工作记忆的 context 中）
            api_key: DashScope API Key（用于 Embedding，仅 dashscope provider 需要）
        """
        # 初始化 Embedder
        if embedder is not None:
            self._embedder = embedder
        else:
            kwargs = embedder_kwargs or {}
            if embedder_provider == "dashscope" and api_key:
                kwargs.setdefault("api_key", api_key)
            self._embedder = create_embedder(embedder_provider, **kwargs)

        # 初始化长期记忆
        self._long_term = LongTermMemory(
            embedder=self._embedder,
            persist_dir=persist_dir,
        )

        # 初始化会话存储
        self._session = SessionStore()

        # 初始化策略管理器
        self._manager = MemoryManager(
            long_term=self._long_term,
            session=self._session,
            max_working_tokens=max_working_tokens,
            system_prompt=system_prompt,
            summarize_func=None,  # 可后续注入 LLMModule 的摘要能力
        )

        logger.info(
            f"MemoryModule 初始化完成，"
            f"embedder={type(self._embedder).__name__}, "
            f"persist={'是' if persist_dir else '否'}"
        )

    # ---- 核心接口 ----

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """
        检索上下文 —— 从三层记忆组装后注入 LLM

        这是最重要的对外接口，返回值直接传给 LLMModule.chat(context=...)。

        Args:
            query: 检索查询（通常是当前用户输入）
            top_k: 长期记忆检索条数

        Returns:
            格式化的上下文字符串
        """
        return self._manager.retrieve(query, top_k=top_k)

    def remember(self, messages: list[dict]) -> None:
        """
        记忆本轮对话 —— 写入工作记忆 + 自动提取长期记忆

        Args:
            messages: 本轮对话消息列表（role: user/assistant）
        """
        self._manager.remember(messages)

    def remember_item(self, content: str, memory_type: str = "knowledge", importance: float = 0.5) -> str:
        """
        手动存入一条长期记忆

        Args:
            content: 记忆内容
            memory_type: 类型 (conversation/preference/entity/knowledge)
            importance: 重要性 0.0~1.0

        Returns:
            记忆项 ID
        """
        from .types import MemoryItem
        item = MemoryItem.create(
            content=content,
            memory_type=memory_type,
            importance=importance,
        )
        self._long_term.remember(item)
        return item.id

    def compress(self) -> str | None:
        """压缩工作记忆的旧消息为摘要"""
        return self._manager.compress()

    # ---- 便捷接口 ----

    def add_preference(self, key: str, value: str) -> None:
        """添加用户偏好"""
        self._manager.add_preference(key, value)

    def get_preference(self, key: str) -> str | None:
        """获取用户偏好"""
        return self._manager.get_preference(key)

    def get_all_preferences(self) -> dict[str, str]:
        """获取所有偏好"""
        return self._manager.get_all_preferences()

    def set_session(self, key: str, value: object, ttl: float | None = None) -> None:
        """设置会话变量"""
        self._manager.set_session(key, value, ttl)

    def get_session(self, key: str, default: object = None) -> object:
        """获取会话变量"""
        return self._manager.get_session(key, default)

    def get_working_context(self) -> list[dict]:
        """获取当前工作记忆窗口（最近对话）"""
        return self._manager.get_working_context()

    def add_working_message(self, message: dict) -> None:
        """向工作记忆追加一条消息（不触发长期记忆写入）"""
        self._manager._working.add(message)

    # ---- 维护 ----

    def run_forgetting(self) -> dict:
        """执行遗忘策略，清理过期/低价值记忆"""
        return self._manager.run_forgetting()

    def save(self) -> None:
        """持久化长期记忆到磁盘"""
        self._long_term.save()

    def clear_session(self) -> None:
        """清空会话状态和短期记忆（长期记忆保留）"""
        self._manager._working.clear()
        self._manager._session.clear()

    def get_stats(self) -> dict:
        """获取记忆统计信息"""
        return self._manager.get_stats()

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"MemoryModule(working_msgs={stats['working_messages']}, "
            f"working_tokens={stats['working_tokens']}, "
            f"ltm_total={stats['long_term']['total']})"
        )
