"""MemoryManager —— 记忆策略层

职责:
    1. 协调三层记忆（WorkingMemory + LongTermMemory + SessionStore）
    2. retrieve(): 从三层记忆组装完整上下文
    3. remember(): 将对话写入短期 + 长期记忆
    4. compress(): 触发工作记忆压缩
    5. 遗忘策略调度

这是记忆模块的核心编排器，MemoryModule 委托给它处理所有策略逻辑。
"""

import logging
from typing import Callable, Optional

from .types import MemoryItem, MemorySearchResult
from .working_memory import WorkingMemory
from .long_term import LongTermMemory
from .session_store import SessionStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    记忆管理器 —— 编排三层记忆的读/写/压缩/遗忘

    使用示例:
        mgr = MemoryManager(
            long_term=ltm,
            max_working_tokens=8000,
            system_prompt="你是助手",
        )
        context = mgr.retrieve("用户之前喜欢什么？")
        mgr.remember([{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}])
    """

    def __init__(
        self,
        long_term: LongTermMemory,
        session: SessionStore | None = None,
        max_working_tokens: int = 8000,
        system_prompt: str | None = None,
        summarize_func: Callable[[list[dict]], str] | None = None,
    ):
        """
        Args:
            long_term: 长期记忆实例
            session: 会话 KV 存储，None 则创建一个空的
            max_working_tokens: 工作记忆最大 token 数
            system_prompt: 系统提示词
            summarize_func: 摘要生成函数（用于压缩旧消息）
        """
        self._long_term = long_term
        self._session = session or SessionStore()
        self._working = WorkingMemory(
            max_tokens=max_working_tokens,
            system_prompt=system_prompt,
            summarize_func=summarize_func,
        )
        logger.info(
            f"MemoryManager 初始化完成，"
            f"max_working_tokens={max_working_tokens}"
        )

    # ---- 核心接口 ----

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """
        检索完整上下文 —— 从三层记忆组装后注入 LLM

        检索策略:
            ① 取工作记忆（最近对话窗口）
            ② 取会话状态（当前任务、中间结果）
            ③ 取长期记忆（语义检索 + 结构化偏好）
            ④ 组装为一条 context 字符串

        Args:
            query: 检索查询（通常是用户最后一条消息）
            top_k: 长期记忆检索条数

        Returns:
            可直接传给 LLMModule.chat(context=...) 的上下文字符串
        """
        parts = []

        # ① 会话状态摘要
        session_summary = self._session.export_summary()
        if session_summary:
            parts.append(session_summary)

        # ② 长期记忆检索
        ltm_context = self._long_term.export_context(query, top_k=top_k)
        if ltm_context:
            parts.append(ltm_context)

        return "\n\n".join(parts) if parts else ""

    def remember(
        self,
        messages: list[dict],
        auto_extract: bool = True,
    ) -> None:
        """
        记忆本轮对话

        策略:
            ① 消息追加到工作记忆
            ② 如果工作记忆超预算 → 触发压缩
            ③ 提取重要内容写入长期记忆（偏好/实体/摘要）

        Args:
            messages: 本轮对话消息列表
            auto_extract: 是否自动提取偏好和实体写入长期记忆
        """
        # ① 追加到工作记忆
        self._working.add_batch(messages)

        # ② 自动提取并写入长期记忆
        if auto_extract:
            self._auto_extract_and_store(messages)

        logger.debug(
            f"已记忆 {len(messages)} 条消息，"
            f"工作记忆: {self._working.total_tokens()} tokens"
        )

    def compress(self) -> str | None:
        """
        压缩工作记忆 —— 将旧消息压缩为摘要，存入长期记忆

        Returns:
            生成的摘要文本，无内容返回 None
        """
        summary = self._working.summarize()
        if summary:
            # 将摘要存入长期记忆
            item = MemoryItem.create(
                content=summary,
                memory_type="conversation",
                importance=0.6,
                metadata={"source": "compression"},
            )
            self._long_term.remember(item)
            logger.info(f"对话摘要已存入长期记忆: {len(summary)} 字符")
        return summary

    # ---- 偏好提取 ----

    def add_preference(self, key: str, value: str) -> None:
        """显式添加用户偏好（同时写入长期记忆）"""
        self._long_term._rel_store.set_preference(key, value)
        item = MemoryItem.create(
            content=f"{key}: {value}",
            memory_type="preference",
            importance=0.7,
            metadata={"key": key},
        )
        self._long_term.remember(item)

    def get_preference(self, key: str) -> str | None:
        """获取用户偏好"""
        return self._long_term._rel_store.get_preference(key)

    def get_all_preferences(self) -> dict[str, str]:
        """获取所有偏好"""
        return self._long_term._rel_store.get_all_preferences()

    # ---- 实体管理 ----

    def remember_entity(
        self, name: str, entity_type: str, properties: dict | None = None
    ) -> None:
        """记住一个实体"""
        props = properties or {}
        content = f"{name} ({entity_type}): {props}"
        item = MemoryItem.create(
            content=content,
            memory_type="entity",
            importance=0.5,
            metadata={"name": name, "entity_type": entity_type, **props},
        )
        self._long_term.remember(item)

    # ---- 会话状态 ----

    def set_session(self, key: str, value: object, ttl: float | None = None) -> None:
        """设置会话变量"""
        self._session.set(key, value, ttl)

    def get_session(self, key: str, default: object = None) -> object:
        """获取会话变量"""
        return self._session.get(key, default)

    # ---- 遗忘策略 ----

    def run_forgetting(self) -> dict:
        """
        执行遗忘策略 —— 清理过期/低价值记忆

        策略:
            1. 超过 30 天未访问且从未使用 → 删除
            2. 重要性低于 0.1 且超过 7 天 → 删除
            3. 容量超过 1000 条 → 按重要性排序淘汰

        Returns:
            {strategy: deleted_count}
        """
        results = {}

        # 策略 1: 30 天未访问
        n = self._long_term.forget_by_access(days_stale=30, min_access=0)
        results["stale_30d"] = n

        # 策略 2: 低重要性 + 超过 7 天
        n = self._long_term.forget_low_importance(threshold=0.1)
        results["low_importance"] = n

        # 策略 3: 容量控制
        max_capacity = 1000
        if self._long_term.count > max_capacity:
            # 按重要性排序，淘汰最低的
            items = sorted(
                self._long_term._vector_store._metadata.values(),
                key=lambda x: (x.importance, x.last_accessed),
            )
            to_remove = items[:self._long_term.count - max_capacity]
            for item in to_remove:
                self._long_term._vector_store.delete(item.id)
            if to_remove:
                self._long_term._vector_store._rebuild_index()
            results["capacity"] = len(to_remove)

        total = sum(results.values())
        if total > 0:
            logger.info(f"遗忘策略执行完成: {results}，共删除 {total} 条")

        return results

    # ---- 查询 ----

    def get_working_context(self) -> list[dict]:
        """获取当前工作记忆窗口"""
        return self._working.get_context()

    def get_stats(self) -> dict:
        """获取记忆统计"""
        return {
            "working_messages": len(self._working),
            "working_tokens": self._working.total_tokens(),
            "session_keys": self._session.keys(),
            "long_term": self._long_term.get_stats(),
        }

    # ---- 内部辅助 ----

    def _auto_extract_and_store(self, messages: list[dict]) -> None:
        """
        从消息中自动提取偏好/实体并存入长期记忆

        当前为规则匹配版本:
            - 包含"我喜欢"、"我偏好"、"我习惯" → 提取为偏好
            - 包含"我是"、"我在"、"我叫" → 提取为实体

        后续可接入 LLM 做更精准的提取。
        """
        for msg in messages:
            content = str(msg.get("content", ""))
            if msg.get("role") != "user" or not content:
                continue

            # 简单规则匹配偏好
            preference_patterns = ["我喜欢", "我偏好", "我习惯", "我不喜欢", "我讨厌"]
            for pattern in preference_patterns:
                if pattern in content:
                    # 截取偏好内容
                    idx = content.index(pattern)
                    snippet = content[idx:idx + 80].strip()
                    item = MemoryItem.create(
                        content=snippet,
                        memory_type="preference",
                        importance=0.6,
                        metadata={"source": "auto_extract", "pattern": pattern},
                    )
                    self._long_term.remember(item)
                    logger.debug(f"自动提取偏好: {snippet[:50]}...")
                    break  # 每条消息只匹配一次

            # 简单规则匹配实体
            entity_patterns = {
                "我叫": "person",
                "我是": "person",
                "我在": "location",
                "我的职业是": "occupation",
            }
            for pattern, etype in entity_patterns.items():
                if pattern in content:
                    idx = content.index(pattern)
                    snippet = content[idx + len(pattern):idx + len(pattern) + 30].strip()
                    # 去掉标点
                    name = snippet.rstrip("。，！？,.!?；;：:")
                    if name:
                        self.remember_entity(name, etype, {"source_text": content[:100]})
                        logger.debug(f"自动提取实体: {name} ({etype})")
                    break  # 每条消息只匹配一次
