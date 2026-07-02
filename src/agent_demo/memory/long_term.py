"""LongTermMemory —— 长期记忆（组合 VectorStore + RelStore）

职责:
    1. 协调 FAISSVectorStore（语义检索）+ RelStore（结构化查询）
    2. 记忆写入路由: 根据 memory_type 分发到对应存储
    3. 记忆检索合并: 语义检索 + 精确匹配 + 过滤
    4. 遗忘策略: 时间衰减 + 重要性阈值
"""

import time
import logging
from typing import Optional

from .types import MemoryItem, MemorySearchResult
from .embedder import Embedder
from .vector_store import FAISSVectorStore
from .rel_store import RelStore

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    长期记忆 —— 管理持久化的语义记忆和结构化记忆

    使用示例:
        ltm = LongTermMemory(embedder=embedder, persist_dir="data/memory")
        ltm.remember(MemoryItem.create(content="用户喜欢简洁回答", memory_type="preference"))
        results = ltm.retrieve("用户偏好什么风格？", top_k=5)
        ltm.export_context()  # → 可注入 LLM 的文本
    """

    def __init__(
        self,
        embedder: Embedder,
        persist_dir: str | None = None,
    ):
        """
        Args:
            embedder: 文本转向量的 Embedder 实例
            persist_dir: 持久化目录，None 则纯内存模式
                        非 None 时: vector 存 {dir}/vector.index,
                                   rel 存 {dir}/rel.db
        """
        self._embedder = embedder

        # 持久化路径
        if persist_dir:
            import os
            os.makedirs(persist_dir, exist_ok=True)
            self._vector_path = os.path.join(persist_dir, "vector.index")
            self._db_path = os.path.join(persist_dir, "rel.db")
        else:
            self._vector_path = None
            self._db_path = ":memory:"

        # 初始化子存储
        self._vector_store = FAISSVectorStore(embedder=embedder)
        self._rel_store = RelStore(self._db_path)

        # 尝试加载已有持久化数据
        if self._vector_path and self._vector_store._index.ntotal == 0:
            self._try_load()

        logger.info(
            f"LongTermMemory 初始化完成，"
            f"向量: {self._vector_store.count} 条，"
            f"偏好: {len(self._rel_store.get_all_preferences())} 条"
        )

    # ---- 写入 ----

    def remember(self, item: MemoryItem) -> str:
        """
        存入一条长期记忆

        写入路由:
            - 所有类型 → FAISSVectorStore（语义检索）
            - preference → 同步写入 RelStore 的 user_preferences
            - entity → 同步写入 RelStore 的 entities+relations

        Args:
            item: 记忆条目

        Returns:
            item.id
        """
        # 1. 写入向量存储
        self._vector_store.add(item)

        # 2. 结构化存储（按类型写入不同表）
        if item.memory_type == "preference":
            # 偏好格式: "key: value" → 解析后写入 RelStore
            content = item.content
            if ":" in content:
                key, value = content.split(":", 1)
                self._rel_store.set_preference(key.strip(), value.strip())
            else:
                self._rel_store.set_preference(
                    item.metadata.get("key", item.id[:8]), content
                )

        elif item.memory_type == "entity":
            entity_type = item.metadata.get("entity_type", "unknown")
            self._rel_store.upsert_entity(
                name=item.metadata.get("name", item.id[:8]),
                entity_type=entity_type,
                properties={
                    "content": item.content,
                    **{k: v for k, v in item.metadata.items()
                       if k not in ("name", "entity_type")},
                },
            )

        logger.debug(
            f"长期记忆已存储: id={item.id}, type={item.memory_type}, "
            f"importance={item.importance:.2f}"
        )
        return item.id

    def remember_batch(self, items: list[MemoryItem]) -> list[str]:
        """批量存入记忆"""
        self._vector_store.add_batch(items)
        for item in items:
            if item.memory_type == "preference" and ":" in item.content:
                key, value = item.content.split(":", 1)
                self._rel_store.set_preference(key.strip(), value.strip())
        return [item.id for item in items]

    # ---- 检索 ----

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        type_filter: str | None = None,
    ) -> list[MemorySearchResult]:
        """
        语义检索长期记忆

        Args:
            query: 查询文本
            top_k: 返回条数
            type_filter: 只返回指定类型的记忆

        Returns:
            按相似度降序的 MemorySearchResult 列表
        """
        return self._vector_store.search(query, top_k, type_filter)

    def export_context(self, query: str = "", top_k: int = 5) -> str:
        """
        导出长期记忆上下文 —— 可直接注入 LLM 的 system prompt

        组装内容:
            1. 用户偏好列表（RelStore 精确查询）
            2. 相关历史记忆（VectorStore 语义检索）

        Args:
            query: 检索查询（空字符串则只返回偏好，不检索）
            top_k: 语义检索条数

        Returns:
            格式化的上下文字符串，无内容返回空字符串
        """
        parts = []

        # 1. 用户偏好
        prefs = self._rel_store.get_all_preferences()
        if prefs:
            pref_lines = ["[用户偏好]"]
            for k, v in prefs.items():
                pref_lines.append(f"  - {k}: {v}")
            parts.append("\n".join(pref_lines))

        # 2. 相关实体
        entities = self._rel_store.list_entities()
        if entities:
            entity_lines = ["[已知实体]"]
            for e in entities[:10]:  # 最多 10 个
                props = e.get("properties", {})
                entity_lines.append(f"  - {e['name']} ({e['type']}): {props.get('content', '')}")
            parts.append("\n".join(entity_lines))

        # 3. 语义检索结果
        if query and query.strip():
            results = self.retrieve(query, top_k=top_k)
            if results:
                mem_lines = ["[相关历史记忆]"]
                for r in results:
                    mem_lines.append(
                        f"  - [{r.item.memory_type}] {r.item.content} "
                        f"(相关性: {r.score:.2f})"
                    )
                parts.append("\n".join(mem_lines))

        return "\n\n".join(parts) if parts else ""

    # ---- 遗忘 ----

    def forget_before(self, timestamp: float) -> int:
        """
        移除指定时间之前的所有记忆

        Args:
            timestamp: Unix 时间戳，此时间之前创建的条目将被删除

        Returns:
            删除的条目数量
        """
        to_delete = [
            item_id for item_id, item in self._vector_store._metadata.items()
            if item.created_at < timestamp
        ]
        for item_id in to_delete:
            self._vector_store.delete(item_id)
        if to_delete:
            self._vector_store._rebuild_index()
        logger.info(f"时间遗忘: 删除 {len(to_delete)} 条记忆 (before {timestamp})")
        return len(to_delete)

    def forget_low_importance(self, threshold: float = 0.2) -> int:
        """
        移除低于重要性阈值的记忆

        Args:
            threshold: 重要性阈值，低于此值的条目将被删除

        Returns:
            删除的条目数量
        """
        to_delete = [
            item_id for item_id, item in self._vector_store._metadata.items()
            if item.importance < threshold
        ]
        for item_id in to_delete:
            self._vector_store.delete(item_id)
        if to_delete:
            self._vector_store._rebuild_index()
        logger.info(
            f"重要性遗忘: 删除 {len(to_delete)} 条低价值记忆 "
            f"(importance < {threshold})"
        )
        return len(to_delete)

    def forget_by_access(
        self, days_stale: int = 30, min_access: int = 0
    ) -> int:
        """
        移除长期未访问的记忆

        Args:
            days_stale: 超过此天数未访问视为过期
            min_access: 最低访问次数，低于此值才删除（0 表示不限制）

        Returns:
            删除的条目数量
        """
        cutoff = time.time() - days_stale * 86400
        to_delete = [
            item_id for item_id, item in self._vector_store._metadata.items()
            if item.last_accessed < cutoff and item.access_count <= min_access
        ]
        for item_id in to_delete:
            self._vector_store.delete(item_id)
        if to_delete:
            self._vector_store._rebuild_index()
        logger.info(
            f"访问遗忘: 删除 {len(to_delete)} 条不活跃记忆 "
            f"(> {days_stale} 天未访问)"
        )
        return len(to_delete)

    # ---- 持久化 ----

    def save(self) -> None:
        """保存向量存储到磁盘（RelStore 基于 SQLite，自动持久化）"""
        if self._vector_path:
            self._vector_store.save(self._vector_path)
            logger.info(f"长期记忆已保存到: {self._vector_path}")

    def _try_load(self) -> None:
        """尝试从磁盘加载已有数据"""
        import os
        if self._vector_path and os.path.exists(self._vector_path):
            try:
                self._vector_store.load(self._vector_path)
                logger.info(f"已加载持久化向量: {self._vector_store.count} 条")
            except Exception as e:
                logger.warning(f"加载向量存储失败: {e}，使用空白存储")

    # ---- 查询 ----

    @property
    def count(self) -> int:
        """向量存储中的记忆总数"""
        return self._vector_store.count

    def get_stats(self) -> dict:
        """获取记忆统计信息"""
        type_counts = {}
        for item in self._vector_store._metadata.values():
            t = item.memory_type
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total": self.count,
            "by_type": type_counts,
            "preferences": len(self._rel_store.get_all_preferences()),
            "entities": len(self._rel_store.list_entities()),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"LongTermMemory(total={stats['total']}, types={stats['by_type']})"
