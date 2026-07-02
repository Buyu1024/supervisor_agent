"""记忆模块 —— 数据类型定义"""

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class MemoryItem:
    """一条长期记忆条目

    存储在向量数据库 + 结构化存储中，支持语义检索。
    """

    id: str                               # 唯一 ID（UUID）
    content: str                          # 记忆的文本内容
    memory_type: str = "knowledge"        # "conversation" | "preference" | "entity" | "knowledge"
    importance: float = 0.5               # 0.0 ~ 1.0，重要性评分
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None  # 缓存的向量，避免重复计算

    @classmethod
    def create(
        cls,
        content: str,
        memory_type: str = "knowledge",
        importance: float = 0.5,
        metadata: dict | None = None,
        embedding: list[float] | None = None,
    ) -> "MemoryItem":
        """快捷创建 MemoryItem（自动生成 ID 和时间戳）"""
        return cls(
            id=str(uuid.uuid4())[:12],  # 缩短 ID，便于日志阅读
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {},
            embedding=embedding,
        )

    def touch(self) -> None:
        """更新最后访问时间和计数"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class MemorySearchResult:
    """检索结果 —— 包含记忆条目和相似度分数"""

    item: MemoryItem
    score: float                         # 相似度分数（越高越相关）

    def __repr__(self) -> str:
        return (
            f"MemorySearchResult(score={self.score:.4f}, "
            f"type={self.item.memory_type}, "
            f"content={self.item.content[:50]}...)"
        )


@dataclass
class SessionEntry:
    """会话 KV 存储的一条记录"""

    key: str
    value: object                        # 任意可序列化的值
    created_at: float = field(default_factory=time.time)
    ttl: float | None = None             # 过期时间（Unix timestamp），None 表示永不过期

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() > self.ttl
