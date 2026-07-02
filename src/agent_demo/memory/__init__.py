"""记忆模块 —— Agent 的记忆系统

三层记忆架构:
    - 短期记忆 (WorkingMemory):  当前对话滑动窗口 + Token 预算管理
    - 长期记忆 (LongTermMemory): FAISS 向量检索 + SQLite 结构化存储
    - 会话存储 (SessionStore):   内存 KV，存储任务状态和中间结果

核心入口: MemoryModule
"""

from .module import MemoryModule
from .types import MemoryItem, MemorySearchResult
from .embedder import Embedder, DashScopeEmbedder, LocalEmbedder, create_embedder

__all__ = [
    "MemoryModule",
    "MemoryItem",
    "MemorySearchResult",
    "Embedder",
    "DashScopeEmbedder",
    "LocalEmbedder",
    "create_embedder",
]
