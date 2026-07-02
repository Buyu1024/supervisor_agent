"""SessionStore —— 内存 KV 存储（会话级别）

职责:
    1. 存储当前任务状态、中间结果、临时变量
    2. 支持 TTL 过期机制
    3. 可序列化为 dict 用于注入 LLM 上下文

使用示例:
    store = SessionStore()
    store.set("current_task", "搜索北京天气")
    store.set("tool_results", {"weather": "晴天"}, ttl=300)  # 5 分钟过期
    store.export()  # → {"current_task": "搜索北京天气", "tool_results": {...}}
"""

import time
import logging
from typing import Any, Optional

from .types import SessionEntry

logger = logging.getLogger(__name__)


class SessionStore:
    """
    会话级 KV 存储 —— 纯内存实现，进程重启后丢失

    设计要点:
        - set() 支持 TTL（秒），超时后自动失效
        - get() 时自动检查过期，过期返回 None
        - export() 导出当前所有有效数据，用于注入 LLM 上下文
    """

    def __init__(self):
        self._store: dict[str, SessionEntry] = {}
        logger.info("SessionStore 初始化完成（内存模式）")

    # ---- CRUD ----

    def set(self, key: str, value: object, ttl: float | None = None) -> None:
        """
        设置键值对

        Args:
            key: 键名
            value: 任意可序列化的值
            ttl: 过期时间（秒），None 表示永不过期
        """
        entry = SessionEntry(
            key=key,
            value=value,
            ttl=(time.time() + ttl) if ttl is not None else None,
        )
        self._store[key] = entry
        logger.debug(f"Session 已设置: {key} (ttl={ttl})")

    def get(self, key: str, default: object = None) -> object:
        """
        获取值，自动检查过期

        Args:
            key: 键名
            default: 键不存在或已过期时的默认值

        Returns:
            存储的值或 default
        """
        entry = self._store.get(key)
        if entry is None:
            return default

        if entry.is_expired():
            del self._store[key]
            logger.debug(f"Session 已过期: {key}")
            return default

        return entry.value

    def delete(self, key: str) -> bool:
        """删除键，返回是否成功删除"""
        if key in self._store:
            del self._store[key]
            logger.debug(f"Session 已删除: {key}")
            return True
        return False

    def exists(self, key: str) -> bool:
        """检查键是否存在且未过期"""
        return self.get(key) is not None

    def clear(self) -> None:
        """清空所有会话数据"""
        self._store.clear()
        logger.info("SessionStore 已清空")

    # ---- 批量操作 ----

    def keys(self) -> list[str]:
        """列出所有有效键名（自动过滤过期）"""
        self._cleanup_expired()
        return list(self._store.keys())

    def export(self) -> dict[str, object]:
        """
        导出所有有效键值对为普通 dict

        用于注入 LLM 上下文时生成会话状态摘要。

        Returns:
            {key: value, ...}
        """
        self._cleanup_expired()
        return {key: entry.value for key, entry in self._store.items()}

    def export_summary(self) -> str:
        """
        导出为可读的文本摘要 —— 拼入 LLM 上下文

        Returns:
            格式化的会话状态文本，空则返回空字符串
        """
        data = self.export()
        if not data:
            return ""

        lines = ["--- 当前会话状态 ---"]
        for key, value in data.items():
            # 截断过长的值
            val_str = str(value)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            lines.append(f"  {key}: {val_str}")
        return "\n".join(lines)

    # ---- 内部辅助 ----

    def _cleanup_expired(self) -> int:
        """清理过期条目，返回清理数量"""
        expired_keys = [
            key for key, entry in self._store.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self._store[key]
        if expired_keys:
            logger.debug(f"清理 {len(expired_keys)} 条过期会话数据")
        return len(expired_keys)

    def __len__(self) -> int:
        self._cleanup_expired()
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return self.exists(key)

    def __repr__(self) -> str:
        return f"SessionStore(keys={len(self)})"
