"""WorkingMemory —— 短期记忆 / 工作记忆

职责:
    1. 维护当前会话的最近 N 轮对话（滑动窗口）
    2. Token 预算管理 —— 超出预算时自动截断旧消息
    3. 支持将旧消息压缩为摘要（需注入 LLMModule 或摘要函数）

设计:
    - 基于 collections.deque，O(1) 追加和左端弹出
    - tiktoken 精确计数
    - 摘要回调注入，保持模块解耦
"""

import logging
from collections import deque
from typing import Callable, Optional

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False

logger = logging.getLogger(__name__)


# tiktoken 未安装时的回退估算: 中文 ~1.5 字符/token, 英文 ~4 字符/token
def _estimate_tokens(text: str) -> int:
    """粗略估算文本的 token 数"""
    if not text:
        return 0
    # 简单启发式: 中文字符 ~1 token，英文单词 ~1.3 token
    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    other_chars = len(text) - chinese_chars
    return chinese_chars + max(1, other_chars // 3)


class WorkingMemory:
    """
    短期记忆 —— 维护对话窗口 + Token 预算控制

    使用示例:
        wm = WorkingMemory(max_tokens=8000)
        wm.add({"role": "user", "content": "你好"})
        wm.add({"role": "assistant", "content": "你好！有什么可以帮助你的？"})
        context = wm.get_context()  # → 返回不超 token 预算的消息列表
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        system_prompt: str | None = None,
        summarize_func: Callable[[list[dict]], str] | None = None,
    ):
        """
        Args:
            max_tokens: 工作记忆的最大 token 预算
            system_prompt: 系统提示词（不计入滑动窗口，但计入 token 预算）
            summarize_func: 压缩旧消息为摘要的回调函数
                           签名: (messages: list[dict]) -> str
                           不传则旧消息直接丢弃，不生成摘要
        """
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._summarize_func = summarize_func

        # 对话消息队列（从旧到新）
        self._messages: deque[dict] = deque()

        # 已压缩的摘要（旧消息的压缩版本）
        self._summary: str | None = None

        # Token 编码器
        if _HAS_TIKTOKEN:
            try:
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoder = None
        else:
            self._encoder = None

        logger.info(
            f"WorkingMemory 初始化完成，max_tokens={max_tokens}，"
            f"tiktoken={'可用' if self._encoder else '不可用（使用估算）'}"
        )

    # ---- 消息管理 ----

    def add(self, message: dict) -> None:
        """追加一条消息到工作记忆"""
        self._messages.append(message)
        self._trim_if_needed()

    def add_batch(self, messages: list[dict]) -> None:
        """批量追加消息"""
        for msg in messages:
            self._messages.append(msg)
        self._trim_if_needed()

    def get_context(self) -> list[dict]:
        """
        获取当前工作记忆上下文 —— 供 LLM 模块使用

        返回的消息列表确保总 token 数不超过 max_tokens。
        返回顺序: 摘要上下文（若有）→ 系统提示 → 对话消息（旧→新）

        Returns:
            list[dict]: 标准 OpenAI messages 格式
        """
        result = []

        # 1. 系统提示词（含摘要）
        system_parts = []
        if self.system_prompt:
            system_parts.append(self.system_prompt)
        if self._summary:
            system_parts.append(
                f"\n--- 历史对话摘要 ---\n{self._summary}"
            )
        if system_parts:
            result.append({"role": "system", "content": "\n".join(system_parts)})

        # 2. 对话消息（已确保不超预算）
        result.extend(list(self._messages))

        return result

    def get_messages(self) -> list[dict]:
        """获取纯消息列表（不含 system prompt 和摘要）"""
        return list(self._messages)

    # ---- Token 管理 ----

    def count_tokens(self, text: str) -> int:
        """计算文本 token 数"""
        if self._encoder:
            return len(self._encoder.encode(text))
        return _estimate_tokens(text)

    def total_tokens(self) -> int:
        """计算当前所有消息 + system prompt 的 token 总数"""
        total = 0
        if self.system_prompt:
            total += self.count_tokens(self.system_prompt)
        if self._summary:
            total += self.count_tokens(self._summary)
        for msg in self._messages:
            total += self.count_tokens(str(msg.get("content", "")))
        return total

    # ---- 压缩 & 截断 ----

    def summarize(self, llm_summarize: Callable[[list[dict]], str] | None = None) -> str | None:
        """
        压缩旧消息为摘要

        压缩策略: 保留最后 N 条消息，其余压缩为一段摘要文本。

        Args:
            llm_summarize: 摘要函数，None 则用初始化时的 summarize_func

        Returns:
            生成的摘要文本，没有需要压缩的内容返回 None
        """
        summarize_fn = llm_summarize or self._summarize_func
        if summarize_fn is None:
            logger.warning("未提供摘要函数，无法压缩消息")
            return None

        # 保留最后 4 条消息（2 轮对话），其余压缩
        keep_count = 4
        if len(self._messages) <= keep_count:
            return None  # 消息太少，不需要压缩

        to_summarize = list(self._messages)[:-keep_count]
        try:
            new_summary = summarize_fn(to_summarize)
        except Exception as e:
            logger.error(f"摘要生成失败: {e}")
            return None

        # 更新状态
        if self._summary:
            self._summary = self._summary + "\n" + new_summary
        else:
            self._summary = new_summary

        # 移除已压缩的消息
        for _ in range(len(to_summarize)):
            self._messages.popleft()

        logger.info(
            f"消息已压缩: {len(to_summarize)} 条 → {len(new_summary)} 字符摘要，"
            f"保留 {len(self._messages)} 条"
        )
        return new_summary

    def _trim_if_needed(self) -> int:
        """
        检查 token 预算，超出则从旧消息开始移除

        Returns:
            移除的消息数量
        """
        removed = 0
        # 为 system prompt 和摘要预留预算
        overhead = 0
        if self.system_prompt:
            overhead += self.count_tokens(self.system_prompt)
        if self._summary:
            overhead += self.count_tokens(self._summary)

        available = self.max_tokens - overhead
        if available <= 0:
            # 极端情况：system prompt 本身就超预算
            logger.warning(
                f"系统提示词+摘要 ({overhead} tokens) 已超预算 ({self.max_tokens})"
            )
            return 0

        while self._messages:
            current = sum(
                self.count_tokens(str(m.get("content", "")))
                for m in self._messages
            )
            if current <= available:
                break
            removed += 1
            self._messages.popleft()

        if removed > 0:
            logger.debug(f"Token 超预算，移除 {removed} 条旧消息")

        return removed

    # ---- 重置 ----

    def clear(self) -> None:
        """清空工作记忆"""
        self._messages.clear()
        self._summary = None
        logger.info("WorkingMemory 已清空")

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"WorkingMemory(messages={len(self._messages)}, "
            f"tokens={self.total_tokens()}/{self.max_tokens}, "
            f"summary={'有' if self._summary else '无'})"
        )
