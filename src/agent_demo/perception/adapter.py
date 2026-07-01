"""输出适配器 —— 将管道处理结果封装为 Message"""

from .message import Message


class OutputAdapter:
    """输出适配器：把管道产出的 content + metadata 封装为标准 Message 对象"""

    def to_message(
        self,
        content: str,
        source_type: str,
        metadata: dict | None = None,
    ) -> Message:
        return Message(
            role="user",
            content=content,
            source_type=source_type,
            metadata=metadata or {},
        )
