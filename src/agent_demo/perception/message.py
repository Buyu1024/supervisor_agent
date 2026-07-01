"""感知模块 - 标准化消息对象"""

from dataclasses import dataclass, field


class RejectException(Exception):
    """内容被拒绝异常（敏感词命中时由 SensitiveFilter 抛出）"""
    pass


@dataclass
class Message:
    """标准化消息对象，感知模块的统一输出格式"""

    role: str = "user"
    content: str = ""
    source_type: str = ""           # "text" / "file"
    metadata: dict = field(default_factory=dict)
    attachments: list = field(default_factory=list)

    # 敏感词拦截标记
    is_rejected: bool = False
    reject_reason: str = ""

    def __repr__(self) -> str:
        status = "REJECTED" if self.is_rejected else "OK"
        return (
            f"<Message role={self.role} source={self.source_type} "
            f"status={status} len={len(self.content)}>"
        )
