"""感知模块 - Agent 的输入感知层

职责：接收外部输入 → 路由分发 → 预处理管道 → 标准化 Message 输出
"""

from .module import PerceptionModule
from .message import Message, RejectException

__all__ = ["PerceptionModule", "Message", "RejectException"]
