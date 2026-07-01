"""感知模块主入口 —— PerceptionModule 整合所有子组件"""

from pathlib import Path
from .router import InputRouter
from .pipeline import PreprocessorPipeline
from .adapter import OutputAdapter
from .message import Message

# 过滤器
from .filters.denoise import DenoiseFilter
from .filters.sensitive import SensitiveFilter
from .filters.truncation import TruncationFilter
from .filters.language import LanguageFilter


class PerceptionModule:
    """
    感知模块 —— Agent 的输入感知层

    职责：接收外部输入 → 路由分发 → 预处理管道 → 标准化 Message

    使用示例:
        pm = PerceptionModule(sensitive_words_path="sensitive_words.txt")
        msg = pm.process("你好，世界！")
        if msg.is_rejected:
            print(msg.content)   # 拒绝提示
        else:
            print(msg.content)   # 清洗后的文本
            print(msg.metadata)  # 语言、长度等元信息
    """

    def __init__(
        self,
        max_length: int = 4000,
        sensitive_words_path: str | Path | None = None,
    ):
        """
        Args:
            max_length: 文本最大字符数，默认 4000
            sensitive_words_path: 敏感词文件路径，None 表示跳过敏感词过滤
        """
        # ---- 初始化各子组件 ----
        self.router = InputRouter()
        self.pipeline = PreprocessorPipeline()
        self.adapter = OutputAdapter()

        # ---- 按序注册预处理过滤器 ----
        # 顺序很重要：先去噪 → 再查敏感词 → 截断 → 语言检测
        self.pipeline.add_filter(DenoiseFilter())
        self.pipeline.add_filter(SensitiveFilter(sensitive_words_path))
        self.pipeline.add_filter(TruncationFilter(max_length))
        self.pipeline.add_filter(LanguageFilter())

    def process(self, raw_input) -> Message:
        """
        处理原始输入，返回标准化 Message

        Args:
            raw_input:
                - str: 文本输入
                - str/Path: 文件路径

        Returns:
            Message 对象。is_rejected=True 表示命中敏感词被拦截
        """
        # 1. 路由：识别输入类型，分发给对应处理器
        processor = self.router.dispatch(raw_input)
        raw_text = processor.read(raw_input)
        source_type = processor.source_type

        # 2. 预处理管道
        result = self.pipeline.run({
            "content": raw_text,
            "metadata": {},
        })

        # 3. 敏感词拦截 → 返回拒绝消息
        if result is None:
            return Message(
                content="您输入的内容包含违规信息，已被系统拦截，请修改后重试。",
                is_rejected=True,
                reject_reason="sensitive_words",
                source_type=source_type,
            )

        # 4. 封装为标准 Message
        return self.adapter.to_message(
            content=result["content"],
            source_type=source_type,
            metadata=result["metadata"],
        )
