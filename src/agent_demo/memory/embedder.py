"""Embedder 抽象层 —— 文本转向量的统一接口

支持两种实现:
    - DashScopeEmbedder: 调用 DashScope text-embedding-v3 API（已有 API Key）
    - LocalEmbedder: 基于 sentence-transformers 的本地模型（离线可用，可选依赖）

使用方式:
    # 默认直接用 DashScope
    embedder = DashScopeEmbedder(api_key="sk-xxx")

    # 本地模型（需 pip install sentence-transformers）
    embedder = LocalEmbedder(model_name="BAAI/bge-small-zh-v1.5")

    # 工厂函数
    embedder = create_embedder("dashscope", api_key="sk-xxx")
    embedder = create_embedder("local", model_name="BAAI/bge-small-zh-v1.5")
"""

import os
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================
# 抽象接口
# ============================================================

class Embedder(ABC):
    """文本转向量的抽象接口 —— 所有 Embedding 实现必须继承此类"""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """将单条文本转换为向量"""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本转向量 —— 默认逐条调用，子类可覆盖为批量 API 调用"""
        return [self.embed(t) for t in texts]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        ...


# ============================================================
# DashScope Embedding 实现
# ============================================================

class DashScopeEmbedder(Embedder):
    """
    基于 DashScope text-embedding-v3 的 Embedding 服务

    复用 QwenClient 的 OpenAI SDK 调用 /embeddings 端点。
    模型: text-embedding-v3，维度: 1024
    """

    MODEL_NAME = "text-embedding-v3"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """
        Args:
            api_key: DashScope API Key，None 则从环境变量 DASHSCOPE_API_KEY 读取
            base_url: API 端点，None 则用默认 DashScope 地址
        """
        from openai import OpenAI

        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DashScope API Key 未设置。请设置环境变量 DASHSCOPE_API_KEY "
                "或通过 DashScopeEmbedder(api_key='sk-xxx') 传入"
            )

        base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"DashScopeEmbedder 初始化完成，模型: {self.MODEL_NAME}")

    @property
    def dimension(self) -> int:
        return 1024

    def embed(self, text: str) -> list[float]:
        """调用 DashScope embedding API"""
        if not text or not text.strip():
            # 空文本返回零向量
            return [0.0] * self.dimension

        try:
            response = self._client.embeddings.create(
                model=self.MODEL_NAME,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding 调用失败: {e}")
            raise

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量调用 —— DashScope 支持一次传入多条文本"""
        if not texts:
            return []

        # 过滤空文本，记录位置
        non_empty_indices = []
        non_empty_texts = []
        for i, t in enumerate(texts):
            if t and t.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(t)

        if not non_empty_texts:
            return [[0.0] * self.dimension for _ in texts]

        try:
            response = self._client.embeddings.create(
                model=self.MODEL_NAME,
                input=non_empty_texts,
            )
            embeddings = [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"批量 Embedding 调用失败: {e}")
            raise

        # 还原到原始位置，空文本位置填充零向量
        result = [[0.0] * self.dimension for _ in texts]
        for idx, emb in zip(non_empty_indices, embeddings):
            result[idx] = emb
        return result


# ============================================================
# 本地 Embedding 实现（可选依赖）
# ============================================================

class LocalEmbedder(Embedder):
    """
    基于 sentence-transformers 的本地 Embedding 模型

    首次使用会自动下载模型文件（~100MB-1GB）。
    默认模型: BAAI/bge-small-zh-v1.5（中文优化，512 维，轻量）

    需要: pip install sentence-transformers
    """

    DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

    def __init__(self, model_name: str | None = None):
        """
        Args:
            model_name: HuggingFace 模型名，None 则用默认 BGE 中文模型
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "本地 Embedding 需要 sentence-transformers 库。"
                "请执行: pip install sentence-transformers"
            )

        model_name = model_name or self.DEFAULT_MODEL
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"LocalEmbedder 初始化完成，模型: {model_name}，维度: {self._dimension}"
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """单条文本转向量"""
        if not text or not text.strip():
            return [0.0] * self.dimension
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本转向量 —— 本地模型批量推理比逐条快很多"""
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


# ============================================================
# 工厂函数
# ============================================================

def create_embedder(
    provider: str = "dashscope",
    **kwargs,
) -> Embedder:
    """
    Embedder 工厂函数 —— 根据 provider 名称创建对应的 Embedder

    Args:
        provider: "dashscope" | "local"
        **kwargs: 传递给具体实现的参数
            - DashScopeEmbedder: api_key, base_url
            - LocalEmbedder: model_name

    Returns:
        Embedder 实例

    Example:
        emb = create_embedder("dashscope", api_key="sk-xxx")
        emb = create_embedder("local", model_name="BAAI/bge-small-zh-v1.5")
    """
    if provider == "dashscope":
        return DashScopeEmbedder(
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
        )
    elif provider == "local":
        return LocalEmbedder(
            model_name=kwargs.get("model_name"),
        )
    else:
        raise ValueError(
            f"不支持的 Embedding 提供商: '{provider}'，可选: 'dashscope', 'local'"
        )
