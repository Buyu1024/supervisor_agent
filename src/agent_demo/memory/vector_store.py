"""FAISSVectorStore —— 基于 FAISS 的向量存储与检索

职责:
    1. 管理 FAISS 索引（创建、增删、持久化）
    2. 文本 → 向量 → 存入索引
    3. 语义检索（查询文本 → 向量 → 搜索 Top-K）
    4. 元数据管理（MemoryItem 除向量外的字段）

内部结构:
    - FAISS IndexFlatIP: 内积相似度（等价于余弦相似度，向量已归一化时）
    - _id_map: FAISS 内部 int ID → MemoryItem 字符串 ID 的映射
    - _metadata: MemoryItem ID → MemoryItem（不含向量）的映射
"""

import os
import pickle
import logging
from typing import Optional

from .types import MemoryItem, MemorySearchResult
from .embedder import Embedder

# 懒加载: numpy 和 faiss 只在首次实例化 FAISSVectorStore 时导入
_np = None
_faiss = None


def _get_np():
    """懒加载 numpy"""
    global _np
    if _np is None:
        import numpy as _np_mod
        _np = _np_mod
    return _np


def _get_faiss():
    """懒加载 faiss"""
    global _faiss
    if _faiss is None:
        try:
            import faiss as _faiss_mod
            _faiss = _faiss_mod
        except ImportError:
            raise ImportError(
                "FAISS 未安装。请执行: pip install faiss-cpu"
            )
    return _faiss

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS 向量存储 —— 封装 FAISS 索引的 CRUD + 持久化

    使用内积相似度 (IndexFlatIP)，向量需先 L2 归一化。
    适用于中小规模数据（< 100K 条），大型数据可换 IndexIVFFlat 等近似索引。

    使用示例:
        store = FAISSVectorStore(embedder=embedder, dimension=1024)
        store.add(MemoryItem.create(content="用户喜欢简洁回答", memory_type="preference"))
        results = store.search("用户偏好", top_k=5)
        store.save("data/memory.index")
        store.load("data/memory.index")
    """

    def __init__(self, embedder: Embedder, dimension: int | None = None):
        """
        Args:
            embedder: 文本转向量的 Embedder 实例
            dimension: 向量维度，None 则从 embedder.dimension 读取
        """
        self._embedder = embedder
        self._dimension = dimension or embedder.dimension

        # FAISS 内积索引（使用内积相似度，等价于归一化后的余弦相似度）
        self._index = _get_faiss().IndexFlatIP(self._dimension)

        # FAISS 内部 int ID → MemoryItem 字符串 ID
        self._id_map: dict[int, str] = {}

        # MemoryItem 字符串 ID → 元数据（不含向量，向量在 FAISS 索引中）
        self._metadata: dict[str, MemoryItem] = {}

        # 下一个 FAISS 内部 ID
        self._next_id: int = 0

        logger.info(
            f"FAISSVectorStore 初始化完成，维度: {self._dimension}，"
            f"索引类型: IndexFlatIP"
        )

    # ---- CRUD ----

    def add(self, item: MemoryItem) -> str:
        """
        添加一条记忆到向量存储

        自动计算 embedding（若 item 未缓存），L2 归一化后加入 FAISS 索引。

        Args:
            item: 要添加的记忆条目

        Returns:
            item.id（与传入的一致）
        """
        # 计算或复用 embedding
        if item.embedding is None:
            item.embedding = self._embedder.embed(item.content)

        vector = _get_np().array(item.embedding, dtype=_get_np().float32)
        # L2 归一化，使内积等价于余弦相似度
        _get_faiss().normalize_L2(vector.reshape(1, -1))

        # 添加到 FAISS 索引
        faiss_id = self._next_id
        self._index.add(vector.reshape(1, -1))
        self._id_map[faiss_id] = item.id
        self._metadata[item.id] = item
        self._next_id += 1

        logger.debug(f"向量已添加: id={item.id}, type={item.memory_type}")
        return item.id

    def add_batch(self, items: list[MemoryItem]) -> list[str]:
        """
        批量添加记忆 —— 批量计算 embedding 后一次性加入 FAISS 索引

        Args:
            items: 要添加的记忆条目列表

        Returns:
            所有 item.id 的列表
        """
        if not items:
            return []

        # 收集需要计算 embedding 的 item
        texts_to_embed = []
        indices_to_embed = []
        for i, item in enumerate(items):
            if item.embedding is None:
                texts_to_embed.append(item.content)
                indices_to_embed.append(i)

        # 批量计算 embedding
        if texts_to_embed:
            embeddings = self._embedder.embed_batch(texts_to_embed)
            for i, emb in zip(indices_to_embed, embeddings):
                items[i].embedding = emb

        # 构建矩阵
        vectors = _get_np().array(
            [item.embedding for item in items], dtype=_get_np().float32
        )
        _get_faiss().normalize_L2(vectors)

        # 批量加入 FAISS 索引
        start_id = self._next_id
        self._index.add(vectors)
        for offset, item in enumerate(items):
            faiss_id = start_id + offset
            self._id_map[faiss_id] = item.id
            self._metadata[item.id] = item
        self._next_id += len(items)

        logger.debug(f"批量添加 {len(items)} 条向量")
        return [item.id for item in items]

    def search(
        self,
        query: str,
        top_k: int = 5,
        type_filter: str | None = None,
    ) -> list[MemorySearchResult]:
        """
        语义检索 —— 返回与查询最相似的 top_k 条记忆

        Args:
            query: 查询文本
            top_k: 返回条数
            type_filter: 只返回指定类型的记忆（None 表示不过滤）

        Returns:
            按相似度降序排列的 MemorySearchResult 列表
        """
        if self._index.ntotal == 0:
            return []

        query_embedding = self._embedder.embed(query)
        query_vector = _get_np().array(query_embedding, dtype=_get_np().float32).reshape(1, -1)
        _get_faiss().normalize_L2(query_vector)

        # 检索时多取一些，方便类型过滤后仍有 top_k 条
        fetch_k = top_k * 3 if type_filter else top_k
        fetch_k = min(fetch_k, self._index.ntotal)

        distances, indices = self._index.search(query_vector, fetch_k)

        results = []
        for dist, faiss_id in zip(distances[0], indices[0]):
            if faiss_id == -1:  # FAISS 无效 ID
                continue
            item_id = self._id_map.get(int(faiss_id))
            if item_id is None:
                continue
            item = self._metadata.get(item_id)
            if item is None:
                continue

            # 类型过滤
            if type_filter and item.memory_type != type_filter:
                continue

            item.touch()  # 更新访问统计
            results.append(MemorySearchResult(item=item, score=float(dist)))

            if len(results) >= top_k:
                break

        return results

    def delete(self, item_id: str) -> bool:
        """
        删除一条记忆

        注意: FAISS IndexFlat 不支持真正的删除，这里做软删除（从元数据中移除）。
        索引中的数据仍存在，但 search 时会跳过。

        Args:
            item_id: 记忆 ID

        Returns:
            是否成功删除（False 表示 ID 不存在）
        """
        if item_id not in self._metadata:
            return False

        del self._metadata[item_id]
        # 清理 _id_map 中的映射
        for faiss_id, mid in list(self._id_map.items()):
            if mid == item_id:
                del self._id_map[faiss_id]
                break

        logger.debug(f"记忆已删除: id={item_id}")
        return True

    def get(self, item_id: str) -> MemoryItem | None:
        """按 ID 获取记忆条目"""
        return self._metadata.get(item_id)

    @property
    def count(self) -> int:
        """已存储的记忆数量"""
        return len(self._metadata)

    def list_ids(self) -> list[str]:
        """列出所有记忆 ID"""
        return list(self._metadata.keys())

    # ---- 持久化 ----

    def save(self, path: str) -> None:
        """
        保存 FAISS 索引和元数据到磁盘

        生成两个文件:
            - {path}: FAISS 索引（二进制）
            - {path}.meta: 元数据（pickle）

        Args:
            path: 保存路径（不含扩展名，实际会生成 .meta 文件）
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # 保存 FAISS 索引
        _get_faiss().write_index(self._index, path)

        # 保存元数据
        meta = {
            "id_map": self._id_map,
            "metadata": self._metadata,
            "next_id": self._next_id,
            "dimension": self._dimension,
        }
        with open(path + ".meta", "wb") as f:
            pickle.dump(meta, f)

        logger.info(
            f"向量存储已保存: {path} (索引) + {path}.meta (元数据)，"
            f"共 {self.count} 条记忆"
        )

    def load(self, path: str) -> None:
        """
        从磁盘加载 FAISS 索引和元数据

        Args:
            path: 之前 save() 时使用的路径
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS 索引文件不存在: {path}")

        meta_path = path + ".meta"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"元数据文件不存在: {meta_path}")

        # 加载 FAISS 索引
        self._index = _get_faiss().read_index(path)

        # 加载元数据
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self._id_map = meta["id_map"]
        self._metadata = meta["metadata"]
        self._next_id = meta["next_id"]
        self._dimension = meta.get("dimension", self._dimension)

        logger.info(
            f"向量存储已加载: {path}，共 {self.count} 条记忆"
        )

    # ---- 内部辅助 ----

    def _rebuild_index(self) -> None:
        """
        重建 FAISS 索引 —— 删除操作较多后使用，清理已删除的向量

        使用场景: 调用 delete() 只是软删除，向量仍在索引中。
        调用此方法会从元数据重新构建索引，移除已删除的条目。
        """
        items = list(self._metadata.values())
        self._index = _get_faiss().IndexFlatIP(self._dimension)
        self._id_map = {}
        self._next_id = 0

        if items:
            self.add_batch(items)

        logger.info(f"索引已重建，当前 {self.count} 条")
