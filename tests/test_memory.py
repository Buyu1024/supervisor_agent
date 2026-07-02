"""记忆模块 —— 集成测试 Demo"""

import os
import sys
import time
import tempfile
from pathlib import Path

# 确保 src 目录在 Python 路径中（支持直接 python tests/test_memory.py 运行）
_SRC_DIR = Path(__file__).parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# 自动加载 .env 文件（如果存在）
_ENV_PATH = Path(__file__).parent.parent / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())


# ============================================================
# 测试用例
# ============================================================

class TestTypes:
    """MemoryItem / MemorySearchResult 数据结构测试"""

    def test_memory_item_create(self):
        """MemoryItem.create() 快捷创建"""
        from agent_demo.memory import MemoryItem

        item = MemoryItem.create(
            content="用户喜欢用中文回答",
            memory_type="preference",
            importance=0.8,
            metadata={"source": "user_input"},
        )
        assert item.id, "应自动生成 ID"
        assert len(item.id) == 12, f"ID 长度应为 12，实际: {len(item.id)}"
        assert item.content == "用户喜欢用中文回答"
        assert item.memory_type == "preference"
        assert item.importance == 0.8
        assert item.metadata["source"] == "user_input"
        assert item.created_at > 0
        assert item.access_count == 0
        print(f"  [PASS] MemoryItem 创建: {item}")

    def test_memory_item_touch(self):
        """touch() 更新访问统计"""
        from agent_demo.memory import MemoryItem

        item = MemoryItem.create(content="测试")
        old_time = item.last_accessed
        time.sleep(0.01)
        item.touch()
        assert item.last_accessed > old_time
        assert item.access_count == 1
        print(f"  [PASS] MemoryItem touch: access_count={item.access_count}")

    def test_memory_search_result(self):
        """MemorySearchResult 创建和显示"""
        from agent_demo.memory import MemoryItem, MemorySearchResult

        item = MemoryItem.create(content="这是一条测试记忆", memory_type="knowledge")
        result = MemorySearchResult(item=item, score=0.85)
        assert result.score == 0.85
        assert result.item.memory_type == "knowledge"
        assert "0.8500" in repr(result)
        print(f"  [PASS] MemorySearchResult: {result}")


class TestSessionStore:
    """SessionStore 会话 KV 测试"""

    def test_set_get(self):
        """基本 set/get 操作"""
        from agent_demo.memory.session_store import SessionStore

        store = SessionStore()
        store.set("key1", "value1")
        store.set("key2", 42)

        assert store.get("key1") == "value1"
        assert store.get("key2") == 42
        assert store.get("nonexistent", "default") == "default"
        print(f"  [PASS] SessionStore set/get: keys={store.keys()}")

    def test_delete_exists(self):
        """delete 和 exists"""
        from agent_demo.memory.session_store import SessionStore

        store = SessionStore()
        store.set("temp", "data")
        assert store.exists("temp")
        assert store.delete("temp")
        assert not store.exists("temp")
        assert not store.delete("temp")  # 二次删除返回 False
        print(f"  [PASS] SessionStore delete/exists")

    def test_ttl_expiry(self):
        """TTL 过期机制"""
        from agent_demo.memory.session_store import SessionStore

        store = SessionStore()
        store.set("ephemeral", "短命数据", ttl=0.01)  # 10ms 过期
        assert store.get("ephemeral") == "短命数据"
        time.sleep(0.02)
        assert store.get("ephemeral") is None, "过期数据应返回 None"
        print(f"  [PASS] SessionStore TTL 过期")

    def test_export(self):
        """export 导出所有数据"""
        from agent_demo.memory.session_store import SessionStore

        store = SessionStore()
        store.set("task", "搜索天气")
        store.set("results", {"city": "北京", "weather": "晴"})

        data = store.export()
        assert data["task"] == "搜索天气"
        assert data["results"]["city"] == "北京"
        print(f"  [PASS] SessionStore export: {len(data)} 个键")

    def test_export_summary(self):
        """export_summary 生成文本摘要"""
        from agent_demo.memory.session_store import SessionStore

        store = SessionStore()
        store.set("当前任务", "帮用户查询北京天气")

        summary = store.export_summary()
        assert "当前任务" in summary
        assert "北京天气" in summary
        print(f"  [PASS] SessionStore export_summary: {len(summary)} 字符")

    def test_clear(self):
        """clear 清空所有数据"""
        from agent_demo.memory.session_store import SessionStore

        store = SessionStore()
        store.set("a", 1)
        store.set("b", 2)
        store.clear()
        assert len(store) == 0
        print(f"  [PASS] SessionStore clear")


class TestRelStore:
    """RelStore SQLite 结构化存储测试"""

    def test_preferences(self):
        """偏好存取"""
        from agent_demo.memory.rel_store import RelStore

        with RelStore(":memory:") as store:
            store.set_preference("语言", "中文")
            store.set_preference("回复风格", "简洁")

            assert store.get_preference("语言") == "中文"
            prefs = store.get_all_preferences()
            assert len(prefs) == 2
            assert prefs["回复风格"] == "简洁"
            print(f"  [PASS] RelStore 偏好: {prefs}")

    def test_delete_preference(self):
        """删除偏好"""
        from agent_demo.memory.rel_store import RelStore

        with RelStore(":memory:") as store:
            store.set_preference("key", "value")
            assert store.delete_preference("key")
            assert store.get_preference("key") is None
            assert not store.delete_preference("key")  # 二次删除返回 False
            print(f"  [PASS] RelStore 删除偏好")

    def test_entities(self):
        """实体 CRUD"""
        from agent_demo.memory.rel_store import RelStore

        with RelStore(":memory:") as store:
            store.upsert_entity("张三", "person", {"age": 25, "city": "北京"})
            store.upsert_entity("Python", "language", {"version": "3.12"})

            entity = store.get_entity("张三")
            assert entity["type"] == "person"
            assert entity["properties"]["age"] == 25

            all_entities = store.list_entities()
            assert len(all_entities) == 2

            # 类型过滤
            persons = store.list_entities(entity_type="person")
            assert len(persons) == 1
            assert persons[0]["name"] == "张三"
            print(f"  [PASS] RelStore 实体: {len(all_entities)} 个实体")

    def test_search_entities(self):
        """关键词搜索实体"""
        from agent_demo.memory.rel_store import RelStore

        with RelStore(":memory:") as store:
            store.upsert_entity("张三", "person", {})
            store.upsert_entity("李四", "person", {})
            store.upsert_entity("公司A", "company", {})

            results = store.search_entities("张")
            assert len(results) == 1
            assert results[0]["name"] == "张三"

            results = store.search_entities("person")
            assert len(results) == 2  # type 匹配
            print(f"  [PASS] RelStore 搜索实体")

    def test_relations(self):
        """实体关系 CRUD"""
        from agent_demo.memory.rel_store import RelStore

        with RelStore(":memory:") as store:
            r1 = store.add_relation("张三", "喜欢", "Python")
            r2 = store.add_relation("张三", "工作于", "公司A")
            store.add_relation("李四", "喜欢", "Python")

            # 按 source 查询
            rels = store.query_relations(source="张三")
            assert len(rels) == 2

            # 按 relation 查询
            rels = store.query_relations(relation="喜欢")
            assert len(rels) == 2

            # 按 source + relation 查询
            rels = store.query_relations(source="张三", relation="喜欢")
            assert len(rels) == 1
            assert rels[0]["target"] == "Python"

            # 删除关系
            assert store.delete_relation(r1)
            rels = store.query_relations(source="张三")
            assert len(rels) == 1
            print(f"  [PASS] RelStore 关系")

    def test_delete_entity_cascades(self):
        """删除实体时级联删除关系"""
        from agent_demo.memory.rel_store import RelStore

        with RelStore(":memory:") as store:
            store.upsert_entity("张三", "person", {})
            store.add_relation("张三", "喜欢", "Python")
            store.add_relation("李四", "认识", "张三")

            store.delete_entity("张三")
            assert store.get_entity("张三") is None
            assert len(store.query_relations(source="张三")) == 0
            assert len(store.query_relations(target="张三")) == 0
            print(f"  [PASS] RelStore 级联删除")


class TestEmbedder:
    """Embedder 抽象层测试"""

    def test_dashscope_embedder_init(self):
        """DashScopeEmbedder 初始化测试（离线）"""
        from agent_demo.memory.embedder import DashScopeEmbedder

        # 无 API Key 应抛异常
        saved = os.environ.pop("DASHSCOPE_API_KEY", None)
        try:
            DashScopeEmbedder(api_key=None)
            assert False, "应抛出 ValueError"
        except ValueError as e:
            assert "API Key" in str(e)
            print(f"  [PASS] DashScopeEmbedder 无 Key 抛异常")
        finally:
            if saved:
                os.environ["DASHSCOPE_API_KEY"] = saved

    def test_dashscope_embedder_with_key(self):
        """DashScopeEmbedder 显式 Key 初始化"""
        from agent_demo.memory.embedder import DashScopeEmbedder

        emb = DashScopeEmbedder(api_key="sk-test-key")
        assert emb.dimension == 1024
        assert emb.MODEL_NAME == "text-embedding-v3"
        print(f"  [PASS] DashScopeEmbedder 初始化: dim={emb.dimension}")

    def test_embed_empty_text(self):
        """空文本返回零向量"""
        from agent_demo.memory.embedder import DashScopeEmbedder

        emb = DashScopeEmbedder(api_key="sk-test-key")
        # 空文本不调 API，直接返回零向量
        vec = emb.embed("")
        assert len(vec) == emb.dimension
        assert all(v == 0.0 for v in vec), "空文本应返回零向量"
        print(f"  [PASS] Embedder 空文本: {len(vec)} 维零向量")

    def test_create_embedder_factory(self):
        """工厂函数 create_embedder"""
        from agent_demo.memory import create_embedder

        emb = create_embedder("dashscope", api_key="sk-test")
        assert emb.dimension == 1024

        # 无效 provider
        try:
            create_embedder("unknown")
            assert False, "应抛出 ValueError"
        except ValueError as e:
            assert "不支持" in str(e)
            print(f"  [PASS] create_embedder 工厂: {type(emb).__name__}")


class TestVectorStore:
    """FAISSVectorStore 向量存储测试"""

    def _make_embedder(self):
        """创建一个假的 Embedder 用于测试（返回确定性向量）"""
        from agent_demo.memory.embedder import Embedder

        class _FakeEmbedder(Embedder):
            dimension = 128

            def embed(self, text: str) -> list[float]:
                import hashlib
                # 用文本 hash 生成确定性向量（循环扩展至目标维度）
                h = hashlib.sha256(text.encode()).digest()
                vec = []
                for i in range(self.dimension):
                    vec.append(float(h[i % len(h)]) / 255.0)
                return vec

        return _FakeEmbedder()

    def test_add_and_search(self):
        """添加记忆后能检索到"""
        from agent_demo.memory.vector_store import FAISSVectorStore
        from agent_demo.memory import MemoryItem

        store = FAISSVectorStore(embedder=self._make_embedder())
        item = MemoryItem.create(content="Python 是一种编程语言", memory_type="knowledge")
        store.add(item)

        assert store.count == 1

        results = store.search("Python", top_k=3)
        assert len(results) == 1
        assert results[0].item.id == item.id
        assert results[0].score > 0  # 应有正相似度
        print(f"  [PASS] VectorStore add+search: {len(results)} 条，score={results[0].score:.4f}")

    def test_add_batch(self):
        """批量添加记忆"""
        from agent_demo.memory.vector_store import FAISSVectorStore
        from agent_demo.memory import MemoryItem

        store = FAISSVectorStore(embedder=self._make_embedder())
        items = [
            MemoryItem.create(content=f"记忆 {i}", memory_type="knowledge")
            for i in range(10)
        ]
        store.add_batch(items)

        assert store.count == 10
        results = store.search("记忆 5", top_k=3)
        assert len(results) == 3
        print(f"  [PASS] VectorStore 批量添加: {store.count} 条")

    def test_search_type_filter(self):
        """类型过滤搜索"""
        from agent_demo.memory.vector_store import FAISSVectorStore
        from agent_demo.memory import MemoryItem

        store = FAISSVectorStore(embedder=self._make_embedder())
        store.add(MemoryItem.create(content="偏好: 喜欢中文", memory_type="preference"))
        store.add(MemoryItem.create(content="知识: 北京是首都", memory_type="knowledge"))
        store.add(MemoryItem.create(content="偏好: 喜欢简洁回答", memory_type="preference"))

        # 只查 preference 类型
        results = store.search("偏好", top_k=5, type_filter="preference")
        assert len(results) == 2
        assert all(r.item.memory_type == "preference" for r in results)
        print(f"  [PASS] VectorStore 类型过滤: {len(results)} 条 preference")

    def test_delete(self):
        """删除记忆"""
        from agent_demo.memory.vector_store import FAISSVectorStore
        from agent_demo.memory import MemoryItem

        store = FAISSVectorStore(embedder=self._make_embedder())
        item = MemoryItem.create(content="测试记忆", memory_type="knowledge")
        store.add(item)

        assert store.delete(item.id)
        assert store.get(item.id) is None
        assert not store.delete("nonexistent")  # 删除不存在的返回 False
        print(f"  [PASS] VectorStore 删除: count={store.count}")

    def test_persistence(self):
        """持久化保存和加载"""
        from agent_demo.memory.vector_store import FAISSVectorStore
        from agent_demo.memory import MemoryItem

        # 创建临时文件
        tmp_dir = tempfile.mkdtemp()
        index_path = os.path.join(tmp_dir, "test.index")

        try:
            # 创建并保存
            store1 = FAISSVectorStore(embedder=self._make_embedder())
            store1.add(MemoryItem.create(content="持久化测试记忆 1"))
            store1.add(MemoryItem.create(content="持久化测试记忆 2"))
            store1.save(index_path)

            # 加载到新实例
            store2 = FAISSVectorStore(embedder=self._make_embedder())
            store2.load(index_path)
            assert store2.count == 2

            results = store2.search("持久化", top_k=3)
            assert len(results) == 2
            print(f"  [PASS] VectorStore 持久化: 保存 {store1.count} → 加载 {store2.count}")
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_rebuild_index(self):
        """重建索引：删除后的清理"""
        from agent_demo.memory.vector_store import FAISSVectorStore
        from agent_demo.memory import MemoryItem

        store = FAISSVectorStore(embedder=self._make_embedder())
        items = [MemoryItem.create(content=f"记忆 {i}") for i in range(5)]
        store.add_batch(items)

        # 删除 2 条
        store.delete(items[0].id)
        store.delete(items[1].id)
        assert store.count == 3  # 软删除后 count 已减

        # 重建索引
        store._rebuild_index()
        assert store.count == 3

        # 验证搜索正常
        results = store.search("记忆", top_k=5)
        assert len(results) == 3
        print(f"  [PASS] VectorStore 重建索引: {store.count} 条")


class TestWorkingMemory:
    """WorkingMemory 短期记忆测试"""

    def test_add_and_get_context(self):
        """添加消息后获取上下文"""
        from agent_demo.memory.working_memory import WorkingMemory

        wm = WorkingMemory(max_tokens=8000, system_prompt="你是助手")
        wm.add({"role": "user", "content": "你好"})
        wm.add({"role": "assistant", "content": "你好！有什么可以帮你的？"})

        ctx = wm.get_context()
        assert len(ctx) == 3  # system + user + assistant
        assert ctx[0]["role"] == "system"
        assert ctx[1]["role"] == "user"
        print(f"  [PASS] WorkingMemory get_context: {len(ctx)} 条消息")

    def test_token_trimming(self):
        """Token 超预算时自动截断旧消息"""
        from agent_demo.memory.working_memory import WorkingMemory

        # 设置有限预算，每条消息 ~15-20 tokens，预算 200 tokens
        wm = WorkingMemory(max_tokens=200, system_prompt="测试助手")
        long_text = "这是一条测试消息内容用于消耗token预算 " * 3  # ~60 chars
        for i in range(15):
            wm.add({"role": "user", "content": f"{long_text} {i}"})

        # 应触发截断，消息数在添加过程中被削减
        msg_count = len(wm)
        assert msg_count < 15, f"应触发截断，实际 {msg_count} 条"
        print(f"  [PASS] WorkingMemory 截断: {len(wm)} 条消息, {wm.total_tokens()} tokens")

    def test_summarize(self):
        """压缩旧消息为摘要"""
        from agent_demo.memory.working_memory import WorkingMemory

        def fake_summarize(messages):
            return f"压缩了 {len(messages)} 条历史对话"

        wm = WorkingMemory(
            max_tokens=8000,
            summarize_func=fake_summarize,
        )
        # 添加 10 条消息（超过保留的 4 条）
        for i in range(10):
            wm.add({"role": "user", "content": f"消息 {i}"})
            wm.add({"role": "assistant", "content": f"回复 {i}"})

        summary = wm.summarize()
        assert summary is not None
        assert "压缩了" in summary
        # 应只保留最后 4 条
        assert len(wm) <= 4
        print(f"  [PASS] WorkingMemory 压缩: {summary}, 剩余 {len(wm)} 条")

    def test_clear(self):
        """清空工作记忆"""
        from agent_demo.memory.working_memory import WorkingMemory

        wm = WorkingMemory(max_tokens=8000)
        wm.add({"role": "user", "content": "测试"})
        wm.clear()
        assert len(wm) == 0
        ctx = wm.get_context()
        assert len(ctx) == 0  # 无 system prompt，应完全为空
        print(f"  [PASS] WorkingMemory clear: {len(wm)} 条")

    def test_no_tiktoken_fallback(self):
        """tiktoken 不可用时的回退估算"""
        from agent_demo.memory.working_memory import _estimate_tokens

        # 中文
        cn_tokens = _estimate_tokens("你好世界")
        assert cn_tokens > 0

        # 英文
        en_tokens = _estimate_tokens("Hello world")
        assert en_tokens > 0

        # 空文本
        assert _estimate_tokens("") == 0
        print(f"  [PASS] Token 估算回退: 中文={cn_tokens}, 英文={en_tokens}")


class TestLongTermMemory:
    """LongTermMemory 长期记忆测试"""

    def _make_embedder(self):
        from agent_demo.memory.embedder import Embedder

        class _FakeEmbedder(Embedder):
            dimension = 64

            def embed(self, text: str) -> list[float]:
                import hashlib
                h = hashlib.sha256(text.encode()).digest()
                vec = []
                for i in range(self.dimension):
                    vec.append(float(h[i % len(h)]) / 255.0)
                return vec

        return _FakeEmbedder()

    def test_remember_and_retrieve(self):
        """记忆存储和检索"""
        from agent_demo.memory.long_term import LongTermMemory
        from agent_demo.memory import MemoryItem

        ltm = LongTermMemory(embedder=self._make_embedder())

        ltm.remember(MemoryItem.create(content="用户喜欢用中文回答", memory_type="preference"))
        ltm.remember(MemoryItem.create(content="用户在北京工作", memory_type="entity"))
        ltm.remember(MemoryItem.create(content="Python 异步编程技巧", memory_type="knowledge"))

        assert ltm.count == 3

        results = ltm.retrieve("用户偏好什么语言？", top_k=3)
        assert len(results) > 0
        # 第一条应是最相关的偏好
        assert "中文" in results[0].item.content
        print(f"  [PASS] LongTermMemory remember+retrieve: {len(results)} 条结果")

    def test_export_context(self):
        """导出上下文字符串"""
        from agent_demo.memory.long_term import LongTermMemory
        from agent_demo.memory import MemoryItem

        ltm = LongTermMemory(embedder=self._make_embedder())
        ltm.remember(MemoryItem.create(content="回复风格: 简洁", memory_type="preference"))
        ltm.remember(MemoryItem.create(content="偏好语言: 中文", memory_type="preference"))

        context = ltm.export_context(query="用户偏好", top_k=3)
        assert "用户偏好" in context
        assert "简洁" in context
        print(f"  [PASS] LongTermMemory export_context: {len(context)} 字符")

    def test_forget_by_access(self):
        """按访问时间遗忘"""
        from agent_demo.memory.long_term import LongTermMemory
        from agent_demo.memory import MemoryItem

        ltm = LongTermMemory(embedder=self._make_embedder())

        # 创建一条记忆，手动将 last_accessed 设到很久以前
        item = MemoryItem.create(content="过时的记忆", memory_type="knowledge")
        item.last_accessed = 0  # Unix epoch
        ltm.remember(item)

        # 添加一条新记忆
        ltm.remember(MemoryItem.create(content="新的记忆", memory_type="knowledge"))

        assert ltm.count == 2
        deleted = ltm.forget_by_access(days_stale=1, min_access=0)
        assert deleted >= 1  # 过时记忆被删除
        assert ltm.count <= 1
        print(f"  [PASS] LongTermMemory forget_by_access: 删除 {deleted} 条")

    def test_forget_low_importance(self):
        """按重要性遗忘"""
        from agent_demo.memory.long_term import LongTermMemory
        from agent_demo.memory import MemoryItem

        ltm = LongTermMemory(embedder=self._make_embedder())
        ltm.remember(MemoryItem.create(content="重要记忆", importance=0.9))
        ltm.remember(MemoryItem.create(content="不重要记忆", importance=0.05))

        deleted = ltm.forget_low_importance(threshold=0.1)
        assert deleted >= 1
        print(f"  [PASS] LongTermMemory forget_low_importance: 删除 {deleted} 条")

    def test_get_stats(self):
        """获取统计信息"""
        from agent_demo.memory.long_term import LongTermMemory
        from agent_demo.memory import MemoryItem

        ltm = LongTermMemory(embedder=self._make_embedder())
        ltm.remember(MemoryItem.create(content="偏好1", memory_type="preference"))
        ltm.remember(MemoryItem.create(content="知识1", memory_type="knowledge"))
        ltm.remember(MemoryItem.create(content="知识2", memory_type="knowledge"))

        stats = ltm.get_stats()
        assert stats["total"] == 3
        assert stats["by_type"]["preference"] == 1
        assert stats["by_type"]["knowledge"] == 2
        print(f"  [PASS] LongTermMemory get_stats: {stats}")


class TestMemoryManager:
    """MemoryManager 策略层测试"""

    def _make_embedder(self):
        from agent_demo.memory.embedder import Embedder

        class _FakeEmbedder(Embedder):
            dimension = 64

            def embed(self, text: str) -> list[float]:
                import hashlib
                h = hashlib.sha256(text.encode()).digest()
                vec = []
                for i in range(self.dimension):
                    vec.append(float(h[i % len(h)]) / 255.0)
                return vec

        return _FakeEmbedder()

    def _make_manager(self, **kwargs):
        from agent_demo.memory.long_term import LongTermMemory
        from agent_demo.memory.manager import MemoryManager

        ltm = LongTermMemory(embedder=self._make_embedder())
        return MemoryManager(long_term=ltm, **kwargs)

    def test_retrieve_with_session(self):
        """检索包含会话状态的上下文"""
        mgr = self._make_manager(system_prompt="你是助手")
        mgr.set_session("当前任务", "查询天气信息")
        mgr.set_session("中间结果", {"city": "北京"})

        context = mgr.retrieve("北京天气", top_k=3)
        assert "当前任务" in context
        assert "北京" in context
        print(f"  [PASS] MemoryManager retrieve+session: {len(context)} 字符")

    def test_remember_with_auto_extract(self):
        """自动提取偏好和实体"""
        mgr = self._make_manager()

        messages = [
            {"role": "user", "content": "我喜欢用简洁的方式回答问题"},
            {"role": "assistant", "content": "好的，我记住了。"},
        ]
        mgr.remember(messages)

        # 工作记忆应有 2 条消息
        assert len(mgr._working) == 2

        # 长期记忆中应自动提取到偏好
        stats = mgr.get_stats()
        assert stats["long_term"]["total"] >= 1, f"应至少提取 1 条长期记忆，实际: {stats}"
        print(f"  [PASS] MemoryManager auto_extract: {stats}")

    def test_add_get_preference(self):
        """显式管理偏好"""
        mgr = self._make_manager()

        mgr.add_preference("回复风格", "幽默")
        assert mgr.get_preference("回复风格") == "幽默"
        assert "回复风格" in mgr.get_all_preferences()
        print(f"  [PASS] MemoryManager 偏好管理: {mgr.get_all_preferences()}")

    def test_get_working_context(self):
        """获取工作记忆上下文"""
        mgr = self._make_manager(system_prompt="你是测试助手")

        mgr._working.add({"role": "user", "content": "测试消息"})
        ctx = mgr.get_working_context()

        assert ctx[0]["role"] == "system"
        assert "测试助手" in ctx[0]["content"]
        assert ctx[1]["role"] == "user"
        print(f"  [PASS] MemoryManager get_working_context: {len(ctx)} 条")


class TestMemoryModule:
    """MemoryModule 主入口集成测试"""

    def test_init_default(self):
        """默认初始化"""
        from agent_demo.memory import MemoryModule
        from agent_demo.memory.embedder import DashScopeEmbedder

        memory = MemoryModule(
            api_key="sk-test-key",
            max_working_tokens=4000,
            system_prompt="你是助手",
        )
        assert isinstance(memory._embedder, DashScopeEmbedder)
        stats = memory.get_stats()
        assert stats["working_messages"] == 0
        assert stats["long_term"]["total"] == 0
        print(f"  [PASS] MemoryModule 初始化: {memory}")

    def test_full_cycle(self):
        """完整记忆周期: 记忆 → 检索 → 偏好"""
        from agent_demo.memory import MemoryModule, MemoryItem
        from agent_demo.memory.embedder import Embedder

        # 使用假 Embedder 避免真实 API 调用
        class _FakeEmbedder(Embedder):
            dimension = 64
            def embed(self, text):
                import hashlib
                h = hashlib.sha256(text.encode()).digest()
                return [float(h[i % len(h)]) / 255.0 for i in range(self.dimension)]

        memory = MemoryModule(embedder=_FakeEmbedder(), system_prompt="你是助手")

        # 1. 存入偏好
        memory.add_preference("语言", "中文")
        memory.add_preference("回复长度", "简短")

        # 2. 手动存入长期记忆
        memory.remember_item(
            content="用户昨天询问了北京天气，最终给出的是晴天 18-26°C",
            memory_type="conversation",
            importance=0.7,
        )

        # 3. 记忆一轮对话
        memory.remember([
            {"role": "user", "content": "我喜欢简单直接的回复"},
            {"role": "assistant", "content": "好的，我会尽量简洁。"},
        ])

        # 4. 检索上下文
        context = memory.retrieve("用户有什么偏好？", top_k=5)
        assert "语言" in context or "中文" in context
        assert len(context) > 0

        # 5. 获取统计
        stats = memory.get_stats()
        assert stats["long_term"]["total"] >= 2  # 手动 1 条 + 自动提取
        print(f"  [PASS] MemoryModule 完整周期: {stats}")
        print(f"    上下文: {context[:200]}...")

    def test_session_management(self):
        """会话状态管理"""
        from agent_demo.memory import MemoryModule

        memory = MemoryModule(api_key="sk-test-key")

        memory.set_session("step", 1)
        memory.set_session("task", "搜索天气", ttl=60)
        assert memory.get_session("step") == 1
        assert memory.get_session("task") == "搜索天气"

        memory.clear_session()
        assert memory.get_session("step") is None
        assert memory.get_session("task") is None
        print(f"  [PASS] MemoryModule 会话管理")


# ============================================================
# 运行入口
# ============================================================

def run_all():
    """运行所有测试并汇总结果"""
    import sys

    test_classes = [
        TestTypes,
        TestSessionStore,
        TestRelStore,
        TestEmbedder,
        TestVectorStore,
        TestWorkingMemory,
        TestLongTermMemory,
        TestMemoryManager,
        TestMemoryModule,
    ]

    total = 0
    passed = 0
    failed = 0

    print("=" * 60)
    print("记忆模块 (Memory Module) 测试")
    print("=" * 60)

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        instance = cls()
        for name in dir(instance):
            if name.startswith("test_"):
                total += 1
                try:
                    getattr(instance, name)()
                    passed += 1
                except Exception as e:
                    failed += 1
                    import traceback
                    print(f"  [FAIL] {name}: {e}")
                    traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"测试完成: {total} 个用例 | 通过: {passed} | 失败: {failed}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    import sys
    sys.exit(0 if success else 1)
