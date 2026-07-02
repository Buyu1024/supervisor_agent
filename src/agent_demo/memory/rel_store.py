"""RelStore —— 基于 SQLite 的结构化记忆存储

职责:
    1. 用户偏好 (user_preferences): key-value 精确存取
    2. 实体管理 (entities): 结构化实体事实
    3. 实体关系 (entity_relations): (source, relation, target) 三元组

基于 sqlite3 标准库，零外部依赖。
"""

import json
import sqlite3
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class RelStore:
    """
    SQLite 结构化存储 —— 精确查询用户偏好、实体、关系

    使用示例:
        store = RelStore("data/memory.db")
        store.set_preference("language", "中文")
        store.upsert_entity("张三", "person", {"age": 25, "city": "北京"})
        store.add_relation("张三", "likes", "Python")
    """

    def __init__(self, db_path: str = ":memory:"):
        """
        Args:
            db_path: SQLite 数据库文件路径，默认 ":memory:" 为内存模式
        """
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"RelStore 初始化完成: {db_path}")

    def _create_tables(self) -> None:
        """创建表结构"""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entities (
                name        TEXT PRIMARY KEY,
                type        TEXT NOT NULL,
                properties  TEXT NOT NULL DEFAULT '{}',
                updated_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entity_relations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT NOT NULL,
                relation    TEXT NOT NULL,
                target      TEXT NOT NULL,
                metadata    TEXT NOT NULL DEFAULT '{}',
                created_at  REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_rel_source
                ON entity_relations(source, relation);
            CREATE INDEX IF NOT EXISTS idx_rel_target
                ON entity_relations(target, relation);
        """)
        self._conn.commit()

    # ---- 用户偏好 ----

    def set_preference(self, key: str, value: str) -> None:
        """设置用户偏好（覆盖写入）"""
        self._conn.execute(
            "INSERT OR REPLACE INTO user_preferences (key, value, updated_at) "
            "VALUES (?, ?, ?)",
            (key, value, time.time()),
        )
        self._conn.commit()
        logger.debug(f"偏好已设置: {key} = {value}")

    def get_preference(self, key: str) -> str | None:
        """获取单个偏好"""
        row = self._conn.execute(
            "SELECT value FROM user_preferences WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def get_all_preferences(self) -> dict[str, str]:
        """获取所有偏好，返回 {key: value}"""
        rows = self._conn.execute(
            "SELECT key, value FROM user_preferences ORDER BY key"
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def delete_preference(self, key: str) -> bool:
        """删除偏好，返回是否成功删除"""
        cursor = self._conn.execute(
            "DELETE FROM user_preferences WHERE key = ?", (key,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ---- 实体管理 ----

    def upsert_entity(self, name: str, entity_type: str, properties: dict | None = None) -> None:
        """创建或更新实体"""
        props_json = json.dumps(properties or {}, ensure_ascii=False)
        self._conn.execute(
            "INSERT OR REPLACE INTO entities (name, type, properties, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (name, entity_type, props_json, time.time()),
        )
        self._conn.commit()
        logger.debug(f"实体已更新: {name} (type={entity_type})")

    def get_entity(self, name: str) -> dict | None:
        """按名称获取实体"""
        row = self._conn.execute(
            "SELECT * FROM entities WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return {
            "name": row["name"],
            "type": row["type"],
            "properties": json.loads(row["properties"]),
            "updated_at": row["updated_at"],
        }

    def delete_entity(self, name: str) -> bool:
        """删除实体及其所有关系"""
        self._conn.execute("DELETE FROM entities WHERE name = ?", (name,))
        self._conn.execute(
            "DELETE FROM entity_relations WHERE source = ? OR target = ?",
            (name, name),
        )
        self._conn.commit()
        logger.debug(f"实体已删除: {name}")
        return True

    def search_entities(self, keyword: str) -> list[dict]:
        """
        关键词搜索实体 —— 按名称模糊匹配

        Args:
            keyword: 搜索关键词

        Returns:
            匹配的实体列表
        """
        rows = self._conn.execute(
            "SELECT * FROM entities WHERE name LIKE ? OR type LIKE ?",
            (f"%{keyword}%", f"%{keyword}%"),
        ).fetchall()
        return [
            {
                "name": row["name"],
                "type": row["type"],
                "properties": json.loads(row["properties"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def list_entities(self, entity_type: str | None = None) -> list[dict]:
        """列出所有实体，可按类型过滤"""
        if entity_type:
            rows = self._conn.execute(
                "SELECT * FROM entities WHERE type = ? ORDER BY name",
                (entity_type,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM entities ORDER BY name"
            ).fetchall()
        return [
            {
                "name": row["name"],
                "type": row["type"],
                "properties": json.loads(row["properties"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    # ---- 实体关系 ----

    def add_relation(
        self,
        source: str,
        relation: str,
        target: str,
        metadata: dict | None = None,
    ) -> int:
        """
        添加实体关系三元组

        Args:
            source: 源实体名称
            relation: 关系类型（如 "likes", "works_at", "friend_of"）
            target: 目标实体名称
            metadata: 附加元数据

        Returns:
            新关系的自增 ID
        """
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        cursor = self._conn.execute(
            "INSERT INTO entity_relations (source, relation, target, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (source, relation, target, meta_json, time.time()),
        )
        self._conn.commit()
        logger.debug(f"关系已添加: {source} --[{relation}]--> {target}")
        return cursor.lastrowid

    def query_relations(
        self,
        source: str | None = None,
        relation: str | None = None,
        target: str | None = None,
    ) -> list[dict]:
        """
        查询关系 —— 按 source/relation/target 过滤，None 表示不做限制

        Returns:
            关系列表，每条为 {id, source, relation, target, metadata, created_at}
        """
        conditions = []
        params = []

        if source is not None:
            conditions.append("source = ?")
            params.append(source)
        if relation is not None:
            conditions.append("relation = ?")
            params.append(relation)
        if target is not None:
            conditions.append("target = ?")
            params.append(target)

        query = "SELECT * FROM entity_relations"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"

        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "id": row["id"],
                "source": row["source"],
                "relation": row["relation"],
                "target": row["target"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def delete_relation(self, relation_id: int) -> bool:
        """删除关系"""
        cursor = self._conn.execute(
            "DELETE FROM entity_relations WHERE id = ?", (relation_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ---- 生命周期 ----

    def close(self) -> None:
        """关闭数据库连接"""
        self._conn.close()
        logger.info(f"RelStore 已关闭: {self._db_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
