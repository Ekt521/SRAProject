"""MemoryManager 单元测试：实体提取 / 置信度 / 过期 / 去重 / 会话持久化"""
import os
import unittest
import tempfile
from datetime import datetime, timedelta

# 先导入触发用户词典加载（否则"写操作"会被 jieba 拆散）
from memory.memory_manager import MemoryManager, HIGH_CONF_SCORE


class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = os.path.join(self.tmpdir, "test.db")
        self.m = MemoryManager(self.db)

    def tearDown(self):
        try:
            os.remove(self.db)
        except OSError:
            pass

    def test_extract_entities_no_fragmentation(self):
        """实体不应再出现截断碎片（如"广州今天的天"）"""
        ents = self.m._extract_entities("广州今天的天气怎么样", "广州今天天气为雷阵雨，气温25度左右")
        self.assertIn("广州", ents)
        self.assertIn("天气", ents)
        self.assertTrue(all(len(e) >= 2 for e in ents), f"存在碎片实体: {ents}")
        # 带单位数字应保留
        self.assertTrue(any("25" in e for e in ents), f"缺少带单位数字: {ents}")

    def test_high_conf_recall(self):
        """近义问题召回应达到高置信阈值"""
        self.m.save("redis写操作速度", "Redis写操作可达81000次每秒")
        mems = self.m.recall("redis写操作性能")
        self.assertTrue(mems, "未召回任何记忆")
        self.assertGreaterEqual(mems[0]["score"], HIGH_CONF_SCORE)
        self.assertIn("写操作", mems[0]["entities"])

    def test_expired_flag(self):
        """100 天前的记忆应标记 expired 且 recency 极低"""
        self.m.save("广州天气", "广州天气晴")
        old = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d %H:%M:%S")
        self.m.cursor.execute("UPDATE episodic_memory SET timestamp=? WHERE id=?", (old, self.m.cursor.lastrowid))
        self.m.conn.commit()
        mems = self.m.recall("广州天气")
        self.assertTrue(mems and mems[0]["expired"], "应标记过期")
        self.assertLess(mems[0]["recency"], 0.1)

    def test_save_dedup(self):
        """1 天内相同 query 应更新而非插入"""
        self.m.save("q1", "a1")
        self.m.save("q1", "a2")
        count = self.m.cursor.execute("SELECT COUNT(*) FROM episodic_memory WHERE query='q1'").fetchone()[0]
        self.assertEqual(count, 1, "相同 query 不应重复插入")
        row = self.m.cursor.execute("SELECT answer FROM episodic_memory WHERE query='q1'").fetchone()
        self.assertEqual(row[0], "a2")

    def test_conversation_roundtrip(self):
        """会话历史应能存取且保持顺序"""
        self.m.save_conversation("user", "你好")
        self.m.save_conversation("assistant", "你好呀")
        hist = self.m.load_conversation()
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0]["role"], "user")
        self.assertEqual(hist[1]["content"], "你好呀")


if __name__ == "__main__":
    unittest.main()
