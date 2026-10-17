"""chunker 单元测试：来源字段 / 分块边界 / 小文本"""
import unittest
from knowledge.chunker import smart_chunk


class TestChunker(unittest.TestCase):
    def test_source_field(self):
        """分块应携带来源元数据"""
        chunks = smart_chunk("# 标题\n正文内容", source="data/sample.md")
        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["source"], "data/sample.md")

    def test_chunk_boundary(self):
        """多个中等章节应切成多块，且每块不超过上限（容差内）"""
        text = "\n".join(f"## 章节{i}\n" + "这是内容" * 40 for i in range(3))
        chunks = smart_chunk(text, source="x.md", max_chunk_size=300)
        self.assertGreater(len(chunks), 1, "长文本应切成多块")
        for c in chunks:
            self.assertLessEqual(len(c["content"]), 320, "分块超过大小上限")

    def test_small_text_single_chunk(self):
        """短文本应保持单块"""
        chunks = smart_chunk("短文本", source="a.md")
        self.assertEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
