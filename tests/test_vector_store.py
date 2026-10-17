"""vector_store 分词测试：jieba 中文分词（无需加载模型，只测静态方法）"""
import unittest

# 先导入 memory_manager 触发用户词典加载（"写操作/Redis"整体切出）
import memory.memory_manager  # noqa: F401
from knowledge.vector_store import HybridVectorStore


class TestVectorStoreTokenize(unittest.TestCase):
    def test_tokenize_keeps_domain_words(self):
        """领域词应整体切出，不被拆散"""
        tokens = HybridVectorStore._tokenize("Redis写操作速度")
        self.assertIn("写操作", tokens)
        self.assertIn("Redis", tokens)

    def test_tokenize_filters_punctuation(self):
        """标点/空白应被过滤"""
        tokens = HybridVectorStore._tokenize("你好，世界！测试123")
        self.assertNotIn("，", tokens)
        self.assertNotIn("！", tokens)
        self.assertNotIn(" ", tokens)

    def test_tokenize_no_empty_tokens(self):
        tokens = HybridVectorStore._tokenize("   ")
        self.assertEqual(tokens, [])


if __name__ == "__main__":
    unittest.main()
