import numpy as np
import jieba
import re
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 项目根目录（本文件位于 knowledge/ 下，父目录的父目录即项目根）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class HybridVectorStore:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        # 优先使用本地已下载的模型，避免每次联网拉取
        local_model = PROJECT_ROOT / "models" / "bge-small-zh-v1.5"
        if local_model.exists():
            model_name = str(local_model)
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Dict[str, str]] = []  # [{"id": 0, "content": "..."}]
        self.embeddings: np.ndarray = None       # (N, dim)
        self.bm25_index: Dict[str, List[int]] = {}  # 倒排索引

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中文分词：jieba 切词 + 过滤空白/纯标点"""
        return [w for w in jieba.cut(text)
                if w.strip() and re.match(r'^[\w一-龥]+$', w)]

    def add_chunks(self, chunks: List[Dict[str, str]]):
        """添加分块并构建索引"""
        self.chunks = chunks
        texts = [c["content"] for c in chunks]
        
        # 1. 计算向量
        print(f"🔢 正在向量化 {len(texts)} 个分块...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # 2. 构建 BM25 倒排索引（纯手写，不依赖库）
        self.bm25_index.clear()
        for idx, text in enumerate(texts):
            words = self._tokenize(text)
            for word in words:
                if word not in self.bm25_index:
                    self.bm25_index[word] = []
                self.bm25_index[word].append(idx)

    def _bm25_score(self, query: str, doc_idx: int) -> float:
        """计算单个文档的 BM25 相关性（简化版，不含 IDF 调参）"""
        words = self._tokenize(query)
        score = 0.0
        doc_text = self.chunks[doc_idx]["content"]
        for word in words:
            if word in self.bm25_index and doc_idx in self.bm25_index[word]:
                # 简单统计词频
                freq = doc_text.count(word)
                score += freq / (freq + 1)  # 简化 BM25 公式
        return score

    def hybrid_search(self, query: str, top_k: int = 3, threshold: float = 0.5) -> List[Dict]:
        """混合检索：向量 + BM25 + RRF 融合"""
        if not self.chunks:
            return []

        # 1. 向量检索（语义）
        q_emb = self.model.encode([query])
        vec_scores = cosine_similarity(q_emb, self.embeddings)[0]  # 形状 (N,)
        
        # 2. BM25 检索（关键词）
        bm25_scores = [self._bm25_score(query, i) for i in range(len(self.chunks))]
        
        # 3. RRF 融合（倒数排名融合）
        # 获取向量检索的排名
        vec_ranked = np.argsort(vec_scores)[::-1]
        bm25_ranked = np.argsort(bm25_scores)[::-1]
        
        combined_scores = {}
        k = 60  # RRF 常数
        for rank, idx in enumerate(vec_ranked):
            if vec_scores[idx] < threshold:  # 向量分数过低直接截断
                break
            combined_scores[idx] = combined_scores.get(idx, 0) + 1 / (k + rank + 1)
        
        for rank, idx in enumerate(bm25_ranked):
            if bm25_scores[idx] <= 0:  # BM25 无匹配则跳过
                continue
            combined_scores[idx] = combined_scores.get(idx, 0) + 1 / (k + rank + 1)
        
        # 排序取 top_k
        sorted_idx = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]
        
        results = []
        for idx in sorted_idx:
            chunk = self.chunks[idx]
            results.append({
                "id": chunk["id"],
                "content": chunk["content"],
                "source": chunk.get("source", "未知来源"),
                "score": combined_scores[idx],
                "vec_score": float(vec_scores[idx]),
                "bm25_score": float(bm25_scores[idx])
            })
        return results