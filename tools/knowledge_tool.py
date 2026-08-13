from knowledge.vector_store import HybridVectorStore
from knowledge.chunker import smart_chunk

# 全局单例（只加载一次）
store = None

def load_knowledge(file_path: str):
    """加载并索引文档（启动时调用一次）"""
    global store
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = smart_chunk(text)
    store = HybridVectorStore()
    store.add_chunks(chunks)
    print(f"✅ 知识库加载完成，共 {len(chunks)} 个分块")

def search_knowledge(query: str, top_k: int = 3) -> str:
    """Agent 调用的检索工具"""
    if not store:
        return "知识库未加载，请先调用 load_knowledge()"
    results = store.hybrid_search(query, top_k=top_k)
    if not results:
        return "未找到相关信息。"
    # 格式化为带编号的引用，方便 Agent 引用
    output = []
    for r in results:
        output.append(f"[citation:{r['id']}] {r['content']}")
    return "\n\n".join(output)