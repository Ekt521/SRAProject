from pathlib import Path
from knowledge.vector_store import HybridVectorStore
from knowledge.chunker import smart_chunk

# 全局单例（只加载一次）
store = None

# 支持的文档扩展名
_SUPPORTED_EXTS = {".md", ".txt", ".markdown"}

def _collect_files(path: Path):
    """收集待加载的文件列表（目录则递归扫描支持的扩展名）"""
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*")
                  if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS)

def load_knowledge(path: str):
    """加载并索引文档（启动时调用一次）。

    支持两种传参：
      - 文件: load_knowledge("data/sample.md")
      - 目录: load_knowledge("data")  → 递归加载 .md/.txt/.markdown
    直接往 data/ 里丢新文档，重启即可自动纳入知识库。
    """
    global store
    p = Path(path)
    files = _collect_files(p)
    if not files:
        raise FileNotFoundError(f"未找到可加载的文档: {path}（支持 {sorted(_SUPPORTED_EXTS)}）")

    all_chunks = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        all_chunks.extend(smart_chunk(text, source=str(f)))  # source 保留相对路径，供引用溯源

    # 每个文件的 chunk id 从 0 开始，合并后需全局重新编号
    for i, c in enumerate(all_chunks):
        c["id"] = i

    store = HybridVectorStore()
    store.add_chunks(all_chunks)
    print(f"✅ 知识库加载完成：{len(files)} 个文件，共 {len(all_chunks)} 个分块")

def search_knowledge(query: str, top_k: int = 3) -> str:
    """Agent 调用的检索工具"""
    if not store:
        return "知识库未加载，请先调用 load_knowledge()"
    results = store.hybrid_search(query, top_k=top_k)
    if not results:
        return "未找到相关信息。"
    # 格式化为带编号的引用（附来源，方便溯源），Agent 用 [citation:编号] 引用
    output = []
    for r in results:
        source = r.get('source', '未知来源')
        output.append(f"[citation:{r['id']}] (来源: {source}) {r['content']}")
    return "\n\n".join(output)