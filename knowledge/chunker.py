# knowledge/chunker.py
import re
from typing import List, Dict

def smart_chunk(text: str, source: str = "unknown", max_chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, str]]:
    """
    语义分块：优先按 Markdown 标题/段落切，保留重叠。
    返回: [{"id": 0, "content": "...", "source": "data/sample.md"}, ...]
    source 记录文档来源，供检索结果溯源。
    """
    # 1. 按一级/二级标题切分（## 或 #）
    sections = re.split(r'(?=^#{1,2}\s+)', text, flags=re.MULTILINE)

    chunks = []
    current_chunk = ""
    chunk_id = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # 如果当前块加上新段落超过限制，先保存当前块
        if len(current_chunk) + len(section) > max_chunk_size and current_chunk:
            chunks.append({"id": chunk_id, "content": current_chunk.strip(), "source": source})
            chunk_id += 1
            # 保留重叠：最后一句；无句号或句末过长时截断到 overlap 字符，避免块膨胀
            sentences = [s for s in current_chunk.split('。') if s]
            overlap_text = sentences[-1] if sentences else ""
            if len(overlap_text) > overlap:
                overlap_text = overlap_text[-overlap:]
            current_chunk = overlap_text + "\n" if overlap_text else ""

        current_chunk += section + "\n"

    # 处理剩余内容
    if current_chunk:
        chunks.append({"id": chunk_id, "content": current_chunk.strip(), "source": source})

    return chunks