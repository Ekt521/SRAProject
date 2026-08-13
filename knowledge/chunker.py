# knowledge/chunker.py
import re
from typing import List, Dict

def smart_chunk(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, str]]:
    """
    语义分块：优先按 Markdown 标题/段落切，保留重叠。
    返回: [{"id": 0, "content": "..."}, ...]
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
            chunks.append({"id": chunk_id, "content": current_chunk.strip()})
            chunk_id += 1
            # 保留重叠（最后一段或最后一句）
            sentences = current_chunk.split('。')
            overlap_text = sentences[-1] if sentences else ""
            current_chunk = overlap_text + "\n" if overlap_text else ""
        
        current_chunk += section + "\n"
    
    # 处理剩余内容
    if current_chunk:
        chunks.append({"id": chunk_id, "content": current_chunk.strip()})
    
    return chunks