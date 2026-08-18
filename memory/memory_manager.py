# memory/memory_manager.py
import sqlite3
import json
import re
import os
from datetime import datetime
from typing import List, Dict, Any

class MemoryManager:
    def __init__(self, db_path: str = "storage/memory.db"):
        # 确保 storage 目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_table()
        
    def _init_table(self):
        """创建记忆表"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                entities TEXT,  -- JSON 数组
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        print("[Memory] ✅ 数据库初始化完成 (storage/memory.db)")

    def _extract_entities(self, text: str) -> List[str]:
        """提取关键实体：技术名词、核心概念、数字"""
        entities = []
        
        # 1. 提取英文技术名词（大写字母开头或全大写）
        eng_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]+\b', text)
        entities.extend(eng_entities)
        
        # 2. 提取数字（包含单独的数字）
        num_entities = re.findall(r'\b\d+\b', text)
        entities.extend(num_entities)
        
        # 3. 提取中文技术名词（2-6个字符，且不以常见动词/助词开头）
        # 关键改进：不截断，完整提取
        chn_matches = re.findall(r'[\u4e00-\u9fa5]{2,6}', text)
        
        # 停用词表（过滤掉无意义的词）
        stopwords = {
            '的', '是', '在', '和', '了', '有', '我', '你', '他', '这', '那', 
            '什么', '怎么', '为什么', '多少', '一个', '可以', '非常', '不过', 
            '实际', '性能', '影响', '官方', '基准', '测试', '显示', '因素',
            '硬件', '网络', '数据', '大小', '等等', '如果', '就是', '例如',
            '还是', '这样', '那样', '因为', '所以', '但是', '而且', '或者'
        }
        for word in chn_matches:
            if word not in stopwords:
                entities.append(word)
        
        # 4. 手动补充：如果文本包含"读操作"，强制加入
        if '读' in text and '操作' in text:
            entities.append('读操作')
        if '写' in text and '操作' in text:
            entities.append('写操作')
        if '持久化' in text:
            entities.append('持久化')
        
        # 5. 去重并限制数量
        seen = set()
        result = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                result.append(e)
        
        return result[:10]

    def save(self, query: str, answer: str) -> None:
        """存储一条记忆"""
        entities = self._extract_entities(query + " " + answer)
        entities_json = json.dumps(entities, ensure_ascii=False)
        
        self.cursor.execute(
            "INSERT INTO episodic_memory (query, answer, entities) VALUES (?, ?, ?)",
            (query, answer, entities_json)
        )
        self.conn.commit()
        print(f"[Memory] 💾 已保存记忆: {query[:30]}... (实体: {entities})")

    def recall(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """根据当前问题召回相关历史记忆"""
        entities = self._extract_entities(query)
        if not entities:
            return []
        
        results = []
        seen_ids = set()
        
        # 逐个实体模糊查询
        for entity in entities[:5]:
            self.cursor.execute(
                "SELECT id, query, answer, timestamp FROM episodic_memory WHERE entities LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f'%"{entity}"%', limit)
            )
            rows = self.cursor.fetchall()
            for row in rows:
                if row[0] not in seen_ids:
                    seen_ids.add(row[0])
                    results.append({
                        "id": row[0],
                        "query": row[1],
                        "answer": row[2],
                        "timestamp": row[3]
                    })
        
        return results[:limit]

    def format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """格式化记忆供 Agent 使用"""
        if not memories:
            return ""
        
        lines = ["【相关历史记忆】"]
        for mem in memories:
            lines.append(f"- 曾问: {mem['query']}")
            lines.append(f"  上次回答: {mem['answer'][:100]}...")
            lines.append("")
        return "\n".join(lines)