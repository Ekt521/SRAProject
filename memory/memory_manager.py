# memory/memory_manager.py
import os
import re
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import jieba
import jieba.posseg as pseg
import jieba.analyse

# ================= 实体提取（jieba 版）=================
# 领域词典：这些词会被 jieba 整体切出，不再被拆散（可编辑 memory/userdict.txt）
_USERDICT_PATH = Path(__file__).resolve().parent / 'userdict.txt'
_DEFAULT_USERDICT = """\
读操作 10 n
写操作 10 n
持久化 10 vn
DeepSeek 10 nz
Redis 10 nz
知识库 10 n
"""

def _load_userdict():
    """加载领域词典，格式：词语 词频 词性（词性可省略）"""
    if not _USERDICT_PATH.exists():
        _USERDICT_PATH.write_text(_DEFAULT_USERDICT, encoding='utf-8')
    jieba.load_userdict(str(_USERDICT_PATH))

_load_userdict()

# 停用词：过滤高频无意义的"名词"（多数来自答案的叙述文本）
_STOPWORDS = {
    '结果', '资料', '测试', '显示', '根据', '官方', '基准', '搜索',
    '内容', '信息', '问题', '回答', '使用', '数据', '方法', '情况',
    '时候', '地方', '方面', '原因', '影响', '性能', '作用',
}

# 数字白名单：数字必须带单位才保留（避免 59093361 这种随机数）
_NUM_UNIT_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*'
    r'(度|℃|摄氏度|次|秒|分钟|小时|天|日|周|月|年|元|块|个|%|GB|MB|KB|万|亿|公里|千米|米|克|千克|公斤)'
)


def _is_stop(word: str) -> bool:
    return word in _STOPWORDS


# ===== 记忆置信度 & 过期参数 =====
HALF_LIFE_DAYS = 7       # 置信度每 7 天减半（软衰减）
EXPIRED_DAYS = 90        # 超过 90 天标注"可能已过时"
HIGH_CONF_SCORE = 0.8    # ≥ 此分 → 高置信，可直接基于记忆回答
LOW_CONF_SCORE = 0.5     # ≥ 此分 → 相关记忆，参考为主


def _keep_pos(word: str, flag: str) -> bool:
    """判断 jieba 词性是否值得作为实体保留"""
    if len(word) < 2:
        return False
    if flag.startswith('n') or flag == 'vn':  # 名词/人名/地名/机构/专名/名动词
        return not _is_stop(word)
    if flag == 'eng':                         # 英文（如 Redis、API）
        return True
    if flag == 'x':                           # 未归类词：只留纯字母数字（产品名）
        return bool(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]+', word))
    return False


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

    def _extract_entities(self, query: str, answer: str = "") -> List[str]:
        """
        用 jieba 词性过滤提取实体，解决正则切词的碎片化问题。

        - query：全量提取（名词/地名/机构/专名/英文）
        - answer：只用 TF-IDF 提 top 关键词，避免把叙述片段当实体
        - 数字：仅保留带单位的（如 25度、81000次）
        """
        entities = []

        # 1. query 全量提取
        for word, flag in pseg.cut(query):
            word = word.strip()
            if _keep_pos(word, flag):
                entities.append(word)

        # 2. answer 只提 TF-IDF 关键词（过滤数字和无意义词）
        if answer:
            for tag in jieba.analyse.extract_tags(answer, topK=3):
                tag = tag.strip()
                if tag.isdigit() or _is_stop(tag) or len(tag) < 2:
                    continue
                entities.append(tag)

        # 3. 数字白名单（query + answer 里带单位的数字）
        for num, unit in _NUM_UNIT_RE.findall(query + " " + answer):
            entities.append(f"{num}{unit}")

        # 在去重之前，过滤掉英文停用词（含 [memory:N] 引用标签）
        english_stopwords = {'citation', 'source', 'http', 'https', 'com', 'org', 'cn', 'www', 'html', 'memory'}

        # 4. 去重 + 限制数量
        seen = set()
        result = []
        for e in entities:
            if e not in seen and e.lower() not in english_stopwords:
                seen.add(e)
                result.append(e)

        return result[:10]

    def save(self, query: str, answer: str) -> None:
        """存储一条记忆，若最近1小时内存在相同 query 则更新，否则插入"""
        entities = self._extract_entities(query, answer)
        entities_json = json.dumps(entities, ensure_ascii=False)

        # 1. 检查最近1天内是否有完全相同的 query
        self.cursor.execute(
            "SELECT id FROM episodic_memory WHERE query = ? AND timestamp >= datetime('now', '-1 day') ORDER BY timestamp DESC LIMIT 1",
            (query,)
        )
        row = self.cursor.fetchone()

        if row:
            # 2. 存在 → 更新该记录（保留最新答案和实体）
            self.cursor.execute(
                "UPDATE episodic_memory SET answer = ?, entities = ?, timestamp = CURRENT_TIMESTAMP WHERE id = ?",
                (answer, entities_json, row[0])
            )
            self.conn.commit()
            print(f"[Memory] 🔄 已更新记忆: {query[:30]}... (实体: {entities})")
        else:
            # 3. 不存在 → 插入新记录
            self.cursor.execute(
                "INSERT INTO episodic_memory (query, answer, entities) VALUES (?, ?, ?)",
                (query, answer, entities_json)
            )
            self.conn.commit()
            print(f"[Memory] 💾 已保存记忆: {query[:30]}... (实体: {entities})")

    def recall(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        召回相关记忆，并计算置信度得分（实体重叠 × 时间衰减）。
        返回带 score / entity_ratio / recency / age_days / expired 的记忆列表。
        """
        query_entities = self._extract_entities(query)
        if not query_entities:
            return []

        # 1. 用 query 实体逐个匹配，按记忆 id 聚合命中的实体集合
        memory_hits: Dict[int, Dict] = {}
        for entity in query_entities[:5]:
            self.cursor.execute(
                "SELECT id, query, answer, timestamp, entities FROM episodic_memory WHERE entities LIKE ?",
                (f'%"{entity}"%',)
            )
            for row in self.cursor.fetchall():
                mem_id = row[0]
                if mem_id not in memory_hits:
                    memory_hits[mem_id] = {"row": row, "hit_entities": set()}
                memory_hits[mem_id]["hit_entities"].add(entity)

        # 2. 计算每条记忆的得分
        results = []
        for mem_id, info in memory_hits.items():
            row = info["row"]
            entity_ratio = len(info["hit_entities"]) / len(query_entities)
            age_days = self._age_days(row[3])
            recency = 0.5 ** (age_days / HALF_LIFE_DAYS)
            results.append({
                "id": row[0],
                "query": row[1],
                "answer": row[2],
                "timestamp": row[3],
                "entities": json.loads(row[4]) if row[4] else [],
                "score": round(entity_ratio * recency, 3),
                "entity_ratio": round(entity_ratio, 3),
                "recency": round(recency, 3),
                "age_days": age_days,
                "expired": age_days > EXPIRED_DAYS,
            })
        results.sort(key=lambda m: m["score"], reverse=True)
        return results[:limit]

    def _age_days(self, timestamp: str) -> int:
        """把 SQLite 时间戳换算成距离今天的天数，解析失败按 0 天处理"""
        try:
            ts = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            return (datetime.now() - ts).days
        except (ValueError, TypeError):
            return 0

    def format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """格式化记忆供 Agent 使用（按置信度分级标注 + [memory:编号] 引用标记）"""
        if not memories:
            return ""

        lines = ["【相关历史记忆】（每条开头有引用标记，回答时请使用 [memory:编号] 引用）"]
        for mem in memories:
            if mem["score"] >= HIGH_CONF_SCORE:
                tag = "✅ 高置信"
            elif mem["score"] >= LOW_CONF_SCORE:
                tag = "ℹ️ 相关"
            else:
                tag = "🗂 弱相关"
            lines.append(f"- {tag} [memory:{mem['id']}] 曾问: {mem['query']} (置信度 {mem['score']:.2f})")
            ans = mem["answer"]
            if len(ans) > 300:
                ans = ans[:300] + "..."
            lines.append(f"  上次回答: {ans}")
            if mem.get("expired"):
                lines.append("  ⚠️ 已超过90天，可能已过时，请谨慎采用。")
            lines.append("")
        return "\n".join(lines)
