import json
import re
import jieba.analyse
from typing import Dict, Any, List
from tools.executor import safe_run_python
from tools.knowledge_tool import search_knowledge
from memory.memory_manager import MemoryManager, HIGH_CONF_SCORE
from tools.web_search import search_web  # 新增

# Critic 事实核对时忽略的引用/链接噪声词
_ENG_NOISE = {'citation', 'source', 'memory', 'http', 'https', 'com', 'org', 'cn', 'www', 'html'}

class SimpleReActAgent:
    def __init__(self, llm_client, memory_db: str = "storage/memory.db"):
        self.llm = llm_client
        self.max_steps = 5  # 防止 Agent 无限思考
        self.memory = MemoryManager(memory_db)
        self.last_retrieved_text = ""   # 已有
        # 跨会话恢复：加载最近的对话历史作为上下文
        self.messages: List[Dict[str, str]] = self.memory.load_conversation(limit=6)
        if self.messages:
            print(f"[Memory] 💬 已恢复 {len(self.messages)} 条历史对话")

    def _finalize(self, user_input: str, answer: str) -> str:
        """统一收尾：保存长期记忆 + 对话历史，追加 assistant 消息并返回"""
        self.memory.save(user_input, answer)
        self.memory.save_conversation("user", user_input)
        self.memory.save_conversation("assistant", answer)
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def _parse_action(self, raw_text: str) -> Dict[str, Any]:
        """鲁棒的 JSON 解析器（增加对 DeepSeek 废话的包容）"""
        # 1. 优先提取 ```json ... ``` 代码块
        match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        # 2. 直接提取包含 "action" 的花括号
        match = re.search(r'\{[^{}]*"action"[^{}]*\}', raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass

        # 3. 强行兜底：让它复述用户问题，避免程序崩溃
        return {"action": "final", "answer": f"系统解析失败。原始返回: {raw_text[:80]}..."}

    def run(self, user_input: str) -> str:
        # 重置步骤计数（每次新问题重新计数）
        step_count = 0
        self.messages.append({"role": "user", "content": user_input})

        # ========== 新增：长期记忆召回 ==========
        memories = self.memory.recall(user_input)
        has_confident_memory = any(m["score"] >= HIGH_CONF_SCORE for m in memories)
        memory_context = self.memory.format_memories(memories)
        if memory_context:
            print(f"[Memory] 📖 已召回 {len(memories)} 条相关记忆"
                  + ("（含高置信，直接回答）" if has_confident_memory else ""))

        system_prompt_content = (
            "你是严格遵循 JSON 格式的智能助理。**每次回答必须只输出一个合法的 JSON 对象，不得输出任何其他文字。**\n\n"
            "【工具调用规则】\n"
            "遇到任何知识性问题（包括但不仅限于：技术名词、历史数据、配置参数、数值），你必须**优先调用 search_knowledge 工具**，而不是依赖你自己的内部知识。\n"
            "即使你觉得自己知道答案，也必须先检索本地知识库来验证。\n"
            "**例外**：如果【重要参考】中已有 ✅ 高置信记忆直接覆盖当前问题，则无需检索，直接基于记忆回答。\n\n"
            "【输出格式】\n"
            "- 如果需要检索知识库：{\"action\": \"search_knowledge\", \"query\": \"关键词\"}\n"
            "- 如果需要联网搜索（知识库无结果时）：{\"action\": \"search_web\", \"query\": \"关键词\"}\n"
            "- 如果需要计算：{\"action\": \"run_python\", \"code\": \"Python代码\"}\n"
            "- 如果已经基于工具返回的结果确认答案：{\"action\": \"final\", \"answer\": \"你的答案\"}\n\n"
            "【特别警告】\n"
            "1. 绝对禁止直接输出非 JSON 格式的自然语言回答。\n"
            "2. 绝对禁止在没有调用 search_knowledge 的情况下回答知识性问题。\n"
            "3. 如果知识库没有返回相关信息，必须如实告知用户'未找到相关资料'。"
        )
        
        # ========== 新增：注入记忆上下文（按置信度分档） ==========
        if memory_context:
            base = f"\n\n【重要参考】\n{memory_context}"
            if has_confident_memory:
                base += (
                    "\n✅ **上述高置信记忆已直接覆盖当前问题。**\n"
                    "你必须直接基于记忆给出 final 回答，**禁止再调用 search_knowledge 或 search_web**。\n"
                    "回答中请保留记忆开头的引用标记 [memory:编号]，例如 [memory:3]。"
                )
            else:
                base += "\n注意：上述记忆仅作为参考。如果与当前知识库冲突，以知识库检索结果为准。"
            system_prompt_content += base
        # ========================================
        
        system_prompt = {"role": "system", "content": system_prompt_content}

        while step_count < self.max_steps:
            step_count += 1
            
            # 组装上下文（只取最近 6 条，控制 token）
            history = self.messages[-6:] if len(self.messages) > 6 else self.messages
            payload = [system_prompt] + history

            # 调用 DeepSeek
            response_text = self.llm.chat(payload)
            print(f"[Debug] 第{step_count}步返回: {response_text[:5000]}...")  # 打印前5000字调试

            # 解析动作
            action = self._parse_action(response_text)
            # 如果解析失败（且不是第一次尝试），强制让模型重新输出 JSON
            if action.get("action") == "final" and "系统解析失败" in action.get("answer", ""):
                self.messages.append({
                    "role": "user",
                    "content": "你的回答必须严格遵循 JSON 格式，请重新输出。例如：{\"action\": \"final\", \"answer\": \"你的答案\"}"
                })
                continue  # 重新走一轮循环
            action_type = action.get("action")

            # ---------- 分发逻辑 ----------
            if action_type == "final":
                answer = action.get("answer", "任务完成")
                # ========== Critic 校验开始 ==========
                # 1. 检查有没有引用标签（支持 citation / source / memory）
                has_citation = bool(re.search(r'\[citation:\d+\]', answer))
                has_source = bool(re.search(r'\[source:https?://[^\]]+\]', answer))
                has_memory = bool(re.search(r'\[memory:\d+\]', answer))

                if not (has_citation or has_source or has_memory):
                    # 如果回答包含“未找到”类关键词，允许通过
                    if any(kw in answer for kw in ['未找到', '没有找到', '无相关']):
                        print("[Critic] ℹ️ 未找到相关信息，允许通过")
                        return self._finalize(user_input, answer)
                    # 否则拦截重写
                    print("[Critic] ⚠️ 拦截：缺少引用，强制重写")
                    self.messages.append({
                        "role": "user",
                        "content": "❌ 你的回答缺少有效引用。请使用 [citation:编号]（本地知识库）、[memory:编号]（历史记忆）或 [source:URL]（联网搜索）来支撑观点。"
                    })
                    continue  # 重新走循环，让 LLM 重新生成

                # 2. 事实核对：数字匹配 + 关键词重叠
                # 记忆回答（[memory:N]）是历史事实，无本次检索原文可比对，跳过核对
                if hasattr(self, 'last_retrieved_text') and not has_memory:
                    retrieved = self.last_retrieved_text

                    # 2a. 数字核对：回答中的数字必须能在原文中找到
                    numbers_in_answer = re.findall(r'\d+', answer)
                    if numbers_in_answer and not any(num in retrieved for num in numbers_in_answer):
                        print("[Critic] ⚠️ 拦截：数字不匹配，怀疑胡编")
                        self.messages.append({
                            "role": "user",
                            "content": f"⚠️ 校验失败：你回答中的数字 {numbers_in_answer} 在引用的原文中未找到对应内容。请重新检查并严格基于检索结果回答。"
                        })
                        continue

                    # 2b. 关键词重叠：回答应共享检索原文的关键名词，防止跑题胡编
                    answer_kw = {t for t in jieba.analyse.extract_tags(answer, topK=8)
                                 if t.lower() not in _ENG_NOISE}
                    retrieved_kw = set(jieba.analyse.extract_tags(retrieved, topK=15))
                    if len(answer_kw) >= 3 and not (answer_kw & retrieved_kw):
                        print("[Critic] ⚠️ 拦截：回答与检索原文主题不符")
                        self.messages.append({
                            "role": "user",
                            "content": "⚠️ 校验失败：你的回答与检索原文主题关联过弱（关键词零重叠）。请严格基于检索结果回答，不要编造或跑题。"
                        })
                        continue
                # ========== Critic 校验结束 ==========

                # 校验通过，返回答案
                print("[Critic] ✅ 校验通过，返回答案")
                return self._finalize(user_input, answer)

            elif action_type == "run_python":
                code = action.get("code", "")
                if not code:
                    observation = "错误：代码为空，请提供有效代码。"
                else:
                    observation = safe_run_python(code)
                
                self.messages.append({
                    "role": "user", 
                    "content": f"代码执行结果 (Observation): {observation}"
                })

            # ---------- search_knowledge 工具 ----------
            elif action_type == "search_knowledge":
                query = action.get("query", "")
                if not query:
                    observation = "错误：查询参数为空，请提供有效的查询关键词。"
                else:
                    print(f"[Tool] 🔍 正在检索: {query}")
                    observation = search_knowledge(query)
                    print(f"[Tool] 📄 检索结果: {observation[:150]}...")  # 打印前150字
                    # 保存原文用于 Critic 校验
                    self.last_retrieved_text = observation
                
                self.messages.append({
                    "role": "user",
                    "content": f"知识库检索结果 (Observation): {observation}"
                })

            # ---------- search_web 工具 ----------
            elif action_type == "search_web":
                query = action.get("query", "")
                if not query:
                    observation = "错误：查询参数为空，请提供有效的搜索关键词。"
                else:
                    print(f"[Tool] 🌐 正在联网搜索: {query}")
                    observation = search_web(query)
                    print(f"[Tool] 📄 搜索结果: {observation[:150]}...")
                    # 保存检索原文（供 Critic 校验引用）
                    self.last_retrieved_text = observation
                
                self.messages.append({
                    "role": "user",
                    "content": f"联网搜索结果 (Observation): {observation}"
                })

            else:
                # 未知动作：列出所有可用动作，引导模型纠正
                self.messages.append({
                    "role": "user",
                    "content": f"错误：未知动作 '{action_type}'。可选动作: final, run_python, search_knowledge, search_web。请重新输出有效 JSON。"
                })

        # 如果超出最大步数：基于已有工具结果强制总结，而不是直接失败
        print("[Agent] ⚠️ 超出最大步数，尝试基于已有结果强制总结...")
        try:
            force_messages = self.messages[-6:]  # 切片副本，不污染 self.messages
            force_messages.append({
                "role": "user",
                "content": "你已尝试多步但未得出结论。请**基于以上已获取的工具结果**直接给出 final 答案（JSON 格式）。若信息不足，明确说明哪些信息无法获取。"
            })
            force_text = self.llm.chat([system_prompt] + force_messages)
            action = self._parse_action(force_text)
            if action.get("action") == "final":
                answer = action.get("answer", "任务完成")
                print("[Agent] ✅ 强制总结成功")
                return self._finalize(user_input, answer)
            print("[Agent] ⚠️ 强制总结未返回 final，放弃")
        except Exception as e:
            print(f"[Agent] ❌ 强制总结失败: {e}")
        return "任务超出最大尝试次数 (5次)，已尝试总结但仍未得到有效答案。请简化问题或检查逻辑。"
