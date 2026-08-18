import json
import re
from typing import Dict, Any, List
from tools.executor import safe_run_python
from tools.knowledge_tool import search_knowledge
from memory.memory_manager import MemoryManager

class SimpleReActAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.messages: List[Dict[str, str]] = []
        self.max_steps = 5  # 防止 Agent 无限思考
        self.memory = MemoryManager()  # 新增
        self.last_retrieved_text = ""   # 已有

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
        memory_context = self.memory.format_memories(memories)
        if memory_context:
            print(f"[Memory] 📖 已召回 {len(memories)} 条相关记忆")

        system_prompt_content = (
            "你是严格遵循 JSON 格式的智能助理。**每次回答必须只输出一个合法的 JSON 对象，不得输出任何其他文字。**\n\n"
            "【工具调用规则】\n"
            "遇到任何知识性问题（包括但不仅限于：技术名词、历史数据、配置参数、数值），你必须**优先调用 search_knowledge 工具**，而不是依赖你自己的内部知识。\n"
            "即使你觉得自己知道答案，也必须先检索本地知识库来验证。\n\n"
            "【输出格式】\n"
            "- 如果需要检索知识库：{\"action\": \"search_knowledge\", \"query\": \"关键词\"}\n"
            "- 如果需要计算：{\"action\": \"run_python\", \"code\": \"Python代码\"}\n"
            "- 如果已经基于工具返回的结果确认答案：{\"action\": \"final\", \"answer\": \"你的答案\"}\n\n"
            "【特别警告】\n"
            "1. 绝对禁止直接输出非 JSON 格式的自然语言回答。\n"
            "2. 绝对禁止在没有调用 search_knowledge 的情况下回答知识性问题。\n"
            "3. 如果知识库没有返回相关信息，必须如实告知用户'未找到相关资料'。"
        )
        
        # ========== 新增：注入记忆上下文 ==========
        if memory_context:
            system_prompt_content += f"\n\n【重要参考】\n{memory_context}\n注意：上述是历史对话的记忆，仅作为参考。如果与当前知识库冲突，以知识库检索结果为准。"
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
                # 1. 检查有没有引用标签
                citations = re.findall(r'\[citation:(\d+)\]', answer)
                if not citations:
                    # 如果是"未找到"类的回答，允许通过
                    if any(kw in answer for kw in ['未找到', '没有找到', '无相关', '无法提供']):
                        print("[Critic] ℹ️ 未找到相关信息，允许通过")
                        self.messages.append({"role": "assistant", "content": answer})
                        return answer
                    # 否则强制重写
                    print("[Critic] ⚠️ 拦截：回答缺少引用，强制重写")
                    self.messages.append({
                        "role": "user",
                        "content": "❌ 你的回答缺少引用。请使用知识库检索结果中的 [citation:编号] 来支撑你的观点。如果知识库没有相关信息，请直接告知用户。"
                    })
                    continue  # 重新走循环，让 LLM 重新生成

                # 2. 简单校验：获取最近一次检索到的原文（存储在 self.last_retrieved 中）
                # 这里简化逻辑，假设你有办法拿到最近的检索文本
                # 检查 answer 中的关键数字/名词是否出现在 citation 对应的原文中
                # （实际生产环境会用 ROUGE-L 或 BERTScore，这里用简单的字符串包含）
                if hasattr(self, 'last_retrieved_text'):
                    # 取 Agent 回答中提到的数字
                    numbers_in_answer = re.findall(r'\d+', answer)
                    # 检查这些数字是否出现在原文中
                    found = False
                    for num in numbers_in_answer:
                        if num in self.last_retrieved_text:
                            found = True
                            break
                    if not found and len(numbers_in_answer) > 0:
                        # 原文没有相关数字，怀疑胡编
                        print(f"[Critic] ⚠️ 拦截：数字不匹配，怀疑胡编")
                        self.messages.append({
                            "role": "user",
                            "content": f"⚠️ 校验失败：你回答中的数字 {numbers_in_answer} 在引用的原文中未找到对应内容。请重新检查并严格基于检索结果回答。"
                        })
                        continue
                # ========== Critic 校验结束 ==========

                # 校验通过，返回答案
                print("[Critic] ✅ 校验通过，返回答案")
                self.memory.save(user_input, answer)  # 新增：保存到长期记忆
                self.messages.append({"role": "assistant", "content": answer})
                return answer

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

            else:
                # 未知动作：列出所有可用动作，引导模型纠正
                self.messages.append({
                    "role": "user",
                    "content": f"错误：未知动作 '{action_type}'。可选动作: final, run_python, search_knowledge。请重新输出有效 JSON。"
                })

        # 如果超出最大步数
        return "任务超出最大尝试次数 (5次)，请简化问题或检查逻辑。"
