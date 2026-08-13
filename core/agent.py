import json
import re
from typing import Dict, Any, List
from tools.executor import safe_run_python
from tools.knowledge_tool import search_knowledge  # 新增导入

class SimpleReActAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.messages: List[Dict[str, str]] = []
        self.max_steps = 5  # 防止 Agent 无限思考

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

        # 构建系统提示词（升级版：包含知识库工具）
        system_prompt = {
            "role": "system",
            "content": (
                "你是严格遵循 JSON 格式的智能助理。\n"
                "你拥有以下工具，遇到知识性问题**必须优先调用工具**：\n"
                "1. search_knowledge: 查询本地知识库。如果用户问及你未掌握或需要依据的问题，必须调用此工具。\n"
                "   调用格式: {\"action\": \"search_knowledge\", \"query\": \"用户问题的关键词\"}\n"
                "2. run_python: 执行计算代码。\n"
                "如果你已经基于工具返回的结果确认答案，输出: {\"action\": \"final\", \"answer\": \"你的答案\"}\n"
                "注意：如果知识库没有返回相关信息，你必须如实告知用户'未找到相关资料'，严禁捏造！"
            )
        }

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
            action_type = action.get("action")

            # ---------- 分发逻辑 ----------
            if action_type == "final":
                answer = action.get("answer", "任务完成")
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

            elif action_type == "search_knowledge":
                query = action.get("query", "")
                if not query:
                    observation = "错误：查询参数为空，请提供有效的查询关键词。"
                else:
                    observation = search_knowledge(query)
                
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
