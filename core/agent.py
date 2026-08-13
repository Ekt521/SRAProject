import json
import re
from typing import Dict, Any, List

class SimpleReActAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.messages: List[Dict[str, str]] = []  # 短期对话记忆

    def _parse_action(self, raw_text: str) -> Dict[str, Any]:
        """极其鲁棒的解析器：专门对付 DeepSeek 可能多说的废话"""
        # 1. 优先提取 ```json ... ``` 代码块
        match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass
        
        # 2. 直接提取最外层的花括号（包含 action 字段）
        match = re.search(r'\{[^{}]*"action"[^{}]*\}', raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass

        # 3. 实在解析不了，强制终结循环，避免死循环
        return {"action": "final", "answer": f"系统解析 LLM 响应失败。原始返回: {raw_text[:50]}..."}

    def run(self, user_input: str) -> str:
        # 将用户问题追加到短期记忆
        self.messages.append({"role": "user", "content": user_input})

        # 构建严格限制输出的 System Prompt（这是 Phase 0 的关键）
        system_prompt = {
            "role": "system", 
            "content": (
                "你是一个智能研究助理。你必须严格按 JSON 格式输出，不要添加任何 Markdown 或解释文字。\n"
                "如果你觉得已经可以回答用户了，输出: {\"action\": \"final\", \"answer\": \"你的回答\"}\n"
                "如果你需要更多信息（当前阶段尚未实现工具，但格式先保留），输出: {\"action\": \"search\", \"query\": \"关键词\"}\n"
                "注意：不要输出多个 JSON，只输出一个！"
            )
        }

        # 组装请求（只取最近 5 条消息，控制 Token 消耗）
        history = self.messages[-5:] if len(self.messages) > 5 else self.messages
        payload = [system_prompt] + history

        # 调用 DeepSeek
        response_text = self.llm.chat(payload)
        print(f"[Debug] DeepSeek 原始返回: {response_text}")  # 调试时可查看

        # 解析动作
        action = self._parse_action(response_text)
        
        if action.get("action") == "final":
            answer = action.get("answer", "任务完成")
            # 将助手回答也存入记忆
            self.messages.append({"role": "assistant", "content": answer})
            return answer
        else:
            # Phase 0 暂未实现工具，给个友好提示
            msg = f"Agent 想要调用工具 [{action.get('action')}]，但当前处于 Phase 0，工具未挂载。"
            self.messages.append({"role": "assistant", "content": msg})
            return msg