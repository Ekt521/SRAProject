import json
import re
from typing import Dict, Any, List
from tools.executor import safe_run_python  # 导入我们的工具

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

        # 构建系统提示词（告诉 Agent 现在有计算器工具了）
        system_prompt = {
            "role": "system",
            "content": (
                "你是严格遵循 JSON 格式的智能助理。\n"
                "每次回答**必须**只输出一个合法的 JSON 对象，不要输出任何其他文字。\n"
                "如果你知道最终答案，输出: {\"action\": \"final\", \"answer\": \"你的答案\"}\n"
                "如果需要计算或逻辑验证，输出: {\"action\": \"run_python\", \"code\": \"Python代码\"}\n"
                "无论问题多么简单，都不允许直接输出非 JSON 的文本。\n"
                "示例：用户问'1+1等于？'，你必须输出 {\"action\": \"final\", \"answer\": \"2\"}，而不是'等于2'。"
                "严禁编造不存在的答案，不确定的问题可以输出 '我不确定' 或 '无法回答'，但仍然要遵循 JSON 格式。"
            )
        }

        while step_count < self.max_steps:
            step_count += 1
            
            # 组装上下文（只取最近 6 条，控制 token）
            history = self.messages[-6:] if len(self.messages) > 6 else self.messages
            payload = [system_prompt] + history

            # 调用 DeepSeek
            response_text = self.llm.chat(payload)
            print(f"[Debug] 第{step_count}步返回: {response_text[:200]}...")  # 打印前200字调试

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
                    # 调用我们的沙盒执行器
                    observation = safe_run_python(code)
                
                # 将执行结果（Observation）塞回消息历史，让 LLM 继续思考
                self.messages.append({
                    "role": "user", 
                    "content": f"代码执行结果 (Observation): {observation}"
                })
                # 循环继续，LLM 会基于这个结果决定是 final 还是修改代码

            else:
                # 未知动作
                self.messages.append({
                    "role": "user",
                    "content": f"错误：未知动作 {action_type}，请输出 final 或 run_python。"
                })

        # 如果超出最大步数
        return "任务超出最大尝试次数 (5次)，请简化问题或检查逻辑。"
