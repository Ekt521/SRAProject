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
                "你是一个智能研究助理，现在你拥有 `run_python` 工具。\n"
                "如果你需要计算或逻辑验证，请输出: "
                '{"action": "run_python", "code": "你的Python代码"} \n'
                "如果你已经知道答案，输出: "
                '{"action": "final", "answer": "你的回答"} \n'
                "注意：代码必须用纯文本写在 code 字段里，不要用 Markdown。"
            )
        }

        while step_count < self.max_steps:
            step_count += 1
            
            # 组装上下文（只取最近 6 条，控制 token）
            history = self.messages[-6:] if len(self.messages) > 6 else self.messages
            payload = [system_prompt] + history

            # 调用 DeepSeek
            response_text = self.llm.chat(payload)
            print(f"[Debug] 第{step_count}步返回: {response_text[:100]}...")  # 打印前100字调试

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
