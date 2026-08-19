import requests
import json
import os
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

logger = logging.getLogger("sra.llm_client")

# ---- 关键改动：自动定位项目根目录下的 .env ----
# 当前文件路径: core/llm_client.py
# parent: core/ 目录
# parent.parent: 项目根目录 (SRAProject)
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class DeepSeekClient:
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        # 优先用传入的，其次从环境变量读（已被 .env 注入）
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "请在项目根目录下创建 .env 文件，并写入 DEEPSEEK_API_KEY=你的密钥"
            )
        
        self.model = model
        self.base_url = "https://api.deepseek.com/chat/completions"
        
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3,
             max_retries: int = 3, timeout: int = 30) -> str:
        """
        手写 DeepSeek 调用，带指数退避重试。
        temperature 设为 0.3：既要逻辑性，又要避免 0 温度导致的无限重复循环
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            # "stream": False  # 不开启流式，方便调试
        }

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=timeout
                )
                response.raise_for_status()  # 触发 HTTP 错误

                result = response.json()
                # DeepSeek 返回格式与 OpenAI 一致
                return result["choices"][0]["message"]["content"]

            except requests.exceptions.RequestException as e:
                # 网络抖动/限流：指数退避后重试（0.5s, 1s, 2s）
                if attempt < max_retries:
                    wait = 0.5 * (2 ** attempt)
                    logger.warning("[LLM] 请求失败 (%s)，%.1fs 后重试 (第%d次)...", e, wait, attempt + 1)
                    time.sleep(wait)
                    continue
                logger.error("[LLM] 请求最终失败: %s", e)
                return f"【网络错误】请求失败: {str(e)}"
            except (KeyError, json.JSONDecodeError) as e:
                logger.error("[LLM] 解析异常: %s", e)
                return f"【解析错误】DeepSeek 返回异常: {str(e)}"