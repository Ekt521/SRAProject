# tools/web_search.py
import os
from pathlib import Path
import requests
from dotenv import load_dotenv

# ---- 自动定位项目根目录下的 .env（与 core/llm_client.py 保持同一方式）----
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

BOCHA_API_URL = "https://api.bochaai.com/v1/web-search"
BOCHA_API_KEY = os.getenv("BOCHA_API_KEY")

# 每条摘要最多保留的字符数，控制发给 LLM 的 token
MAX_SNIPPET = 300


def search_web(query: str, max_results: int = 3) -> str:
    """
    联网搜索：优先使用博查 API（国内稳定），失败时回退到 DuckDuckGo。
    如果全部失败，返回明确错误信息，绝不伪造结果。
    """
    # 1. 优先博查 API（.env 中配置了 BOCHA_API_KEY）
    if BOCHA_API_KEY:
        result = _search_bocha(query, max_results)
        if result is not None:
            return result
        print("[警告] 博查搜索失败，回退到 DuckDuckGo...")
    else:
        print("[提示] 未配置 BOCHA_API_KEY，使用 DuckDuckGo 搜索...")

    # 2. 回退 DuckDuckGo
    result = _search_ddg(query, max_results)
    if result is not None:
        return result

    # 3. 全部失败：明确告知，而不是返回伪造数据
    return (
        f"❌ 联网搜索失败：博查 API 与 DuckDuckGo 均不可用，"
        f"无法获取 '{query}' 的最新信息。请检查网络连接或 .env 中的 BOCHA_API_KEY 配置。"
    )


def _search_bocha(query: str, max_results: int) -> str | None:
    """调用博查 Web Search API，失败返回 None（由上层决定回退）。"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BOCHA_API_KEY}",
        }
        payload = {
            "query": query,
            "summary": True,          # 返回长文本摘要，方便 Agent 引用
            "count": max(1, min(max_results, 50)),
            "freshness": "noLimit",   # 不限时间范围，保证有结果
        }
        resp = requests.post(BOCHA_API_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()

        data = resp.json()
        if data.get("code") != 200:
            print(f"[警告] 博查返回异常 code={data.get('code')}: {data.get('msg')}")
            return None

        pages = data.get("data", {}).get("webPages", {}).get("value") or []
        if not pages:
            return f"未找到与 '{query}' 相关的网页信息。"

        output = []
        for r in pages[:max_results]:
            title = r.get("name") or "无标题"
            url = r.get("url") or ""
            snippet = r.get("snippet") or r.get("summary") or "无摘要"
            if isinstance(snippet, (dict, list)):
                snippet = str(snippet)
            if len(snippet) > MAX_SNIPPET:
                snippet = snippet[:MAX_SNIPPET] + "..."
            output.append(f"[source:{url}] {title}\n{snippet}")
        return "\n\n".join(output)

    except Exception as e:
        print(f"[警告] 博查搜索异常: {e}")
        return None


def _search_ddg(query: str, max_results: int) -> str | None:
    """调用 DuckDuckGo（ddgs 库），失败返回 None。"""
    try:
        from ddgs import DDGS
    except ImportError:
        print("[提示] 未安装 ddgs 库，跳过 DuckDuckGo 回退。安装: pip install ddgs")
        return None

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, region='cn-zh', timeout=20))
            if not results:
                # 中文区域无结果时，尝试无区域限制
                results = list(ddgs.text(query, max_results=max_results, timeout=20))
            if not results:
                return f"未找到与 '{query}' 相关的网页信息。"

            output = []
            for r in results:
                title = r.get('title', '无标题')
                url = r.get('href', '')
                snippet = r.get('body', '无摘要')
                if len(snippet) > MAX_SNIPPET:
                    snippet = snippet[:MAX_SNIPPET] + "..."
                output.append(f"[source:{url}] {title}\n{snippet}")
            return "\n\n".join(output)

    except Exception as e:
        print(f"[警告] DuckDuckGo 搜索失败: {e}")
        return None


# 单独运行时可测试：python tools/web_search.py
if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "DeepSeek"
    print(search_web(q))
