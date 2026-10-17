import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.agent import SimpleReActAgent
from core.llm_client import DeepSeekClient
from tools.knowledge_tool import load_knowledge

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
app = FastAPI(title="SRA Assistant", version="1.0.0")

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)

class ChatResponse(BaseModel):
    response: str

_agent = None
_agent_error = None

def get_agent():
    global _agent, _agent_error
    if _agent is not None:
        return _agent
    if _agent_error is not None:
        raise RuntimeError(_agent_error)
    try:
        load_knowledge(str(ROOT / "data"))
        _agent = SimpleReActAgent(DeepSeekClient())
        return _agent
    except Exception as exc:
        _agent_error = str(exc)
        raise RuntimeError(_agent_error) from exc

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "configured": bool(os.getenv("DEEPSEEK_API_KEY")),
        "endpoint": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions"),
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        agent = get_agent()
        # MemoryManager keeps a SQLite connection bound to this server thread.
        # Keep the synchronous agent on it rather than moving it to a worker thread.
        answer = agent.run(request.message.strip())
        return ChatResponse(response=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")
