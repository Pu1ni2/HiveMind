"""
FastAPI server — serves the frontend + provides WebSocket
for real-time pipeline streaming + interactive agent chat.
"""

import asyncio
import logging
import os
import time
import threading
import queue
import json
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from orchestrator.events import EventBus, set_bus
from orchestrator.pipeline import run_task
from orchestrator.capabilities import OUTPUT_DIR
from orchestrator.config import OPENAI_API_KEY
from orchestrator.memory import MemoryManager
from orchestrator.rag_engine import process_upload, query_rag, get_agent_files
from fastapi import UploadFile, File

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hivemind.api")

app = FastAPI(title="HIVEMIND")

# ── Memory manager (persistent across runs) ───────────────────────
memory_manager = MemoryManager(data_dir="data")

# ── CORS ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session storage for post-run interactive chat ─────────────────
_sessions: dict[str, dict] = {}
_SESSION_TTL = 3600       # 1 hour
_MAX_SESSIONS = 20
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
_ALLOWED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".html", ".ics", ".xlsx", ".py"}


def _cleanup_sessions() -> None:
    """Evict expired and oldest sessions."""
    now = time.time()
    expired = [k for k, v in _sessions.items() if now - v.get("created_at", 0) > _SESSION_TTL]
    for k in expired:
        del _sessions[k]
    while len(_sessions) > _MAX_SESSIONS:
        oldest = min(_sessions, key=lambda k: _sessions[k].get("created_at", 0))
        del _sessions[oldest]


# ── Request models ─────────────────────────────────────────────────

class TaskRequest(BaseModel):
    task: str
    mcp_servers: dict | None = None

    @field_validator("task")
    @classmethod
    def task_not_empty(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Task must be at least 3 characters")
        if len(v) > 10_000:
            raise ValueError("Task must be under 10,000 characters")
        return v


class ChatRequest(BaseModel):
    session_id: str
    agent_id: str
    message: str

    @field_validator("session_id", "agent_id")
    @classmethod
    def ids_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message must not be empty")
        if len(v) > 4_000:
            raise ValueError("Message must be under 4,000 characters")
        return v


class FeedbackRequest(BaseModel):
    episode_id: str
    feedback: str
    score: float  # 0.0 – 10.0

    @field_validator("score")
    @classmethod
    def score_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 10.0):
            raise ValueError("Score must be between 0.0 and 10.0")
        return v

    @field_validator("episode_id", "feedback")
    @classmethod
    def fields_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()


class RAGQueryRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question must not be empty")
        if len(v) > 2_000:
            raise ValueError("Question must be under 2,000 characters")
        return v


# ── REST: run task (sync fallback) ────────────────────────────────

@app.post("/api/run")
async def run_task_endpoint(req: TaskRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_task(req.task, mcp_servers=req.mcp_servers, memory=memory_manager),
        )
        return result
    except Exception as exc:
        logger.exception("Error in POST /api/run")
        raise HTTPException(status_code=500, detail=str(exc))


# ── WebSocket: real-time streaming ────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        raw = await ws.receive_text()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send_json({"type": "error", "data": {"message": "Invalid JSON payload"}})
            await ws.close()
            return

        task = data.get("task", "").strip()
        if len(task) < 3:
            await ws.send_json({"type": "error", "data": {"message": "Task must be at least 3 characters"}})
            await ws.close()
            return
        if len(task) > 10_000:
            await ws.send_json({"type": "error", "data": {"message": "Task too long (max 10,000 chars)"}})
            await ws.close()
            return

        session_id = uuid.uuid4().hex[:12]
        bus = EventBus()
        done = threading.Event()
        result_holder: dict = {}

        def pipeline_thread():
            set_bus(bus)
            try:
                bus.emit("pipeline_start", {"task": task, "session_id": session_id})
                result = run_task(task, event_bus=bus, memory=memory_manager)
                result_holder["result"] = result

                _cleanup_sessions()
                _sessions[session_id] = {
                    "task": task,
                    "plan": result.get("plan", {}),
                    "agent_outputs": result.get("agent_outputs", {}),
                    "chat_histories": {},
                    "created_at": time.time(),
                    "episode_id": result.get("metadata", {}).get("episode_id", ""),
                }

                bus.emit("pipeline_done", {
                    "session_id": session_id,
                    "final_output": result.get("final_output", ""),
                    "coverage_report": result.get("coverage_report", {}),
                    "known_issues": result.get("known_issues", []),
                    "metadata": result.get("metadata", {}),
                    "agent_outputs": {
                        k: {"role": v.get("role", k), "output": v.get("output", "")}
                        for k, v in result.get("agent_outputs", {}).items()
                    },
                })
            except Exception as exc:
                logger.exception("Pipeline error in WebSocket thread")
                bus.emit("pipeline_error", {"error": str(exc)})
            finally:
                set_bus(None)
                done.set()

        thread = threading.Thread(target=pipeline_thread, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        while not done.is_set() or not bus.is_empty():
            try:
                event = await loop.run_in_executor(None, lambda: bus.get(timeout=0.3))
                if event:
                    await ws.send_json(event)
            except queue.Empty:
                continue
            except (WebSocketDisconnect, Exception):
                break

        # Drain any remaining events
        while not bus.is_empty():
            event = bus.get(timeout=0.1)
            if event:
                try:
                    await ws.send_json(event)
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("Unhandled WebSocket error")
        try:
            await ws.send_json({"type": "error", "data": {"message": str(exc)}})
        except Exception:
            pass


# ── Interactive agent chat ────────────────────────────────────────

@app.post("/api/chat")
async def chat_with_agent(req: ChatRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Run a task first.")

    agent_output = session["agent_outputs"].get(req.agent_id)
    if not agent_output:
        raise HTTPException(status_code=404, detail=f"Agent '{req.agent_id}' not found in session.")

    if req.agent_id not in session["chat_histories"]:
        session["chat_histories"][req.agent_id] = []
    history = session["chat_histories"][req.agent_id]

    # Find agent spec from plan
    agent_spec = next(
        (a for a in session["plan"].get("agents", []) if a.get("id") == req.agent_id),
        {},
    )

    role = agent_output.get("role", req.agent_id)
    persona = agent_spec.get("persona", "")
    objective = agent_spec.get("objective", "")
    output = agent_output.get("output", "")

    system_content = (
        f"You are {role}.\n\n"
        f"{persona}\n\n"
        f"Your objective was: {objective}\n\n"
        f"The original task was: {session['task']}\n\n"
        f"Here is the work you produced:\n---\n{output[:6000]}\n---\n\n"
        "The user wants to discuss your work, ask follow-up questions, or request changes. "
        "Be specific, reference your findings, and provide actionable insights."
    )

    messages = [SystemMessage(content=system_content)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=req.message))

    try:
        model = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.5)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: model.invoke(messages))
        reply = response.content

        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": reply})

        return {"response": reply, "agent_id": req.agent_id, "role": role}
    except Exception as exc:
        logger.exception("Error in POST /api/chat")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Output file browser ──────────────────────────────────────────

@app.get("/api/files")
async def list_output_files():
    if not os.path.exists(OUTPUT_DIR):
        return {"files": []}
    files = []
    for f in sorted(os.listdir(OUTPUT_DIR)):
        filepath = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(filepath):
            files.append({
                "name": f,
                "size": os.path.getsize(filepath),
                "modified": os.path.getmtime(filepath),
            })
    return {"files": files}


@app.get("/api/files/{filename}")
async def read_output_file(filename: str):
    safe_name = os.path.basename(filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=403, detail=f"File type '{ext}' not allowed")

    filepath = os.path.join(OUTPUT_DIR, safe_name)
    resolved = os.path.realpath(filepath)
    if not resolved.startswith(os.path.realpath(OUTPUT_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(resolved) or not os.path.isfile(resolved):
        raise HTTPException(status_code=404, detail="File not found")

    file_size = os.path.getsize(resolved)
    if file_size > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large ({file_size:,} bytes, max {_MAX_FILE_SIZE:,})")

    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return {"name": safe_name, "content": content, "size": len(content)}


# ── Memory endpoints ─────────────────────────────────────────────

@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """User rates a completed task — creates learnings for future runs."""
    try:
        memory_manager.record_feedback(req.episode_id, req.feedback, req.score)
        return {"status": "ok", "message": "Feedback recorded. Future runs will learn from this."}
    except Exception as exc:
        logger.exception("Error in POST /api/feedback")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/memory/episodes")
async def list_episodes(limit: int = 20, domain: str | None = None):
    """Browse past task executions."""
    if not (1 <= limit <= 500):
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    try:
        episodes = memory_manager.get_episode_history(limit=limit, domain=domain)
        return {
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "task": ep.task[:200],
                    "task_domain": ep.task_domain,
                    "task_complexity": ep.task_complexity,
                    "success_score": ep.success_score,
                    "user_feedback": ep.user_feedback,
                    "timestamp": ep.timestamp,
                    "tags": ep.tags,
                    "agent_count": len(ep.plan.get("agents", [])),
                    "metadata": ep.metadata,
                }
                for ep in episodes
            ]
        }
    except Exception as exc:
        logger.exception("Error in GET /api/memory/episodes")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/memory/search")
async def search_memory(query: str, n_results: int = 5):
    """Semantic search across all past experience."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    if not (1 <= n_results <= 50):
        raise HTTPException(status_code=400, detail="n_results must be between 1 and 50")
    try:
        results = memory_manager.search_memory(query, n_results=n_results)
        return {"results": results}
    except Exception as exc:
        logger.exception("Error in GET /api/memory/search")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/memory/stats")
async def memory_stats():
    """Memory system statistics."""
    try:
        episodes = memory_manager.store.list_episodes(limit=1000)
        entries = memory_manager.store.get_all_entries(limit=1000)
        return {
            "total_episodes": len(episodes),
            "total_memory_entries": len(entries),
            "entry_types": {
                mtype: len([e for e in entries if e.memory_type == mtype])
                for mtype in set(e.memory_type for e in entries)
            } if entries else {},
            "domains": list(set(ep.task_domain for ep in episodes if ep.task_domain)),
            "vector_search_available": memory_manager.index.available,
        }
    except Exception as exc:
        logger.exception("Error in GET /api/memory/stats")
        raise HTTPException(status_code=500, detail=str(exc))


# ── RAG Agent endpoints ──────────────────────────────────────────

@app.post("/api/agents/{agent_id}/upload")
async def upload_to_agent(agent_id: str, file: UploadFile = File(...)):
    """Upload a file to an agent's RAG knowledge base."""
    if not agent_id.strip():
        raise HTTPException(status_code=400, detail="agent_id must not be empty")
    content = await file.read()
    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit")
    result = process_upload(agent_id, file.filename or "upload", content)
    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result.get("message", "Processing error"))
    return result


@app.post("/api/agents/{agent_id}/query")
async def query_agent_rag(agent_id: str, req: RAGQueryRequest):
    """Query an agent's RAG knowledge base."""
    if not agent_id.strip():
        raise HTTPException(status_code=400, detail="agent_id must not be empty")

    agent_role = ""
    agent_persona = ""
    agent_objective = ""
    for session in _sessions.values():
        ao = session.get("agent_outputs", {}).get(agent_id)
        if ao:
            agent_role = ao.get("role", agent_id)
            for a in session.get("plan", {}).get("agents", []):
                if a.get("id") == agent_id:
                    agent_persona = a.get("persona", "")
                    agent_objective = a.get("objective", "")
                    break
            break

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: query_rag(
                agent_id, req.question,
                agent_role=agent_role,
                agent_persona=agent_persona,
                agent_objective=agent_objective,
            ),
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=422, detail=result.get("answer", "RAG error"))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in POST /api/agents/{agent_id}/query")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/agents/{agent_id}/files")
async def list_agent_files(agent_id: str):
    """List all files in an agent's RAG knowledge base."""
    files = get_agent_files(agent_id)
    return {"files": files}


@app.get("/api/agents/{agent_id}/info")
async def get_agent_info(agent_id: str):
    """Get full agent spec from any active session."""
    for session in _sessions.values():
        for a in session.get("plan", {}).get("agents", []):
            if a.get("id") == agent_id:
                ao = session.get("agent_outputs", {}).get(agent_id, {})
                return {
                    "id": agent_id,
                    "role": a.get("role", ""),
                    "persona": a.get("persona", ""),
                    "objective": a.get("objective", ""),
                    "tools": [t.get("name") for t in a.get("tools_needed", [])],
                    "model_tier": a.get("model_tier", ""),
                    "agent_type": a.get("agent_type", "standard"),
                    "output": ao.get("output", ""),
                    "depends_on": a.get("depends_on", []),
                    "expected_output": a.get("expected_output", ""),
                }
    raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found in any active session")


# ── Health check ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Deployment health check — used by Render and load balancers."""
    return {"status": "ok", "service": "hivemind"}


# ── Frontend ─────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")

os.makedirs("output", exist_ok=True)
app.mount("/output", StaticFiles(directory="output", html=True), name="output")
