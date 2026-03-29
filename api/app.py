"""
FastAPI server — serves the frontend + provides WebSocket
for real-time pipeline streaming + interactive agent chat.
"""

import asyncio
import os
import time
import threading
import queue
import json
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from orchestrator.events import EventBus, set_bus
from orchestrator.pipeline import run_task
from orchestrator.capabilities import OUTPUT_DIR
from orchestrator.config import OPENAI_API_KEY
from orchestrator.memory import MemoryManager
from orchestrator.rag_engine import process_upload, query_rag, get_agent_files

app = FastAPI(title="HIVEMIND")

# ── Memory manager (persistent across runs) ───────────────────────
memory_manager = MemoryManager(data_dir="data")

# ── CORS (for hackathon demo flexibility) ─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session storage for post-run interactive chat ───────────────────
_sessions: dict[str, dict] = {}
_SESSION_TTL = 3600  # 1 hour
_MAX_SESSIONS = 20


def _cleanup_sessions():
    """Evict oldest sessions when over limit or expired."""
    now = time.time()
    expired = [k for k, v in _sessions.items() if now - v.get("created_at", 0) > _SESSION_TTL]
    for k in expired:
        del _sessions[k]
    while len(_sessions) > _MAX_SESSIONS:
        oldest = min(_sessions, key=lambda k: _sessions[k].get("created_at", 0))
        del _sessions[oldest]


# ── REST endpoint (fallback / testing) ──────────────────────────────

class TaskRequest(BaseModel):
    task: str
    mcp_servers: dict | None = None


@app.post("/api/run")
async def run_task_endpoint(req: TaskRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: run_task(req.task, mcp_servers=req.mcp_servers, memory=memory_manager)
    )
    return result


# ── WebSocket endpoint (real-time streaming) ────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        raw = await ws.receive_text()
        data = json.loads(raw)
        task = data.get("task", "")

        if not task:
            await ws.send_json({"type": "error", "data": {"message": "No task provided"}})
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

                # Store session for interactive chat
                _cleanup_sessions()
                _sessions[session_id] = {
                    "task": task,
                    "plan": result.get("plan", {}),
                    "agent_outputs": result.get("agent_outputs", {}),
                    "chat_histories": {},  # {agent_id: [messages]}
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
                bus.emit("pipeline_error", {"error": str(exc)})
            finally:
                set_bus(None)
                done.set()

        thread = threading.Thread(target=pipeline_thread, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()

        while not done.is_set() or not bus.is_empty():
            try:
                event = await loop.run_in_executor(
                    None, lambda: bus.get(timeout=0.3)
                )
                if event:
                    await ws.send_json(event)
            except queue.Empty:
                continue
            except (WebSocketDisconnect, Exception):
                break

        # Drain remaining events
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
        try:
            await ws.send_json({"type": "error", "data": {"message": str(exc)}})
        except Exception:
            pass


# ── Interactive agent chat ──────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    agent_id: str
    message: str


@app.post("/api/chat")
async def chat_with_agent(req: ChatRequest):
    session = _sessions.get(req.session_id)
    if not session:
        return {"error": "Session not found. Run a task first."}

    agent_output = session["agent_outputs"].get(req.agent_id)
    if not agent_output:
        return {"error": f"Agent {req.agent_id} not found in session."}

    # Get or create chat history for this agent
    if req.agent_id not in session["chat_histories"]:
        session["chat_histories"][req.agent_id] = []

    history = session["chat_histories"][req.agent_id]

    # Find agent spec from plan
    agent_spec = {}
    for a in session["plan"].get("agents", []):
        if a.get("id") == req.agent_id:
            agent_spec = a
            break

    role = agent_output.get("role", req.agent_id)
    persona = agent_spec.get("persona", "")
    objective = agent_spec.get("objective", "")
    output = agent_output.get("output", "")

    # Build system message with full agent context
    system_content = (
        f"You are {role}.\n\n"
        f"{persona}\n\n"
        f"Your objective was: {objective}\n\n"
        f"The original task was: {session['task']}\n\n"
        f"Here is the work you produced:\n"
        f"---\n{output[:6000]}\n---\n\n"
        f"The user wants to discuss your work, ask follow-up questions, "
        f"or request changes. Answer based on your expertise and the "
        f"work you did. Be specific, reference your findings, and "
        f"provide actionable insights."
    )

    messages = [SystemMessage(content=system_content)]

    # Add chat history
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Add new user message
    messages.append(HumanMessage(content=req.message))

    # Call LLM
    model = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.5)
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: model.invoke(messages))
        reply = response.content

        # Save to history
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": reply})

        return {"response": reply, "agent_id": req.agent_id, "role": role}
    except Exception as exc:
        return {"error": str(exc)}


# ── Output file browser ─────────────────────────────────────────────

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
    # Sanitize — prevent path traversal
    safe_name = os.path.basename(filename)
    filepath = os.path.join(OUTPUT_DIR, safe_name)
    resolved = os.path.realpath(filepath)
    if not resolved.startswith(os.path.realpath(OUTPUT_DIR)):
        return PlainTextResponse("Access denied", status_code=403)
    if not os.path.exists(resolved) or not os.path.isfile(resolved):
        return PlainTextResponse("File not found", status_code=404)
    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return {"name": safe_name, "content": content, "size": len(content)}


# ── Memory API endpoints ────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    episode_id: str
    feedback: str
    score: float  # 0.0 - 10.0


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """User rates a completed task — creates learnings for future runs."""
    try:
        memory_manager.record_feedback(req.episode_id, req.feedback, req.score)
        return {"status": "ok", "message": "Feedback recorded. Future runs will learn from this."}
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/api/memory/episodes")
async def list_episodes(limit: int = 20, domain: str | None = None):
    """Browse past task executions."""
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
        return {"episodes": [], "error": str(exc)}


@app.get("/api/memory/search")
async def search_memory(query: str, n_results: int = 5):
    """Semantic search across all past experience."""
    try:
        results = memory_manager.search_memory(query, n_results=n_results)
        return {"results": results}
    except Exception as exc:
        return {"results": [], "error": str(exc)}


@app.get("/api/memory/stats")
async def memory_stats():
    """Get memory system statistics."""
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
        return {"error": str(exc)}


# ── RAG Agent endpoints ─────────────────────────────────────────────

@app.post("/api/agents/{agent_id}/upload")
async def upload_to_agent(agent_id: str, file: UploadFile = File(...)):
    """Upload a file to an agent's RAG knowledge base."""
    content = await file.read()
    result = process_upload(agent_id, file.filename, content)
    return result


class RAGQueryRequest(BaseModel):
    question: str


@app.post("/api/agents/{agent_id}/query")
async def query_agent_rag(agent_id: str, req: RAGQueryRequest):
    """Query an agent's RAG knowledge base."""
    # Get agent context from session
    agent_role = ""
    agent_persona = ""
    agent_objective = ""

    for session in _sessions.values():
        ao = session.get("agent_outputs", {}).get(agent_id)
        if ao:
            agent_role = ao.get("role", agent_id)
            # Get full spec from plan
            for a in session.get("plan", {}).get("agents", []):
                if a.get("id") == agent_id:
                    agent_persona = a.get("persona", "")
                    agent_objective = a.get("objective", "")
                    break
            break

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: query_rag(
            agent_id, req.question,
            agent_role=agent_role,
            agent_persona=agent_persona,
            agent_objective=agent_objective,
        )
    )
    return result


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
    return {"error": "Agent not found"}


# ── Serve frontend ──────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")

# Serve output files (so generated HTML forms can be opened in browser)
os.makedirs("output", exist_ok=True)
app.mount("/output", StaticFiles(directory="output", html=True), name="output")
