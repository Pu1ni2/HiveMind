"""
FastAPI server — serves the frontend + provides WebSocket
for real-time pipeline streaming + interactive agent chat.
"""

import asyncio
import os
import threading
import queue
import json
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from orchestrator.events import EventBus, set_bus
from orchestrator.pipeline import run_task
from orchestrator.capabilities import OUTPUT_DIR
from orchestrator.config import OPENAI_API_KEY

app = FastAPI(title="YCONIC")

# ── Session storage for post-run interactive chat ───────────────────
_sessions: dict[str, dict] = {}


# ── REST endpoint (fallback / testing) ──────────────────────────────

class TaskRequest(BaseModel):
    task: str
    mcp_servers: dict | None = None


@app.post("/api/run")
async def run_task_endpoint(req: TaskRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: run_task(req.task, mcp_servers=req.mcp_servers)
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
                result = run_task(task, event_bus=bus)
                result_holder["result"] = result

                # Store session for interactive chat
                _sessions[session_id] = {
                    "task": task,
                    "plan": result.get("plan", {}),
                    "agent_outputs": result.get("agent_outputs", {}),
                    "chat_histories": {},  # {agent_id: [messages]}
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
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        return PlainTextResponse("File not found", status_code=404)
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return {"name": filename, "content": content, "size": len(content)}


# ── Serve frontend ──────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")

# Serve output files (so generated HTML forms can be opened in browser)
import os as _os
_os.makedirs("output", exist_ok=True)
app.mount("/output", StaticFiles(directory="output", html=True), name="output")
