# HiveMind

**Autonomous multi-agent orchestration with adversarial plan validation, runtime tool synthesis, RAG-enabled agents, persistent cross-run memory, and real-time execution streaming.**

Live demo: [https://hivemind-v9fr.onrender.com](https://hivemind-v9fr.onrender.com)

---

## What is HiveMind?

HiveMind is a general-purpose task execution engine. You give it any complex task in natural language; it figures out what agents to spawn, what tools those agents need, builds those tools from code it writes itself, debates its own plan before executing, and synthesises a final deliverable — all while streaming every decision to a live UI.

It solves three failures common to every existing multi-agent framework:

| Failure | Root cause | HiveMind's fix |
|---|---|---|
| Wrong agents for the task | Static predefined rosters | Agents are spawned at runtime from the plan |
| Bad plans execute silently | No pre-execution review | Adversarial DA ↔ EA debate gate |
| Agents work in silos | Isolated context windows | Shared workspace + cross-run episodic memory |

---

## Full Execution Pipeline

```
User Task (natural language)
        │
        ▼
┌───────────────────┐
│  Quick Classifier  │  GPT-4o, temp=0, JSON mode
│  (quick_actions.py)│  Asks: can this be solved with 1-3 direct tool calls?
└────────┬──────────┘
         │ quick?          │ full pipeline?
         ▼                 ▼
   Direct execution   ┌─────────────────────┐
   (send email,       │   Adversarial Debate  │
    search web, etc.) │   (debate.py)         │
                      │   DA proposes plan    │
                      │   EA critiques 7 dims │
                      │   ≤3 rounds, score≥6  │
                      └──────────┬────────────┘
                                 ▼
                      ┌─────────────────────┐
                      │    Tool Forge         │
                      │  (tool_forge.py)      │
                      │  GPT-4o writes Python │
                      │  AST safety validation│
                      │  Parallel (6 threads) │
                      └──────────┬────────────┘
                                 ▼
                      ┌─────────────────────┐
                      │   Agent Factory       │
                      │  (agent_factory.py)   │
                      │  create_react_agent() │
                      │  per spec, forged     │
                      │  tools + memory tools │
                      └──────────┬────────────┘
                                 ▼
                      ┌─────────────────────┐
                      │   Graph Builder       │
                      │  (graph_builder.py)   │
                      │  LangGraph StateGraph │
                      │  dependency DAG       │
                      │  cycle detection      │
                      └──────────┬────────────┘
                                 ▼
                      ┌─────────────────────┐
                      │    Execution          │
                      │  LangGraph .invoke()  │
                      │  parallel fan-out     │
                      │  shared workspace     │
                      │  token streaming      │
                      └──────────┬────────────┘
                                 ▼
                      ┌─────────────────────┐
                      │     Compiler          │
                      │  (compiler.py)        │
                      │  GPT-4o synthesises   │
                      │  all agent outputs    │
                      │  + coverage report    │
                      └──────────┬────────────┘
                                 ▼
                       Final output + episode
                       saved to memory
```

Everything from the debate through compilation streams to the frontend over a single WebSocket connection.

---

## Architecture

### Module Map

```
orchestrator/
├── pipeline.py          Entry point: run_task(task, event_bus, memory)
├── debate.py            DA ↔ EA adversarial debate loop
├── quick_actions.py     Pre-pipeline classifier + direct executor
├── tool_forge.py        LLM code generation → StructuredTool
├── agent_factory.py     create_react_agent() per spec
├── graph_builder.py     LangGraph StateGraph + cycle detection
├── compiler.py          Final output synthesis node
├── rag_engine.py        Per-agent document Q&A (ChromaDB + LLM)
├── capabilities.py      13 built-in functions (search, file I/O, compute …)
├── integrations.py      8 real-world integrations (email, Slack, calendar …)
├── mcp_client.py        Model Context Protocol (stdio + SSE)
├── events.py            Thread-safe EventBus for WebSocket streaming
├── state.py             LangGraph OrchestratorState TypedDict
├── prompts.py           All system prompts (DA, EA, agents, compiler, forge)
├── config.py            Models, limits, env loading
├── utils.py             parse_json_response, truncate, call_llm
└── memory/
    ├── __init__.py      MemoryManager — orchestrates all memory layers
    ├── types.py         Episode, MemoryEntry, SharedMemoryItem dataclasses
    ├── store.py         SQLite — episodes + distilled memory entries
    ├── episodic.py      EpisodeRecorder — captures events during a run
    ├── long_term.py     Cross-run learning — distill + retrieve patterns
    ├── short_term.py    SharedWorkspace — in-run key-value store
    └── embeddings.py    ChromaDB semantic index (optional, degrades gracefully)

api/
└── app.py               FastAPI server — REST + WebSocket + session management

frontend/
├── index.html           Dashboard UI
├── css/style.css        Dark theme
└── js/app.js            WebSocket client + real-time rendering
```

### LangGraph State (`orchestrator/state.py`)

```python
class OrchestratorState(TypedDict):
    task: str
    plan: dict
    agent_outputs: Annotated[dict, merge_dicts]   # parallel fan-in
    shared_memory: Annotated[dict, merge_dicts]   # workspace snapshots
    final_output: str
    coverage_report: dict
    known_issues: Annotated[list, merge_lists]
    metadata: dict
```

`merge_dicts` and `merge_lists` are custom reducers that merge concurrent agent outputs without overwrites — essential for the parallel fan-out/fan-in pattern.

---

## Core Components

### 1. Adversarial Debate (`orchestrator/debate.py`)

The Dynamic Agent (DA) and Evaluator Agent (EA) are two separate GPT-4o instances with fundamentally different system prompts. The DA is optimised to plan; the EA is explicitly instructed to find problems.

**DA config:** `temperature=0.7`, `response_format=json_object` — produces an `ExecutionPlan` JSON with agents, tools, dependencies, and model tiers.

**EA config:** `temperature=0.3`, scores the plan across 7 dimensions:
1. Coverage — does the plan address every aspect?
2. Agent roles — right expertise, no redundancy?
3. Tool feasibility — can each tool be a Python function?
4. Dependency logic — correct ordering, maximum parallelism?
5. Overkill — needlessly complex?
6. Underkill — too thin for the task?
7. Tool description specificity — vague descriptions catch here, not at forge time

**Convergence:** `score >= 6` with no CRITICAL issues, or a hard cap of `MAX_DEBATE_ROUNDS = 3`. When the EA provides a `modified_plan` field, that rewritten plan is used directly — skipping a DA revision round.

**Memory injection:** The DA's first prompt is augmented with `memory_context` — relevant past episodes and distilled plan patterns retrieved from `LongTermMemory`.

### 2. Quick Action Intelligence (`orchestrator/quick_actions.py`)

Before any debate starts, a GPT-4o classifier (temp=0, JSON mode) determines whether the task maps to 1–3 direct built-in tool calls. The full `_TOOL_MAP` contains 13 entries matching the capability and integration functions. If mode is `"quick"`, the actions are executed in sequence and the pipeline returns immediately.

This eliminates unnecessary debate/forge/graph overhead for tasks like:
- "Send an email to alice@example.com saying the meeting is cancelled"
- "Create a calendar event for tomorrow at 3pm"
- "Search the web for OpenAI pricing"

The classifier decision (mode, reason, action list) is streamed to the frontend as `quick_detect_done`.

### 3. Tool Forge (`orchestrator/tool_forge.py`)

After the plan is approved, every `tools_needed` spec across all agents is collected. **All tools are forged in parallel** using a `ThreadPoolExecutor` with up to 6 workers — one thread per unique tool name.

For each spec:
1. GPT-4o (`FORGE_MODEL`, temp=0, max_tokens=2048) writes a Python function body matching the spec's name, description, parameters, and return type.
2. `ast.parse()` validates syntax.
3. An AST visitor walks the tree and blocks:
   - **Forbidden imports:** `subprocess`, `shutil`, `ctypes`, `importlib`, `pickle`, `shelve`, `multiprocessing`, `signal`, `socket`
   - **Forbidden calls:** `os.system`, `os.remove`, `os.rmdir`, `os.unlink`, `shutil.rmtree`, `__import__`, `eval`, `exec`
4. `exec()` runs the code in a namespace pre-populated with `CAPABILITY_NAMESPACE` — the 13 real capability functions. This is what allows forged tools to call `search_web()`, `save_file()`, `compute()`, etc.
5. The function is wrapped in `_make_safe_wrapper` (catches all exceptions, returns error strings) and converted to a `LangChain StructuredTool`.
6. On any failure, one retry with the error message fed back to the LLM.

Tools are cached by name — identical tools requested by multiple agents are generated once.

### 4. Agent Factory (`orchestrator/agent_factory.py`)

Each agent spec from the approved plan becomes a LangGraph `create_react_agent()` instance:

- **Model selection:** `TIER_TO_MODEL` maps `FAST → gpt-4o-mini`, `BALANCED/HEAVY → gpt-4o`
- **Tools:** forged tools + optional MCP tools + `remember`/`recall` memory tools
- **Memory tools:** `remember(key, value, tags)` and `recall(key)` are `StructuredTool` instances backed by `SharedWorkspace`. Agents use these to pass data to downstream agents without needing direct message passing.
- **System prompt:** `AGENT_SYSTEM_PROMPT` is formatted with role, persona, objective, tool list, and any relevant past experience retrieved from `LongTermMemory`.
- **Streaming:** `AgentStreamHandler` (a `BaseCallbackHandler`) emits `agent_token`, `agent_tool_call`, and `agent_tool_result` events on every token and tool interaction.

### 5. Graph Builder (`orchestrator/graph_builder.py`)

Converts the flat agent list + dependency specs into a `LangGraph StateGraph`:

1. **Cycle detection:** DFS over the `depends_on` map before adding any edges. Raises `ValueError` with the full cycle path if found.
2. **Root agents** (no dependencies) get edges from `START` and run in parallel.
3. **Dependency edges** connect each agent to its prerequisites — LangGraph's fan-in waits for all predecessors before firing a node.
4. **Leaf agents** (nothing depends on them) connect to the `compiler` node.
5. Compiled with `MemorySaver` for LangGraph checkpoint persistence.

### 6. Compiler (`orchestrator/compiler.py`)

The final node in the graph. Receives the full `OrchestratorState` (all agent outputs, shared workspace snapshot, plan) and calls GPT-4o (`COMPILER_MODEL`, temp=0.3, max_tokens=4096, JSON mode) to synthesise a structured deliverable:

```json
{
  "final_output": "<markdown deliverable>",
  "coverage_report": { "quality_assessment": "...", ... },
  "known_issues": ["..."],
  "recommendations": ["..."]
}
```

Falls back to raw concatenation of agent outputs if JSON parsing fails — the compiler never crashes the pipeline.

### 7. RAG Engine (`orchestrator/rag_engine.py`)

Any agent can be given a persistent document knowledge base. Files are uploaded via `POST /api/agents/{agent_id}/upload` and indexed into a per-agent ChromaDB collection.

**Supported formats:** PDF (via pdfplumber), Excel (via openpyxl), CSV, TXT, MD, JSON.

**Chunking:** Text is split by paragraphs first, then sentences for oversized paragraphs. Overlapping windows (100-char overlap) prevent context loss at boundaries. Chunks are upserted in batches of 40.

**Query flow:** ChromaDB retrieves the top-N most similar chunks by cosine distance. The LLM receives an identity-anchored system prompt (the agent's role + persona + objective) alongside the retrieved context, and is instructed to cite `[Source N]` references.

### 8. Memory System (`orchestrator/memory/`)

Two layers operate simultaneously:

**Short-term — `ShortTermMemory` (`short_term.py`):**
A thread-safe `SharedWorkspace` dict. Agents write with `remember(key, value, tags)` and read with `recall(key)`. The workspace is snapshotted into `OrchestratorState.shared_memory` after every agent node, so downstream agents always see what upstream agents stored.

**Long-term — `LongTermMemory` (`long_term.py` + `store.py` + `embeddings.py`):**
Every pipeline run is saved as an `Episode` in SQLite (`data/hivemind_memory.db`). After saving, `LongTermMemory.record_episode()` distills the episode into typed `MemoryEntry` records:

| Type | Trigger | Used by |
|---|---|---|
| `plan_pattern` | Each run | DA — pre-planning context |
| `lesson_learned` | Runs with known_issues | DA + compiler |
| `agent_strategy` | Each agent's output | Agent factory — per-role context |
| `user_preference` | User feedback via `/api/feedback` | DA — shapes future plans |

Retrieval uses ChromaDB cosine search (if available) with SQLite full-text fallback. The DA, each agent, and the compiler each receive their own scoped context slice before execution.

### 9. Event Bus (`orchestrator/events.py`)

A thread-safe `queue.Queue`-backed `EventBus`. The pipeline thread calls `emit(event_type, data)` throughout execution; the WebSocket coroutine drains the queue and pushes JSON to the client. This decouples the synchronous pipeline from the async FastAPI layer.

All 25+ event types are documented in the API reference below.

### 10. MCP Integration (`orchestrator/mcp_client.py`)

`load_mcp_tools(config)` supports both:
- **stdio** — spawns a local process (`npx @modelcontextprotocol/server-filesystem`, etc.) and communicates over stdin/stdout
- **SSE** — connects to a remote HTTP MCP server

Each MCP tool is wrapped as a `LangChain StructuredTool` and added to every agent's tool list. If MCP is unavailable or the config file is missing, the system continues without it.

---

## Built-in Capabilities (`orchestrator/capabilities.py`)

These 13 functions are injected into every forged tool's `exec()` namespace, making them callable from any LLM-generated code:

| Function | What it does |
|---|---|
| `search_web(query, max_results=8)` | DuckDuckGo Instant Answer + HTML scraping fallback |
| `scrape_url(url, max_chars=8000)` | HTTP GET + BeautifulSoup text extraction |
| `save_file(filename, content)` | Write to `output/` (path is sandboxed) |
| `read_file(filepath)` | Read from `output/` |
| `list_files(directory="")` | List `output/` contents |
| `fetch_json(url)` | HTTP GET → parsed JSON |
| `compute(code_str)` | Restricted Python eval — math, statistics, datetime, collections, re, json |
| `create_html_form(filename, title, fields, submit_action)` | Dark-themed HTML form with localStorage |

## Real-World Integrations (`orchestrator/integrations.py`)

| Function | What it does | Fallback |
|---|---|---|
| `send_email(to, subject, body, cc, html)` | SMTP (Gmail-compatible) | Draft file in `output/` |
| `send_slack_message(message, channel)` | Webhook POST | Draft file in `output/` |
| `create_calendar_event(title, start, end, ...)` | `.ics` file | Always works |
| `create_spreadsheet(filename, headers, rows)` | Excel via openpyxl or CSV | CSV fallback |
| `create_kanban_board(title, columns)` | Drag-and-drop HTML board | Always works |
| `send_webhook(url, payload)` | HTTP POST to any URL | — |
| `read_pdf(filepath)` | Text extraction via pdfplumber | PyPDF2 fallback |
| `parse_resume(text)` | Email, phone, sections, years of experience | Always works |

Every integration degrades gracefully — a useful artifact is always produced.

---

## API Reference

### REST

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/run` | Synchronous task execution (REST fallback for WebSocket) |
| `POST` | `/api/chat` | Chat with a specific agent post-run |
| `GET` | `/api/files` | List generated output files |
| `GET` | `/api/files/{filename}` | Read a file from `output/` (10 MB cap, extension allowlist) |
| `POST` | `/api/feedback` | Submit a rating for a completed run |
| `GET` | `/api/memory/episodes` | List past executions (limit 1–500) |
| `GET` | `/api/memory/search` | Semantic search across all memory (n_results 1–50) |
| `GET` | `/api/memory/stats` | Memory system statistics |
| `POST` | `/api/agents/{agent_id}/upload` | Upload a file to an agent's RAG knowledge base |
| `POST` | `/api/agents/{agent_id}/query` | Query an agent's RAG knowledge base |
| `GET` | `/api/agents/{agent_id}/files` | List files indexed for an agent |
| `GET` | `/api/agents/{agent_id}/info` | Get agent spec + output from active session |

All error responses use proper HTTP status codes: 400 (bad input), 403 (access denied), 404 (not found), 413 (too large), 422 (processing error), 500 (server error).

### WebSocket

```
WS /ws
Client sends:  {"task": "<string>"}
Server sends:  {"type": "<event_type>", "data": {...}, "ts": "<ISO timestamp>"}
```

**Event stream:**

```
pipeline_start
  quick_detect_start → quick_detect_done
    [quick path]  quick_start → quick_action(×N) → quick_done
    [full path]   memory_recall
                  debate_start → debate_da_response → debate_eval_response → ... → debate_complete
                  forge_start → forge_tool_start(×N) → forge_tool_done(×N) → forge_complete
                  agents_created
                  agent_start → agent_token(×N) → agent_tool_call → agent_tool_result → agent_done
                  memory_store
                  compile_start → compile_done
                  episode_saved
pipeline_done | pipeline_error
```

---

## Setup

### Requirements

- Python 3.10+
- OpenAI API key

```bash
git clone <repo-url>
cd Dynamic-Agent

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env — set OPENAI_API_KEY at minimum
```

### Environment Variables

**Required:**
```env
OPENAI_API_KEY=sk-...
```

**Optional integrations:**
```env
# Email — degrades to draft files if not set
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@example.com
SMTP_PASS=your-app-password
SMTP_FROM=you@example.com

# Slack — degrades to draft files if not set
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# MCP servers config
MCP_CONFIG_PATH=mcp_servers.json
```

**Optional MCP servers:**
```bash
cp mcp_servers.json.example mcp_servers.json
# Enable filesystem, Slack, GitHub, Notion, etc.
```

### Running

```bash
# Web server (recommended — real-time streaming UI)
python run_server.py
# Open http://localhost:8000

# CLI
python main.py "Plan a product launch for a B2B SaaS tool"

# Benchmark (single LLM vs full pipeline)
python evaluate.py
python run_benchmark.py
```

---

## System Configuration

[orchestrator/config.py](orchestrator/config.py):

```python
PLANNER_MODEL   = "gpt-4o"    # DA + Quick Classifier
EVALUATOR_MODEL = "gpt-4o"    # EA
COMPILER_MODEL  = "gpt-4o"    # Compiler
FORGE_MODEL     = "gpt-4o"    # Tool Forge

TIER_TO_MODEL = {
    "FAST":     "gpt-4o-mini",
    "BALANCED": "gpt-4o",
    "HEAVY":    "gpt-4o",
}

MAX_DEBATE_ROUNDS = 3
MAX_AGENTS        = 8
MAX_AGENT_STEPS   = 25   # LangGraph recursion limit per agent
```

---

## Security

### Tool Forge AST Validation

Every LLM-generated function is validated before execution. The AST visitor blocks:

- **Forbidden imports:** `subprocess`, `shutil`, `ctypes`, `importlib`, `pickle`, `shelve`, `multiprocessing`, `signal`, `socket`
- **Forbidden calls:** `os.system`, `os.remove`, `os.rmdir`, `os.unlink`, `shutil.rmtree`, `__import__`, `eval`, `exec`

Generated code runs in a capability namespace with only the 13 pre-approved functions available — it cannot import arbitrary modules.

### API Layer

- Input validation on all request models via Pydantic `@field_validator`
- File access sandboxed to `output/` with `os.path.realpath` traversal check
- File extension allowlist (`.txt`, `.md`, `.json`, `.csv`, `.html`, `.ics`, `.xlsx`, `.py`)
- 10 MB upload and file-read size cap
- Session TTL: 1 hour; max 20 concurrent sessions with LRU eviction
- All errors returned as structured HTTP exceptions with appropriate status codes

---

## Competitive Positioning

| System | What HiveMind adds |
|---|---|
| LangGraph | Adversarial debate gate; dynamic agent spawning; runtime tool synthesis |
| CrewAI | No static roster; plan validation before execution; cross-run memory |
| AutoGen | Structured confidence-scored approval gate vs. unstructured group chat |
| Reflexion | Separate adversarial critic vs. same-agent self-reflection |
| MetaGPT | General-purpose, not code-generation-specific |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph 1.0+, LangChain 1.2+ |
| LLM | OpenAI GPT-4o (gpt-4o-mini for FAST tier) |
| Backend | FastAPI 0.110+, Uvicorn, Pydantic 2.0+ |
| Memory | SQLite (episodic store), ChromaDB (vector search) |
| RAG | ChromaDB + pdfplumber + openpyxl |
| Integrations | SMTP, Slack webhooks, iCalendar, openpyxl, pdfplumber |
| MCP | `mcp` 1.0+ (stdio + SSE) |
| Frontend | Vanilla HTML/CSS/JS, WebSocket, Marked.js |
| Search | DuckDuckGo (primary), requests (scraping) |
