# HiveMind — Technical System Design

**A dynamic multi-agent orchestration engine with adversarial plan validation, runtime tool synthesis, RAG document intelligence, persistent cross-run memory, and full-stack real-time streaming.**

Live: [https://hivemind-v9fr.onrender.com](https://hivemind-v9fr.onrender.com)

---

## 1. Problem Statement

Multi-agent AI systems suffer from three specific, simultaneously unsolved failures that show up in production:

### 1.1 Static Agent Rosters (Rigid)

Frameworks like CrewAI, AutoGen, and LangGraph's multi-agent templates require agents to be predefined before the task is known. A Researcher, Writer, and Analyst are hardcoded — but a different task needs a Data Cleaner and Fact-Checker. The roster cannot adapt; the task is forced to fit the roster. This is the most common production friction point, evidenced by hundreds of open GitHub issues across LangChain (180K+ stars), CrewAI (25K+ stars), and AutoGen (40K+ stars).

### 1.2 Unvalidated Task Decomposition (Fragile)

Orchestrators decompose tasks and execute immediately with no quality gate. The plan itself is never challenged. Bad decompositions produce: redundant agents, missing agents, incorrect dependency orderings, and agents working at cross-purposes. The failure is discovered only after execution completes — wasting time and API tokens.

### 1.3 Memory Isolation (Incoherent)

Sub-agents operate in isolated context windows. Agent A produces market research; Agent B writes a report — but B never accesses A's findings. Outputs become contradictory or shallow. No existing framework provides a context-aware memory broker that routes relevant outputs between agents, and no framework remembers anything across runs.

### 1.4 Measured Evidence

The benchmark suite (`evaluate.py`, `run_benchmark.py`) runs identical tasks through single-pass decomposition and HiveMind's debate pipeline across four categories: mathematical reasoning, code generation, software design, and multi-domain research. In every research/analysis task, single-pass decomposition exhibited at least one structural plan error that adversarial debate caught before a single sub-agent token was spent: missing roles, incorrect dependency ordering, redundant agents, and vague tool specifications.

---

## 2. System Overview

HiveMind is structured as a sequential pipeline with parallel execution at the agent layer. The entry point is `orchestrator/pipeline.py:run_task()`. Every phase emits structured events to a thread-safe `EventBus`, which the WebSocket layer drains and pushes to the frontend.

### 2.1 Pipeline Phases

```
1.  Quick Check        orchestrator/quick_actions.py
2.  Memory Recall      orchestrator/memory/long_term.py
3.  Debate             orchestrator/debate.py
4.  Tool Forge         orchestrator/tool_forge.py
5.  Agent Creation     orchestrator/agent_factory.py
6.  MCP Load           orchestrator/mcp_client.py
7.  Graph Build        orchestrator/graph_builder.py
8.  Execution          LangGraph StateGraph.invoke()
9.  Compilation        orchestrator/compiler.py
10. Memory Persist     orchestrator/memory/episodic.py + long_term.py
```

### 2.2 Data Flow

```
pipeline.py:run_task(task, event_bus, memory)
    │
    ├─ try_quick_execute(task) → {result} or None
    │
    ├─ memory.get_planning_context(task) → memory_context str
    │
    ├─ run_debate(task, memory_context) → plan: dict
    │     ├─ DA: ChatOpenAI(gpt-4o, temp=0.7, json_mode) → ExecutionPlan JSON
    │     └─ EA: ChatOpenAI(gpt-4o, temp=0.3, json_mode) → Critique JSON
    │
    ├─ forge_tools_for_plan(plan) → {agent_id: [StructuredTool]}
    │     └─ ThreadPoolExecutor(max_workers=6) — parallel per unique tool
    │
    ├─ create_all_agents(plan, agent_tools, mcp_tools, memory)
    │     └─ {agent_id: {"agent": CompiledGraph, "spec": dict}}
    │
    ├─ build_graph(plan, agent_bundles, memory) → CompiledGraph
    │     ├─ cycle detection (DFS)
    │     └─ StateGraph with dependency edges
    │
    ├─ graph.invoke({"task": task, "plan": plan}, config)
    │     ├─ parallel root agents (fan-out)
    │     ├─ dependency-ordered agents
    │     └─ compiler node (fan-in)
    │
    └─ memory.end_run(result) → episode saved + distilled
```

---

## 3. Component Design

### 3.1 Adversarial Debate (`orchestrator/debate.py`)

The debate subsystem is the architectural core of HiveMind. Two separate LLM instances with fundamentally different mandates interact in a structured loop.

**Dynamic Agent (DA):**
- Model: `gpt-4o`, temperature=0.7, JSON mode
- Mandate: produce an `ExecutionPlan` JSON
- Memory injection: receives `memory_context` from `LongTermMemory.get_planning_context()` before the first generation, containing similar past tasks and lessons learned

**Evaluator Agent (EA):**
- Model: `gpt-4o`, temperature=0.3, JSON mode
- Mandate: "Your job is to find problems" — explicitly adversarial system prompt
- Output schema: `{score: int, approved: bool, verdict: str, issues: [{severity, dimension, description}], strengths: [], modified_plan: dict|null}`

**Seven critique dimensions:**
1. Coverage — does the plan address every aspect of the task?
2. Agent roles — right number, right expertise, any redundancy?
3. Tool feasibility — can each tool be implemented as a Python function using standard libraries?
4. Dependency logic — correct ordering? could more agents run in parallel?
5. Overkill — unnecessarily complex?
6. Underkill — too thin for what's being asked?
7. Tool description specificity — vague descriptions that would cause forge failures

**Convergence logic (`debate.py:run_debate`):**
```python
if approved or verdict == "APPROVED":
    if critique.get("modified_plan") and critique["modified_plan"].get("agents"):
        plan = critique["modified_plan"]   # Use EA's rewrite directly
    return plan                             # Done

# EA provides revised plan — use it, skip DA revision round
if critique.get("modified_plan") and critique["modified_plan"].get("agents"):
    plan = critique["modified_plan"]
    continue

# DA revises based on critique
da_messages.append(AIMessage(content=json.dumps(plan)))
da_messages.append(HumanMessage(content=f"Critique:\n{json.dumps(critique)}\n\nRevise..."))
plan = parse_json_response(da_model.invoke(da_messages).content)
```

Hard cap: `MAX_DEBATE_ROUNDS = 3`. If the cap is hit, the last plan is used regardless of score.

**Frontend visibility:** Every round streams `debate_da_response` (plan preview) and `debate_eval_response` (score + issues). The UI renders character-level diffs between plan versions and displays issue descriptions as tooltips on changed sections.

### 3.2 Quick Action Intelligence (`orchestrator/quick_actions.py`)

A GPT-4o classifier (temp=0, JSON mode) runs before the debate. It determines whether the task can be resolved with 1–3 direct built-in tool calls.

**Classifier prompt defines the decision boundary:**
- Single action → `"quick"` (send email, create event, do one search)
- 2–3 sequential simple actions → `"quick"` (search + email, event + slack)
- Multi-step research/analysis → `"full_pipeline"`
- When in doubt → `"full_pipeline"`

**Execution (`_TOOL_MAP`):** 13 entries mapping tool names to actual Python functions from `capabilities.py` and `integrations.py`. Actions are executed sequentially; each result is captured and emitted as a `quick_action` event.

This phase eliminates the single biggest criticism of multi-agent systems — spawning a swarm for what one function call could do.

### 3.3 Tool Forge (`orchestrator/tool_forge.py`)

The forge takes every `tools_needed` spec from the approved plan and synthesises working Python functions at runtime. Key design decisions:

**Parallel generation:** All unique tools are generated simultaneously using `concurrent.futures.ThreadPoolExecutor(max_workers=min(len(specs), 6))`. For a plan with 8 agents and 12 unique tools, all 12 tools are generated in one batch.

**Safety pipeline (per tool):**
```
LLM generates code (GPT-4o, temp=0, max_tokens=2048)
    │
    ▼
ast.parse()  ← SyntaxError → retry with error
    │
    ▼
ASTSafetyVisitor  ← forbidden import/call → retry with reason
    │
    ▼
exec(code, {**CAPABILITY_NAMESPACE})  ← runtime error → retry
    │
    ▼
_extract_function(namespace, preferred_name)
    │
    ▼
_make_safe_wrapper()  ← all exceptions → return error string
    │
    ▼
StructuredTool.from_function()
```

The `CAPABILITY_NAMESPACE` populated into the exec namespace contains the 13 capability functions. Generated code can call `search_web()`, `save_file()`, `compute()` etc. without importing anything — they're already in scope.

**Tool caching:** `tool_cache: dict[str, StructuredTool]` ensures identical tool names are generated once and shared across agents.

**Error containment:** `_make_safe_wrapper` wraps every forged function. Any runtime exception becomes a string like `"[Tool search_competitors error] ..."` — agents never receive an unhandled exception from a tool call.

### 3.4 Agent Factory (`orchestrator/agent_factory.py`)

Each agent in the approved plan becomes a separate `create_react_agent()` instance:

```python
model = ChatOpenAI(
    model=TIER_TO_MODEL[spec.get("model_tier", "BALANCED")],  # gpt-4o-mini or gpt-4o
    api_key=OPENAI_API_KEY,
    temperature=0.5,
    streaming=True,   # enables AgentStreamHandler token events
)
tools = forged_tools + mcp_tools + memory_tools  # remember + recall
agent = create_react_agent(model=model, tools=tools, prompt=system_prompt)
```

**Memory tools (`_build_memory_tools`):** `remember(key, value, tags)` and `recall(key)` are `StructuredTool` instances wrapping `SharedWorkspace.write()` and `SharedWorkspace.read()`. They emit `memory_store` and `memory_recall` events for UI display.

**`AgentStreamHandler` (extends `BaseCallbackHandler`):**
- `on_llm_new_token(token)` → emits `agent_token`
- `on_tool_start(serialized, input_str)` → emits `agent_tool_call`
- `on_tool_end(output)` → emits `agent_tool_result`

This provides token-level streaming for every agent without any changes to the LangChain/LangGraph internals.

**Context building (`make_agent_node`):** The `node_fn` closure reads upstream agent outputs from `OrchestratorState.agent_outputs` (filtered by `spec.depends_on`), truncates each to 8,000 chars, and prepends them to the agent's prompt. Shared workspace summary is also injected.

### 3.5 Graph Builder (`orchestrator/graph_builder.py`)

Translates the flat agent list + `depends_on` links into a `LangGraph StateGraph`.

**Cycle detection (DFS, 3-colour algorithm):**
```python
WHITE, GRAY, BLACK = 0, 1, 2

def dfs(node):
    color[node] = GRAY
    path.append(node)
    for dep in depends_on.get(node, []):
        if color[dep] == GRAY:        # back edge → cycle
            path.append(dep)
            return True
        if color[dep] == WHITE and dfs(dep):
            return True
    color[node] = BLACK
    path.pop()
    return False
```

Raises `ValueError` with the full cycle path before building any edges. This is a hard guardrail — a circular plan can never reach execution.

**Edge construction:**
- Agents with empty `depends_on` → `START → agent`
- Dependency pairs → `dep → agent`
- Agents not depended on by anything → `agent → compiler`

LangGraph handles parallel fan-out automatically: all edges from `START` fire concurrently. Fan-in happens at nodes with multiple incoming edges — LangGraph waits for all predecessors.

### 3.6 LangGraph State (`orchestrator/state.py`)

```python
class OrchestratorState(TypedDict):
    task: str
    plan: dict
    agent_outputs: Annotated[dict, merge_dicts]
    shared_memory: Annotated[dict, merge_dicts]
    final_output: str
    coverage_report: dict
    known_issues: Annotated[list, merge_lists]
    metadata: dict
```

`merge_dicts` and `merge_lists` are custom reducers invoked by LangGraph when multiple parallel nodes write to the same state field simultaneously. Without them, parallel writes would clobber each other.

### 3.7 RAG Engine (`orchestrator/rag_engine.py`)

Per-agent document intelligence. Each agent gets its own isolated ChromaDB collection (`rag_{agent_id}`).

**Ingestion pipeline:**
1. File bytes received via `POST /api/agents/{agent_id}/upload`
2. Format detection → PDF (pdfplumber) / Excel (openpyxl) / CSV / plain text / JSON
3. `_chunk_text(text, chunk_size=800, overlap=100)` — paragraph-first splitting, sentence fallback for oversized paragraphs, 100-char overlapping windows
4. ChromaDB `collection.upsert()` in batches of 40 (ChromaDB batch limit)
5. Raw file saved to `output/` for download

**Query pipeline:**
1. `collection.query(query_texts=[question], n_results=N)` — cosine distance retrieval
2. Retrieved chunks assembled into `[Source N: filename, chunk_index]\nchunk_text` context blocks
3. Agent-identity-anchored system prompt: role + persona + objective injected before context
4. GPT-4o (temp=0.3) generates a grounded answer with `[Source N]` citations
5. Returns `{answer, sources: [{filename, chunk_index, relevance, preview}], status}`

### 3.8 Memory System (`orchestrator/memory/`)

The memory subsystem spans two layers with different scopes and persistence strategies.

#### Short-term: `SharedWorkspace` (`short_term.py`)

An in-process thread-safe dictionary. Agents write to it via the `remember` tool and read via `recall`. After each agent node executes, `ws_snapshot = memory.get_workspace().to_dict()` is merged into `OrchestratorState.shared_memory`. The Compiler reads the workspace snapshot to include cross-agent findings in its synthesis.

```python
class SharedWorkspace:
    _lock: threading.RLock
    _items: dict[str, SharedMemoryItem]  # key → {value, author, tags, timestamp}

    def write(key, value, author, tags) → str
    def read(key) → str
    def search_by_tag(tag) → list[SharedMemoryItem]
    def get_summary() → str   # used in agent prompts
    def to_dict() → dict      # snapshotted into LangGraph state
```

#### Long-term: SQLite + ChromaDB (`store.py`, `long_term.py`, `embeddings.py`)

Every completed run produces an `Episode` dataclass:
```python
@dataclass
class Episode:
    episode_id: str
    task: str
    task_domain: str        # extracted by LLM during planning
    task_complexity: str    # "simple" | "moderate" | "complex"
    plan: dict
    agent_outputs: dict
    final_output: str
    known_issues: list
    success_score: float
    user_feedback: str
    metadata: dict
    tags: list[str]
    timestamp: str
```

`LongTermMemory.record_episode()` distills the episode into typed `MemoryEntry` records and indexes them:

| `memory_type` | Content | Retrieved by |
|---|---|---|
| `plan_pattern` | Successful plan structure for this task type | DA pre-planning |
| `lesson_learned` | Known issues from past runs | DA + Compiler |
| `agent_strategy` | Per-agent output patterns | Agent factory (role-scoped) |
| `user_preference` | Preferences from `/api/feedback` | DA — shapes future plans |

**Retrieval (`SemanticIndex` in `embeddings.py`):**
ChromaDB cosine similarity search over embedded memory entries. Falls back to SQLite `LIKE`-based full-text search if ChromaDB is unavailable. The `available` flag is checked at import time — the rest of the system never crashes if ChromaDB is missing.

### 3.9 Event Bus (`orchestrator/events.py`)

```python
class EventBus:
    _queue: queue.Queue          # thread-safe
    def emit(event_type, data)   # called from pipeline thread
    def get(timeout) → dict      # called from asyncio coroutine
    def is_empty() → bool
```

`set_bus(bus)` stores the bus in a thread-local, allowing any module to call `emit(...)` directly without passing the bus through the call stack. The WebSocket coroutine runs `bus.get(timeout=0.3)` in a thread pool executor to bridge the sync/async boundary:

```python
while not done.is_set() or not bus.is_empty():
    event = await loop.run_in_executor(None, lambda: bus.get(timeout=0.3))
    if event:
        await ws.send_json(event)
```

### 3.10 API Layer (`api/app.py`)

FastAPI with two connection modes:

**REST (`POST /api/run`):** Runs the pipeline in a thread pool executor and returns the complete result. Used for testing and integrations that don't support WebSocket.

**WebSocket (`/ws`):** Spawns a daemon thread for the pipeline, bridges events to the WebSocket coroutine via the EventBus. Sessions (including agent outputs and chat histories) are stored in `_sessions` with 1-hour TTL and LRU eviction at 20 sessions.

**Input validation (Pydantic `@field_validator`):**
- `TaskRequest.task`: 3–10,000 chars
- `ChatRequest.message`: non-empty, ≤4,000 chars
- `FeedbackRequest.score`: 0.0–10.0
- `RAGQueryRequest.question`: non-empty, ≤2,000 chars

**Error handling:** All endpoints use `raise HTTPException(status_code=N, detail=...)` for structured error responses. All exception handlers log via `logging.getLogger("hivemind.api")`.

**File access controls (`GET /api/files/{filename}`):**
- `os.path.basename()` strips path components
- `os.path.realpath()` comparison prevents symlink traversal
- Extension allowlist: `.txt`, `.md`, `.json`, `.csv`, `.html`, `.ics`, `.xlsx`, `.py`
- 10 MB file size cap

---

## 4. ExecutionPlan Schema

The DA produces this JSON structure, validated by the EA:

```json
{
  "task_analysis": {
    "domain": "marketing",
    "complexity": "moderate",
    "key_objectives": ["..."],
    "success_criteria": ["..."]
  },
  "agents": [
    {
      "id": "agent_1",
      "role": "Market Research Analyst",
      "persona": "Expert in competitive intelligence...",
      "objective": "Identify top 5 competitors and their pricing",
      "model_tier": "BALANCED",
      "agent_type": "standard",
      "parallel_group": 1,
      "depends_on": [],
      "tools_needed": [
        {
          "name": "search_competitors",
          "description": "Search the web for competitor pricing pages",
          "parameters": [{"name": "query", "type": "str"}],
          "returns": "str"
        }
      ],
      "expected_output": "Structured comparison of competitor pricing"
    }
  ],
  "execution_strategy": {
    "total_agents": 3,
    "parallel_groups": 2,
    "critical_path": ["agent_1", "agent_3"]
  }
}
```

The `parallel_group` integer determines display grouping in the frontend, but actual parallelism is derived entirely from the `depends_on` graph structure.

---

## 5. Concurrency Model

HiveMind mixes three concurrency contexts:

| Layer | Mechanism | Usage |
|---|---|---|
| FastAPI | `asyncio` coroutines | WebSocket I/O, REST endpoints |
| Pipeline | `threading.Thread` (daemon) | Synchronous LangGraph execution |
| Tool Forge | `ThreadPoolExecutor` (max 6) | Parallel tool generation |
| LangGraph | Internal threading | Parallel agent node execution |
| Event Bridge | `loop.run_in_executor` | Sync queue → async WebSocket |

The pipeline runs in a daemon thread so it doesn't block the asyncio event loop. The EventBus queue is the only shared state between the pipeline thread and the WebSocket coroutine.

---

## 6. Persistence Layout

```
data/
├── hivemind_memory.db        SQLite — episodes + memory_entries + preferences
└── hivemind_vectors/         ChromaDB persistent client
    ├── chroma.sqlite3
    └── [collection UUIDs]/   One per domain type

data/hivemind_rag/            ChromaDB RAG collections
    └── rag_{agent_id}/       One per agent

output/                       Generated files (sandboxed)
    ├── email_draft_*.txt
    ├── form_*.html
    ├── kanban_*.html
    ├── *.csv / *.xlsx
    └── calendar_event_*.ics
```

SQLite tables:
- `episodes` — full run records, JSON-serialised plan + agent_outputs
- `memory_entries` — distilled learnings (typed, tagged, timestamped)
- `preferences` — simple key-value pairs from user feedback

---

## 7. Evaluation

### `evaluate.py` — Single-Task Comparison

Runs two execution paths on the same task and prints a side-by-side comparison:

**Baseline path (`direct_llm_call`):**
```python
model = ChatOpenAI(model=PLANNER_MODEL, api_key=OPENAI_API_KEY, temperature=0.7)
response = model.invoke([SystemMessage(...), HumanMessage(content=task)])
```

**HiveMind path (`hivemind_call`):**
```python
result = run_task(task)
# Internally: quick_check → debate → forge → agents → compile
```

**Metrics reported:**
- Wall-clock time (total + per-phase: debate / forge / execute)
- Output length (chars)
- Agents spawned, tools generated
- Known issues flagged by the Compiler
- Time multiplier and length ratio

Results saved to `output/evaluate_results.json`.

### `run_benchmark.py` — Four-Task Suite

Runs both paths across 4 task categories and produces a summary table:

| Category | What it measures |
|---|---|
| Mathematical reasoning | Token efficiency on verifiable tasks |
| Code generation | Plan quality for implementation tasks |
| Complex software design | Coherence on architecture decisions |
| Multi-domain research | Plan structural error detection |

For each task the comparison captures: time multiplier, output length ratio, and issues caught by the Compiler. All results saved to `output/benchmark_results.json`.

### Known Baseline Failures (Caught by Debate)

On multi-domain research tasks, single-pass decomposition consistently produces:
- **Missing roles** — competitive analysis with no competitor-identification agent
- **Incorrect dependency ordering** — report-writing agent scheduled before its research dependency
- **Redundant agents** — two agents with near-identical objectives assigned to the same domain
- **Vague tool specs** — tool descriptions like "do research" that produce non-functional forge code

The EA catches all of these in round 1 before a single execution token is spent.

---

## 8. Competitive Positioning

| System | Approach | Specific gap HiveMind fills |
|---|---|---|
| LangGraph | Graph-based orchestration | No adversarial debate; new templates still require predefined patterns |
| CrewAI | Role-based predefined crews | Static roster; no plan validation; no shared memory broker |
| AutoGen v0.4 | Conversational group chat | Unstructured — no confidence-scored approval gate |
| Reflexion | Same-agent self-critique | Shares proposer's biases; not truly adversarial |
| MetaGPT | Software-dev multi-agent | Domain-specific; no general-purpose spawning |

**HiveMind is the only system combining:**
- Adversarial debate as a mandatory pre-execution gate
- Plan-driven dynamic agent creation (agents don't exist until the plan is approved)
- Runtime LLM tool synthesis with AST safety validation
- Context-aware shared workspace (intra-run) + cross-run episodic learning
- Per-agent RAG document intelligence
- Full-stack real-time execution transparency over WebSocket
- MCP protocol interoperability

---

## 9. Extension Points

### Adding a new built-in capability

1. Implement the function in `orchestrator/capabilities.py`
2. Add it to `CAPABILITY_NAMESPACE` at the bottom of that file
3. It becomes available to all forged tools automatically (injected into `exec()` namespace)
4. Add it to `_TOOL_MAP` in `quick_actions.py` if it should be usable as a quick action
5. Document it in `QUICK_DETECT_PROMPT` so the classifier knows it exists

### Adding a new integration

1. Implement with graceful degradation in `orchestrator/integrations.py`
2. Import and add to `_TOOL_MAP` in `quick_actions.py`
3. Add to `QUICK_DETECT_PROMPT`

### Adding an MCP server

1. Add entry to `mcp_servers.json`:
```json
{
  "my-server": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-my-server"]
  }
}
```
2. Set `MCP_CONFIG_PATH=mcp_servers.json` in `.env`
3. All tools from the server are automatically loaded and added to every agent

### Changing models

Edit `orchestrator/config.py`. Model tiers are per-agent and specified in the ExecutionPlan. The DA, EA, Forge, and Compiler can each use a different model.

---

## 10. Known Limitations

| Limitation | Location | Severity | Notes |
|---|---|---|---|
| Blocking `graph.invoke()` | `pipeline.py` | Low | Mitigated by `run_in_executor` in API layer |
| No persistent session state | `api/app.py` | Medium | Sessions are in-memory; cleared on restart |
| No authentication | `api/app.py` | Medium | CORS allows `*`; suitable for demos, not production |
| `exec()` without timeout | `tool_forge.py` | Low | Forged code with infinite loops block the forge thread |
| New event loop per MCP call | `mcp_client.py` | Low | `asyncio.new_event_loop()` per call; not ideal under high load |
| ChromaDB optional | `embeddings.py` | Low | Degrades silently to SQLite full-text; semantic search unavailable |
| Agent context truncation | `agent_factory.py` | Low | Dependency outputs capped at 8,000 chars |
