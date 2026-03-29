/* ═══════════════════════════════════════════════════════════════════
   YCONIC Frontend — WebSocket-driven reactive UI
   ═══════════════════════════════════════════════════════════════════ */

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// ── State ──────────────────────────────────────────────────────────
const state = {
    phase: "idle",        // idle | debate | forge | execute | compile | done | error
    task: "",
    ws: null,
    logCount: 0,
    agents: [],           // agent specs from plan
    agentStatuses: {},    // {agent_id: "waiting"|"running"|"done"|"error"}
    agentOutputs: {},     // {agent_id: preview}
    agentStreams: {},      // {agent_id: "streaming text so far"}
    sessionId: null,       // for interactive chat
    chatAgentId: null,     // currently chatting with
};

// ── Init ───────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    // Submit
    $("#submitBtn").addEventListener("click", handleSubmit);
    $("#taskInput").addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
    });

    // Chips
    $$(".chip").forEach((chip) =>
        chip.addEventListener("click", () => {
            $("#taskInput").value = chip.dataset.task;
            $("#taskInput").focus();
        })
    );

    // Log toggle
    $("#logToggle").addEventListener("click", () => $("#logPanel").classList.add("open"));

    // File refresh
    $("#refreshFiles").addEventListener("click", loadOutputFiles);

    // Chat modal
    $("#chatClose").addEventListener("click", closeChat);
    $("#chatSend").addEventListener("click", sendChat);
    $("#chatInput").addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendChat();
    });
    $("#chatModal").addEventListener("click", (e) => {
        if (e.target === $("#chatModal")) closeChat();
    });
    $("#logClose").addEventListener("click", () => $("#logPanel").classList.remove("open"));
});

// ── Submit ─────────────────────────────────────────────────────────
function handleSubmit() {
    const task = $("#taskInput").value.trim();
    if (!task || state.phase !== "idle") return;

    state.task = task;
    $("#submitBtn").disabled = true;
    setStatus("processing", "Connecting...");

    // Show pipeline, hide hero
    $("#heroSection").classList.add("hidden");
    $("#pipelineSection").classList.remove("hidden");
    $("#logToggle").classList.remove("hidden");
    $("#taskBarText").textContent = task;

    connectWS(task);
}

// ── WebSocket ──────────────────────────────────────────────────────
function connectWS(task) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    state.ws = ws;

    ws.onopen = () => {
        ws.send(JSON.stringify({ task }));
        setStatus("processing", "Running...");
    };

    ws.onmessage = (evt) => {
        try {
            const event = JSON.parse(evt.data);
            handleEvent(event);
        } catch (e) {
            console.error("WS parse error", e);
        }
    };

    ws.onerror = () => {
        setStatus("error", "Connection error");
        addLog("error", "WebSocket error");
    };

    ws.onclose = () => {
        if (state.phase !== "done" && state.phase !== "error") {
            setStatus("error", "Disconnected");
        }
    };
}

// ── Event router ───────────────────────────────────────────────────
function handleEvent(event) {
    const { type, data } = event;

    switch (type) {
        case "pipeline_start":
            setPhase("debate");
            if (data.session_id) state.sessionId = data.session_id;
            addLog("info", `Pipeline started: ${data.task?.substring(0, 60)}...`);
            break;

        case "debate_start":
            setPhase("debate");
            addLog("info", `Debate starting (max ${data.max_rounds} rounds)`);
            break;

        case "debate_da_response":
            renderDAMessage(data);
            addLog("info", `DA plan: ${data.plan?.agents?.length || "?"} agents (round ${data.round})`);
            break;

        case "debate_eval_response":
            renderEvalMessage(data);
            addLog(data.approved ? "success" : "warn",
                `Evaluator: ${data.score}/10 — ${data.verdict}`);
            break;

        case "debate_complete":
            $("#debateBadge").textContent = data.approved ? "Approved" : "Max rounds";
            $("#debateBadge").classList.add("success");
            addLog("success", `Plan approved (${data.rounds} round${data.rounds > 1 ? "s" : ""})`);
            break;

        case "forge_start":
            setPhase("forge");
            addLog("info", `Forging ${data.total_specs} tool(s)...`);
            break;

        case "forge_tool_start":
            addForgeCard(data, "forging");
            addLog("info", `Forging: ${data.tool_name}`);
            break;

        case "forge_tool_done":
            updateForgeCard(data.tool_name, data.success ? "created" : "failed");
            addLog(data.success ? "success" : "error",
                `${data.tool_name}: ${data.success ? "created" : "FAILED"}`);
            break;

        case "forge_complete":
            $("#forgeBadge").textContent = `${data.total_tools} tools`;
            $("#forgeBadge").classList.add("success");
            addLog("success", `Forge complete: ${data.total_tools} tool(s)`);
            break;

        case "agents_created":
            setPhase("execute");
            state.agents = data.agents || [];
            state.agents.forEach((a) => (state.agentStatuses[a.id] = "waiting"));
            renderAgentCards();
            $("#agentsBadge").textContent = `${state.agents.length} agents`;
            addLog("success", `${state.agents.length} agent(s) created`);
            break;

        case "graph_built":
            addLog("info", `Graph built: ${data.total_nodes} nodes`);
            break;

        case "agent_start":
            state.agentStatuses[data.agent_id] = "running";
            updateAgentCard(data.agent_id);
            addLog("info", `${data.role} started`);
            break;

        case "agent_done":
            state.agentStatuses[data.agent_id] = "done";
            state.agentOutputs[data.agent_id] = data.output_preview;
            updateAgentCard(data.agent_id);
            addLog("success", `${data.role} done`);
            break;

        case "agent_error":
            state.agentStatuses[data.agent_id] = "error";
            updateAgentCard(data.agent_id);
            addLog("error", `${data.role} error: ${data.error?.substring(0, 80)}`);
            break;

        case "agent_token":
            appendAgentStream(data.agent_id, data.token);
            break;

        case "agent_tool_call":
            appendAgentStream(data.agent_id, `\n[calling ${data.tool_name}...]\n`, true);
            addLog("info", `${data.agent_id} calling ${data.tool_name}`);
            break;

        case "agent_tool_result":
            appendAgentStream(data.agent_id, `[done]\n`, true);
            break;

        case "compile_start":
            setPhase("compile");
            addLog("info", "Compiling final output...");
            break;

        case "compile_done":
            addLog("success", "Compilation done");
            break;

        case "pipeline_done":
            setPhase("done");
            if (data.session_id) state.sessionId = data.session_id;
            renderOutput(data);
            addLog("success", "Pipeline complete!");
            break;

        case "pipeline_error":
            setPhase("error");
            setStatus("error", "Error");
            addLog("error", data.error || "Unknown error");
            break;

        default:
            addLog("info", type);
    }
}

// ── Phase management ───────────────────────────────────────────────
function setPhase(phase) {
    state.phase = phase;

    const progressMap = {
        debate: 15, forge: 35, execute: 60, compile: 85, done: 100, error: 0,
    };
    $("#progressFill").style.width = (progressMap[phase] || 0) + "%";

    // Update phase dots
    const order = ["debate", "forge", "execute", "compile"];
    const idx = order.indexOf(phase);
    $$(".phase-item").forEach((el) => {
        const p = el.dataset.phase;
        const pi = order.indexOf(p);
        el.classList.remove("active", "done");
        if (pi < idx || phase === "done") el.classList.add("done");
        else if (pi === idx) el.classList.add("active");
    });

    // Show / hide containers
    const containers = {
        debate: "debateContainer",
        forge: "forgeContainer",
        execute: "agentsContainer",
        compile: "executionContainer",
        done: "outputContainer",
    };

    // Show current and keep previous visible
    Object.entries(containers).forEach(([p, id]) => {
        const el = $(`#${id}`);
        if (!el) return;
        const pi = order.indexOf(p);
        if (pi <= idx || phase === "done") el.classList.remove("hidden");
    });

    // Status
    const statusMap = {
        debate: "Planning...",
        forge: "Forging tools...",
        execute: "Agents running...",
        compile: "Compiling...",
        done: "Complete",
        error: "Error",
    };
    if (phase === "done") setStatus("done", "Complete");
    else if (phase === "error") setStatus("error", "Error");
    else setStatus("processing", statusMap[phase] || "Running...");
}

function setStatus(type, text) {
    const dot = $("#statusDot");
    dot.className = "status-dot";
    if (type === "processing") dot.classList.add("processing");
    else if (type === "error") dot.classList.add("error");
    $("#statusLabel").textContent = text;
}

// ── Debate rendering ───────────────────────────────────────────────
function renderDAMessage(data) {
    const container = $("#debateMessages");
    const agents = data.plan?.agents || [];

    const agentChips = agents
        .map((a) => `<span class="agent-chip">${a.role}</span>`)
        .join("");

    const msg = document.createElement("div");
    msg.className = "debate-msg da";
    msg.innerHTML = `
        <div class="debate-msg-header">
            <span class="debate-msg-role">Dynamic Agent</span>
            <span class="debate-msg-round">Round ${data.round}</span>
        </div>
        <div>Proposed <strong>${agents.length}</strong> agent(s) for the task.</div>
        <div class="debate-agents-preview">${agentChips}</div>
    `;
    container.appendChild(msg);
    msg.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderEvalMessage(data) {
    const container = $("#debateMessages");
    const scoreClass = data.score >= 7 ? "high" : data.score >= 5 ? "mid" : "low";

    let issuesHTML = "";
    if (data.issues?.length) {
        issuesHTML = `<div class="debate-issues">${data.issues
            .map(
                (i) =>
                    `<div class="debate-issue">
                        <span class="issue-sev ${i.severity || ""}">${i.severity || "?"}</span>
                        <span>${i.description || ""}</span>
                    </div>`
            )
            .join("")}</div>`;
    }

    const msg = document.createElement("div");
    msg.className = "debate-msg evaluator";
    msg.innerHTML = `
        <div class="debate-msg-header">
            <span class="debate-msg-role">Evaluator</span>
            <span class="debate-msg-round">Round ${data.round}</span>
            <span class="debate-score ${scoreClass}">${data.score}/10</span>
        </div>
        <div><strong>${data.verdict}</strong></div>
        ${issuesHTML}
    `;
    container.appendChild(msg);
    msg.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Forge rendering ────────────────────────────────────────────────
function addForgeCard(data, status) {
    const grid = $("#forgeGrid");
    const icon = status === "forging" ? "\u2692" : status === "created" ? "\u2713" : "\u2717";
    const statusLabel = status === "forging" ? "Generating code..." : status === "created" ? "Ready" : "Failed";

    const card = document.createElement("div");
    card.className = `forge-card ${status} stagger-${grid.children.length % 8 + 1}`;
    card.id = `forge-${data.tool_name}`;
    card.innerHTML = `
        <div class="forge-card-header">
            <div class="forge-card-icon">${icon}</div>
            <div class="forge-card-name">${escapeHTML(data.tool_name)}</div>
            <div class="forge-card-status-label ${status}">${statusLabel}</div>
        </div>
        <div class="forge-card-desc">${escapeHTML(data.description || "Dynamic tool")}</div>
        <div class="forge-card-detail">
            <span class="forge-card-detail-label">For agent:</span>
            <span class="forge-card-detail-value">${escapeHTML(data.agent_id || "?")}</span>
        </div>
        <div class="forge-card-badge">
            <span class="tool-tag">LLM-generated Python</span>
            <span class="tool-tag">Runtime compiled</span>
        </div>
    `;
    grid.appendChild(card);
}

function updateForgeCard(toolName, status) {
    const card = $(`#forge-${toolName}`);
    if (!card) return;
    card.className = `forge-card ${status}`;
    const icon = card.querySelector(".forge-card-icon");
    if (icon) icon.textContent = status === "created" ? "\u2713" : "\u2717";
}

// ── Agent rendering ────────────────────────────────────────────────
function renderAgentCards() {
    const grid = $("#agentsGrid");
    grid.innerHTML = "";

    state.agents.forEach((agent, i) => {
        const status = state.agentStatuses[agent.id] || "waiting";
        const toolTags = (agent.tools || [])
            .map((t) => `<span class="tool-tag">${t}</span>`)
            .join("");

        const depsText = (agent.depends_on || []).length
            ? `<div class="agent-card-row"><span class="agent-card-label">Depends on:</span> <span>${agent.depends_on.join(", ")}</span></div>`
            : "";

        const card = document.createElement("div");
        card.className = `agent-card ${status} stagger-${(i % 8) + 1}`;
        card.id = `agent-${agent.id}`;
        card.innerHTML = `
            <div class="agent-card-header">
                <div class="agent-card-role">${escapeHTML(agent.role || agent.id)}</div>
                <div class="agent-card-status ${status}">${status}</div>
            </div>
            <div class="agent-card-id">${escapeHTML(agent.id)}</div>
            <div class="agent-card-section">
                <div class="agent-card-label">Persona</div>
                <div class="agent-card-persona">${escapeHTML(agent.persona || "Autonomous AI agent")}</div>
            </div>
            <div class="agent-card-section">
                <div class="agent-card-label">Objective</div>
                <div class="agent-card-objective">${escapeHTML(agent.objective || "")}</div>
            </div>
            <div class="agent-card-section">
                <div class="agent-card-label">Tools (${(agent.tools || []).length})</div>
                <div class="agent-card-tools">${toolTags || '<span class="text-faint">None</span>'}</div>
            </div>
            ${depsText}
            <div class="agent-card-footer">
                <span class="agent-card-meta-item">
                    <span class="agent-card-label">Group</span> ${agent.parallel_group || "?"}
                </span>
                <span class="agent-card-meta-item">
                    <span class="agent-card-label">Model</span> ${agent.model_tier || "BALANCED"}
                </span>
            </div>
        `;

        // Click to expand/collapse details
        card.addEventListener("click", () => card.classList.toggle("expanded"));
        grid.appendChild(card);
    });

    // Clone for execution grid
    const execGrid = $("#executionGrid");
    execGrid.innerHTML = "";
    state.agents.forEach((agent, i) => {
        const status = state.agentStatuses[agent.id] || "waiting";
        const card = document.createElement("div");
        card.className = `agent-card ${status} stagger-${(i % 8) + 1}`;
        card.id = `agent-${agent.id}`;
        card.innerHTML = `
            <div class="agent-card-header">
                <div class="agent-card-role">${escapeHTML(agent.role || agent.id)}</div>
                <div class="agent-card-status ${status}">${status}</div>
            </div>
            <div class="agent-card-persona-short">${escapeHTML((agent.persona || "").substring(0, 80))}${(agent.persona || "").length > 80 ? "..." : ""}</div>
        `;
        execGrid.appendChild(card);
    });
}

function updateAgentCard(agentId) {
    const status = state.agentStatuses[agentId] || "waiting";

    // Update in both grids
    [`#agent-${agentId}`, `#executionGrid #agent-${agentId}`].forEach((sel) => {
        // Try both containers
        document.querySelectorAll(`[id="agent-${agentId}"]`).forEach((card) => {
            card.className = `agent-card ${status}`;
            const badge = card.querySelector(".agent-card-status");
            if (badge) {
                badge.className = `agent-card-status ${status}`;
                badge.textContent = status;
            }

            // Add output preview if done
            if (status === "done" && state.agentOutputs[agentId]) {
                if (!card.querySelector(".agent-output-preview")) {
                    const preview = document.createElement("div");
                    preview.className = "agent-output-preview";
                    preview.textContent = state.agentOutputs[agentId].substring(0, 200) + "...";
                    card.appendChild(preview);
                }
            }
        });
    });

    // Update progress during execution
    const total = state.agents.length;
    const doneCount = Object.values(state.agentStatuses).filter((s) => s === "done" || s === "error").length;
    const pct = 45 + Math.round((doneCount / Math.max(total, 1)) * 40);
    $("#progressFill").style.width = pct + "%";
    $("#execBadge").textContent = `${doneCount}/${total} done`;
}

// ── Output rendering ───────────────────────────────────────────────
function renderOutput(data) {
    // ── Render agent work products ─────────────────────────────────
    const agentOutputs = data.agent_outputs || {};
    if (Object.keys(agentOutputs).length > 0) {
        $("#agentOutputsContainer").classList.remove("hidden");
        $("#outputsBadge").textContent = `${Object.keys(agentOutputs).length} agents`;

        const list = $("#agentOutputsList");
        list.innerHTML = "";

        Object.entries(agentOutputs).forEach(([id, info]) => {
            const item = document.createElement("div");
            item.className = "agent-output-item";
            item.innerHTML = `
                <div class="agent-output-header">
                    <div>
                        <span class="agent-output-role">${escapeHTML(info.role || id)}</span>
                        <span class="agent-output-id">${escapeHTML(id)}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <button class="agent-output-chat-btn" data-id="${escapeHTML(id)}" data-role="${escapeHTML(info.role || id)}">Chat</button>
                        <span class="agent-output-toggle">&#9660;</span>
                    </div>
                </div>
                <div class="agent-output-body">
                    <div class="agent-output-content">${escapeHTML(info.output || "No output")}</div>
                </div>
            `;
            item.querySelector(".agent-output-header").addEventListener("click", (e) => {
                if (e.target.classList.contains("agent-output-chat-btn")) return;
                item.classList.toggle("open");
            });
            item.querySelector(".agent-output-chat-btn").addEventListener("click", (e) => {
                e.stopPropagation();
                openChat(e.target.dataset.id, e.target.dataset.role);
            });
            list.appendChild(item);
        });
    }

    // ── Load generated files ───────────────────────────────────────
    loadOutputFiles();

    // ── Render final deliverable ───────────────────────────────────
    const raw = data.final_output || "No output produced.";
    const html = typeof marked !== "undefined" ? marked.parse(raw) : raw.replace(/\n/g, "<br>");
    $("#outputContent").innerHTML = html;

    // Stats
    const meta = data.metadata || {};
    const issues = data.known_issues || [];
    $("#outputMeta").innerHTML = `
        <div class="stat">
            <div class="stat-value">${meta.total_time_s || "?"}s</div>
            <div class="stat-label">Total Time</div>
        </div>
        <div class="stat">
            <div class="stat-value">${meta.total_agents || "?"}</div>
            <div class="stat-label">Agents</div>
        </div>
        <div class="stat">
            <div class="stat-value">${meta.total_tools || "?"}</div>
            <div class="stat-label">Tools Forged</div>
        </div>
        <div class="stat">
            <div class="stat-value">${meta.debate_time_s || "?"}s</div>
            <div class="stat-label">Debate</div>
        </div>
        <div class="stat">
            <div class="stat-value">${meta.exec_time_s || "?"}s</div>
            <div class="stat-label">Execution</div>
        </div>
        ${issues.length ? `<div class="stat">
            <div class="stat-value">${issues.length}</div>
            <div class="stat-label">Issues</div>
        </div>` : ""}
    `;

    // Scroll to agent outputs first
    if (Object.keys(agentOutputs).length > 0) {
        $("#agentOutputsContainer").scrollIntoView({ behavior: "smooth" });
    } else {
        $("#outputContainer").scrollIntoView({ behavior: "smooth" });
    }

    // Re-enable submit for another run
    $("#submitBtn").disabled = false;
}

// ── File browser ───────────────────────────────────────────────────
async function loadOutputFiles() {
    try {
        const resp = await fetch("/api/files");
        const data = await resp.json();
        const files = data.files || [];

        if (files.length === 0) return;

        $("#filesContainer").classList.remove("hidden");
        const list = $("#filesList");
        list.innerHTML = "";

        files.forEach((file) => {
            const sizeStr = file.size > 1024
                ? `${(file.size / 1024).toFixed(1)} KB`
                : `${file.size} B`;

            const item = document.createElement("div");
            item.className = "file-item";
            item.innerHTML = `
                <div class="file-item-name">
                    <span class="file-item-icon">\u{1F4C4}</span>
                    <span>${escapeHTML(file.name)}</span>
                </div>
                <span class="file-item-size">${sizeStr}</span>
            `;
            item.addEventListener("click", () => loadFilePreview(file.name));
            list.appendChild(item);
        });
    } catch (e) {
        console.error("Failed to load files", e);
    }
}

async function loadFilePreview(filename) {
    try {
        const resp = await fetch(`/api/files/${encodeURIComponent(filename)}`);
        const data = await resp.json();

        const preview = $("#filePreview");
        preview.classList.remove("hidden");
        preview.innerHTML = `
            <div class="file-preview-header">
                <span>\u{1F4C4} ${escapeHTML(data.name)}</span>
                <span class="file-item-size">${data.size} chars</span>
            </div>
            <div class="file-preview-content">${escapeHTML(data.content)}</div>
        `;
        preview.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (e) {
        console.error("Failed to load file", e);
    }
}

// ── Activity log ───────────────────────────────────────────────────
function addLog(level, text) {
    state.logCount++;
    $("#logCount").textContent = state.logCount;

    const container = $("#logMessages");
    const now = new Date().toLocaleTimeString("en-US", { hour12: false });

    const entry = document.createElement("div");
    entry.className = "log-entry";
    entry.innerHTML = `<span class="log-ts">${now}</span> <span class="log-type ${level}">${level}</span> ${escapeHTML(text)}`;
    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
}

// ── Agent streaming ────────────────────────────────────────────────
function appendAgentStream(agentId, text, isToolEvent = false) {
    if (!state.agentStreams[agentId]) state.agentStreams[agentId] = "";
    state.agentStreams[agentId] += text;

    // Find stream container in execution grid
    const cards = document.querySelectorAll(`[id="agent-${agentId}"]`);
    cards.forEach((card) => {
        let stream = card.querySelector(".agent-stream");
        if (!stream) {
            stream = document.createElement("div");
            stream.className = "agent-stream";
            card.appendChild(stream);
        }

        if (isToolEvent) {
            stream.innerHTML += `<span class="tool-call">${escapeHTML(text)}</span>`;
        } else {
            // Remove old cursor, add text, add new cursor
            const cursor = stream.querySelector(".cursor");
            if (cursor) cursor.remove();
            stream.appendChild(document.createTextNode(text));
            const c = document.createElement("span");
            c.className = "cursor";
            stream.appendChild(c);
        }

        stream.scrollTop = stream.scrollHeight;
    });
}

// ── Interactive chat ───────────────────────────────────────────────
function openChat(agentId, role) {
    state.chatAgentId = agentId;
    $("#chatAgentRole").textContent = role;
    $("#chatAgentId").textContent = agentId;
    $("#chatMessages").innerHTML = "";
    $("#chatInput").value = "";
    $("#chatModal").classList.remove("hidden");

    // Add welcome message
    addChatMsg("agent", `Hi! I'm the ${role}. I just completed my work on this task. Ask me anything about my findings, or request changes and deeper analysis.`);
    $("#chatInput").focus();
}

function closeChat() {
    $("#chatModal").classList.add("hidden");
    state.chatAgentId = null;
}

function addChatMsg(role, text) {
    const container = $("#chatMessages");
    const msg = document.createElement("div");
    msg.className = `chat-msg ${role}`;
    if (typeof marked !== "undefined" && role === "agent") {
        msg.innerHTML = marked.parse(text);
    } else {
        msg.textContent = text;
    }
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
    return msg;
}

async function sendChat() {
    const input = $("#chatInput");
    const message = input.value.trim();
    if (!message || !state.chatAgentId || !state.sessionId) return;

    input.value = "";
    addChatMsg("user", message);

    // Show typing indicator
    const typing = addChatMsg("agent", "Thinking...");
    typing.classList.add("typing");
    $("#chatSend").disabled = true;

    try {
        const resp = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: state.sessionId,
                agent_id: state.chatAgentId,
                message: message,
            }),
        });
        const data = await resp.json();

        // Replace typing with actual response
        typing.remove();

        if (data.error) {
            addChatMsg("agent", `Error: ${data.error}`);
        } else {
            addChatMsg("agent", data.response || "No response.");
        }
    } catch (e) {
        typing.remove();
        addChatMsg("agent", `Connection error: ${e.message}`);
    }

    $("#chatSend").disabled = false;
    input.focus();
}

// ── Helpers ────────────────────────────────────────────────────────
function escapeHTML(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}
