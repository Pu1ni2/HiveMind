/* ===================================================================
   HIVEMIND — Dashboard Frontend
   =================================================================== */

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const state = {
    phase: "idle",
    task: "",
    ws: null,
    logCount: 0,
    agents: [],
    agentStatuses: {},
    agentOutputs: {},
    agentStreams: {},
    sessionId: null,
    chatAgentId: null,
    timerStart: null,
    timerInterval: null,
    currentPage: "pipeline",
};

// ── Init ──────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    // Sidebar navigation
    $$(".sb-item").forEach((btn) =>
        btn.addEventListener("click", () => navigateTo(btn.dataset.page))
    );

    // Submit
    $("#submitBtn").addEventListener("click", handleSubmit);
    $("#taskInput").addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
    });

    // Chips
    $$(".chip").forEach((c) =>
        c.addEventListener("click", () => { $("#taskInput").value = c.dataset.task; $("#taskInput").focus(); })
    );

    // Theme toggles
    $$("#themeToggle, #themeToggle2").forEach((btn) =>
        btn.addEventListener("click", toggleTheme)
    );

    // Restore saved theme
    const saved = localStorage.getItem("hivemind-theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);

    // Log
    $("#logToggle").addEventListener("click", () => $("#logPanel").classList.add("open"));
    $("#logClose").addEventListener("click", () => $("#logPanel").classList.remove("open"));

    // Files
    $("#refreshFiles").addEventListener("click", loadOutputFiles);

    // Chat
    $("#chatClose").addEventListener("click", closeChat);
    $("#chatSend").addEventListener("click", sendChat);
    $("#chatInput").addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });
    $("#chatModal").addEventListener("click", (e) => { if (e.target === $("#chatModal")) closeChat(); });

    // New task
    $("#newTaskBtn").addEventListener("click", resetForNewTask);

    // Error toast
    $("#errorToastClose").addEventListener("click", () => $("#errorToast").classList.add("hidden"));

    // Agent detail / RAG
    $("#agentDetailBack").addEventListener("click", () => navigateTo("agents"));
    $("#ragSend").addEventListener("click", sendRagQuery);
    $("#ragInput").addEventListener("keydown", (e) => { if (e.key === "Enter") sendRagQuery(); });

    // File upload (click + drag-and-drop)
    const dropZone = $("#ragDropZone");
    const fileInput = $("#ragFileInput");
    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
        Array.from(e.target.files).forEach(f => uploadRagFile(f));
        fileInput.value = "";
    });
    dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault(); dropZone.classList.remove("dragover");
        Array.from(e.dataTransfer.files).forEach(f => uploadRagFile(f));
    });
});

// ── Theme ─────────────────────────────────────────────────────────
function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("hivemind-theme", next);
}

// ── Navigation ────────────────────────────────────────────────────
function navigateTo(page) {
    state.currentPage = page;
    $$(".sb-item").forEach((b) => b.classList.toggle("active", b.dataset.page === page));
    $$(".page").forEach((p) => p.classList.toggle("active", p.id === `page-${page}`));

    if (page === "files") loadOutputFiles();
    if (page === "memory") loadMemory();
}

// ── Submit ────────────────────────────────────────────────────────
function handleSubmit() {
    const task = $("#taskInput").value.trim();
    if (!task || (state.phase !== "idle" && state.phase !== "done" && state.phase !== "error")) return;

    state.task = task;
    state.phase = "idle"; state.logCount = 0;
    state.agents = []; state.agentStatuses = {}; state.agentOutputs = {}; state.agentStreams = {}; state.sessionId = null;

    $("#submitBtn").disabled = true;
    setStatus("processing", "Connecting...");

    // Reset pipeline containers
    ["quickContainer","debateContainer","forgeContainer","graphContainer",
     "executionContainer","outputContainer","completionCta","agentOutputsContainer"
    ].forEach((id) => $(`#${id}`).classList.add("hidden"));
    ["debateMessages","forgeGrid","executionGrid","agentsGrid","toolsGrid",
     "agentOutputsList","quickActions"
    ].forEach((id) => { const el = $(`#${id}`); if(el) el.innerHTML = ""; });
    $("#graphSvg").innerHTML = "";
    $("#logMessages").innerHTML = "";
    $("#logCount").textContent = "0";
    $$("#phaseRibbon .phase-step").forEach((s) => s.classList.remove("active","done"));
    $("#agentsPageEmpty").classList.remove("hidden");
    $("#toolsPageEmpty").classList.remove("hidden");
    $("#taskPill").textContent = task.substring(0, 80) + (task.length > 80 ? "..." : "");

    // Transition: Hero → Dashboard
    $("#heroScreen").classList.add("hidden");
    $("#dashboardScreen").classList.remove("hidden");
    navigateTo("pipeline");
    $("#logToggle").classList.remove("hidden");

    startTimer();
    connectWS(task);
}

// ── Timer ─────────────────────────────────────────────────────────
function startTimer() {
    state.timerStart = Date.now();
    $("#liveTimer").classList.remove("hidden");
    if (state.timerInterval) clearInterval(state.timerInterval);
    state.timerInterval = setInterval(() => {
        const s = Math.floor((Date.now() - state.timerStart) / 1000);
        $("#timerValue").textContent = `${Math.floor(s/60)}:${(s%60).toString().padStart(2,"0")}`;
    }, 1000);
}

function stopTimer() { if (state.timerInterval) { clearInterval(state.timerInterval); state.timerInterval = null; } }

function resetForNewTask() {
    stopTimer();
    state.phase = "idle";
    // Transition: Dashboard → Hero
    $("#dashboardScreen").classList.add("hidden");
    $("#heroScreen").classList.remove("hidden");
    $("#logToggle").classList.add("hidden");
    $("#logPanel").classList.remove("open");
    $("#liveTimer").classList.add("hidden");
    $("#submitBtn").disabled = false;
    $("#taskInput").value = "";
    $("#taskInput").focus();
    setStatus("ready", "Ready");
}

// ── WebSocket ─────────────────────────────────────────────────────
function connectWS(task) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    state.ws = ws;
    ws.onopen = () => { ws.send(JSON.stringify({ task })); setStatus("processing", "Running..."); };
    ws.onmessage = (evt) => { try { handleEvent(JSON.parse(evt.data)); } catch(e) { console.error(e); } };
    ws.onerror = () => { setStatus("error", "Error"); showError("Connection failed"); stopTimer(); };
    ws.onclose = () => { if (state.phase !== "done" && state.phase !== "error") { setStatus("error", "Disconnected"); stopTimer(); } };
}

function showError(msg) { $("#errorToastMsg").textContent = msg; $("#errorToast").classList.remove("hidden"); setTimeout(() => $("#errorToast").classList.add("hidden"), 8000); }

// ── Event router ──────────────────────────────────────────────────
function handleEvent(event) {
    const { type, data } = event;
    switch (type) {
        case "pipeline_start":
            setPhase("analyze");
            if (data.session_id) state.sessionId = data.session_id;
            addLog("info", "Pipeline started"); break;
        case "debate_start":
            setPhase("debate"); addLog("info", `Debate (max ${data.max_rounds} rounds)`); break;
        case "debate_da_response":
            renderDAMessage(data); addLog("info", `DA: ${data.plan?.agents?.length||"?"} agents`); break;
        case "debate_eval_response":
            renderEvalMessage(data); addLog(data.approved?"success":"warn", `Eval: ${data.score}/10`); break;
        case "debate_complete":
            $("#debateBadge").textContent = data.approved ? "Approved" : "Max rounds";
            $("#debateBadge").classList.add("badge-green");
            addLog("success", `Plan approved (${data.rounds} rounds)`); break;
        case "forge_start":
            setPhase("forge"); addLog("info", `Forging ${data.total_specs} tools`); break;
        case "forge_tool_start":
            addForgeCard(data, "forging"); addLog("info", `Forging: ${data.tool_name}`); break;
        case "forge_tool_done":
            updateForgeCard(data.tool_name, data.success?"created":"failed");
            addLog(data.success?"success":"error", `${data.tool_name}: ${data.success?"ready":"FAILED"}`); break;
        case "forge_complete":
            $("#forgeBadge").textContent = `${data.total_tools} tools`;
            $("#forgeBadge").classList.add("badge-green");
            addLog("success", `Forge done: ${data.total_tools} tools`); break;
        case "agents_created":
            setPhase("execute");
            state.agents = data.agents || [];
            state.agents.forEach((a) => (state.agentStatuses[a.id] = "waiting"));
            renderAgentCards(); renderGraph();
            addLog("success", `${state.agents.length} agents created`);
            updateNavBadges(); break;
        case "graph_built": addLog("info", `Graph: ${data.total_nodes} nodes`); break;
        case "agent_start":
            state.agentStatuses[data.agent_id] = "running"; updateAgentCard(data.agent_id);
            addLog("info", `${data.role} started`); break;
        case "agent_done":
            state.agentStatuses[data.agent_id] = "done";
            state.agentOutputs[data.agent_id] = data.output_preview;
            updateAgentCard(data.agent_id); addLog("success", `${data.role} done`); break;
        case "agent_error":
            state.agentStatuses[data.agent_id] = "error"; updateAgentCard(data.agent_id);
            addLog("error", `${data.role} error`); break;
        case "agent_token": appendAgentStream(data.agent_id, data.token); break;
        case "agent_tool_call":
            appendAgentStream(data.agent_id, `\n[${data.tool_name}...]\n`, true);
            addLog("info", `${data.agent_id} -> ${data.tool_name}`); break;
        case "agent_tool_result": appendAgentStream(data.agent_id, `[done]\n`, true); break;
        case "compile_start": setPhase("compile"); updateGraphNode("compiler","running"); addLog("info","Compiling..."); break;
        case "compile_done": updateGraphNode("compiler","done"); addLog("success","Compiled"); break;
        case "pipeline_done":
            setPhase("done"); stopTimer();
            if (data.session_id) state.sessionId = data.session_id;
            renderOutput(data); addLog("success","Complete!"); break;
        case "pipeline_error":
            setPhase("error"); stopTimer(); setStatus("error","Error");
            showError(data.error||"Error"); addLog("error", data.error||"Error"); break;
        case "quick_detect_start": addLog("info","Analyzing task..."); setStatus("processing","Analyzing..."); break;
        case "quick_detect_done":
            if (data.mode==="quick") { setPhase("quick"); addLog("success",`Quick: ${data.reason}`); }
            else addLog("info",`Full pipeline: ${data.reason}`); break;
        case "quick_start": setPhase("quick"); renderQuickActions(data); addLog("info",`${data.action_count} quick actions`); break;
        case "quick_action": updateQuickAction(data);
            if(data.status==="done") addLog("success",`${data.tool} done`);
            else if(data.status==="error") addLog("error",`${data.tool} failed`);
            else addLog("info",`Running: ${data.tool}`); break;
        case "quick_done": addLog("success",`Quick done (${data.results_count} results)`); break;
        case "memory_recall": addLog("info",`Memory: ${data.chars} chars recalled`); break;
        case "memory_store": addLog("info",`Memory: ${data.key} stored by ${data.author}`); break;
        case "episode_saved": addLog("success",`Episode ${data.episode_id} saved`); break;
        default: addLog("info", type);
    }
}

// ── Phase management ──────────────────────────────────────────────
function setPhase(phase) {
    state.phase = phase;
    const order = ["analyze","debate","forge","execute","compile"];
    const idx = order.indexOf(phase);
    $$("#phaseRibbon .ph").forEach((el) => {
        const p = el.dataset.phase; const pi = order.indexOf(p);
        el.classList.remove("active","done");
        if (pi < idx || phase === "done") el.classList.add("done");
        else if (pi === idx) el.classList.add("active");
    });
    if (phase === "quick") { $("#quickContainer").classList.remove("hidden"); }
    if (phase === "debate"||phase==="quick") { $("#debateContainer").classList.remove("hidden"); }
    if (phase === "forge") { $("#forgeContainer").classList.remove("hidden"); $("#debateContainer").classList.remove("hidden"); }
    if (phase === "execute") { ["debateContainer","forgeContainer","graphContainer","executionContainer"].forEach(id=>$(`#${id}`).classList.remove("hidden")); }
    if (phase === "compile") { ["debateContainer","forgeContainer","graphContainer","executionContainer"].forEach(id=>$(`#${id}`).classList.remove("hidden")); }
    if (phase === "done") { ["debateContainer","forgeContainer","graphContainer","executionContainer","outputContainer"].forEach(id=>$(`#${id}`).classList.remove("hidden")); }

    const statusMap = { analyze:"Analyzing...", debate:"Planning...", forge:"Forging...", execute:"Executing...", compile:"Compiling...", done:"Complete", error:"Error", quick:"Quick action..." };
    if (phase==="done") setStatus("done","Complete");
    else if (phase==="error") setStatus("error","Error");
    else setStatus("processing", statusMap[phase]||"Running...");
}

function setStatus(type, text) {
    const dot = $("#statusDot");
    dot.className = "status-dot";
    if (type==="processing") dot.classList.add("processing");
    else if (type==="error") dot.classList.add("error");
    else if (type==="done") dot.classList.add("done");
    $("#statusLabel").textContent = text;
}

function updateNavBadges() {
    if (state.agents.length) { $("#agentsBadgeNav").textContent = state.agents.length; $("#agentsBadgeNav").classList.remove("hidden"); }
    const toolCount = state.agents.reduce((s,a) => s + (a.tools||[]).length, 0);
    if (toolCount) { $("#toolsBadgeNav").textContent = toolCount; $("#toolsBadgeNav").classList.remove("hidden"); }
}

// ── Debate rendering ──────────────────────────────────────────────
function renderDAMessage(data) {
    const agents = data.plan?.agents||[];
    const chips = agents.map(a=>`<span class="agent-chip">${escapeHTML(a.role)}</span>`).join("");
    const msg = document.createElement("div"); msg.className = "debate-msg da";
    msg.innerHTML = `<div class="debate-msg-header"><span class="debate-msg-role">Dynamic Agent</span><span class="debate-msg-round">R${data.round}</span></div><div>Proposed <strong>${agents.length}</strong> agents</div><div class="debate-agents-preview">${chips}</div>`;
    $("#debateMessages").appendChild(msg); msg.scrollIntoView({behavior:"smooth",block:"nearest"});
}
function renderEvalMessage(data) {
    const sc = data.score>=7?"high":data.score>=5?"mid":"low";
    let issues = ""; if(data.issues?.length) issues=`<div class="debate-issues">${data.issues.map(i=>`<div class="debate-issue"><span class="issue-sev ${(i.severity||"").toLowerCase()}">${escapeHTML(i.severity||"?")}</span><span>${escapeHTML(i.description||"")}</span></div>`).join("")}</div>`;
    let strengths = ""; if(data.strengths?.length) strengths=`<div class="debate-strengths">${data.strengths.map(s=>`<span class="strength-chip">${escapeHTML(s)}</span>`).join("")}</div>`;
    const msg = document.createElement("div"); msg.className = "debate-msg evaluator";
    msg.innerHTML = `<div class="debate-msg-header"><span class="debate-msg-role">Evaluator</span><span class="debate-msg-round">R${data.round}</span><span class="debate-score ${sc}">${data.score}/10</span></div><div><strong>${escapeHTML(data.verdict)}</strong></div>${strengths}${issues}`;
    $("#debateMessages").appendChild(msg); msg.scrollIntoView({behavior:"smooth",block:"nearest"});
}

// ── Forge rendering ───────────────────────────────────────────────
function addForgeCard(data, status) {
    const card = document.createElement("div"); card.className = `forge-card ${status}`;
    card.id = `forge-${CSS.escape(data.tool_name)}`;
    card.innerHTML = `<div class="forge-card-header"><div class="forge-card-icon forge-spinner"></div><div class="forge-card-name">${escapeHTML(data.tool_name)}</div><div class="forge-card-status-label ${status}">Generating...</div></div><div class="forge-card-desc">${escapeHTML(data.description||"")}</div><div class="forge-card-badge"><span class="tool-tag">LLM-generated</span><span class="tool-tag">AST-validated</span></div>`;
    $("#forgeGrid").appendChild(card);
    // Also add to tools page
    addToolToPage(data);
}
function updateForgeCard(name, status) {
    const card = document.getElementById(`forge-${CSS.escape(name)}`); if(!card) return;
    card.className = `forge-card ${status}`;
    const icon = card.querySelector(".forge-card-icon"); if(icon){icon.classList.remove("forge-spinner"); icon.textContent = status==="created"?"\u2713":"\u2717";}
    const lbl = card.querySelector(".forge-card-status-label"); if(lbl){lbl.className=`forge-card-status-label ${status}`; lbl.textContent=status==="created"?"Ready":"Failed";}
    // Update tools page
    const tp = document.getElementById(`tool-${CSS.escape(name)}`);
    if(tp) { tp.querySelector(".forge-card-status-label").className=`forge-card-status-label ${status}`; tp.querySelector(".forge-card-status-label").textContent=status==="created"?"Ready":"Failed"; }
}
function addToolToPage(data) {
    $("#toolsPageEmpty").classList.add("hidden");
    const card = document.createElement("div"); card.className = "forge-card forging"; card.id = `tool-${CSS.escape(data.tool_name)}`;
    card.innerHTML = `<div class="forge-card-header"><div class="forge-card-icon forge-spinner"></div><div class="forge-card-name">${escapeHTML(data.tool_name)}</div><div class="forge-card-status-label forging">Generating...</div></div><div class="forge-card-desc">${escapeHTML(data.description||"")}</div><div class="forge-card-detail"><span class="forge-card-detail-label">Agent:</span> <span class="forge-card-detail-value">${escapeHTML(data.agent_id||"?")}</span></div><div class="forge-card-badge"><span class="tool-tag">LLM-generated</span><span class="tool-tag">AST-validated</span><span class="tool-tag">Runtime compiled</span></div>`;
    $("#toolsGrid").appendChild(card);
}

// ── Agent rendering ───────────────────────────────────────────────
function renderAgentCards() {
    // Agents page (full cards)
    const grid = $("#agentsGrid"); grid.innerHTML = "";
    $("#agentsPageEmpty").classList.add("hidden");
    state.agents.forEach((a) => {
        const st = state.agentStatuses[a.id]||"waiting";
        const tags = (a.tools||[]).map(t=>`<span class="tool-tag">${escapeHTML(t)}</span>`).join("");
        const card = document.createElement("div"); card.className = `agent-card ${st}`; card.id = `agents-${a.id}`;
        const typeLabel = a.agent_type && a.agent_type !== "standard" ? `<span class="badge ${a.agent_type==="rag"?"bg-blue":"bg-green"}" style="margin-left:6px;font-size:.55rem">${a.agent_type.toUpperCase()}</span>` : "";
        card.innerHTML = `<div class="agent-card-header"><div class="agent-card-role">${escapeHTML(a.role||a.id)}${typeLabel}</div><div class="agent-card-status ${st}">${st}</div></div><div class="agent-card-id">${escapeHTML(a.id)} &mdash; click to open</div><div class="agent-card-section"><div class="agent-card-label">Persona</div><div class="agent-card-persona">${escapeHTML(a.persona||"")}</div></div><div class="agent-card-section"><div class="agent-card-label">Objective</div><div class="agent-card-objective">${escapeHTML(a.objective||"")}</div></div><div class="agent-card-section"><div class="agent-card-label">Tools (${(a.tools||[]).length})</div><div class="agent-card-tools">${tags||'<span class="text-muted">None</span>'}</div></div><div class="agent-card-footer"><span class="agent-card-meta-item"><span class="agent-card-label">Group</span> ${a.parallel_group||"?"}</span><span class="agent-card-meta-item"><span class="agent-card-label">Model</span> ${a.model_tier||"BALANCED"}</span></div>`;
        card.addEventListener("click", () => openAgentDetail(a.id));
        grid.appendChild(card);
    });
    // Execution grid (compact)
    const eg = $("#executionGrid"); eg.innerHTML = "";
    state.agents.forEach((a) => {
        const st = state.agentStatuses[a.id]||"waiting";
        const card = document.createElement("div"); card.className = `agent-card ${st}`; card.id = `exec-${a.id}`;
        card.innerHTML = `<div class="agent-card-header"><div class="agent-card-role">${escapeHTML(a.role||a.id)}</div><div class="agent-card-status ${st}">${st}</div></div><div class="agent-card-persona-short">${escapeHTML((a.persona||"").substring(0,60))}</div>`;
        eg.appendChild(card);
    });
}
function updateAgentCard(agentId) {
    const st = state.agentStatuses[agentId]||"waiting";
    updateGraphNode(agentId, st);
    document.querySelectorAll(`#agents-${CSS.escape(agentId)}, #exec-${CSS.escape(agentId)}`).forEach((card) => {
        card.className = card.className.replace(/\b(waiting|running|done|error)\b/,st);
        const b = card.querySelector(".agent-card-status"); if(b){b.className=`agent-card-status ${st}`;b.textContent=st;}
        if(st==="done"&&state.agentOutputs[agentId]&&!card.querySelector(".agent-output-preview")){
            const p=document.createElement("div");p.className="agent-output-preview";p.textContent=state.agentOutputs[agentId].substring(0,150)+"...";card.appendChild(p);
        }
    });
    const total=state.agents.length; const done=Object.values(state.agentStatuses).filter(s=>s==="done"||s==="error").length;
    $("#execBadge").textContent=`${done}/${total}`;
}

// ── Quick Actions ─────────────────────────────────────────────────
function renderQuickActions(data) {
    const c=$("#quickActions"); c.innerHTML="";
    for(let i=0;i<data.action_count;i++){const d=document.createElement("div");d.className="quick-action-card waiting";d.id=`quick-action-${i}`;d.innerHTML=`<div class="quick-action-icon"><div class="forge-spinner"></div></div><div class="quick-action-info"><div class="quick-action-tool">Preparing...</div><div class="quick-action-status-text">Waiting</div></div>`;c.appendChild(d);}
}
function updateQuickAction(data) {
    const card=document.getElementById(`quick-action-${data.index}`); if(!card)return;
    const icons={send_email:"\u2709\uFE0F",send_slack:"\uD83D\uDCAC",create_calendar_event:"\uD83D\uDCC5",search_web:"\uD83D\uDD0D",create_spreadsheet:"\uD83D\uDCCA",create_kanban_board:"\uD83D\uDCCB",create_form:"\uD83D\uDCCB",save_file:"\uD83D\uDCBE"};
    card.className=`quick-action-card ${data.status}`;
    if(data.status==="running") card.querySelector(".quick-action-icon").innerHTML=`<div class="forge-spinner"></div>`;
    else if(data.status==="done") card.querySelector(".quick-action-icon").innerHTML=`<span>${icons[data.tool]||"\u26A1"}</span>`;
    else if(data.status==="error") card.querySelector(".quick-action-icon").innerHTML=`<span>\u274C</span>`;
    let params="";if(data.params){const pp=Object.entries(data.params).filter(([,v])=>v!=="***").map(([k,v])=>`<span class="quick-param">${escapeHTML(k)}: ${escapeHTML(v)}</span>`);if(pp.length)params=`<div class="quick-action-params">${pp.join("")}</div>`;}
    let preview="";if(data.status==="done"&&data.preview)preview=`<div class="quick-action-preview">${escapeHTML(data.preview.substring(0,200))}</div>`;
    card.querySelector(".quick-action-info").innerHTML=`<div class="quick-action-tool">${escapeHTML(data.tool)}</div>${params}<div class="quick-action-status-text ${data.status}">${data.status==="running"?"Running...":data.status==="done"?"Done":data.status==="error"?"Error: "+(data.error||""):data.status}</div>${preview}`;
}

// ── Graph ─────────────────────────────────────────────────────────
function renderGraph(){const svg=$("#graphSvg");svg.innerHTML="";$("#graphContainer").classList.remove("hidden");const agents=state.agents;if(!agents.length)return;const groups={};agents.forEach(a=>{const g=a.parallel_group||1;if(!groups[g])groups[g]=[];groups[g].push(a);});const keys=Object.keys(groups).map(Number).sort((a,b)=>a-b);const W=140,H=50,CG=160,RG=68;let maxPC=1;keys.forEach(k=>{if(groups[k].length>maxPC)maxPC=groups[k].length;});const cols=keys.length+2;const svgW=cols*(W+CG)+30;const svgH=Math.max(maxPC*(H+RG)+40,180);svg.setAttribute("viewBox",`0 0 ${svgW} ${svgH}`);svg.style.width="100%";svg.style.height=svgH+"px";const pos={};const sX=20,sY=svgH/2;pos["__start__"]={x:sX,y:sY};keys.forEach((gk,ci)=>{const col=ci+1;const ag=groups[gk];const cX=col*(W+CG);const tH=ag.length*H+(ag.length-1)*RG;const oY=(svgH-tH)/2;ag.forEach((a,ri)=>{pos[a.id]={x:cX,y:oY+ri*(H+RG)};});});const compCol=keys.length+1;const compX=compCol*(W+CG);const compY=svgH/2-H/2;pos["compiler"]={x:compX,y:compY};const defs=sEl("defs");defs.innerHTML=`<marker id="ah" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="rgba(129,140,248,0.4)"/></marker>`;svg.appendChild(defs);const eg=sEl("g");agents.forEach(a=>{const deps=(a.depends_on||[]).filter(d=>pos[d]);if(!deps.length)eg.appendChild(dEdge(pos["__start__"].x+30,pos["__start__"].y,pos[a.id].x,pos[a.id].y+H/2));deps.forEach(d=>eg.appendChild(dEdge(pos[d].x+W,pos[d].y+H/2,pos[a.id].x,pos[a.id].y+H/2)));});const ds=new Set();agents.forEach(a=>(a.depends_on||[]).forEach(d=>ds.add(d)));agents.forEach(a=>{if(!ds.has(a.id))eg.appendChild(dEdge(pos[a.id].x+W,pos[a.id].y+H/2,pos["compiler"].x,pos["compiler"].y+H/2));});svg.appendChild(eg);const ng=sEl("g");ng.appendChild(dSpec(pos["__start__"].x,pos["__start__"].y-14,28,28,"START","start"));agents.forEach(a=>{const p=pos[a.id];const st=state.agentStatuses[a.id]||"waiting";const g=sEl("g",{id:`graph-node-${a.id}`,class:`graph-node ${st}`,transform:`translate(${p.x},${p.y})`});g.appendChild(sEl("rect",{width:W,height:H,rx:8,ry:8,class:"graph-node-bg"}));g.appendChild(sEl("circle",{cx:12,cy:H/2,r:4,class:"graph-node-dot"}));const r=sEl("text",{x:22,y:20,class:"graph-node-role"});r.textContent=a.role?.substring(0,18)||a.id;g.appendChild(r);const m=sEl("text",{x:22,y:36,class:"graph-node-meta"});m.textContent=`${a.id} | grp ${a.parallel_group||"?"}`;g.appendChild(m);const s=sEl("text",{x:W-6,y:14,class:"graph-node-status","text-anchor":"end"});s.textContent=st;g.appendChild(s);ng.appendChild(g);});ng.appendChild(dSpec(pos["compiler"].x,pos["compiler"].y,W,H,"COMPILER","compiler","compiler"));svg.appendChild(ng);}
function dEdge(x1,y1,x2,y2){const dx=(x2-x1)*.4;return sEl("path",{d:`M${x1},${y1} C${x1+dx},${y1} ${x2-dx},${y2} ${x2},${y2}`,class:"graph-edge","marker-end":"url(#ah)"});}
function dSpec(x,y,w,h,lbl,type,id){const g=sEl("g",{class:`graph-special-node ${type}`,transform:`translate(${x},${y})`});if(id)g.id=`graph-node-${id}`;g.appendChild(sEl("rect",{width:w,height:h,rx:type==="start"?14:8,ry:type==="start"?14:8,class:"graph-special-bg"}));const t=sEl("text",{x:w/2,y:h/2+4,class:"graph-special-label","text-anchor":"middle"});t.textContent=lbl;g.appendChild(t);return g;}
function updateGraphNode(id,st){const n=document.getElementById(`graph-node-${CSS.escape(id)}`);if(!n)return;n.className.baseVal=n.className.baseVal.replace(/\b(waiting|running|done|error)\b/,"").trim()+` ${st}`;const s=n.querySelector(".graph-node-status");if(s)s.textContent=st;}
function sEl(tag,attrs){const el=document.createElementNS("http://www.w3.org/2000/svg",tag);if(attrs)Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));return el;}

// ── Output rendering ──────────────────────────────────────────────
function renderOutput(data) {
    const ao=data.agent_outputs||{};
    if(Object.keys(ao).length>0){
        $("#agentOutputsContainer").classList.remove("hidden");
        $("#outputsBadge").textContent=`${Object.keys(ao).length} agents`;
        const list=$("#agentOutputsList");list.innerHTML="";
        Object.entries(ao).forEach(([id,info])=>{
            const item=document.createElement("div");item.className="agent-output-item";
            item.innerHTML=`<div class="agent-output-header"><div><span class="agent-output-role">${escapeHTML(info.role||id)}</span><span class="agent-output-id">${escapeHTML(id)}</span></div><div style="display:flex;align-items:center;gap:6px;"><button class="agent-output-chat-btn" data-id="${escapeHTML(id)}" data-role="${escapeHTML(info.role||id)}">Chat</button><span class="agent-output-toggle">&#9660;</span></div></div><div class="agent-output-body"><div class="agent-output-content">${renderMd(info.output||"No output")}</div></div>`;
            item.querySelector(".agent-output-header").addEventListener("click",e=>{if(e.target.classList.contains("agent-output-chat-btn"))return;item.classList.toggle("open");});
            item.querySelector(".agent-output-chat-btn").addEventListener("click",e=>{e.stopPropagation();openChat(e.target.dataset.id,e.target.dataset.role);});
            list.appendChild(item);
        });
    }
    loadOutputFiles();
    const raw=data.final_output||"No output.";
    $("#outputContent").innerHTML=renderMd(raw);
    const meta=data.metadata||{};const issues=data.known_issues||[];const elapsed=state.timerStart?Math.round((Date.now()-state.timerStart)/1000):meta.total_time_s||"?";
    $("#outputMeta").innerHTML=`<div class="stat"><div class="stat-value">${elapsed}s</div><div class="stat-label">Time</div></div><div class="stat"><div class="stat-value">${meta.total_agents||"?"}</div><div class="stat-label">Agents</div></div><div class="stat"><div class="stat-value">${meta.total_tools||"?"}</div><div class="stat-label">Tools</div></div><div class="stat"><div class="stat-value">${meta.debate_time_s||"?"}s</div><div class="stat-label">Debate</div></div><div class="stat"><div class="stat-value">${meta.exec_time_s||"?"}s</div><div class="stat-label">Execution</div></div>${issues.length?`<div class="stat"><div class="stat-value">${issues.length}</div><div class="stat-label">Issues</div></div>`:""}`;
    $("#completionCta").classList.remove("hidden");
    $("#submitBtn").disabled = false;
}

// ── Agent streaming ───────────────────────────────────────────────
function appendAgentStream(agentId, text, isTool=false) {
    if(!state.agentStreams[agentId])state.agentStreams[agentId]="";state.agentStreams[agentId]+=text;
    const card=document.getElementById(`exec-${CSS.escape(agentId)}`);if(!card)return;
    let s=card.querySelector(".agent-stream");if(!s){s=document.createElement("div");s.className="agent-stream";card.appendChild(s);}
    if(isTool)s.innerHTML+=`<span class="tool-call">${escapeHTML(text)}</span>`;
    else{const c=s.querySelector(".cursor");if(c)c.remove();s.appendChild(document.createTextNode(text));const nc=document.createElement("span");nc.className="cursor";s.appendChild(nc);}
    s.scrollTop=s.scrollHeight;
}

// ── Files ─────────────────────────────────────────────────────────
async function loadOutputFiles(){try{const r=await fetch("/api/files");const d=await r.json();const files=d.files||[];const list=$("#filesList");list.innerHTML="";if(!files.length){list.innerHTML='<div class="empty-state">No files yet.</div>';return;}files.forEach(f=>{const ext=f.name.split(".").pop().toLowerCase();const icons={md:"\uD83D\uDCC4",json:"\uD83D\uDCCA",html:"\uD83C\uDF10",csv:"\uD83D\uDCCA",xlsx:"\uD83D\uDCCA",ics:"\uD83D\uDCC5",txt:"\uD83D\uDCC3"};const sz=f.size>1024?`${(f.size/1024).toFixed(1)} KB`:`${f.size} B`;const item=document.createElement("div");item.className="file-item";item.innerHTML=`<div class="file-item-name"><span class="file-item-icon">${icons[ext]||"\uD83D\uDCC4"}</span><span>${escapeHTML(f.name)}</span></div><span class="file-item-size">${sz}</span>`;item.addEventListener("click",()=>loadFilePreview(f.name));list.appendChild(item);});}catch(e){}}
async function loadFilePreview(name){try{const r=await fetch(`/api/files/${encodeURIComponent(name)}`);const d=await r.json();const p=$("#filePreview");p.classList.remove("hidden");const ext=name.split(".").pop().toLowerCase();const content=ext==="md"?renderMd(d.content):`<pre>${escapeHTML(d.content)}</pre>`;p.innerHTML=`<div class="fprev-hd"><span>${escapeHTML(d.name)}</span><span class="file-item-size">${(d.size/1024).toFixed(1)} KB</span></div><div class="fprev-body">${content}</div>`;p.scrollIntoView({behavior:"smooth",block:"nearest"});}catch(e){}}

// ── Memory page ───────────────────────────────────────────────────
async function loadStats(){try{await fetch("/api/memory/stats");}catch(e){}}
async function loadMemory(){
    try{const sr=await fetch("/api/memory/stats");const sd=await sr.json();
    $("#memoryStats").innerHTML=`<div class="mstat"><div class="mstat-n">${sd.total_episodes||0}</div><div class="mstat-l">Episodes</div></div><div class="mstat"><div class="mstat-n">${sd.total_memory_entries||0}</div><div class="mstat-l">Memories</div></div><div class="mstat"><div class="mstat-n">${(sd.domains||[]).length}</div><div class="mstat-l">Domains</div></div><div class="mstat"><div class="mstat-n">${sd.vector_search_available?"ON":"OFF"}</div><div class="mstat-l">Vector DB</div></div>`;
    const er=await fetch("/api/memory/episodes?limit=20");const ed=await er.json();
    const eps=ed.episodes||[];const container=$("#memoryEpisodes");container.innerHTML="";
    if(!eps.length){container.innerHTML='<div class="empty-state">No episodes yet. Run a task to start building memory.</div>';return;}
    eps.forEach(ep=>{const card=document.createElement("div");card.className="ep-card";
    card.innerHTML=`<div class="ep-task">${escapeHTML(ep.task)}</div><div class="ep-meta"><span>${ep.task_domain||"?"}</span><span>${ep.task_complexity||"?"}</span><span>${ep.agent_count||"?"} agents</span><span>${new Date(ep.timestamp).toLocaleDateString()}</span>${ep.success_score?`<span>Score: ${ep.success_score}/10</span>`:""}</div>`;
    container.appendChild(card);});}catch(e){}}

// ── Chat ──────────────────────────────────────────────────────────
function openChat(id,role){state.chatAgentId=id;$("#chatAgentRole").textContent=role;$("#chatAgentId").textContent=id;$("#chatMessages").innerHTML="";$("#chatInput").value="";$("#chatModal").classList.remove("hidden");addChatMsg("agent",`Hi! I'm the **${role}**. Ask me anything about my work.`);$("#chatInput").focus();}
function closeChat(){$("#chatModal").classList.add("hidden");state.chatAgentId=null;}
function addChatMsg(role,text){const c=$("#chatMessages");const m=document.createElement("div");m.className=`chat-msg ${role}`;if(role==="agent")m.innerHTML=renderMd(text);else m.textContent=text;c.appendChild(m);c.scrollTop=c.scrollHeight;return m;}
async function sendChat(){const input=$("#chatInput");const msg=input.value.trim();if(!msg||!state.chatAgentId||!state.sessionId)return;input.value="";addChatMsg("user",msg);const typing=addChatMsg("agent","Thinking...");typing.classList.add("typing");$("#chatSend").disabled=true;try{const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:state.sessionId,agent_id:state.chatAgentId,message:msg})});const d=await r.json();typing.remove();addChatMsg("agent",d.error?`Error: ${d.error}`:d.response||"No response.");}catch(e){typing.remove();addChatMsg("agent",`Error: ${e.message}`);}$("#chatSend").disabled=false;input.focus();}

// ── Agent Detail / RAG ────────────────────────────────────────────
let _ragAgentId = null;

function openAgentDetail(agentId) {
    _ragAgentId = agentId;
    // Show the detail page
    $$(".sb-item").forEach(b => b.classList.remove("active"));
    $$(".page").forEach(p => p.classList.remove("active"));
    $("#page-agent-detail").classList.add("active");

    // Load agent info
    fetch(`/api/agents/${agentId}/info`).then(r=>r.json()).then(info => {
        if (info.error) { $("#agentDetailTitle").textContent = agentId; return; }

        $("#agentDetailTitle").textContent = info.role || agentId;
        const agentType = info.agent_type || "standard";
        $("#agentTypeBadge").textContent = agentType.toUpperCase();
        $("#agentTypeBadge").className = `badge ${agentType === "rag" ? "bg-blue" : ""}`;

        $("#agentDetailMeta").innerHTML = `
            <div><strong>Persona:</strong> ${escapeHTML(info.persona)}</div>
            <div><strong>Objective:</strong> ${escapeHTML(info.objective)}</div>
            <div><strong>Tools:</strong> ${(info.tools||[]).map(t=>`<span class="tool-tag">${escapeHTML(t)}</span>`).join(" ")}</div>
            <div><strong>Model:</strong> ${info.model_tier} &nbsp; <strong>Type:</strong> ${agentType}</div>
        `;

        // Only RAG agents get the upload/document Q&A interface
        if (agentType === "rag") {
            $("#ragSection").classList.remove("hidden");
            $("#ragMessages").innerHTML = "";
            addRagMsg("agent", `I'm the **${info.role}**. Upload documents and I'll analyze them. My expertise: ${info.persona?.substring(0, 100) || "document analysis"}`);
            loadRagFiles(agentId);
        } else {
            $("#ragSection").classList.add("hidden");
        }

        // Show output for all agents
        if (info.output) {
            $("#agentOutputSection").classList.remove("hidden");
            $("#agentDetailOutput").innerHTML = renderMd(info.output);
        } else {
            $("#agentOutputSection").classList.add("hidden");
        }
    });
}

function loadRagFiles(agentId) {
    fetch(`/api/agents/${agentId}/files`).then(r=>r.json()).then(data => {
        const list = $("#ragFilesList");
        const files = data.files || [];
        if (!files.length) { list.innerHTML = ""; return; }
        list.innerHTML = files.map(f => `
            <div class="rag-file-item">
                <span class="rag-file-name">${escapeHTML(f.filename)}</span>
                <span class="rag-file-chunks">${f.chunks} chunks</span>
                <span class="rag-file-status indexed">Indexed</span>
            </div>
        `).join("");
    });
}

async function uploadRagFile(file) {
    if (!_ragAgentId) return;
    // Show uploading state
    const list = $("#ragFilesList");
    const item = document.createElement("div");
    item.className = "rag-file-item";
    item.id = `rag-file-${file.name.replace(/\W/g, "_")}`;
    item.innerHTML = `<span class="rag-file-name">${escapeHTML(file.name)}</span><span class="rag-file-status indexing">Indexing...</span>`;
    list.appendChild(item);

    const formData = new FormData();
    formData.append("file", file);

    try {
        const resp = await fetch(`/api/agents/${_ragAgentId}/upload`, { method: "POST", body: formData });
        const result = await resp.json();
        item.innerHTML = `
            <span class="rag-file-name">${escapeHTML(file.name)}</span>
            <span class="rag-file-chunks">${result.chunks || 0} chunks</span>
            <span class="rag-file-status ${result.status === "ok" ? "indexed" : "error"}">${result.status === "ok" ? "Indexed" : "Error"}</span>
        `;
        if (result.status === "ok") {
            addRagMsg("agent", `Indexed **${file.name}** — ${result.chunks} chunks, ${result.chars} chars. Ask me anything about it.`);
        } else {
            addRagMsg("agent", `Failed to process ${file.name}: ${result.message}`);
        }
    } catch (e) {
        item.querySelector(".rag-file-status").textContent = "Error";
        item.querySelector(".rag-file-status").className = "rag-file-status error";
    }
}

async function sendRagQuery() {
    const input = $("#ragInput");
    const q = input.value.trim();
    if (!q || !_ragAgentId) return;
    input.value = "";
    addRagMsg("user", q);

    const typing = addRagMsg("agent", "Thinking...");
    typing.classList.add("typing");
    $("#ragSend").disabled = true;

    try {
        const resp = await fetch(`/api/agents/${_ragAgentId}/query`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ question: q }),
        });
        const data = await resp.json();
        typing.remove();

        let answer = data.answer || "No answer.";
        if (data.sources?.length) {
            answer += `\n\n<div class="rag-sources"><strong>Sources:</strong> ${data.sources.map(s => `<span class="rag-source-item">${escapeHTML(s.filename)} (${Math.round(s.relevance * 100)}%)</span>`).join("")}</div>`;
        }
        const msg = addRagMsg("agent", "");
        msg.innerHTML = renderMd(data.answer || "No answer.");
        if (data.sources?.length) {
            msg.innerHTML += `<div class="rag-sources"><strong>Sources:</strong> ${data.sources.map(s => `<span class="rag-source-item">${escapeHTML(s.filename)} (${Math.round(s.relevance * 100)}%)</span>`).join("")}</div>`;
        }
    } catch (e) {
        typing.remove();
        addRagMsg("agent", `Error: ${e.message}`);
    }
    $("#ragSend").disabled = false;
    input.focus();
}

function addRagMsg(role, text) {
    const container = $("#ragMessages");
    const msg = document.createElement("div");
    msg.className = `rag-msg ${role}`;
    if (role === "agent" && text) msg.innerHTML = `<div class="md-body">${renderMd(text)}</div>`;
    else if (text) msg.textContent = text;
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
    return msg;
}

// ── Log ───────────────────────────────────────────────────────────
function addLog(level,text){state.logCount++;$("#logCount").textContent=state.logCount;const c=$("#logMessages");const now=new Date().toLocaleTimeString("en-US",{hour12:false});const e=document.createElement("div");e.className="log-entry";e.innerHTML=`<span class="log-ts">${now}</span> <span class="log-type ${level}">${level}</span> ${escapeHTML(text)}`;c.appendChild(e);c.scrollTop=c.scrollHeight;}

// ── Helpers ───────────────────────────────────────────────────────
function escapeHTML(s){if(!s)return"";const d=document.createElement("div");d.textContent=s;return d.innerHTML;}
function renderMd(t){if(typeof marked!=="undefined")return marked.parse(t);return escapeHTML(t).replace(/\n/g,"<br>");}
