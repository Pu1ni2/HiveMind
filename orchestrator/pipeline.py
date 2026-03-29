"""
HIVEMIND Pipeline — top-level entry point.

    task  -->  debate  -->  forge tools  -->  build agents
              (DA+Eval)    (Tool Forge)       (Agent Factory)
                                                   |
                                                   v
    result <-- compile <-- execute graph <-- build graph
              (Compiler)   (LangGraph)      (Graph Builder)
"""

import time
import uuid

from .debate import run_debate
from .tool_forge import forge_tools_for_plan
from .agent_factory import create_all_agents
from .graph_builder import build_graph
from .mcp_client import load_mcp_tools
from .events import EventBus, set_bus, emit
from .quick_actions import try_quick_execute


def run_task(task: str, mcp_servers: dict | None = None, event_bus: EventBus | None = None, memory=None) -> dict:
    """Run the full HIVEMIND orchestration pipeline.

    Parameters
    ----------
    task : str
        The user's task in natural language.
    mcp_servers : dict, optional
        MCP server configs.
    event_bus : EventBus, optional
        If provided, pipeline events are emitted for real-time streaming.
    memory : MemoryManager, optional
        If provided, enables persistent memory across runs.

    Returns
    -------
    dict with keys: final_output, coverage_report, known_issues, plan,
                    agent_outputs, metadata
    """
    if event_bus is not None:
        set_bus(event_bus)

    start_time = time.time()

    print("=" * 70)
    print("  HIVEMIND — Autonomous Multi-Agent Orchestration Engine")
    print("=" * 70)
    print(f"\n  Task: {task}\n")

    # ── Initialize memory for this run ──────────────────────────────
    planning_context = ""
    if memory is not None:
        memory.begin_run(task)
        planning_context = memory.get_planning_context(task)
        if planning_context:
            print(f"  [MEMORY] Retrieved {len(planning_context)} chars of past experience")
            emit("memory_recall", {"source": "long_term", "chars": len(planning_context)})

    # ── Quick action check ───────────────────────────────────────────
    print("-" * 70)
    print("  Checking for quick action ...")
    print("-" * 70)

    quick_result = try_quick_execute(task)
    if quick_result is not None:
        total_time = time.time() - start_time
        quick_result["metadata"]["total_time_s"] = round(total_time, 2)

        # Persist episode if memory is available
        if memory is not None:
            episode = memory.end_run(quick_result)
            quick_result["metadata"]["episode_id"] = episode.episode_id
            emit("episode_saved", {"episode_id": episode.episode_id})

        print(f"\n  Quick action completed in {total_time:.1f}s")
        return quick_result

    # ── Phase 1-2: DA <-> Evaluator debate ──────────────────────────
    print("-" * 70)
    print("  PHASE 1-2: Planning (DA <-> Evaluator Debate)")
    print("-" * 70)

    plan = run_debate(task, memory_context=planning_context)
    debate_time = time.time() - start_time

    agents_spec = plan.get("agents", [])
    print(f"\n  Plan approved: {len(agents_spec)} agent(s)")

    # ── Load MCP tools (if configured) ──────────────────────────────
    mcp_tools = load_mcp_tools(mcp_servers)

    # ── Phase 3: Tool Forge ─────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  PHASE 3: Tool Forge (Dynamic Tool Creation)")
    print("-" * 70)

    forge_start = time.time()
    agent_tools = forge_tools_for_plan(plan)
    forge_time = time.time() - forge_start

    total_tools = sum(len(t) for t in agent_tools.values())
    print(f"\n  Forged {total_tools} tool(s) in {forge_time:.1f}s")

    # ── Phase 4: Agent Factory ──────────────────────────────────────
    print("\n" + "-" * 70)
    print("  PHASE 4: Agent Factory (Creating Sub-Agents)")
    print("-" * 70)

    agent_bundles = create_all_agents(plan, agent_tools, mcp_tools, memory=memory)

    # ── Phase 5: Build and Execute Graph ────────────────────────────
    print("\n" + "-" * 70)
    print("  PHASE 5: Execution (LangGraph Dynamic Orchestration)")
    print("-" * 70)

    exec_start = time.time()
    graph = build_graph(plan, agent_bundles, memory=memory)

    emit("graph_built", {
        "total_nodes": len(agent_bundles) + 1,
    })

    initial_state = {
        "task": task,
        "plan": plan,
        "agent_outputs": {},
        "final_output": "",
        "coverage_report": {},
        "known_issues": [],
        "metadata": {},
        "shared_memory": {},
    }

    thread_id = uuid.uuid4().hex
    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )
    exec_time = time.time() - exec_start

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Debate:    {debate_time:.1f}s")
    print(f"  Forge:     {forge_time:.1f}s")
    print(f"  Execution: {exec_time:.1f}s")
    print(f"  Total:     {total_time:.1f}s")
    print(f"  Agents:    {len(agents_spec)}")
    print(f"  Tools:     {total_tools}")

    # ── Persist episode to memory ───────────────────────────────────
    episode_id = ""
    if memory is not None:
        pipeline_result = {
            "final_output": result.get("final_output", ""),
            "coverage_report": result.get("coverage_report", {}),
            "known_issues": result.get("known_issues", []),
            "plan": plan,
            "agent_outputs": result.get("agent_outputs", {}),
            "metadata": {
                "debate_time_s": round(debate_time, 2),
                "forge_time_s": round(forge_time, 2),
                "exec_time_s": round(exec_time, 2),
                "total_time_s": round(total_time, 2),
                "total_agents": len(agents_spec),
                "total_tools": total_tools,
            },
        }
        episode = memory.end_run(pipeline_result)
        episode_id = episode.episode_id
        print(f"  [MEMORY] Episode {episode_id} saved")
        emit("episode_saved", {"episode_id": episode_id})

    return {
        "final_output": result.get("final_output", ""),
        "coverage_report": result.get("coverage_report", {}),
        "known_issues": result.get("known_issues", []),
        "plan": plan,
        "agent_outputs": result.get("agent_outputs", {}),
        "metadata": {
            "debate_time_s": round(debate_time, 2),
            "forge_time_s": round(forge_time, 2),
            "exec_time_s": round(exec_time, 2),
            "total_time_s": round(total_time, 2),
            "total_agents": len(agents_spec),
            "total_tools": total_tools,
            "mcp_tools": len(mcp_tools),
            "episode_id": episode_id,
            **(result.get("metadata", {})),
        },
    }
