"""
Graph Builder — dynamically constructs a LangGraph StateGraph
from the approved plan and agent bundles.

Parallel groups become fan-out / fan-in edges:
  START ─┬─> agent_1 ─┬─> agent_3 ─┬─> compiler ─> END
         └─> agent_2 ─┘            │
                                    └─ (fan-in)

Dependencies are encoded as edges.  If agent_3 depends on agent_1,
agent_3 only starts after agent_1 completes.
"""

from __future__ import annotations
from collections import defaultdict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import OrchestratorState
from .agent_factory import make_agent_node
from .compiler import compile_node, set_compiler_memory


def _detect_cycles(depends_on: dict[str, list[str]]) -> list[str] | None:
    """DFS-based cycle detection.  Returns the cycle path or None if acyclic."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in depends_on}
    path: list[str] = []

    def dfs(node: str) -> bool:
        color[node] = GRAY
        path.append(node)
        for dep in depends_on.get(node, []):
            if color[dep] == GRAY:
                path.append(dep)
                return True
            if color[dep] == WHITE and dfs(dep):
                return True
        color[node] = BLACK
        path.pop()
        return False

    for node in list(depends_on):
        if color[node] == WHITE:
            if dfs(node):
                return path
    return None


def build_graph(plan: dict, agent_bundles: dict[str, dict], memory=None):
    """Build and compile the orchestration graph.

    Parameters
    ----------
    plan : dict
        The approved plan from the debate phase.
    agent_bundles : dict
        {agent_id: {"agent": CompiledGraph, "spec": dict}} from agent_factory.

    Returns
    -------
    Compiled LangGraph graph ready for .invoke().
    """
    # Set compiler memory reference
    if memory is not None:
        set_compiler_memory(memory)

    graph = StateGraph(OrchestratorState)

    # ── Add agent nodes ─────────────────────────────────────────────────
    agent_ids = []
    for agent_id, bundle in agent_bundles.items():
        node_fn = make_agent_node(agent_id, bundle, memory=memory)
        graph.add_node(agent_id, node_fn)
        agent_ids.append(agent_id)

    # ── Add compiler node ───────────────────────────────────────────────
    graph.add_node("compiler", compile_node)

    if not agent_ids:
        graph.add_edge(START, "compiler")
        graph.add_edge("compiler", END)
        return graph.compile(checkpointer=MemorySaver())

    # ── Build dependency map ────────────────────────────────────────────
    depends_on: dict[str, list[str]] = {}
    depended_by: dict[str, list[str]] = defaultdict(list)

    for agent_id, bundle in agent_bundles.items():
        deps = bundle["spec"].get("depends_on", [])
        # Filter out invalid dep references
        deps = [d for d in deps if d in agent_bundles]
        depends_on[agent_id] = deps
        for d in deps:
            depended_by[d].append(agent_id)

    # ── Cycle detection ─────────────────────────────────────────────────
    cycle = _detect_cycles(depends_on)
    if cycle:
        cycle_str = " -> ".join(cycle)
        raise ValueError(
            f"[GRAPH] Circular dependency detected in plan: {cycle_str}. "
            "The Evaluator should have caught this — re-run the debate."
        )

    # ── Add edges ───────────────────────────────────────────────────────
    root_agents = [aid for aid in agent_ids if not depends_on.get(aid)]
    for aid in root_agents:
        graph.add_edge(START, aid)

    for agent_id, deps in depends_on.items():
        for dep in deps:
            graph.add_edge(dep, agent_id)

    leaf_agents = [aid for aid in agent_ids if not depended_by.get(aid)]
    for aid in leaf_agents:
        graph.add_edge(aid, "compiler")

    graph.add_edge("compiler", END)

    # ── Compile ─────────────────────────────────────────────────────────
    compiled = graph.compile(checkpointer=MemorySaver())

    print(f"\n[GRAPH] Built graph: {len(agent_ids)} agent(s) + compiler")
    print(f"  Roots (from START): {root_agents}")
    print(f"  Leaves (to compiler): {leaf_agents}")

    return compiled
