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
from .compiler import compile_node


def build_graph(plan: dict, agent_bundles: dict[str, dict]):
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
    graph = StateGraph(OrchestratorState)

    # ── Add agent nodes ─────────────────────────────────────────────────
    agent_ids = []
    for agent_id, bundle in agent_bundles.items():
        node_fn = make_agent_node(agent_id, bundle)
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

    # ── Add edges ───────────────────────────────────────────────────────
    # Agents with no dependencies get edges from START
    root_agents = [aid for aid in agent_ids if not depends_on.get(aid)]
    for aid in root_agents:
        graph.add_edge(START, aid)

    # Add dependency edges
    for agent_id, deps in depends_on.items():
        for dep in deps:
            graph.add_edge(dep, agent_id)

    # Agents with no downstream dependents connect to compiler
    leaf_agents = [aid for aid in agent_ids if not depended_by.get(aid)]
    for aid in leaf_agents:
        graph.add_edge(aid, "compiler")

    # compiler -> END
    graph.add_edge("compiler", END)

    # ── Compile ─────────────────────────────────────────────────────────
    compiled = graph.compile(checkpointer=MemorySaver())

    print(f"\n[GRAPH] Built graph: {len(agent_ids)} agent(s) + compiler")
    print(f"  Roots (from START): {root_agents}")
    print(f"  Leaves (to compiler): {leaf_agents}")

    return compiled
