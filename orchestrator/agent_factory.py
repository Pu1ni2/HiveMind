"""
Agent Factory — creates LangGraph react agents from plan specs + forged tools.
"""

from __future__ import annotations
import inspect
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

from .config import OPENAI_API_KEY, TIER_TO_MODEL, MAX_AGENT_STEPS
from .prompts import AGENT_SYSTEM_PROMPT
from .state import OrchestratorState
from .utils import truncate
from .events import emit


class AgentStreamHandler(BaseCallbackHandler):
    """Streams tokens and tool events via the global event bus."""

    def __init__(self, agent_id: str, role: str):
        self.agent_id = agent_id
        self.role = role

    def on_llm_new_token(self, token: str, **kwargs):
        emit("agent_token", {"agent_id": self.agent_id, "token": token})

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown") if isinstance(serialized, dict) else "tool"
        emit("agent_tool_call", {
            "agent_id": self.agent_id,
            "tool_name": tool_name,
        })

    def on_tool_end(self, output, **kwargs):
        emit("agent_tool_result", {
            "agent_id": self.agent_id,
            "result_preview": str(output)[:300],
        })


def _build_memory_tools(memory, agent_id: str) -> list[StructuredTool]:
    """Create remember/recall tools backed by the shared workspace."""
    ws = memory.get_workspace()
    if ws is None:
        return []

    def remember(key: str, value: str, tags: str = "") -> str:
        """Store information in shared memory so other agents can access it.

        Args:
            key: A descriptive key for this data (e.g. 'market_data', 'competitor_list')
            value: The information to store
            tags: Optional comma-separated tags for categorization
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        result = ws.write(key, value, agent_id, tag_list)
        emit("memory_store", {"key": key, "author": agent_id, "chars": len(value)})
        return result

    def recall(key: str = "") -> str:
        """Retrieve information from shared memory written by any agent.

        Args:
            key: Specific key to retrieve. Leave empty to see all stored data.
        """
        if key:
            return ws.read(key)
        return ws.get_summary()

    remember_tool = StructuredTool.from_function(
        func=remember,
        name="remember",
        description=(
            "Store information in shared memory for other agents to access. "
            "Use descriptive keys like 'market_research', 'candidate_scores'. "
            "Other agents can retrieve this data using the recall tool."
        ),
    )

    recall_tool = StructuredTool.from_function(
        func=recall,
        name="recall",
        description=(
            "Retrieve information from shared memory written by any agent. "
            "Pass a specific key to get that data, or call with empty key "
            "to see everything stored so far."
        ),
    )

    return [remember_tool, recall_tool]


def create_all_agents(
    plan: dict,
    agent_tools: dict[str, list],
    mcp_tools: list | None = None,
    memory=None,
) -> dict[str, dict]:
    """Build react agents for every agent in the plan.
    Returns {agent_id: {"agent": CompiledGraph, "spec": dict}}
    """
    agents = {}
    task = plan.get("task_analysis", {}).get("domain", "")

    for spec in plan.get("agents", []):
        agent_id = spec["id"]
        model_name = TIER_TO_MODEL.get(spec.get("model_tier", "BALANCED"), "gpt-4o")

        model = ChatOpenAI(
            model=model_name,
            api_key=OPENAI_API_KEY,
            temperature=0.5,
            streaming=True,
        )

        tools = list(agent_tools.get(agent_id, []))
        if mcp_tools:
            tools.extend(mcp_tools)

        # Add shared memory tools (remember / recall)
        if memory is not None:
            tools.extend(_build_memory_tools(memory, agent_id))

        tool_names = ", ".join(t.name for t in tools) if tools else "none"

        # Build memory context section
        memory_section = ""
        if memory is not None:
            agent_memory = memory.get_agent_context(
                spec.get("role", ""), spec.get("objective", "")
            )
            if agent_memory:
                memory_section = f"--- RELEVANT PAST EXPERIENCE ---\n{agent_memory}"

        system_prompt = AGENT_SYSTEM_PROMPT.format(
            role=spec.get("role", "Agent"),
            persona=spec.get("persona", ""),
            objective=spec.get("objective", ""),
            task=task,
            context_section="",
            memory_section=memory_section,
            tool_names=tool_names,
            expected_output=spec.get("expected_output", ""),
        )

        agent = create_react_agent(
            model=model,
            tools=tools if tools else [],
            prompt=system_prompt,
        )

        agents[agent_id] = {"agent": agent, "spec": spec}
        mem_flag = " +memory" if memory_section else ""
        print(f"[FACTORY] {agent_id} ({spec.get('role', '?')}) created "
              f"| model={model_name} | tools={len(tools)}{mem_flag}")

    emit("agents_created", {
        "agents": [
            {
                "id": s["id"],
                "role": s.get("role", ""),
                "persona": s.get("persona", "")[:150],
                "tools": [t.name for t in agent_tools.get(s["id"], [])]
                         + (["remember", "recall"] if memory else []),
                "model_tier": s.get("model_tier", "BALANCED"),
                "parallel_group": s.get("parallel_group", 1),
                "depends_on": s.get("depends_on", []),
                "objective": s.get("objective", "")[:200],
                "agent_type": s.get("agent_type", "standard"),
            }
            for s in plan.get("agents", [])
        ]
    })

    return agents


def make_agent_node(agent_id: str, agent_bundle: dict, memory=None) -> Any:
    """Return a node function compatible with the outer OrchestratorState graph."""
    agent = agent_bundle["agent"]
    spec = agent_bundle["spec"]
    depends_on = spec.get("depends_on", [])
    role = spec.get("role", agent_id)
    objective = spec.get("objective", "")

    def node_fn(state: OrchestratorState) -> dict:
        context_parts = []
        for dep_id in depends_on:
            dep_output = state.get("agent_outputs", {}).get(dep_id)
            if dep_output:
                dep_role = dep_output.get("role", dep_id)
                dep_text = truncate(dep_output.get("output", ""), 8000)
                context_parts.append(f"=== Output from {dep_role} ({dep_id}) ===\n{dep_text}")

        context_block = "\n\n".join(context_parts) if context_parts else ""

        # Include shared memory summary if available
        shared_mem_summary = ""
        if memory and memory.get_workspace():
            ws_summary = memory.get_workspace().get_summary()
            if "empty" not in ws_summary.lower():
                shared_mem_summary = f"\n\nSHARED MEMORY:\n{ws_summary}"

        user_msg = (
            f"TASK: {state['task']}\n\n"
            f"YOUR ROLE: {role}\n"
            f"YOUR OBJECTIVE: {objective}\n"
        )
        if context_block:
            user_msg += f"\nCONTEXT FROM OTHER AGENTS:\n{context_block}\n"
        if shared_mem_summary:
            user_msg += shared_mem_summary
        user_msg += "\nExecute your objective now.  Use your tools as needed."

        print(f"\n[AGENT] {agent_id} ({role}) starting ...")
        emit("agent_start", {"agent_id": agent_id, "role": role})

        stream_handler = AgentStreamHandler(agent_id, role)

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_msg)]},
                config={
                    "recursion_limit": MAX_AGENT_STEPS,
                    "callbacks": [stream_handler],
                },
            )

            final_content = ""
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, "content") and msg.content and msg.type == "ai":
                    final_content = msg.content
                    break

            if not final_content:
                final_content = "[Agent produced no output]"

            print(f"[AGENT] {agent_id} done ({len(final_content)} chars)")
            emit("agent_done", {
                "agent_id": agent_id,
                "role": role,
                "output_preview": final_content[:500],
            })

            # Record in episode
            if memory and memory.recorder:
                memory.recorder.record_agent_output(agent_id, role, final_content)

        except Exception as exc:
            final_content = f"[Agent {agent_id} error: {exc}]"
            print(f"[AGENT] {agent_id} FAILED: {exc}")
            emit("agent_error", {"agent_id": agent_id, "role": role, "error": str(exc)})
            if memory and memory.recorder:
                memory.recorder.record_error(agent_id, str(exc))

        # Snapshot shared memory contributions into state
        ws_snapshot = {}
        if memory and memory.get_workspace():
            ws_snapshot = memory.get_workspace().to_dict()

        return {
            "agent_outputs": {
                agent_id: {"role": role, "output": final_content}
            },
            "shared_memory": ws_snapshot,
        }

    node_fn.__name__ = agent_id
    return node_fn
