"""
Agent Factory — creates LangGraph react agents from plan specs + forged tools.
"""

from __future__ import annotations
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
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


def create_all_agents(
    plan: dict,
    agent_tools: dict[str, list],
    mcp_tools: list | None = None,
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

        tool_names = ", ".join(t.name for t in tools) if tools else "none"
        system_prompt = AGENT_SYSTEM_PROMPT.format(
            role=spec.get("role", "Agent"),
            persona=spec.get("persona", ""),
            objective=spec.get("objective", ""),
            task=task,
            context_section="",
            tool_names=tool_names,
            expected_output=spec.get("expected_output", ""),
        )

        agent = create_react_agent(
            model=model,
            tools=tools if tools else [],
            prompt=system_prompt,
        )

        agents[agent_id] = {"agent": agent, "spec": spec}
        print(f"[FACTORY] {agent_id} ({spec.get('role', '?')}) created "
              f"| model={model_name} | tools={len(tools)}")

    emit("agents_created", {
        "agents": [
            {
                "id": s["id"],
                "role": s.get("role", ""),
                "persona": s.get("persona", "")[:150],
                "tools": [t.name for t in agent_tools.get(s["id"], [])],
                "model_tier": s.get("model_tier", "BALANCED"),
                "parallel_group": s.get("parallel_group", 1),
                "depends_on": s.get("depends_on", []),
                "objective": s.get("objective", "")[:200],
            }
            for s in plan.get("agents", [])
        ]
    })

    return agents


def make_agent_node(agent_id: str, agent_bundle: dict) -> Any:
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

        user_msg = (
            f"TASK: {state['task']}\n\n"
            f"YOUR ROLE: {role}\n"
            f"YOUR OBJECTIVE: {objective}\n"
        )
        if context_block:
            user_msg += f"\nCONTEXT FROM OTHER AGENTS:\n{context_block}\n"
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

        except Exception as exc:
            final_content = f"[Agent {agent_id} error: {exc}]"
            print(f"[AGENT] {agent_id} FAILED: {exc}")
            emit("agent_error", {"agent_id": agent_id, "role": role, "error": str(exc)})

        return {
            "agent_outputs": {
                agent_id: {"role": role, "output": final_content}
            }
        }

    node_fn.__name__ = agent_id
    return node_fn
