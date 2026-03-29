import json
import concurrent.futures
from .config import VALIDATION_MODEL, TIER_TO_MODEL
from .llm_client import call_llm_json
from .prompts import DA_COMPILE_FINAL_OUTPUT_PROMPT
from .sub_agent import SubAgent


class DynamicAgent:
    def __init__(self, plan: dict, tool_executor=None):
        """
        Args:
            plan: Full pipeline context. Keys:
                  task, requirements, plan (list of agents),
                  execution_strategy
            tool_executor: function(tool_name, tool_args) -> str
        """
        self.task = plan["task"]
        self.requirements = plan.get("requirements", {})
        self.agents_config = plan["plan"]
        self.execution_strategy = plan.get("execution_strategy", {})
        self.tool_executor = tool_executor

    def run(self) -> dict:
        # Build agent configs with resolved models
        agents = self._resolve_agents()

        # Execute based on parallel groups
        parallel_groups = self.execution_strategy.get("parallel_groups", {})

        if parallel_groups:
            results = self._run_grouped(agents, parallel_groups)
        else:
            results = self._run_sequential(agents)

        # Phase 6: DA validates and compiles
        validated = self._validate(results)
        return validated

    def _resolve_agents(self) -> dict:
        """Convert plan agents into a dict keyed by id, with model resolved."""
        agents = {}
        for a in self.agents_config:
            a["model"] = TIER_TO_MODEL.get(a.get("model_tier", "BALANCED"), TIER_TO_MODEL["BALANCED"])
            agents[a["id"]] = a
        return agents

    def _run_grouped(self, agents: dict, parallel_groups: dict) -> dict:
        """Run agents by parallel groups. Groups run sequentially, agents within a group run in parallel."""
        outputs = {}  # agent_id -> output

        for group_num in sorted(parallel_groups.keys(), key=lambda x: int(x)):
            agent_ids = parallel_groups[group_num]
            print(f"\n  Group {group_num}: agents {agent_ids}")

            group_agents = [agents[aid] for aid in agent_ids if aid in agents]

            with concurrent.futures.ThreadPoolExecutor() as pool:
                futures = {}
                for config in group_agents:
                    context = self._build_context(config, outputs)
                    agent = SubAgent(config, tool_executor=self.tool_executor)
                    print(f"  [{agent.role}] running...")
                    futures[pool.submit(agent.run, self.task, context)] = config

                for future in concurrent.futures.as_completed(futures):
                    config = futures[future]
                    output = future.result()
                    outputs[config["id"]] = {"role": config["role"], "output": output}
                    print(f"  [{config['role']}] done")

        return outputs

    def _run_sequential(self, agents: dict) -> dict:
        """Fallback: run all agents one by one."""
        outputs = {}
        for agent_id, config in agents.items():
            context = self._build_context(config, outputs)
            agent = SubAgent(config, tool_executor=self.tool_executor)
            print(f"\n  [{agent.role}] running...")
            output = agent.run(self.task, context)
            outputs[agent_id] = {"role": config["role"], "output": output}
            print(f"  [{config['role']}] done")
        return outputs

    def _build_context(self, config: dict, outputs: dict) -> str:
        """Build context string from agents this one depends on."""
        deps = config.get("context_from_agents", [])
        if not deps:
            return ""

        parts = []
        for dep_id in deps:
            if dep_id in outputs:
                dep = outputs[dep_id]
                parts.append(f"[{dep['role']}]:\n{dep['output']}")

        return "\n\n".join(parts)

    def _validate(self, outputs: dict) -> dict:
        """Phase 6: DA compiles and validates all sub-agent outputs."""
        agent_outputs = ""
        for aid, data in outputs.items():
            agent_outputs += f"\n[Agent {aid} — {data['role']}]:\n{data['output']}\n"

        print("\n  [DA] Validating and compiling final output...")

        result = call_llm_json(
            VALIDATION_MODEL,
            messages=[
                {"role": "system", "content": DA_COMPILE_FINAL_OUTPUT_PROMPT},
                {"role": "user", "content": (
                    f"Original user request: {self.task}\n\n"
                    f"Requirements:\n{json.dumps(self.requirements, indent=2)}\n\n"
                    f"Plan:\n{json.dumps(self.agents_config, indent=2)}\n\n"
                    f"Sub-agent outputs:\n{agent_outputs}"
                )},
            ],
        )

        print("  [DA] Validation complete")
        return {
            "final_output": result.get("final_output", ""),
            "coverage_report": result.get("coverage_report", {}),
            "known_issues": result.get("known_issues", []),
            "agent_outputs": outputs,
        }
