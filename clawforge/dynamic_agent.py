import json
import time
import concurrent.futures
from .config import VALIDATION_MODEL, TIER_TO_MODEL
from .llm_client import call_llm_json
from .prompts import DA_COMPILE_FINAL_OUTPUT_PROMPT
from .sub_agent import SubAgent


class DynamicAgent:
    def __init__(self, plan: dict, tool_executor=None):
        self.task = plan["task"]
        self.requirements = plan.get("requirements", {})
        self.agents_config = plan["plan"]
        self.execution_strategy = plan.get("execution_strategy", {})
        self.tool_executor = tool_executor

    def run(self) -> dict:
        metrics = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "time_seconds": 0}
        start = time.time()

        # Build agent configs with resolved models
        agents = self._resolve_agents()

        # Execute based on parallel groups
        parallel_groups = self.execution_strategy.get("parallel_groups", {})

        if parallel_groups:
            results, sub_agents = self._run_grouped(agents, parallel_groups)
        else:
            results, sub_agents = self._run_sequential(agents)

        # Collect sub-agent tokens
        for sa in sub_agents:
            self._accumulate(metrics, sa.token_usage)

        # Phase 6: DA validates and compiles
        validated, val_usage = self._validate(results)
        self._accumulate(metrics, val_usage)

        metrics["time_seconds"] = round(time.time() - start, 2)
        validated["metrics_phase_5_6"] = metrics
        return validated

    def _resolve_agents(self) -> dict:
        agents = {}
        for a in self.agents_config:
            a["model"] = TIER_TO_MODEL.get(a.get("model_tier", "BALANCED"), TIER_TO_MODEL["BALANCED"])
            agents[a["id"]] = a
        return agents

    def _run_grouped(self, agents: dict, parallel_groups: dict) -> tuple[dict, list]:
        outputs = {}
        all_sub_agents = []

        for group_num in sorted(parallel_groups.keys(), key=lambda x: int(x)):
            agent_ids = parallel_groups[group_num]
            print(f"\n  Group {group_num}: agents {agent_ids}")

            group_agents = [agents[aid] for aid in agent_ids if aid in agents]

            with concurrent.futures.ThreadPoolExecutor() as pool:
                futures = {}
                for config in group_agents:
                    context = self._build_context(config, outputs)
                    agent = SubAgent(config, tool_executor=self.tool_executor)
                    all_sub_agents.append(agent)
                    print(f"  [{agent.role}] running...")
                    futures[pool.submit(agent.run, self.task, context)] = (config, agent)

                for future in concurrent.futures.as_completed(futures):
                    config, agent = futures[future]
                    output = future.result()
                    outputs[config["id"]] = {"role": config["role"], "output": output}
                    print(f"  [{config['role']}] done")

        return outputs, all_sub_agents

    def _run_sequential(self, agents: dict) -> tuple[dict, list]:
        outputs = {}
        all_sub_agents = []

        for agent_id, config in agents.items():
            context = self._build_context(config, outputs)
            agent = SubAgent(config, tool_executor=self.tool_executor)
            all_sub_agents.append(agent)
            print(f"\n  [{agent.role}] running...")
            output = agent.run(self.task, context)
            outputs[agent_id] = {"role": config["role"], "output": output}
            print(f"  [{config['role']}] done")

        return outputs, all_sub_agents

    def _build_context(self, config: dict, outputs: dict) -> str:
        deps = config.get("context_from_agents", [])
        if not deps:
            return ""

        parts = []
        for dep_id in deps:
            if dep_id in outputs:
                dep = outputs[dep_id]
                parts.append(f"[{dep['role']}]:\n{dep['output']}")

        return "\n\n".join(parts)

    def _validate(self, outputs: dict) -> tuple[dict, dict]:
        """Phase 6: DA compiles and validates all sub-agent outputs.
        Returns (result_dict, token_usage).
        """
        agent_outputs = ""
        for aid, data in outputs.items():
            agent_outputs += f"\n[Agent {aid} — {data['role']}]:\n{data['output']}\n"

        print("\n  [DA] Validating and compiling final output...")

        result, usage = call_llm_json(
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
        }, usage

    def _accumulate(self, metrics: dict, usage: dict) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            metrics[key] += usage.get(key, 0)
