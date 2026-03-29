import json
import time
from .debate import run_debate
from .dynamic_agent import DynamicAgent
from .prompts import (
    DA_GENERATE_REQUIREMENTS_PROMPT,
    EVALUATOR_CRITIQUE_REQUIREMENTS_PROMPT,
    DA_GENERATE_SUBAGENTS_PROMPT,
    EVALUATOR_CRITIQUE_SUBAGENTS_PROMPT,
)


def run_pipeline(task: str, tool_executor=None) -> dict:
    """
    Full ClawForge pipeline:
        Phase 1-2: Debate requirements
        Phase 3-4: Debate sub-agent plan
        Phase 5:   Execute sub-agents
        Phase 6:   DA validates and compiles final output

    Returns:
        Dict with final_output, coverage_report, known_issues, agent_outputs, metrics.
    """
    pipeline_start = time.time()
    total_metrics = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "phase_1_2": {},
        "phase_3_4": {},
        "phase_5_6": {},
    }

    # Phase 1-2: Requirements debate
    print("\n" + "=" * 60)
    print("  PHASE 1-2: Requirements Debate")
    print("=" * 60)

    requirements, req_metrics = run_debate(
        task=task,
        generator_prompt=DA_GENERATE_REQUIREMENTS_PROMPT,
        critic_prompt=EVALUATOR_CRITIQUE_REQUIREMENTS_PROMPT,
        modified_key="modified_requirements",
    )

    total_metrics["phase_1_2"] = req_metrics
    _accumulate(total_metrics, req_metrics)
    print(f"\n  Requirements approved.")

    # Phase 3-4: Plan debate
    print("\n" + "=" * 60)
    print("  PHASE 3-4: Sub-Agent Plan Debate")
    print("=" * 60)

    req_context = f"Approved Requirements:\n{json.dumps(requirements, indent=2)}"

    plan_result, plan_metrics = run_debate(
        task=task,
        generator_prompt=DA_GENERATE_SUBAGENTS_PROMPT,
        critic_prompt=EVALUATOR_CRITIQUE_SUBAGENTS_PROMPT,
        modified_key="modified_plan",
        input_context=req_context,
    )

    total_metrics["phase_3_4"] = plan_metrics
    _accumulate(total_metrics, plan_metrics)

    agents_list = plan_result.get("plan", [])
    execution_strategy = plan_result.get("execution_strategy", {})

    print(f"\n  Plan approved: {len(agents_list)} agents")

    # Phase 5-6: Execute sub-agents + DA validation
    print("\n" + "=" * 60)
    print("  PHASE 5-6: Execution + Validation")
    print("=" * 60)

    full_plan = {
        "task": task,
        "requirements": requirements,
        "plan": agents_list,
        "execution_strategy": execution_strategy,
    }

    da = DynamicAgent(full_plan, tool_executor=tool_executor)
    result = da.run()

    exec_metrics = result.pop("metrics_phase_5_6", {})
    total_metrics["phase_5_6"] = exec_metrics
    _accumulate(total_metrics, exec_metrics)

    total_metrics["time_seconds"] = round(time.time() - pipeline_start, 2)
    result["metrics"] = total_metrics

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

    return result


def _accumulate(total: dict, phase: dict) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] += phase.get(key, 0)
