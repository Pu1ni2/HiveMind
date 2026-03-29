"""
HiveMind Evaluation — compares the full orchestration pipeline
against a single direct GPT-4o call on the same task.

Measures:
  - Wall-clock time per approach and per phase
  - Output length and structural completeness
  - Plan structural errors caught by the debate gate

Usage:
    python evaluate.py
    python evaluate.py "Your custom task here"
"""

import sys
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from orchestrator.config import OPENAI_API_KEY, PLANNER_MODEL
from orchestrator.pipeline import run_task

DEFAULT_TASK = (
    "Research the top 3 AI agent frameworks in 2025 — LangGraph, CrewAI, and AutoGen. "
    "Compare their architecture, strengths, weaknesses, and ideal use cases. "
    "Produce a structured report with a recommendation framework."
)

TASK_TIMEOUT_S = 300


def _run_with_timeout(fn, *args, timeout: int = TASK_TIMEOUT_S):
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            raise TimeoutError(f"Call timed out after {timeout}s")


def direct_llm_call(task: str) -> dict:
    """Baseline: single GPT-4o call with no planning, debate, or agents."""
    model = ChatOpenAI(model=PLANNER_MODEL, api_key=OPENAI_API_KEY, temperature=0.7)
    start = time.time()
    response = model.invoke([
        SystemMessage(content="You are a helpful assistant. Answer the task thoroughly."),
        HumanMessage(content=task),
    ])
    elapsed = time.time() - start
    output = response.content or ""
    if not output:
        raise ValueError("Direct LLM returned empty response")
    return {
        "output": output,
        "time_seconds": round(elapsed, 2),
        "output_length": len(output),
    }


def hivemind_call(task: str) -> dict:
    """Full HiveMind pipeline: quick-check → debate → forge → agents → compile."""
    result = run_task(task)
    meta = result.get("metadata", {})
    output = result.get("final_output", "")
    return {
        "output": output,
        "time_seconds": 0,  # filled by caller from wall clock
        "output_length": len(output),
        "debate_time_s": meta.get("debate_time_s", 0),
        "forge_time_s": meta.get("forge_time_s", 0),
        "exec_time_s": meta.get("exec_time_s", 0),
        "total_agents": meta.get("total_agents", 0),
        "total_tools": meta.get("total_tools", 0),
        "known_issues": result.get("known_issues", []),
        "coverage_report": result.get("coverage_report", {}),
    }


def main():
    task = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else DEFAULT_TASK

    print("=" * 70)
    print("  HIVEMIND EVALUATION — Pipeline vs Direct LLM")
    print("=" * 70)
    print(f"\n  Task: {task}\n")
    print(f"  Timeout per call: {TASK_TIMEOUT_S}s\n")

    # ── 1. Direct LLM baseline ─────────────────────────────────────
    print("-" * 70)
    print("  1. DIRECT GPT-4o CALL (no planning, no agents)")
    print("-" * 70)

    direct = None
    direct_error = None
    try:
        start = time.time()
        direct = _run_with_timeout(direct_llm_call, task)
        direct["time_seconds"] = round(time.time() - start, 2)
    except (TimeoutError, Exception) as exc:
        direct_error = str(exc)
        print(f"\n  ERROR: {exc}")
        direct = {"output": "", "time_seconds": 0, "output_length": 0}

    if direct_error:
        print(f"\n  Direct call failed: {direct_error}")
    else:
        print(f"\n  Time   : {direct['time_seconds']}s")
        print(f"  Length : {direct['output_length']} chars")
        print(f"\n  Output preview:\n")
        preview = direct["output"][:600]
        print(preview)
        if direct["output_length"] > 600:
            print(f"\n  ... ({direct['output_length']} chars total)")

    # ── 2. HiveMind pipeline ───────────────────────────────────────
    print("\n" + "-" * 70)
    print("  2. HIVEMIND PIPELINE (debate → forge → agents → compile)")
    print("-" * 70)

    pipeline = None
    pipeline_error = None
    try:
        start = time.time()
        pipeline = _run_with_timeout(hivemind_call, task)
        pipeline["time_seconds"] = round(time.time() - start, 2)
    except (TimeoutError, Exception) as exc:
        pipeline_error = str(exc)
        print(f"\n  ERROR: {exc}")
        pipeline = {
            "output": "", "time_seconds": 0, "output_length": 0,
            "debate_time_s": 0, "forge_time_s": 0, "exec_time_s": 0,
            "total_agents": 0, "total_tools": 0,
            "known_issues": [], "coverage_report": {},
        }

    if pipeline_error:
        print(f"\n  Pipeline call failed: {pipeline_error}")
    else:
        print(f"\n  Time breakdown:")
        print(f"    Debate  : {pipeline['debate_time_s']}s")
        print(f"    Forge   : {pipeline['forge_time_s']}s")
        print(f"    Execute : {pipeline['exec_time_s']}s")
        print(f"    Total   : {pipeline['time_seconds']}s")
        print(f"  Agents    : {pipeline['total_agents']}")
        print(f"  Tools     : {pipeline['total_tools']}")
        print(f"  Length    : {pipeline['output_length']} chars")

        if pipeline["known_issues"]:
            print(f"\n  Known issues flagged by Compiler:")
            for issue in pipeline["known_issues"]:
                print(f"    - {issue}")

        print(f"\n  Output preview:\n")
        print(pipeline["output"][:600])
        if pipeline["output_length"] > 600:
            print(f"\n  ... ({pipeline['output_length']} chars total)")

    # ── 3. Comparison ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    direct_t = direct.get("time_seconds", 0)
    pipeline_t = pipeline.get("time_seconds", 0)
    direct_len = direct.get("output_length", 0)
    pipeline_len = pipeline.get("output_length", 0)

    time_ratio = round(pipeline_t / direct_t, 1) if direct_t > 0.01 else "N/A"
    length_ratio = round(pipeline_len / direct_len, 1) if direct_len > 0 else "N/A"

    print(f"""
  {'Metric':<30} {'Direct GPT-4o':<20} {'HiveMind Pipeline':<20}
  {'-' * 72}
  {'Time (seconds)':<30} {direct_t:<20} {pipeline_t:<20}
  {'Output length (chars)':<30} {direct_len:<20} {pipeline_len:<20}
  {'Agents spawned':<30} {'1 (implicit)':<20} {pipeline['total_agents']:<20}
  {'Tools generated':<30} {'0':<20} {pipeline['total_tools']:<20}
  {'Known issues flagged':<30} {'N/A':<20} {len(pipeline['known_issues']):<20}
  {'-' * 72}
  {'Time multiplier':<30} {'1x':<20} {time_ratio}{'x' if isinstance(time_ratio, float) else ''}
  {'Output length ratio':<30} {'1x':<20} {length_ratio}{'x' if isinstance(length_ratio, float) else ''}
""")

    if direct_error or pipeline_error:
        print("  ERRORS:")
        if direct_error:
            print(f"    Direct: {direct_error}")
        if pipeline_error:
            print(f"    Pipeline: {pipeline_error}")

    # ── 4. Save results ────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    results = {
        "task": task,
        "direct": direct,
        "hivemind": pipeline,
        "comparison": {
            "time_multiplier": time_ratio,
            "length_ratio": length_ratio,
            "direct_error": direct_error,
            "pipeline_error": pipeline_error,
        },
    }
    output_path = os.path.join("output", "evaluate_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Full results saved to {output_path}")

    return 0 if not (direct_error or pipeline_error) else 1


if __name__ == "__main__":
    sys.exit(main())
