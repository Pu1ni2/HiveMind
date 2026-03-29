"""
HiveMind Benchmark Suite — runs HiveMind vs direct GPT-4o across
four task categories and produces a structured comparison report.

Task categories:
  1. Mathematical reasoning  — verifiable answer, tests token efficiency
  2. Code generation         — functional correctness, tests plan quality
  3. Software design         — architecture decisions, tests coherence
  4. Multi-domain research   — cross-domain synthesis, tests plan structure

Usage:
    python run_benchmark.py
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from orchestrator.config import OPENAI_API_KEY, PLANNER_MODEL
from orchestrator.pipeline import run_task

TASKS = [
    {
        "name": "Mathematical Reasoning",
        "task": (
            "Solve step by step: What is the derivative of f(x) = 3x^4 - 2x^2 + 7x - 5? "
            "Find f'(2), determine the critical points, and classify them as minima or maxima."
        ),
    },
    {
        "name": "Code Generation",
        "task": (
            "Write a Python implementation of a thread-safe LRU cache with a configurable max size. "
            "Include type hints, docstrings, a complete test suite using pytest, and usage examples."
        ),
    },
    {
        "name": "Complex Software Design",
        "task": (
            "Design the architecture for a real-time collaborative document editing system "
            "supporting 10,000 concurrent users. Cover the data model, conflict resolution strategy "
            "(CRDT or OT), WebSocket handling, storage layer, and deployment topology."
        ),
    },
    {
        "name": "Multi-Domain Research",
        "task": (
            "Produce a comprehensive competitive analysis of LangGraph, CrewAI, and AutoGen as "
            "multi-agent frameworks. Cover architecture, developer experience, production readiness, "
            "community size, and licensing. Provide a decision framework for choosing between them."
        ),
    },
]

# Per-task wall-clock timeout in seconds
TASK_TIMEOUT_S = 300


def _timed_call(fn, *args, timeout: int = TASK_TIMEOUT_S) -> tuple[dict, float]:
    """Run fn(*args) with a timeout. Returns (result_dict, elapsed_s)."""
    start = time.time()
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args)
        try:
            result = future.result(timeout=timeout)
        except FuturesTimeout:
            raise TimeoutError(f"Task timed out after {timeout}s")
    return result, time.time() - start


def direct_llm_call(task: str) -> dict:
    """Baseline: single GPT-4o call with no pipeline."""
    model = ChatOpenAI(model=PLANNER_MODEL, api_key=OPENAI_API_KEY, temperature=0.7)
    start = time.time()
    response = model.invoke([
        SystemMessage(content="You are a helpful assistant. Answer the task thoroughly."),
        HumanMessage(content=task),
    ])
    elapsed = time.time() - start
    output = response.content or ""
    return {
        "output": output,
        "time_seconds": round(elapsed, 2),
        "output_length": len(output),
    }


def hivemind_call(task: str) -> dict:
    """Full HiveMind pipeline."""
    result = run_task(task)
    meta = result.get("metadata", {})
    output = result.get("final_output", "")
    return {
        "output": output,
        "output_length": len(output),
        "debate_time_s": meta.get("debate_time_s", 0),
        "forge_time_s": meta.get("forge_time_s", 0),
        "exec_time_s": meta.get("exec_time_s", 0),
        "total_agents": meta.get("total_agents", 0),
        "total_tools": meta.get("total_tools", 0),
        "known_issues": result.get("known_issues", []),
        "coverage_report": result.get("coverage_report", {}),
    }


def run_single(entry: dict, checkpoint_dir: str = "output") -> dict:
    task = entry["task"]
    name = entry["name"]

    print("\n" + "=" * 70)
    print(f"  TASK: {name}")
    print("=" * 70)
    print(f"  {task[:120]}...\n")

    # ── Direct call ────────────────────────────────────────────────
    print("  [1/2] Direct GPT-4o call ...")
    direct_error = None
    direct: dict = {}
    try:
        direct, elapsed = _timed_call(direct_llm_call, task)
        direct["time_seconds"] = round(elapsed, 2)
        print(f"        Done — {direct['time_seconds']}s, {direct['output_length']} chars")
    except TimeoutError as exc:
        direct_error = str(exc)
        direct = {"output": "", "time_seconds": TASK_TIMEOUT_S, "output_length": 0, "error": direct_error}
        print(f"        TIMEOUT: {exc}")
    except Exception as exc:
        direct_error = str(exc)
        direct = {"output": "", "time_seconds": 0, "output_length": 0, "error": direct_error}
        print(f"        ERROR: {exc}")

    # ── HiveMind call ──────────────────────────────────────────────
    print("  [2/2] HiveMind pipeline ...")
    pipeline_error = None
    pipeline: dict = {}
    try:
        pipeline, elapsed = _timed_call(hivemind_call, task)
        pipeline["time_seconds"] = round(elapsed, 2)
        print(f"        Done — {pipeline['time_seconds']}s, {pipeline['output_length']} chars, "
              f"{pipeline['total_agents']} agents, {pipeline['total_tools']} tools")
    except TimeoutError as exc:
        pipeline_error = str(exc)
        pipeline = {
            "output": "", "output_length": 0, "time_seconds": TASK_TIMEOUT_S,
            "debate_time_s": 0, "forge_time_s": 0, "exec_time_s": 0,
            "total_agents": 0, "total_tools": 0,
            "known_issues": [], "coverage_report": {}, "error": pipeline_error,
        }
        print(f"        TIMEOUT: {exc}")
    except Exception as exc:
        pipeline_error = str(exc)
        pipeline = {
            "output": "", "output_length": 0, "time_seconds": 0,
            "debate_time_s": 0, "forge_time_s": 0, "exec_time_s": 0,
            "total_agents": 0, "total_tools": 0,
            "known_issues": [], "coverage_report": {}, "error": pipeline_error,
        }
        print(f"        ERROR: {exc}")

    direct_t = direct.get("time_seconds", 0)
    pipeline_t = pipeline.get("time_seconds", 0)
    direct_len = direct.get("output_length", 0)
    pipeline_len = pipeline.get("output_length", 0)

    time_multiplier = round(pipeline_t / direct_t, 2) if direct_t > 0.01 else 0
    length_ratio = round(pipeline_len / direct_len, 2) if direct_len > 0 else 0

    result = {
        "name": name,
        "task": task,
        "direct": direct,
        "hivemind": pipeline,
        "comparison": {
            "time_multiplier": time_multiplier,
            "length_ratio": length_ratio,
            "issues_caught": len(pipeline.get("known_issues", [])),
            "direct_error": direct_error,
            "pipeline_error": pipeline_error,
        },
    }

    # ── Checkpoint after each task so a crash doesn't lose data ───
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"benchmark_{name.lower().replace(' ', '_')}.json")
    with open(checkpoint_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    print("=" * 70)
    print("  HIVEMIND BENCHMARK SUITE")
    print("=" * 70)
    print(f"  Running {len(TASKS)} tasks — HiveMind pipeline vs direct GPT-4o\n")
    print(f"  Per-task timeout: {TASK_TIMEOUT_S}s\n")

    all_results = []
    for entry in TASKS:
        result = run_single(entry)
        all_results.append(result)

    # ── Summary table ─────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  BENCHMARK SUMMARY")
    print("=" * 100)

    print(f"\n  {'Task':<28} {'Direct(s)':>10} {'Pipeline(s)':>12} {'TimeX':>7} "
          f"{'Direct Chars':>13} {'Pipeline Chars':>15} {'LenX':>6} {'Issues':>8} {'Errors':>8}")
    print("  " + "-" * 110)

    time_multipliers = []
    length_ratios = []
    total_issues = 0
    error_count = 0

    for r in all_results:
        p = r["hivemind"]
        d = r["direct"]
        c = r["comparison"]
        has_error = bool(c.get("direct_error") or c.get("pipeline_error"))
        if has_error:
            error_count += 1
        if not has_error:
            time_multipliers.append(c["time_multiplier"])
            length_ratios.append(c["length_ratio"])
        total_issues += c["issues_caught"]
        print(
            f"  {r['name']:<28} "
            f"{d['time_seconds']:>10} "
            f"{p['time_seconds']:>12} "
            f"{c['time_multiplier']:>6}x "
            f"{d['output_length']:>13} "
            f"{p['output_length']:>15} "
            f"{c['length_ratio']:>5}x "
            f"{c['issues_caught']:>8} "
            f"{'ERR' if has_error else 'ok':>8}"
        )

    avg_time_x = round(sum(time_multipliers) / len(time_multipliers), 2) if time_multipliers else 0
    avg_len_x = round(sum(length_ratios) / len(length_ratios), 2) if length_ratios else 0

    print("  " + "-" * 110)
    print(f"\n  Average time multiplier    : {avg_time_x}x  (over {len(time_multipliers)} completed tasks)")
    print(f"  Average output length ratio: {avg_len_x}x")
    print(f"  Total issues flagged       : {total_issues}")
    print(f"  Tasks with errors          : {error_count}/{len(TASKS)}")

    # ── Save full results ─────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "tasks": all_results,
                "summary": {
                    "avg_time_multiplier": avg_time_x,
                    "avg_length_ratio": avg_len_x,
                    "total_issues_caught": total_issues,
                    "tasks_with_errors": error_count,
                    "completed_tasks": len(time_multipliers),
                },
            },
            f,
            indent=2,
        )
    print(f"\n  Full results saved to {output_path}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
