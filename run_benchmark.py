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
import time
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


def direct_llm_call(task: str) -> dict:
    """Baseline: single GPT-4o call with no pipeline."""
    model = ChatOpenAI(model=PLANNER_MODEL, api_key=OPENAI_API_KEY, temperature=0.7)
    start = time.time()
    response = model.invoke([
        SystemMessage(content="You are a helpful assistant. Answer the task thoroughly."),
        HumanMessage(content=task),
    ])
    elapsed = time.time() - start
    output = response.content
    return {
        "output": output,
        "time_seconds": round(elapsed, 2),
        "output_length": len(output),
    }


def hivemind_call(task: str) -> dict:
    """Full HiveMind pipeline."""
    start = time.time()
    result = run_task(task)
    elapsed = time.time() - start

    meta = result.get("metadata", {})
    return {
        "output": result.get("final_output", ""),
        "time_seconds": round(elapsed, 2),
        "output_length": len(result.get("final_output", "")),
        "debate_time_s": meta.get("debate_time_s", 0),
        "forge_time_s": meta.get("forge_time_s", 0),
        "exec_time_s": meta.get("exec_time_s", 0),
        "total_agents": meta.get("total_agents", 0),
        "total_tools": meta.get("total_tools", 0),
        "known_issues": result.get("known_issues", []),
        "coverage_report": result.get("coverage_report", {}),
    }


def run_single(entry: dict) -> dict:
    task = entry["task"]
    name = entry["name"]

    print("\n" + "=" * 70)
    print(f"  TASK: {name}")
    print("=" * 70)
    print(f"  {task[:120]}...\n")

    print("  [1/2] Direct GPT-4o call ...")
    direct = direct_llm_call(task)
    print(f"        Done — {direct['time_seconds']}s, {direct['output_length']} chars")

    print("  [2/2] HiveMind pipeline ...")
    pipeline = hivemind_call(task)
    print(f"        Done — {pipeline['time_seconds']}s, {pipeline['output_length']} chars, "
          f"{pipeline['total_agents']} agents, {pipeline['total_tools']} tools")

    time_multiplier = (
        round(pipeline["time_seconds"] / direct["time_seconds"], 2)
        if direct["time_seconds"] > 0 else 0
    )
    length_ratio = (
        round(pipeline["output_length"] / direct["output_length"], 2)
        if direct["output_length"] > 0 else 0
    )

    return {
        "name": name,
        "task": task,
        "direct": direct,
        "hivemind": pipeline,
        "comparison": {
            "time_multiplier": time_multiplier,
            "length_ratio": length_ratio,
            "issues_caught": len(pipeline["known_issues"]),
        },
    }


def main():
    print("=" * 70)
    print("  HIVEMIND BENCHMARK SUITE")
    print("=" * 70)
    print(f"  Running {len(TASKS)} tasks — HiveMind pipeline vs direct GPT-4o\n")

    all_results = []
    for entry in TASKS:
        result = run_single(entry)
        all_results.append(result)

    # ── Summary table ───────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  BENCHMARK SUMMARY")
    print("=" * 100)

    print(f"\n  {'Task':<28} {'Direct(s)':>10} {'Pipeline(s)':>12} {'TimeX':>7} "
          f"{'Direct Chars':>13} {'Pipeline Chars':>15} {'LenX':>6} {'Issues':>8}")
    print("  " + "-" * 97)

    for r in all_results:
        p = r["hivemind"]
        d = r["direct"]
        c = r["comparison"]
        print(
            f"  {r['name']:<28} "
            f"{d['time_seconds']:>10} "
            f"{p['time_seconds']:>12} "
            f"{c['time_multiplier']:>6}x "
            f"{d['output_length']:>13} "
            f"{p['output_length']:>15} "
            f"{c['length_ratio']:>5}x "
            f"{c['issues_caught']:>8}"
        )

    # Averages
    avg_time_x = round(sum(r["comparison"]["time_multiplier"] for r in all_results) / len(all_results), 2)
    avg_len_x = round(sum(r["comparison"]["length_ratio"] for r in all_results) / len(all_results), 2)
    total_issues = sum(r["comparison"]["issues_caught"] for r in all_results)

    print("  " + "-" * 97)
    print(f"\n  Average time multiplier  : {avg_time_x}x")
    print(f"  Average output length ratio: {avg_len_x}x")
    print(f"  Total issues flagged by compiler: {total_issues}")

    # ── Save results ────────────────────────────────────────────────
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
                },
            },
            f,
            indent=2,
        )
    print(f"\n  Full results saved to {output_path}")


if __name__ == "__main__":
    main()
