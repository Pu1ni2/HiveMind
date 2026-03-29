"""
Benchmark: ClawForge Pipeline vs Direct LLM Call

Compares token usage, time, and output between:
  1. The full ClawForge pipeline (debate + sub-agents + validation)
  2. A single direct LLM call with the same task
"""

import json
import time
import os
from .config import DEBATE_MODEL
from .llm_client import call_llm


def call_direct(task: str) -> tuple[str, dict]:
    """Single direct LLM call — no pipeline, no debate."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Respond thoroughly to the user's request."},
        {"role": "user", "content": task},
    ]
    content, usage = call_llm(DEBATE_MODEL, messages)
    return content, usage


def run_benchmark(task: str, tool_executor=None) -> dict:
    # Import here to avoid circular import
    from .pipeline import run_pipeline

    print("=" * 60)
    print("BENCHMARK: ClawForge Pipeline vs Direct LLM Call")
    print("=" * 60)
    print(f"Task: {task}\n")

    # --- Run 1: Full Pipeline ---
    print("-" * 60)
    print("Running: CLAWFORGE PIPELINE")
    print("-" * 60)

    pipeline_start = time.time()
    pipeline_result = run_pipeline(task, tool_executor=tool_executor)
    pipeline_time = time.time() - pipeline_start
    pipeline_metrics = pipeline_result.get("metrics", {})

    # --- Run 2: Direct LLM Call ---
    print("\n" + "-" * 60)
    print("Running: DIRECT LLM CALL")
    print("-" * 60)

    direct_start = time.time()
    direct_response, direct_usage = call_direct(task)
    direct_time = time.time() - direct_start

    print(f"\n[Direct LLM]\n{direct_response[:500]}...")

    # --- Comparison ---
    results = {
        "task": task,
        "pipeline": {
            "time_seconds": round(pipeline_time, 2),
            "prompt_tokens": pipeline_metrics.get("prompt_tokens", 0),
            "completion_tokens": pipeline_metrics.get("completion_tokens", 0),
            "total_tokens": pipeline_metrics.get("total_tokens", 0),
            "phase_1_2": pipeline_metrics.get("phase_1_2", {}),
            "phase_3_4": pipeline_metrics.get("phase_3_4", {}),
            "phase_5_6": pipeline_metrics.get("phase_5_6", {}),
        },
        "direct": {
            "time_seconds": round(direct_time, 2),
            "prompt_tokens": direct_usage["prompt_tokens"],
            "completion_tokens": direct_usage["completion_tokens"],
            "total_tokens": direct_usage["total_tokens"],
        },
        "comparison": {
            "time_multiplier": round(pipeline_time / direct_time, 2) if direct_time > 0 else 0,
            "token_multiplier": round(
                pipeline_metrics.get("total_tokens", 0) / direct_usage["total_tokens"], 2
            ) if direct_usage["total_tokens"] > 0 else 0,
        },
    }

    # --- Print Summary ---
    p = results["pipeline"]
    d = results["direct"]
    c = results["comparison"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Pipeline':>15} {'Direct':>15} {'Multiplier':>12}")
    print("-" * 67)
    print(f"{'Time (seconds)':<25} {p['time_seconds']:>15} {d['time_seconds']:>15} {c['time_multiplier']:>11}x")
    print(f"{'Prompt tokens':<25} {p['prompt_tokens']:>15} {d['prompt_tokens']:>15}")
    print(f"{'Completion tokens':<25} {p['completion_tokens']:>15} {d['completion_tokens']:>15}")
    print(f"{'Total tokens':<25} {p['total_tokens']:>15} {d['total_tokens']:>15} {c['token_multiplier']:>11}x")

    # Phase breakdown
    print(f"\n{'Phase Breakdown':<25} {'Tokens':>15} {'Time (s)':>15} {'Rounds':>12}")
    print("-" * 67)
    for phase_name, phase_key in [("Phase 1-2 (Reqs)", "phase_1_2"), ("Phase 3-4 (Plan)", "phase_3_4"), ("Phase 5-6 (Exec)", "phase_5_6")]:
        pm = p.get(phase_key, {})
        print(f"{phase_name:<25} {pm.get('total_tokens', 0):>15} {pm.get('time_seconds', 0):>15} {pm.get('rounds', 0):>12}")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "benchmark_results.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFull results saved to {output_path}")
    return results
