"""
Evaluation script â€” compares ClawForge dynamic pipeline vs a direct LLM call.
Measures: tokens used, time taken, output quality.

Usage:
    python evaluate.py
    python evaluate.py "Your custom task here"
"""

import sys
import time
from dotenv import load_dotenv

load_dotenv()

from clawforge.llm_client import get_client, reset_token_count, get_token_count
from clawforge.config import DEBATE_MODEL
from clawforge import run_pipeline


DEFAULT_TASK = (
    "Research the top 3 AI agent frameworks in 2025 and "
    "summarize their pros and cons."
)


def direct_llm_call(task: str) -> dict:
    """Single LLM call â€” same model as the debate, same task."""
    client = get_client()

    start = time.time()

    response = client.chat.completions.create(
        model=DEBATE_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the task thoroughly."},
            {"role": "user",   "content": task},
        ],
    )

    elapsed = time.time() - start

    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    output = response.choices[0].message.content

    return {
        "output": output,
        "time_seconds": round(elapsed, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def pipeline_call(task: str) -> dict:
    """Full ClawForge pipeline â€” debate + sub-agents + validation."""
    reset_token_count()

    start = time.time()
    result = run_pipeline(task)
    elapsed = time.time() - start

    tokens = get_token_count()

    return {
        "output": result.get("final_output", ""),
        "time_seconds": round(elapsed, 2),
        "input_tokens": tokens["input_tokens"],
        "output_tokens": tokens["output_tokens"],
        "total_tokens": tokens["input_tokens"] + tokens["output_tokens"],
        "known_issues": result.get("known_issues", []),
    }


def main():
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TASK

    print("=" * 70)
    print("  CLAWFORGE EVALUATION")
    print("=" * 70)
    print(f"\n  Task: {task}\n")

    # --- Direct LLM call ---
    print("-" * 70)
    print("  1. DIRECT LLM CALL")
    print("-" * 70)

    direct = direct_llm_call(task)

    print(f"\n  Time   : {direct['time_seconds']}s")
    print(f"  Tokens : {direct['total_tokens']} (in: {direct['input_tokens']}, out: {direct['output_tokens']})")
    print(f"\n  Output:\n")
    print(direct["output"][:500])
    if len(direct["output"]) > 500:
        print(f"\n  ... ({len(direct['output'])} chars total)")

    # --- Pipeline call ---
    print("\n" + "-" * 70)
    print("  2. CLAWFORGE PIPELINE (debate + sub-agents + validation)")
    print("-" * 70)

    pipeline = pipeline_call(task)

    print(f"\n  Time   : {pipeline['time_seconds']}s")
    print(f"  Tokens : {pipeline['total_tokens']} (in: {pipeline['input_tokens']}, out: {pipeline['output_tokens']})")
    print(f"\n  Output:\n")
    print(pipeline["output"][:500])
    if len(pipeline["output"]) > 500:
        print(f"\n  ... ({len(pipeline['output'])} chars total)")

    # --- Comparison ---
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    time_ratio = pipeline["time_seconds"] / direct["time_seconds"] if direct["time_seconds"] > 0 else 0
    token_ratio = pipeline["total_tokens"] / direct["total_tokens"] if direct["total_tokens"] > 0 else 0

    print(f"""
  {'Metric':<25} {'Direct':<20} {'Pipeline':<20}
  {'-'*65}
  {'Time (seconds)':<25} {direct['time_seconds']:<20} {pipeline['time_seconds']:<20}
  {'Input tokens':<25} {direct['input_tokens']:<20} {pipeline['input_tokens']:<20}
  {'Output tokens':<25} {direct['output_tokens']:<20} {pipeline['output_tokens']:<20}
  {'Total tokens':<25} {direct['total_tokens']:<20} {pipeline['total_tokens']:<20}
  {'Output length (chars)':<25} {len(direct['output']):<20} {len(pipeline['output']):<20}
  {'-'*65}
  {'Time ratio':<25} {'1x':<20} {f'{time_ratio:.1f}x':<20}
  {'Token ratio':<25} {'1x':<20} {f'{token_ratio:.1f}x':<20}
""")


if __name__ == "__main__":
    main()
