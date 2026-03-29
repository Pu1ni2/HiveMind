import json
import os
from clawforge.benchmark import run_benchmark

TASKS = [
    {
        "name": "Simple Math",
        "task": "Solve this step by step: What is the derivative of f(x) = 3x^4 - 2x^2 + 7x - 5? Then find f'(2).",
    },
    {
        "name": "Code Generation",
        "task": "Write a Python function that implements binary search on a sorted list. Include error handling and type hints.",
    },
    {
        "name": "Complex Program",
        "task": "Design and implement a REST API for a task management system with endpoints for CRUD operations, user authentication, and task assignment. Provide the full code using FastAPI and SQLAlchemy.",
    },
    {
        "name": "Research & Analysis",
        "task": "Compare microservices vs monolithic architecture. Analyze trade-offs in scalability, deployment complexity, team structure, and debugging. Provide a recommendation framework for choosing between them.",
    },
]


def main():
    all_results = []

    for i, entry in enumerate(TASKS, 1):
        print("\n" + "#" * 70)
        print(f"  TASK {i}/{len(TASKS)}: {entry['name']}")
        print("#" * 70)

        result = run_benchmark(entry["task"])
        result["name"] = entry["name"]
        all_results.append(result)

    # --- Final Summary Table ---
    print("\n" + "=" * 80)
    print("FINAL COMPARISON ACROSS ALL TASKS")
    print("=" * 80)

    print(f"\n{'Task':<25} {'Pipeline (s)':>14} {'Direct (s)':>14} {'Time X':>10} {'Pipeline Tok':>14} {'Direct Tok':>14} {'Tok X':>10}")
    print("-" * 101)

    for r in all_results:
        p = r["pipeline"]
        d = r["direct"]
        c = r["comparison"]
        print(
            f"{r['name']:<25} "
            f"{p['time_seconds']:>14} "
            f"{d['time_seconds']:>14} "
            f"{c['time_multiplier']:>9}x "
            f"{p['total_tokens']:>14} "
            f"{d['total_tokens']:>14} "
            f"{c['token_multiplier']:>9}x"
        )

    # Save all results
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "benchmark_all_tasks.json")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {output_path}")


if __name__ == "__main__":
    main()
