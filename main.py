import sys
from clawforge import run_pipeline

task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
    "Write a Python function that takes a list of numbers and returns the sum of the squares of those numbers."
)

result = run_pipeline(task)

print("\n--- FINAL OUTPUT ---")
print(result["final_output"])

if result["known_issues"]:
    print("\n--- KNOWN ISSUES ---")
    for issue in result["known_issues"]:
        print(f"  - {issue}")
