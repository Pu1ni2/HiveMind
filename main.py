import sys
from clawforge import run_pipeline

task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
    
"Consider a function ( f : \mathbb{R} \to \mathbb{R} ) defined on all real numbers. The function satisfies the functional equation ( f(x + y) + f(x - y) = 2f(x)f(y) ) for every pair of real numbers ( x ) and ( y ), along with the initial condition ( f(0) = 1 ). Your task is to determine all possible functions ( f(x) ) that satisfy both the functional equation and the given condition. The solution should include all valid forms of ( f(x) ) expressed in closed form, and must hold true for every real input. This problem requires careful reasoning about functional equations, exploring substitutions, and recognizing patterns that may relate to known function families such as exponential or trigonometric functions."

)

result = run_pipeline(task)

print("\n--- FINAL OUTPUT ---")
print(result["final_output"])

if result["known_issues"]:
    print("\n--- KNOWN ISSUES ---")
    for issue in result["known_issues"]:
        print(f"  - {issue}")
