import json
import os
from config import MAX_ROUNDS
from agents import call_dynamic_agent, call_evaluator_agent


def run_debate(task: str) -> dict:
    """
    Phase 1-2: DA generates requirements, Evaluator critiques them.
    Returns the final state with approved/modified requirements.
    """
    state = {
        "round": 0,
        "task": task,
        "requirements": None,
        "history": [],
        "approved": False,
    }

    while state["round"] < MAX_ROUNDS:
        state["round"] += 1
        print(f"\n--- Round {state['round']} ---")

        # Phase 1: DA generates requirements
        try:
            da_response = call_dynamic_agent(task, state["history"])
        except (json.JSONDecodeError, Exception) as e:
            print(f"\n[ERROR] Dynamic Agent returned invalid response: {e}")
            print("Using last valid requirements.")
            break

        state["requirements"] = da_response.get("requirements")
        da_content = json.dumps(da_response)
        state["history"].append({"role": "dynamic_agent", "content": da_content})
        print(f"[Dynamic Agent]\n{json.dumps(da_response, indent=2)}")

        # Phase 2: Evaluator critiques requirements
        try:
            eval_response = call_evaluator_agent(task, state["history"])
        except (json.JSONDecodeError, Exception) as e:
            print(f"\n[ERROR] Evaluator Agent returned invalid response: {e}")
            print("Using current requirements as final.")
            break

        eval_content = json.dumps(eval_response)
        state["history"].append({"role": "evaluator_agent", "content": eval_content})
        print(f"\n[Evaluator Agent]\n{json.dumps(eval_response, indent=2)}")

        if eval_response.get("approved"):
            state["approved"] = True
            state["requirements"] = eval_response.get("modified_requirements")
            print("\nEvaluator APPROVED the requirements.")
            break

        # Evaluator rejected — update requirements with modified version for next round
        state["requirements"] = eval_response.get("modified_requirements")
        print(f"\nEvaluator REJECTED. Critique: {eval_response.get('critique')}")

    if not state["approved"]:
        print(f"\nMax rounds ({MAX_ROUNDS}) reached. Using last requirements as-is.")

    # Write requirements to JSON
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "approved_requirements.json")

    with open(output_path, "w") as f:
        json.dump(
            {"approved": state["approved"], "requirements": state["requirements"]},
            f,
            indent=2,
        )

    print(f"\nRequirements written to {output_path}")
    return state


if __name__ == "__main__":
    task = "Research the top 3 AI frameworks in 2025 and summarize their pros and cons."
    final_state = run_debate(task)
    print(f"\nApproved: {final_state['approved']}")
