import json
import os
from config import MAX_ROUNDS
from agents import call_dynamic_agent, call_evaluator_agent


def run_debate(task: str) -> dict:
    """
    Run the Dynamic Agent <-> Evaluator Agent debate loop.
    Returns the approved plan as a dict, or the last plan if max rounds hit.
    """
    state = {
        "round": 0,
        "task": task,
        "plan": None,
        "history": [],
        "approved": False,
    }

    while state["round"] < MAX_ROUNDS:
        state["round"] += 1
        print(f"\n--- Round {state['round']} ---")

        # Dynamic Agent proposes or revises the plan
        da_response = call_dynamic_agent(task, state["history"])
        state["plan"] = da_response.get("plan")
        da_content = json.dumps(da_response)
        state["history"].append({"role": "dynamic_agent", "content": da_content})
        print(f"[Dynamic Agent]  {da_content}")

        # Evaluator Agent critiques or approves
        eval_response = call_evaluator_agent(task, state["history"])
        eval_content = json.dumps(eval_response)
        state["history"].append({"role": "evaluator_agent", "content": eval_content})
        print(f"[Evaluator Agent] {eval_content}")

        if eval_response.get("approved"):
            state["approved"] = True
            print("\nEvaluator APPROVED the plan.")
            break

        print(f"\nEvaluator REJECTED. Critique: {eval_response.get('critique')}")

    if not state["approved"]:
        print(f"\nMax rounds ({MAX_ROUNDS}) reached. Using last plan as-is.")

    # Write the approved plan to a JSON file
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "approved_plan.json")

    with open(output_path, "w") as f:
        json.dump({"approved": state["approved"], "plan": state["plan"]}, f, indent=2)

    print(f"\nPlan written to {output_path}")

    return state


if __name__ == "__main__":
    task = "Research the top 3 AI frameworks in 2025 and summarize their pros and cons."
    final_state = run_debate(task)
    print(f"\nApproved: {final_state['approved']}")
    print(f"Plan: {json.dumps(final_state['plan'], indent=2)}")
