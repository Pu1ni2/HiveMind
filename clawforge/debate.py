import json
from .config import DEBATE_MODEL, MAX_ROUNDS
from .llm_client import call_llm_json


def run_debate(task: str, generator_prompt: str, critic_prompt: str,
               modified_key: str, input_context: str = "") -> dict:
    """
    Generic debate loop — works for both requirements (Phase 1-2)
    and plan (Phase 3-4).

    Args:
        task:             The original user task.
        generator_prompt: System prompt for the DA (generator side).
        critic_prompt:    System prompt for the Evaluator (critic side).
        modified_key:     Key the evaluator uses for its output
                          ("modified_requirements" or "modified_plan").
        input_context:    Extra context to prepend (e.g. approved requirements
                          when debating the plan).

    Returns:
        The evaluator's approved output (or last version if max rounds hit).
    """
    history = []
    result = None
    user_content = f"Task: {task}"
    if input_context:
        user_content += f"\n\n{input_context}"

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n  Round {round_num}/{MAX_ROUNDS}")

        # Generator (DA) proposes
        gen_messages = _build_messages(user_content, history, "generator")
        gen_messages.insert(0, {"role": "system", "content": generator_prompt})

        try:
            gen_response = call_llm_json(DEBATE_MODEL, gen_messages)
        except Exception as e:
            print(f"  [DA] Error: {e}")
            break

        history.append({"role": "generator", "content": json.dumps(gen_response)})
        result = gen_response
        print(f"  [DA] Proposed")

        # Critic (Evaluator) reviews
        crit_messages = _build_messages(user_content, history, "critic")
        crit_messages.insert(0, {"role": "system", "content": critic_prompt})

        try:
            crit_response = call_llm_json(DEBATE_MODEL, crit_messages)
        except Exception as e:
            print(f"  [Evaluator] Error: {e}")
            break

        history.append({"role": "critic", "content": json.dumps(crit_response)})

        if crit_response.get("approved"):
            print(f"  [Evaluator] APPROVED")
            return crit_response.get(modified_key, result)

        result = crit_response.get(modified_key, result)
        critique = crit_response.get("critique", "")
        print(f"  [Evaluator] REJECTED — {critique[:100]}")

    print(f"  Max rounds reached. Using last version.")
    return result


def _build_messages(user_content: str, history: list, speaker: str) -> list:
    messages = [{"role": "user", "content": user_content}]
    for entry in history:
        role = "assistant" if entry["role"] == speaker else "user"
        messages.append({"role": role, "content": entry["content"]})
    return messages
