import json
import time
from .config import DEBATE_MODEL, MAX_ROUNDS
from .llm_client import call_llm_json


def run_debate(task: str, generator_prompt: str, critic_prompt: str,
               modified_key: str, input_context: str = "") -> tuple[dict, dict]:
    """
    Generic debate loop — works for both requirements (Phase 1-2)
    and plan (Phase 3-4).

    Returns:
        (approved_output, metrics) where metrics has token_usage, time, rounds.
    """
    history = []
    result = None
    user_content = f"Task: {task}"
    if input_context:
        user_content += f"\n\n{input_context}"

    metrics = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "rounds": 0,
        "time_seconds": 0,
    }

    start_time = time.time()

    for round_num in range(1, MAX_ROUNDS + 1):
        metrics["rounds"] = round_num
        print(f"\n  Round {round_num}/{MAX_ROUNDS}")

        # Generator (DA) proposes
        gen_messages = _build_messages(user_content, history, "generator")
        gen_messages.insert(0, {"role": "system", "content": generator_prompt})

        try:
            gen_response, gen_usage = call_llm_json(DEBATE_MODEL, gen_messages)
        except Exception as e:
            print(f"  [DA] Error: {e}")
            break

        _accumulate(metrics, gen_usage)
        history.append({"role": "generator", "content": json.dumps(gen_response)})
        result = gen_response
        print(f"  [DA] Proposed")

        # Critic (Evaluator) reviews
        crit_messages = _build_messages(user_content, history, "critic")
        crit_messages.insert(0, {"role": "system", "content": critic_prompt})

        try:
            crit_response, crit_usage = call_llm_json(DEBATE_MODEL, crit_messages)
        except Exception as e:
            print(f"  [Evaluator] Error: {e}")
            break

        _accumulate(metrics, crit_usage)
        history.append({"role": "critic", "content": json.dumps(crit_response)})

        if crit_response.get("approved"):
            print(f"  [Evaluator] APPROVED")
            result = crit_response.get(modified_key, result)
            break

        result = crit_response.get(modified_key, result)
        critique = crit_response.get("critique", "")
        print(f"  [Evaluator] REJECTED — {critique[:100]}")

    else:
        print(f"  Max rounds reached. Using last version.")

    metrics["time_seconds"] = round(time.time() - start_time, 2)
    return result, metrics


def _accumulate(metrics: dict, usage: dict) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        metrics[key] += usage.get(key, 0)


def _build_messages(user_content: str, history: list, speaker: str) -> list:
    messages = [{"role": "user", "content": user_content}]
    for entry in history:
        role = "assistant" if entry["role"] == speaker else "user"
        messages.append({"role": role, "content": entry["content"]})
    return messages
