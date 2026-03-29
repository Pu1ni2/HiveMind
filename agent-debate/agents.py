import json
from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL
from prompts import DA_GENERATE_REQUIREMENTS_PROMPT, EVALUATOR_CRITIQUE_REQUIREMENTS_PROMPT

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)


def call_dynamic_agent(task: str, history: list) -> dict:
    """Phase 1: DA generates structured requirements from user request."""
    messages = _build_messages(task, history, speaker="dynamic_agent")
    messages.insert(0, {"role": "system", "content": DA_GENERATE_REQUIREMENTS_PROMPT})

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=messages,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def call_evaluator_agent(task: str, history: list) -> dict:
    """Phase 2: Evaluator critiques requirements and returns modified version."""
    messages = _build_messages(task, history, speaker="evaluator_agent")
    messages.insert(0, {"role": "system", "content": EVALUATOR_CRITIQUE_REQUIREMENTS_PROMPT})

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=messages,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def _build_messages(task: str, history: list, speaker: str) -> list:
    """Convert debate history into OpenAI-compatible message format."""
    messages = [{"role": "user", "content": f"Task: {task}"}]

    for entry in history:
        role = "assistant" if entry["role"] == speaker else "user"
        messages.append({"role": role, "content": entry["content"]})

    return messages
