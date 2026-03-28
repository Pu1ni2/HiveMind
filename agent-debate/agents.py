import json
import re
from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL
from prompts import DYNAMIC_AGENT_PROMPT, EVALUATOR_AGENT_PROMPT

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)


def call_dynamic_agent(task: str, history: list) -> dict:
    """Ask the Dynamic Agent to propose or revise a plan."""
    messages = _build_messages(task, history, speaker="dynamic_agent")
    messages.insert(0, {"role": "system", "content": DYNAMIC_AGENT_PROMPT})

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=messages,
    )

    return _parse_json(response.choices[0].message.content)


def call_evaluator_agent(task: str, history: list) -> dict:
    """Ask the Evaluator Agent to critique or approve the plan."""
    messages = _build_messages(task, history, speaker="evaluator_agent")
    messages.insert(0, {"role": "system", "content": EVALUATOR_AGENT_PROMPT})

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=messages,
    )

    return _parse_json(response.choices[0].message.content)


def _parse_json(text: str) -> dict:
    """Extract JSON from model response, stripping markdown fences if present."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)


def _build_messages(task: str, history: list, speaker: str) -> list:
    """Convert debate history into OpenAI-compatible message format."""
    messages = [{"role": "user", "content": f"Task: {task}"}]

    for entry in history:
        role = "assistant" if entry["role"] == speaker else "user"
        messages.append({"role": role, "content": entry["content"]})

    return messages
