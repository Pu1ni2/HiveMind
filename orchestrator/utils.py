"""Shared helpers used across the orchestrator package."""

import json
import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


def parse_json_response(text: str) -> dict:
    """Parse JSON from an LLM response, stripping markdown fences if present."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def truncate(text: str, max_chars: int = 12000) -> str:
    """Truncate text to max_chars, appending an ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n... [truncated]"


def call_llm(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    *,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    json_mode: bool = False,
) -> dict[str, Any]:
    """Invoke a ChatOpenAI model and return the parsed JSON response.

    Centralises the repeated ChatOpenAI(...).invoke([System, Human]) +
    parse_json_response() pattern used in debate, compiler, quick_actions, etc.

    Parameters
    ----------
    model_name  : OpenAI model identifier (e.g. "gpt-4o").
    system_prompt : Content for the SystemMessage.
    user_prompt   : Content for the HumanMessage.
    api_key       : OpenAI API key.
    temperature   : Sampling temperature (default 0 for deterministic JSON calls).
    max_tokens    : Optional token cap for the completion.
    json_mode     : When True, instructs the model to return strict JSON via
                    response_format={"type": "json_object"}.

    Returns
    -------
    Parsed dict from the model response.

    Raises
    ------
    json.JSONDecodeError  : If the response is not valid JSON.
    Exception             : Any LangChain / OpenAI transport error.
    """
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    model = ChatOpenAI(**kwargs)
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return parse_json_response(response.content)
