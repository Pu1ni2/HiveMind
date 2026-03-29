import json
from openai import OpenAI
from .config import API_KEY, BASE_URL

_client = None

MAX_RETRIES = 2


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def _extract_usage(response) -> dict:
    """Pull token counts from the API response."""
    if response.usage:
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def call_llm(model: str, messages: list, json_mode: bool = False, max_tokens: int = 2048) -> tuple[str, dict]:
    """Returns (content, token_usage)."""
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(MAX_RETRIES + 1):
        response = get_client().chat.completions.create(**kwargs)
        content = response.choices[0].message.content

        if content and content.strip():
            return content, _extract_usage(response)

        print(f"[WARN] Empty response from {model} (attempt {attempt + 1}/{MAX_RETRIES + 1})")

    raise RuntimeError(f"Model {model} returned empty response after {MAX_RETRIES + 1} attempts")


def call_llm_json(model: str, messages: list, max_tokens: int = 2048) -> tuple[dict, dict]:
    """Returns (parsed_json, token_usage)."""
    raw, usage = call_llm(model, messages, json_mode=True, max_tokens=max_tokens)
    return json.loads(raw), usage
