import json
from openai import OpenAI
from .config import API_KEY, BASE_URL

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def call_llm(model: str, messages: list, json_mode: bool = False, max_tokens: int = 2048) -> str:
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = get_client().chat.completions.create(**kwargs)
    return response.choices[0].message.content


def call_llm_json(model: str, messages: list, max_tokens: int = 2048) -> dict:
    raw = call_llm(model, messages, json_mode=True, max_tokens=max_tokens)
    return json.loads(raw)
