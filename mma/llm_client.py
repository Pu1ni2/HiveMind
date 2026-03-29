import json

from openai import OpenAI

from config import LM_STUDIO_BASE_URL

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="lm-studio")
    return _client


def call_llm(model: str, messages: list, temperature: float | None = None, max_tokens: int = 2048) -> str:
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = get_client().chat.completions.create(**kwargs)
    return response.choices[0].message.content


def call_llm_json(model: str, messages: list, temperature: float | None = None, max_tokens: int = 2048) -> dict:
    raw = call_llm(model, messages, temperature=temperature, max_tokens=max_tokens)
    return json.loads(raw)
