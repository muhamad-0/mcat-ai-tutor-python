from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from openai import OpenAI

from .config import settings


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    use_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    if use_openrouter:
        api_key = os.getenv("OPENROUTER_API_KEY")
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set either OPENROUTER_API_KEY or OPENAI_API_KEY in backend/.env"
        )
    return OpenAI(api_key=api_key)


def chat_completion(messages: List[dict], temperature: float = 0.3) -> str:
    client = _client()
    model = (
        settings.openrouter_model
        if os.getenv("OPENROUTER_API_KEY")
        else settings.openai_model
    )
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""
