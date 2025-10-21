import os
from functools import lru_cache
from typing import Any, Dict, Optional

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _build_default_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_SITE_URL")
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.environ.get("OPENROUTER_SITE_NAME")
    if title:
        headers["X-Title"] = title
    return headers


@lru_cache(maxsize=1)
def get_openrouter_client() -> OpenAI:
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable not set.")

    headers = _build_default_headers()
    # OpenAI() accepts None, but passing {} avoids mutation surprises.
    default_headers: Optional[Dict[str, str]] = headers or {}

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=default_headers,
    )


def chat_completion(*, model: str, messages: Any, **kwargs: Any):
    """
    Convenience helper to create an OpenRouter chat completion.
    Additional OpenAI parameters (max_tokens, response_format, etc.) can be
    supplied via kwargs.
    """
    client = get_openrouter_client()
    return client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
