import logging
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
except ImportError:  # pragma: no cover - fallback for older openai releases
    APIConnectionError = APIStatusError = APITimeoutError = RateLimitError = ()  # type: ignore[misc]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_MESSAGE_TOKENS = (
    "busy",
    "capacity",
    "overloaded",
    "rate limit",
    "temporarily unavailable",
    "timeout",
)

logger = logging.getLogger(__name__)


def _build_default_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_SITE_URL")
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.environ.get("OPENROUTER_SITE_NAME")
    if title:
        headers["X-Title"] = title
    return headers


def _load_api_keys_from_env(*, raise_if_missing: bool = True) -> List[Tuple[str, str]]:
    """
    Returns the list of configured API keys ordered by priority.
    Primary key (API_KEY) comes first, followed by API_KEY_<number> sorted numerically.
    """
    keys: List[Tuple[str, str]] = []
    primary = os.environ.get("API_KEY")
    if primary:
        keys.append(("API_KEY", primary.strip()))

    suffix_entries: List[Tuple[int, str, str]] = []
    for name, value in os.environ.items():
        if not name.startswith("API_KEY_"):
            continue
        if not value:
            continue
        suffix = name.split("API_KEY_", 1)[-1]
        try:
            order = int(suffix)
        except ValueError:
            # Non-numeric suffixes are placed at the end but kept deterministic.
            order = 10_000
        suffix_entries.append((order, name, value.strip()))

    for _, name, value in sorted(suffix_entries, key=lambda item: (item[0], item[1])):
        keys.append((name, value))

    if not keys and raise_if_missing:
        raise ValueError(
            "At least one API_KEY environment variable must be set. "
            "Supported variables: API_KEY, API_KEY_2, API_KEY_3, ..."
        )
    return keys


def get_available_api_keys(*, allow_empty: bool = False) -> List[str]:
    """
    Returns the configured API keys in priority order.
    When allow_empty=True the function returns an empty list instead of raising.
    """
    keys = _load_api_keys_from_env(raise_if_missing=not allow_empty)
    return [value for _, value in keys]


@lru_cache(maxsize=None)
def _get_client_for_api_key(api_key: str) -> OpenAI:
    headers = _build_default_headers()
    # OpenAI() accepts None, but passing {} avoids mutation surprises.
    default_headers: Optional[Dict[str, str]] = headers or {}

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=default_headers,
    )


def get_openrouter_client() -> OpenAI:
    """
    Returns an OpenRouter OpenAI client using the primary configured API key.
    """
    keys = _load_api_keys_from_env()
    return _get_client_for_api_key(keys[0][1])


def _iter_api_keys() -> Iterable[Tuple[str, str]]:
    return _load_api_keys_from_env()


def _extract_status_code(exc: Exception) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if isinstance(resp_status, int):
            return resp_status
    return None


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        status_code = _extract_status_code(exc)
        if status_code in RETRYABLE_STATUS_CODES:
            return True
    else:
        status_code = _extract_status_code(exc)
        if status_code in RETRYABLE_STATUS_CODES:
            return True

    message = str(exc).lower()
    return any(token in message for token in _RETRYABLE_MESSAGE_TOKENS)


def chat_completion(*, model: str, messages: Any, **kwargs: Any):
    """
    Convenience helper to create an OpenRouter chat completion.
    Additional OpenAI parameters (max_tokens, response_format, etc.) can be
    supplied via kwargs.
    """
    throttled_errors: List[Tuple[str, Exception]] = []
    for env_name, api_key in _iter_api_keys():
        client = _get_client_for_api_key(api_key)
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - network interactions
            if not _is_retryable_error(exc):
                raise
            throttled_errors.append((env_name, exc))
            logger.warning(
                "OpenRouter API key %s encountered a transient error (%s). Trying next key.",
                env_name,
                exc,
            )
            continue

    if throttled_errors:
        summary = "; ".join(f"{name}: {error}" for name, error in throttled_errors)
        last_error = throttled_errors[-1][1]
        raise RuntimeError(
            f"All configured API keys failed due to transient OpenRouter errors: {summary}"
        ) from last_error

    # Should be unreachable because _iter_api_keys raises if empty,
    # but keeping a fallback defensive guard.
    raise RuntimeError("No OpenRouter API keys are configured.")
