import logging
import os
from typing import Callable, Iterable, List, Sequence, Tuple

import requests

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 524}
_RETRYABLE_MESSAGE_TOKENS = (
    "busy",
    "capacity",
    "overloaded",
    "rate limit",
    "temporarily unavailable",
    "timeout",
    "too many requests",
)

logger = logging.getLogger(__name__)


def _load_jina_api_key_pairs(*, raise_if_missing: bool = True) -> List[Tuple[str, str]]:
    keys: List[Tuple[str, str]] = []
    primary = os.environ.get("JINA_API_KEY")
    if primary:
        keys.append(("JINA_API_KEY", primary.strip()))

    suffix_entries: List[Tuple[int, str, str]] = []
    for name, value in os.environ.items():
        if not name.startswith("JINA_API_KEY_"):
            continue
        if not value:
            continue
        suffix = name.split("JINA_API_KEY_", 1)[-1]
        try:
            order = int(suffix)
        except ValueError:
            order = 10_000
        suffix_entries.append((order, name, value.strip()))

    for _, name, value in sorted(suffix_entries, key=lambda item: (item[0], item[1])):
        keys.append((name, value))

    if not keys and raise_if_missing:
        raise ValueError(
            "At least one JINA_API_KEY environment variable must be set. "
            "Supported variables: JINA_API_KEY, JINA_API_KEY_2, JINA_API_KEY_3, ..."
        )
    return keys


def get_available_jina_api_keys(*, allow_empty: bool = False) -> List[str]:
    pairs = _load_jina_api_key_pairs(raise_if_missing=not allow_empty)
    return [value for _, value in pairs]


def get_named_jina_api_keys(*, allow_empty: bool = False) -> List[Tuple[str, str]]:
    return _load_jina_api_key_pairs(raise_if_missing=not allow_empty)


def _is_retryable_jina_error(exc: requests.exceptions.RequestException) -> bool:
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True

    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int) and status_code in RETRYABLE_STATUS_CODES:
            return True

    message = str(exc).lower()
    return any(token in message for token in _RETRYABLE_MESSAGE_TOKENS)


class JinaAPIKeyRotator:
    """
    Helper that encapsulates multiple Jina API keys and retries requests using fallbacks.
    """

    def __init__(self, api_keys: Sequence[str] | Sequence[Tuple[str, str]] | str | None = None):
        if api_keys is None:
            pairs = _load_jina_api_key_pairs()
        else:
            pairs = self._normalize_keys(api_keys)

        if not pairs:
            raise ValueError("No Jina API keys provided.")

        self.api_keys: List[Tuple[str, str]] = pairs
        for name, key in self.api_keys:
            if not key.startswith("jina_"):
                logger.warning(
                    "Jina API key %s does not appear to use the expected format.", name
                )

    @staticmethod
    def _normalize_keys(
        raw_keys: Sequence[str] | Sequence[Tuple[str, str]] | str
    ) -> List[Tuple[str, str]]:
        normalized: List[Tuple[str, str]] = []
        if isinstance(raw_keys, str):
            raw_keys_list: List[str | Tuple[str, str]] = [raw_keys]
        else:
            raw_keys_list = list(raw_keys)

        if not raw_keys_list:
            return normalized

        first = raw_keys_list[0]
        if isinstance(first, tuple):
            for name, value in raw_keys_list:  # type: ignore[misc]
                if value:
                    normalized.append((name, value.strip()))
        else:
            for value in raw_keys_list:  # type: ignore[misc]
                if value:
                    normalized.append(("JINA_API_KEY", value.strip()))
        return normalized

    def execute(self, request_fn: Callable[[str], Iterable | dict | list | str | requests.Response]):
        """
        Executes a callable that performs a Jina API request using each key until one succeeds.
        """
        throttled_errors: List[Tuple[str, requests.exceptions.RequestException]] = []
        for env_name, api_key in self.api_keys:
            try:
                return request_fn(api_key)
            except requests.exceptions.RequestException as exc:
                if not _is_retryable_jina_error(exc):
                    raise
                throttled_errors.append((env_name, exc))
                logger.warning(
                    "Jina API key %s encountered a transient error (%s). Trying next key.",
                    env_name,
                    exc,
                )
                continue

        if throttled_errors:
            summary = "; ".join(f"{name}: {error}" for name, error in throttled_errors)
            last_error = throttled_errors[-1][1]
            raise RuntimeError(
                f"All configured Jina API keys failed due to transient errors: {summary}"
            ) from last_error

        raise RuntimeError("No Jina API keys succeeded for the request.")
