from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List


def env_str(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def env_bool(key: str, default: bool = False) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def env_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def env_list(key: str, default: Iterable[str]) -> List[str]:
    value = os.environ.get(key)
    if value is None or not value.strip():
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def env_path_list(key: str, default: Iterable[Path]) -> List[Path]:
    default_paths = list(default)
    value = os.environ.get(key)
    if value is None or not value.strip():
        return default_paths
    return [Path(item.strip()) for item in value.split(",") if item.strip()]
