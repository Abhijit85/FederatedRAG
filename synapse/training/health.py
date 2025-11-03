from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque, List, Optional

from synapse.utils import env_int


@dataclass
class HealthConfig:
    """
    Health monitoring configuration for client runtimes.
    """

    heartbeat_interval_seconds: int = 30
    retry_queue_limit: int = 5
    offline_grace_period_seconds: int = 120

    @classmethod
    def from_env(cls, prefix: str = "SYNAPSE") -> "HealthConfig":
        defaults = cls()
        return cls(
            heartbeat_interval_seconds=env_int(f"{prefix}_HEARTBEAT_INTERVAL", defaults.heartbeat_interval_seconds),
            retry_queue_limit=env_int(f"{prefix}_RETRY_QUEUE_LIMIT", defaults.retry_queue_limit),
            offline_grace_period_seconds=env_int(f"{prefix}_OFFLINE_GRACE", defaults.offline_grace_period_seconds),
        )


@dataclass
class RetryRecord:
    round_hint: int
    attempts: int = 0
    last_attempt_ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HealthAgent:
    """
    Tracks heartbeat state, retry queues, and resume checkpoints for clients.
    """

    def __init__(self, config: HealthConfig | None = None) -> None:
        self.config = config or HealthConfig()
        self._last_heartbeat: Optional[datetime] = None
        self._retry_queue: Deque[RetryRecord] = deque(maxlen=self.config.retry_queue_limit)
        self._last_ack_version: Optional[str] = None

    def heartbeat(self, timestamp: Optional[datetime] = None) -> None:
        self._last_heartbeat = timestamp or datetime.now(timezone.utc)

    def record_retry(self, round_hint: int) -> None:
        for record in self._retry_queue:
            if record.round_hint == round_hint:
                record.attempts += 1
                record.last_attempt_ts = datetime.now(timezone.utc)
                return
        self._retry_queue.append(RetryRecord(round_hint=round_hint, attempts=1))

    def mark_ack(self, version: str) -> None:
        self._last_ack_version = version

    def is_healthy(self, now: Optional[datetime] = None) -> bool:
        if self._last_heartbeat is None:
            return False
        now = now or datetime.now(timezone.utc)
        elapsed = (now - self._last_heartbeat).total_seconds()
        return elapsed <= self.config.offline_grace_period_seconds

    def pending_retries(self) -> List[RetryRecord]:
        return list(self._retry_queue)

    @property
    def last_ack_version(self) -> Optional[str]:
        return self._last_ack_version
