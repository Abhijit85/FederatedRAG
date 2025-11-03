from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque, Iterable, List, Optional, Sequence, Tuple
from collections import deque

from .contracts import AdapterUpdate


@dataclass
class SchedulerConfig:
    """
    Configuration for the asynchronous window scheduler.
    """

    window_seconds: int = 900
    max_pending_windows: int = 4
    late_decay: float = 0.7  # Weight factor applied to windows emitted late.


class AsyncWindowScheduler:
    """
    Buckets adapter updates into asynchronous aggregation windows.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()
        self._queue: Deque[Tuple[datetime, AdapterUpdate]] = deque()
        self._window_start: datetime | None = None

    def offer(self, update: AdapterUpdate, arrival: Optional[datetime] = None) -> None:
        """
        Buffer an update for the next aggregation window.
        """
        if arrival is None:
            arrival = datetime.now(timezone.utc)
        self._queue.append((arrival, update))
        if self._window_start is None:
            self._window_start = arrival

    def _pop_window(self, cutoff: datetime) -> List[AdapterUpdate]:
        batch: List[AdapterUpdate] = []
        while self._queue and self._queue[0][0] <= cutoff:
            _, update = self._queue.popleft()
            batch.append(update)
        if not self._queue:
            self._window_start = None
        else:
            self._window_start = self._queue[0][0]
        return batch

    def ready(self, now: Optional[datetime] = None) -> bool:
        if not self._queue:
            return False
        now = now or datetime.now(timezone.utc)
        if self._window_start is None:
            self._window_start = self._queue[0][0]
        elapsed = (now - self._window_start).total_seconds()
        return elapsed >= self.config.window_seconds

    def collect_ready(self, now: Optional[datetime] = None) -> List[AdapterUpdate]:
        """
        Retrieve the updates whose window has elapsed.
        """
        if not self.ready(now):
            return []
        now = now or datetime.now(timezone.utc)
        cutoff = self._window_start + timedelta(seconds=self.config.window_seconds)
        return self._pop_window(cutoff)

    def drain(self) -> List[AdapterUpdate]:
        """
        Flush every pending update, used for shutdown or tests.
        """
        drained: List[AdapterUpdate] = [update for _, update in self._queue]
        self._queue.clear()
        self._window_start = None
        return drained

    @property
    def size(self) -> int:
        return len(self._queue)
