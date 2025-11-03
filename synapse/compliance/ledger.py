from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class LedgerEntry:
    """
    Immutable record of a federated aggregation release.
    """

    round_id: int
    timestamp: datetime
    epsilon: float
    delta: float
    participant_count: int
    layers_updated: List[str]
    spectral_k: int
    dp_sigma: float
    release_notes: str
    telemetry_snapshot: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


class ComplianceLedger:
    """
    In-memory ledger with optional persistence hook.
    """

    def __init__(self) -> None:
        self._entries: List[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> None:
        self._entries.append(entry)

    def latest(self) -> Optional[LedgerEntry]:
        return self._entries[-1] if self._entries else None

    def to_list(self) -> List[Dict[str, object]]:
        return [entry.to_dict() for entry in self._entries]
