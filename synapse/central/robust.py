from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class GradientAuditRecord:
    """
    Captures diagnostic information for trimmed/flagged gradients.
    """

    batch_id: str
    reason: str
    score: float


@dataclass
class GradientAuditor:
    """
    Applies batch-level robustness checks prior to aggregation.
    """

    trim_percent: float = 0.05
    cosine_tau: float = 0.65
    ema_decay: float = 0.9
    _ema_vector: np.ndarray | None = field(default=None, init=False)
    _history: List[GradientAuditRecord] = field(default_factory=list, init=False)

    def audit(
        self,
        batch_updates: Sequence[Tuple[str, np.ndarray]],
    ) -> Tuple[List[Tuple[str, np.ndarray]], List[GradientAuditRecord]]:
        """
        Filter or down-weight pathological gradients.
        """
        kept: List[Tuple[str, np.ndarray]] = []
        records: List[GradientAuditRecord] = []

        if not batch_updates:
            return kept, records

        norms = np.array([np.linalg.norm(vector) for _, vector in batch_updates], dtype=np.float64)
        if norms.size == 0:
            return kept, records

        # Trim largest norms (potential high-loss batches).
        trim = max(int(len(batch_updates) * self.trim_percent), 0)
        sorted_indices = np.argsort(norms)
        trimmed_indices = set(sorted_indices[-trim:]) if trim > 0 else set()

        for idx, (batch_id, vector) in enumerate(batch_updates):
            reason = None
            score = 0.0

            if idx in trimmed_indices:
                reason = "trimmed_norm"
                score = float(norms[idx])

            cosine = self._cosine(vector)
            if cosine is not None and cosine < self.cosine_tau:
                reason = "cosine_disagreement"
                score = float(cosine)

            if reason:
                records.append(GradientAuditRecord(batch_id=batch_id, reason=reason, score=score))
                continue

            kept.append((batch_id, vector))
            self._update_ema(vector)

        self._history.extend(records)
        return kept, records

    def _cosine(self, vector: np.ndarray) -> float | None:
        if self._ema_vector is None:
            return None
        denom = np.linalg.norm(vector) * np.linalg.norm(self._ema_vector)
        if denom <= 0:
            return None
        return float(np.dot(vector, self._ema_vector) / denom)

    def _update_ema(self, vector: np.ndarray) -> None:
        if self._ema_vector is None:
            self._ema_vector = vector.copy()
            return
        self._ema_vector = (self.ema_decay * self._ema_vector) + ((1 - self.ema_decay) * vector)

    @property
    def history(self) -> List[GradientAuditRecord]:
        return list(self._history)
