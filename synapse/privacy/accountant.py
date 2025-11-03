from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RDPConfig:
    """
    Configuration for Rényi differential privacy accounting.
    """

    delta: float = 1e-6
    alpha: float = 10.0


class RDPAccountant:
    """
    Minimal RDP accountant tracking cumulative privacy loss.
    """

    def __init__(self, config: RDPConfig | None = None) -> None:
        self.config = config or RDPConfig()
        self._total_rdp: float = 0.0
        self._rounds: int = 0

    def accumulate(self, participation_rate: float, sigma: float) -> None:
        """
        Add the contribution of a single aggregation round.
        """
        sigma = max(sigma, 1e-6)
        participation_rate = max(min(participation_rate, 1.0), 1e-6)
        increment = (self.config.alpha * (participation_rate ** 2)) / (2 * sigma ** 2)
        self._total_rdp += increment
        self._rounds += 1

    def epsilon(self) -> float:
        return float(self._total_rdp + math.log(1 / self.config.delta) / (self.config.alpha - 1))

    @property
    def rounds(self) -> int:
        return self._rounds
