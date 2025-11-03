from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from synapse.utils import env_float


@dataclass
class DPConfig:
    """
    Differential privacy hyperparameters for client-side guards.
    """

    clip_norm: float = 0.5
    noise_multiplier: float = 1.2
    sample_rate: float = 0.1

    @classmethod
    def from_env(cls, prefix: str = "SYNAPSE") -> "DPConfig":
        defaults = cls()
        return cls(
            clip_norm=env_float(f"{prefix}_DP_CLIP", defaults.clip_norm),
            noise_multiplier=env_float(f"{prefix}_DP_NOISE", defaults.noise_multiplier),
            sample_rate=env_float(f"{prefix}_DP_SAMPLE_RATE", defaults.sample_rate),
        )


class DifferentialPrivacyGuard:
    """
    Applies per-layer clipping and Gaussian noise to LoRA deltas.
    """

    def __init__(self, config: DPConfig | None = None, rng: np.random.Generator | None = None) -> None:
        self.config = config or DPConfig()
        self._rng = rng or np.random.default_rng()

    def sanitize(self, vector: np.ndarray) -> np.ndarray:
        clipped = self._clip(vector)
        return self._add_noise(clipped)

    def _clip(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm <= self.config.clip_norm or norm == 0:
            return vector
        scale = self.config.clip_norm / norm
        return vector * scale

    def _add_noise(self, vector: np.ndarray) -> np.ndarray:
        sigma = self.config.noise_multiplier * self.config.clip_norm
        noise = self._rng.normal(scale=sigma, size=vector.shape)
        return vector + noise

    def estimate_local_epsilon(self, steps: int) -> float:
        """
        Rough ε estimate using the strong composition theorem.
        """
        if self.config.noise_multiplier <= 0 or self.config.sample_rate <= 0:
            return float("inf")
        steps = max(steps, 1)
        return float(
            steps
            * (self.config.sample_rate ** 2)
            / max(self.config.noise_multiplier, 1e-6)
        )

    def clip_norm(self) -> float:
        return float(self.config.clip_norm)

    def sigma(self) -> float:
        return float(self.config.noise_multiplier * self.config.clip_norm)
