from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from synapse.hyfical.contracts import LayerUpdate
from synapse.secure import SecureAggregationProvider, build_secure_provider
from synapse.utils import env_int, env_str


@dataclass
class SecAggConfig:
    """
    Placeholder secure aggregation configuration.
    """

    protocol: str = "SecAgg+"
    key_rotation_rounds: int = 20
    provider: str = "simple"
    secret: str = "synapse-shared-secret"
    attestation_key: str = ""

    @classmethod
    def from_env(cls, prefix: str = "SYNAPSE") -> "SecAggConfig":
        defaults = cls()
        return cls(
            protocol=env_str(f"{prefix}_SECAGG_PROTOCOL", defaults.protocol) or defaults.protocol,
            key_rotation_rounds=env_int(f"{prefix}_SECAGG_KEY_ROTATION", defaults.key_rotation_rounds),
            provider=env_str(f"{prefix}_SECAGG_PROVIDER", defaults.provider) or defaults.provider,
            secret=env_str(f"{prefix}_SECAGG_SECRET", defaults.secret) or defaults.secret,
            attestation_key=env_str(f"{prefix}_SECAGG_ATTESTATION", defaults.attestation_key) or defaults.attestation_key,
        )


class SecAggAdapter:
    """
    Lightweight SecAgg stub that serializes vectors without true encryption.

    The intent is to provide a seam for integrating a production-grade
    secure aggregation implementation. Until then, the adapter performs
    identity masking so higher layers can be exercised end-to-end.
    """

    def __init__(self, config: SecAggConfig | None = None, provider: SecureAggregationProvider | None = None) -> None:
        self.config = config or SecAggConfig()
        self._provider = provider or build_secure_provider(
            self.config.provider,
            self.config.secret,
            attestation_key=self.config.attestation_key,
        )
        self._round_counter: int = 0

    def mask(self, client_id: str, round_hint: int, layer: str, vector: np.ndarray) -> Tuple[bytes, Dict[str, str], float]:
        """
        Serialize a vector and return its norm for telemetry.
        """
        masked, metadata = self._provider.mask(client_id, layer, round_hint, vector.astype(np.float32))
        norm = float(np.linalg.norm(vector))
        return masked, metadata, norm

    def unmask(self, layer_update: LayerUpdate) -> np.ndarray:
        return self._provider.unmask(layer_update).astype(np.float64)

    def next_round(self) -> None:
        self._round_counter += 1
        if self._round_counter % max(self.config.key_rotation_rounds, 1) == 0:
            self.rotate_keys()

    def rotate_keys(self) -> None:
        """
        Stub for key rotation. Production environment should integrate crypto primitives.
        """
        # No-op placeholder.
        return

    @property
    def provider(self) -> SecureAggregationProvider:
        return self._provider
