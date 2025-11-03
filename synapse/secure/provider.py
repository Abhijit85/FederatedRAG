from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from synapse.hyfical.contracts import LayerUpdate


class SecureAggregationProvider:
    """
    Interface for masking/unmasking adapter updates. Concrete implementations can
    forward to cryptographic SecAgg systems or trusted enclaves.
    """

    def mask(
        self,
        client_id: str,
        layer: str,
        round_hint: int,
        vector: np.ndarray,
    ) -> Tuple[bytes, Dict[str, str]]:
        raise NotImplementedError

    def unmask(self, layer_update: LayerUpdate) -> np.ndarray:
        raise NotImplementedError

    def attest(self) -> Dict[str, str]:
        """Return attestation metadata (if applicable)."""

        return {}


@dataclass
class SimpleMaskingProvider(SecureAggregationProvider):
    """
    Demonstration provider that applies deterministic masks derived from a shared
    secret. Designed for testing; not cryptographically secure.
    """

    secret: str

    def _rng(self, client_id: str, layer: str, round_hint: int, size: int) -> np.random.Generator:
        material = f"{self.secret}:{client_id}:{layer}:{round_hint}:{size}".encode("utf-8")
        digest = hashlib.sha256(material).digest()[:8]
        seed = int.from_bytes(digest, "big")
        return np.random.default_rng(seed)

    def mask(
        self,
        client_id: str,
        layer: str,
        round_hint: int,
        vector: np.ndarray,
    ) -> Tuple[bytes, Dict[str, str]]:
        rng = self._rng(client_id, layer, round_hint, vector.size)
        mask = rng.standard_normal(size=vector.size, dtype=np.float32).reshape(vector.shape)
        masked = (vector + mask).astype(np.float32)
        metadata = {
            "shape": ";".join(str(dim) for dim in vector.shape),
            "round": str(round_hint),
            "client": client_id,
        }
        return masked.tobytes(), metadata

    def unmask(self, layer_update: LayerUpdate) -> np.ndarray:
        shape = tuple(int(dim) for dim in layer_update.mask_metadata.get("shape", "").split(";") if dim)
        if not shape:
            raise ValueError("Missing shape metadata for secure aggregation unmasking")
        client_id = layer_update.mask_metadata.get("client", "unknown")
        round_hint = int(layer_update.mask_metadata.get("round", "0"))
        vector = np.frombuffer(layer_update.masked_delta, dtype=np.float32).reshape(shape)
        rng = self._rng(client_id, layer_update.layer, round_hint, vector.size)
        mask = rng.standard_normal(size=vector.size, dtype=np.float32).reshape(vector.shape)
        return vector - mask


@dataclass
class TEEAggregationProvider(SimpleMaskingProvider):
    """
    Simulated trusted-enclave provider. In production this would proxy updates to
    an attested enclave for decryption and robust aggregation. Here we reuse the
    simple masking logic but expose attestation hooks for orchestration.
    """

    attestation_key: str = ""

    def attest(self) -> Dict[str, str]:
        token = hashlib.sha256((self.secret + self.attestation_key).encode("utf-8")).hexdigest()
        return {"enclave_attestation": token[:16]}


def build_secure_provider(mode: str, secret: str, **kwargs) -> SecureAggregationProvider:
    normalized = (mode or "simple").strip().lower()
    if normalized in {"tee", "enclave"}:
        return TEEAggregationProvider(secret=secret, attestation_key=kwargs.get("attestation_key", ""))
    return SimpleMaskingProvider(secret=secret)
