from __future__ import annotations

import math
import re
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional

from synapse.knowledge.compendium import KnowledgeArtifact
from synapse.privacy.encryption import SynapseEncryptor


def _laplace(scale: float) -> float:
    # Inverse transform sampling for Laplace(0, scale)
    u = random.random() - 0.5
    return -scale * (1 if u > 0 else -1) * math.log(1 - 2 * abs(u))


@dataclass
class PrivacyPolicy:
    """
    Placeholder privacy policy that can redact or drop artifacts.
    """

    redact_sensitive_metadata: bool = True
    drop_pii_text: bool = True
    dp_epsilon: Optional[float] = None
    encryption_secret: Optional[str] = None

    def __post_init__(self) -> None:
        self._encryptor: Optional[SynapseEncryptor] = None
        if self.encryption_secret:
            self._encryptor = SynapseEncryptor(self.encryption_secret)

    def enforce(self, artifacts: Iterable[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        """
        Apply lightweight redaction of metadata flagged as sensitive.

        Future versions will integrate DP noise and encryption hooks.
        """
        sanitized: List[KnowledgeArtifact] = []
        for artifact in artifacts:
            text = artifact.text
            if self.drop_pii_text and self._looks_like_pii(text):
                continue

            metadata = artifact.metadata or {}
            if self.redact_sensitive_metadata:
                filtered_meta = {
                    key: value
                    for key, value in metadata.items()
                    if not key.lower().startswith("pii")
                }
            else:
                filtered_meta = metadata

            sanitized.append(
                KnowledgeArtifact(
                    signature=artifact.signature,
                    text=text,
                    structured_payload=artifact.structured_payload,
                    metadata=filtered_meta,
                )
            )
        return sanitized

    def apply_dp_noise(self, artifacts: Iterable[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        """
        Apply Laplace noise to numeric metadata/structured payload entries.
        """
        if not self.dp_epsilon:
            return list(artifacts)

        scale = 1.0 / max(self.dp_epsilon, 1e-6)
        protected: List[KnowledgeArtifact] = []

        for artifact in artifacts:
            metadata = dict(artifact.metadata)
            for key, value in list(metadata.items()):
                if isinstance(value, (int, float)):
                    metadata[key] = value + _laplace(scale)

            structured = artifact.structured_payload
            if isinstance(structured, dict):
                structured = self._apply_noise_to_payload(structured, scale)

            protected.append(
                KnowledgeArtifact(
                    signature=artifact.signature,
                    text=artifact.text,
                    structured_payload=structured,
                    metadata=metadata,
                )
            )
        return protected

    def _apply_noise_to_payload(self, payload: dict, scale: float) -> dict:
        updated = {}
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                noise = random.laplacevariate(0.0, scale)
                updated[key] = value + noise
            elif isinstance(value, list):
                updated[key] = [
                    item + _laplace(scale) if isinstance(item, (int, float)) else item
                    for item in value
                ]
            else:
                updated[key] = value
        return updated

    def encrypt_artifacts(self, artifacts: Iterable[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        if not self._encryptor:
            return list(artifacts)

        encrypted: List[KnowledgeArtifact] = []
        for artifact in artifacts:
            encrypted.append(
                KnowledgeArtifact(
                    signature=artifact.signature,
                    text=self._encryptor.encrypt_text(artifact.text),
                    structured_payload=artifact.structured_payload,
                    metadata={**artifact.metadata, "_encrypted": True},
                )
            )
        return encrypted

    def decrypt_artifacts(self, artifacts: Iterable[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        if not self._encryptor:
            return list(artifacts)

        decrypted: List[KnowledgeArtifact] = []
        for artifact in artifacts:
            metadata = dict(artifact.metadata)
            is_encrypted = metadata.pop("_encrypted", False)
            text = self._encryptor.decrypt_text(artifact.text) if is_encrypted else artifact.text
            decrypted.append(
                KnowledgeArtifact(
                    signature=artifact.signature,
                    text=text,
                    structured_payload=artifact.structured_payload,
                    metadata=metadata,
                )
            )
        return decrypted

    def _looks_like_pii(self, text: str) -> bool:
        """
        Naive PII detector that flags long numeric sequences or email patterns.
        """
        if re.search(r"\b\d{9,}\b", text):
            return True
        if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
            return True
        return False
