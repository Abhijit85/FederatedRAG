from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from synapse.knowledge.compendium import KnowledgeArtifact


def _laplace(scale: float) -> float:
    """
    Sample noise from a Laplace(0, scale) distribution using inverse transform sampling.
    """
    u = random.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))


@dataclass
class PrivacyPolicy:
    """
    Redaction and differential privacy controls applied before sharing artifacts.
    """

    redact_sensitive_metadata: bool = True
    drop_pii_text: bool = True
    dp_epsilon: Optional[float] = None

    def enforce(self, artifacts: Iterable[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        """
        Apply privacy protections (PII redaction and optional DP noise).
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

        if self.dp_epsilon and self.dp_epsilon > 0:
            return self._apply_dp_noise(sanitized)
        return sanitized

    def _apply_dp_noise(self, artifacts: Iterable[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        scale = 1.0 / max(self.dp_epsilon, 1e-6)
        privatized: List[KnowledgeArtifact] = []
        for artifact in artifacts:
            metadata = dict(artifact.metadata or {})
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    metadata[key] = value + _laplace(scale)

            payload = artifact.structured_payload
            if isinstance(payload, dict):
                privatized_payload = {}
                for key, value in payload.items():
                    if isinstance(value, (int, float)):
                        privatized_payload[key] = value + _laplace(scale)
                    elif isinstance(value, list):
                        privatized_payload[key] = [
                            item + _laplace(scale) if isinstance(item, (int, float)) else item
                            for item in value
                        ]
                    else:
                        privatized_payload[key] = value
                payload = privatized_payload

            privatized.append(
                KnowledgeArtifact(
                    signature=artifact.signature,
                    text=artifact.text,
                    structured_payload=payload,
                    metadata=metadata,
                )
            )
        return privatized

    def _looks_like_pii(self, text: str) -> bool:
        """
        Naive PII detector that flags long numeric sequences or email patterns.
        """
        if re.search(r"\b\d{9,}\b", text):
            return True
        if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
            return True
        return False
