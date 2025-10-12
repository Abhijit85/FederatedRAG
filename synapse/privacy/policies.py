from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from synapse.knowledge.compendium import KnowledgeArtifact


@dataclass
class PrivacyPolicy:
    """
    Placeholder privacy policy that can redact or drop artifacts.
    """

    redact_sensitive_metadata: bool = True
    drop_pii_text: bool = True

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

    def _looks_like_pii(self, text: str) -> bool:
        """
        Naive PII detector that flags long numeric sequences or email patterns.
        """
        if re.search(r"\b\d{9,}\b", text):
            return True
        if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
            return True
        return False
