from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from synapse.knowledge.compendium import KnowledgeArtifact


@dataclass
class RetrievalConfig:
    """Configuration for dynamic context retrieval."""

    max_artifacts: int = 5
    enable_symbolic_queries: bool = True


class RetrievalPlanner:
    """
    Coordinates retrieval of relevant knowledge artifacts for a new query.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None) -> None:
        self.config = config or RetrievalConfig()

    def select(self, query: str, artifacts: Iterable[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        """
        Placeholder retrieval strategy that returns the first N artifacts.

        Later revisions will combine embedding similarity, symbolic
        matching, and uncertainty-aware heuristics.
        """
        ranked = sorted(
            artifacts,
            key=lambda art: self._score_artifact(query, art),
            reverse=True,
        )
        return ranked[: self.config.max_artifacts]

    def _tokenize(self, text: str) -> Counter:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return Counter(tokens)

    def _score_artifact(self, query: str, artifact: KnowledgeArtifact) -> float:
        query_tokens = self._tokenize(query)
        artifact_tokens = self._tokenize(artifact.text)

        overlap = sum((query_tokens & artifact_tokens).values())
        length_norm = max(len(artifact_tokens), 1)
        token_score = overlap / length_norm

        structured_bonus = 0.0
        if self.config.enable_symbolic_queries and artifact.structured_payload:
            payload = artifact.structured_payload
            if isinstance(payload, dict):
                for value in payload.values():
                    if isinstance(value, str) and value.lower() in query.lower():
                        structured_bonus += 0.5
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and item.lower() in query.lower():
                                structured_bonus += 0.2

        return token_score + structured_bonus
