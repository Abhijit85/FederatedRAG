from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

from synapse.knowledge.compendium import KnowledgeArtifact
from synapse.retrieval.vector_store import HashedVectorStore


@dataclass
class RetrievalConfig:
    """Configuration for dynamic context retrieval."""

    max_artifacts: int = 5
    enable_symbolic_queries: bool = True


class RetrievalPlanner:
    """
    Coordinates retrieval of relevant knowledge artifacts for a new query.
    """

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        vector_store: Optional[HashedVectorStore] = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.vector_store = vector_store or HashedVectorStore()

    def update_artifacts(self, artifacts: Iterable[KnowledgeArtifact]) -> None:
        self.vector_store.bulk_upsert(artifacts)

    def select(self, query: str, artifacts: Iterable[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        self.update_artifacts(artifacts)
        vector_hits = self.vector_store.query(query, top_k=self.config.max_artifacts * 3)

        ranked: List[Tuple[KnowledgeArtifact, float]] = []
        for artifact, vector_score in vector_hits:
            symbolic_score = self._score_artifact(query, artifact)
            ranked.append((artifact, vector_score + symbolic_score))

        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return [artifact for artifact, _ in ranked[: self.config.max_artifacts]]

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
