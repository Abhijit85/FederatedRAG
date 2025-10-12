from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np

from synapse.knowledge.compendium import KnowledgeArtifact


class HashedVectorStore:
    """
    Lightweight vector store using hashed bag-of-words embeddings.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self._vectors: Dict[str, np.ndarray] = {}
        self._artifacts: Dict[str, KnowledgeArtifact] = {}

    def _tokenize(self, text: str) -> Counter:
        tokens = text.lower().split()
        return Counter(tokens)

    def _hash_token(self, token: str) -> int:
        return hash(token) % self.dim

    def _embed_text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        counts = self._tokenize(text)
        total = sum(counts.values()) or 1
        for token, count in counts.items():
            idx = self._hash_token(token)
            vector[idx] += count / total
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector

    def _embed_artifact(self, artifact: KnowledgeArtifact) -> np.ndarray:
        vector = self._embed_text(artifact.text)
        payload = artifact.structured_payload
        if isinstance(payload, dict):
            image_embedding = payload.get("image_embedding")
            if isinstance(image_embedding, list):
                image_vec = np.array(image_embedding, dtype=np.float32)
                if image_vec.size > 0:
                    image_vec = image_vec / (np.linalg.norm(image_vec) or 1)
                    # Blend text and image embeddings.
                    overlay = np.zeros(self.dim, dtype=np.float32)
                    overlay[: min(len(image_vec), self.dim)] = image_vec[: self.dim]
                    vector = (vector + overlay) / 2.0
        return vector

    def upsert(self, artifact: KnowledgeArtifact) -> None:
        embedding = self._embed_artifact(artifact)
        self._vectors[artifact.signature] = embedding
        self._artifacts[artifact.signature] = artifact

    def bulk_upsert(self, artifacts: Iterable[KnowledgeArtifact]) -> None:
        for artifact in artifacts:
            self.upsert(artifact)

    def query(self, text: str, top_k: int = 5) -> List[Tuple[KnowledgeArtifact, float]]:
        if not self._vectors:
            return []

        query_vec = self._embed_text(text)
        scores: List[Tuple[KnowledgeArtifact, float]] = []
        for signature, vector in self._vectors.items():
            score = float(np.dot(query_vec, vector))
            scores.append((self._artifacts[signature], score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]
