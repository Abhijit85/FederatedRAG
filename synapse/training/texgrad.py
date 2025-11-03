from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np

from synapse.hyfical.contracts import TexGradMetrics


@dataclass
class TexGradConfig:
    """
    Controls weighting and output dimensionality for TexGrad signals.
    """

    lambdas: Dict[str, float] = field(default_factory=lambda: {"ent": 0.5, "attr": 0.5, "ctr": 0.3})
    semantic_fingerprint_dim: int = 64


@dataclass
class TexGradSample:
    """
    Compact representation of a single RAG training example.
    """

    question: str
    answer: str
    positive_contexts: Sequence[str]
    negative_contexts: Sequence[str]
    entailment_score: float
    citation_coverage: float
    contrastive_margin: float
    retrieval_entropy: float

    @classmethod
    def from_strings(
        cls,
        question: str,
        answer: str,
        positives: Sequence[str],
        negatives: Sequence[str],
    ) -> "TexGradSample":
        """
        Create heuristic scores from plain text inputs.
        """
        entailment = cls._normalized_overlap(answer, positives)
        citation = cls._normalized_overlap(question, positives)
        contrastive = max(entailment - cls._normalized_overlap(answer, negatives), 0.0)
        entropy = cls._estimated_entropy(positives, negatives)
        return cls(
            question=question,
            answer=answer,
            positive_contexts=list(positives),
            negative_contexts=list(negatives),
            entailment_score=entailment,
            citation_coverage=citation,
            contrastive_margin=contrastive,
            retrieval_entropy=entropy,
        )

    @classmethod
    def blank(cls) -> "TexGradSample":
        return cls(
            question="",
            answer="",
            positive_contexts=[],
            negative_contexts=[],
            entailment_score=0.5,
            citation_coverage=0.5,
            contrastive_margin=0.3,
            retrieval_entropy=0.5,
        )

    @staticmethod
    def _normalized_overlap(seed: str, texts: Sequence[str]) -> float:
        if not texts:
            return 0.1
        seed_tokens = set(seed.lower().split())
        overlaps: List[float] = []
        for text in texts:
            tokens = set(text.lower().split())
            intersect = seed_tokens.intersection(tokens)
            ratio = len(intersect) / max(len(tokens), 1)
            overlaps.append(ratio)
        return float(sum(overlaps) / len(overlaps) if overlaps else 0.1)

    @staticmethod
    def _estimated_entropy(positives: Sequence[str], negatives: Sequence[str]) -> float:
        vocab = set()
        for text in positives:
            vocab.update(text.split())
        for text in negatives:
            vocab.update(text.split())
        vocab_size = max(len(vocab), 1)
        return float(min(math.log(vocab_size + 1.0) / 10.0, 1.0))


class TexGradHead:
    """
    Lightweight TexGrad estimator that simulates entailment and attribution signals.
    """

    def __init__(self, config: TexGradConfig | None = None) -> None:
        self.config = config or TexGradConfig()

    def _fingerprint(self, samples: Sequence[TexGradSample]) -> np.ndarray:
        dim = self.config.semantic_fingerprint_dim
        if not samples:
            return np.zeros(dim, dtype=np.float64)
        accumulator = np.zeros(dim, dtype=np.float64)
        for sample in samples:
            text = " ".join([sample.question, sample.answer, *sample.positive_contexts])
            hasher = hashlib.sha256(text.encode("utf-8"))
            seed = int(hasher.hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            accumulator += rng.normal(size=dim)
        return accumulator / max(len(samples), 1)

    def aggregate_metrics(self, samples: Iterable[TexGradSample]) -> TexGradMetrics:
        sample_list = list(samples)
        if not sample_list:
            sample_list = [TexGradSample.blank()]

        entailment = np.mean([sample.entailment_score for sample in sample_list])
        citation = np.mean([sample.citation_coverage for sample in sample_list])
        contrastive = np.mean([sample.contrastive_margin for sample in sample_list])
        entropy = np.mean([sample.retrieval_entropy for sample in sample_list])
        fingerprint = self._fingerprint(sample_list)

        return TexGradMetrics(
            entailment=float(entailment),
            citation_coverage=float(citation),
            contrastive_margin=float(contrastive),
            retrieval_entropy=float(entropy),
            semantic_fingerprint=fingerprint.tolist(),
        )
