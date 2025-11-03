from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

from synapse.utils import env_str

try:  # pragma: no cover - optional dependency
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore


LOGGER = logging.getLogger(__name__)


DEFAULT_ENT_MODEL = "textattack/roberta-base-MNLI"
DEFAULT_CITATION_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class EntailmentScorer:
    """Optional entailment classifier backed by Hugging Face models."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        self.model_name = model_name or env_str("SYNAPSE_TEXGRAD_ENT_MODEL", DEFAULT_ENT_MODEL)
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        self._available = torch is not None and AutoModelForSequenceClassification is not None
        if not self._available:
            LOGGER.info("EntailmentScorer running in fallback mode; install transformers for richer signals.")
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
            self.ent_index = label2id.get("entailment", label2id.get("ent", 2))
        except Exception as exc:  # pragma: no cover - network/offline
            LOGGER.warning("Failed to load entailment model %s (%s); falling back to heuristics.", self.model_name, exc)
            self._available = False

    def score(self, answer: str, contexts: Sequence[str]) -> float:
        if not contexts:
            contexts = [answer]
        if not self._available:
            # Simple lexical overlap heuristic (fallback)
            answer_tokens = set(answer.lower().split())
            scores = [len(answer_tokens.intersection(ctx.lower().split())) / (len(answer_tokens) + 1e-6) for ctx in contexts]
            return float(max(scores) if scores else 0.5)

        with torch.no_grad():
            best = 0.0
            for context in contexts:
                inputs = self.tokenizer(answer, context, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                score = probs[0, self.ent_index].item()
                best = max(best, score)
        return float(best)


class CitationAligner:
    """Optional citation aligner leveraging cross-encoder relevance scoring."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        self.model_name = model_name or env_str("SYNAPSE_TEXGRAD_CITATION_MODEL", DEFAULT_CITATION_MODEL)
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        self._available = torch is not None and AutoModelForSequenceClassification is not None
        if not self._available:
            LOGGER.info("CitationAligner running in fallback mode; install transformers for richer signals.")
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to load citation model %s (%s); falling back to heuristics.", self.model_name, exc)
            self._available = False

    def coverage(self, answer: str, contexts: Sequence[str]) -> float:
        if not contexts:
            return 0.0
        if not self._available:
            # Fallback coverage: proportion of answer tokens contained in union of context tokens
            answer_tokens = set(answer.lower().split())
            context_tokens = set(" ".join(contexts).lower().split())
            if not answer_tokens:
                return 0.0
            return float(len(answer_tokens.intersection(context_tokens)) / len(answer_tokens))

        with torch.no_grad():
            scores: List[float] = []
            for context in contexts:
                inputs = self.tokenizer(answer, context, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                # Assume binary relevance; take probability of class 1 if available
                if probs.shape[-1] > 1:
                    score = probs[0, 1].item()
                else:
                    score = torch.sigmoid(probs).item()
                scores.append(score)
        return float(np.mean(scores))
