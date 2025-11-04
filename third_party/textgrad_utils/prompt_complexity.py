"""
Lightweight helpers for estimating prompt complexity without heavyweight ML deps.
The original upstream module relied on nltk, scikit-learn, transformers, and torch;
downloading those models is overkill for metadata we only log locally. This module
keeps the public API shape but uses inexpensive heuristics so it works in minimal
environments.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from typing import Dict, Tuple


def _basic_word_tokenize(text: str) -> list[str]:
    """Regex-based tokenizer that roughly matches wordpiece boundaries."""
    if not text:
        return []
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def calculate_entropy(text: str) -> float:
    """Shannon entropy over individual characters."""
    if not text:
        return 0.0
    counts = Counter(text)
    total_chars = sum(counts.values())
    probs = (count / total_chars for count in counts.values())
    return -sum(p * math.log2(p) for p in probs if p > 0)


def calculate_compression_rate(text: str) -> float:
    """Compression ratio using zlib as a crude redundancy proxy."""
    if not text:
        return 0.0
    raw = text.encode("utf-8")
    compressed = zlib.compress(raw)
    return len(compressed) / len(raw)


def calculate_tfidf(text: str) -> Tuple[float, Dict[str, float]]:
    """
    Approximates TF-IDF scores using a single-document corpus. We scale each token's
    term frequency by a pseudo-IDF so downstream code still receives a dict of scores.
    """
    tokens = _basic_word_tokenize(text.lower())
    if not tokens:
        return 0.0, {}

    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    scores = {}
    for token, count in token_counts.items():
        tf = count / total_tokens
        idf = math.log((total_tokens + 1) / (count + 1)) + 1.0
        scores[token] = tf * idf

    average_score = sum(scores.values()) / len(scores)
    return average_score, scores


def calculate_perplexity(text: str) -> float:
    """
    Estimates perplexity from token entropy. This is not a full language-model
    perplexity but retains the same qualitative meaning: higher implies less predictable.
    """
    tokens = _basic_word_tokenize(text)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    probs = (count / total for count in counts.values())
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return 2 ** entropy


def calculate_token_length(text: str) -> Tuple[int, int]:
    """
    Returns both simple word/token counts and a rough approximation of GPT-style
    subword length (empirically ~4 chars per token for English).
    """
    regex_tokens = _basic_word_tokenize(text)
    approx_gpt_tokens = max(1, round(len(text.encode("utf-8")) / 4)) if text else 0
    return len(regex_tokens), approx_gpt_tokens


def calculate_text_complexity(text: str) -> Dict[str, float | Dict[str, float]]:
    """
    Bundles together the cheap heuristics above so callers can stash a consistent
    metadata blob alongside their TextGrad artifacts.
    """
    entropy = calculate_entropy(text)
    compression_rate = calculate_compression_rate(text)
    average_tfidf, _ = calculate_tfidf(text)
    perplexity = calculate_perplexity(text)
    nltk_token_length, gpt_token_length = calculate_token_length(text)

    return {
        "Information Entropy": entropy,
        "Compression Rate": compression_rate,
        "Average TF-IDF": average_tfidf,
        "Perplexity": perplexity,
        "NLTK Token Length": nltk_token_length,
        "GPT-2 Token Length": gpt_token_length,
    }
