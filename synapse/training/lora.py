from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .texgrad import TexGradSample
from synapse.utils import env_int, env_list


@dataclass
class LoRALayerConfig:
    """
    Configuration for LoRA adapter generation.
    """

    target_modules: Sequence[str] = (
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "mlp.gate_proj",
    )
    rank_choices: Sequence[int] = (4, 8, 16)
    base_dimension: int = 32

    def resolve_layers(self) -> List[str]:
        return list(self.target_modules)

    @classmethod
    def from_env(cls, prefix: str = "SYNAPSE") -> "LoRALayerConfig":
        """
        Build a configuration from environment variables.
        """
        defaults = cls()
        target_key = f"{prefix}_LORA_TARGETS"
        rank_key = f"{prefix}_LORA_RANKS"
        base_dim_key = f"{prefix}_LORA_BASE_DIM"

        targets = env_list(target_key, defaults.target_modules)
        ranks_raw = env_list(rank_key, [str(rank) for rank in defaults.rank_choices])

        try:
            rank_choices = tuple(sorted({int(value) for value in ranks_raw}))
        except ValueError:
            rank_choices = tuple(defaults.rank_choices)

        base_dim = env_int(base_dim_key, defaults.base_dimension)

        return cls(
            target_modules=tuple(targets),
            rank_choices=rank_choices,
            base_dimension=base_dim,
        )


class LoRAUpdatePlanner:
    """
    Deterministic pseudo-training of LoRA adapters from TexGrad samples.

    The planner does not run gradient descent; instead it derives stable
    pseudo-updates by hashing sample content. This makes it feasible to
    unit test higher layers without heavy model dependencies while still
    reflecting differences across datasets.
    """

    def __init__(self, config: LoRALayerConfig | None = None) -> None:
        self.config = config or LoRALayerConfig()

    def _hash_to_rng(self, *items: str) -> np.random.Generator:
        hasher = hashlib.sha256()
        for item in items:
            hasher.update(item.encode("utf-8"))
        seed = int(hasher.hexdigest()[:16], 16)
        return np.random.default_rng(seed)

    def _sample_vector(self, layer: str, rank: int, sample: TexGradSample) -> np.ndarray:
        dimension = self.config.base_dimension * rank
        rng = self._hash_to_rng(
            layer,
            sample.question,
            "".join(sample.positive_contexts),
            "".join(sample.negative_contexts),
        )
        vector = rng.normal(loc=0.0, scale=0.5, size=dimension)
        # Weight by entailment / citation heuristics to keep determinism but reflect quality.
        scale = (sample.entailment_score + sample.citation_coverage + 1e-3) / 2.0
        return vector * float(scale)

    def build_layer_updates(
        self,
        samples: Iterable[TexGradSample],
        rank: int,
    ) -> Dict[str, np.ndarray]:
        """
        Produce deterministic pseudo LoRA deltas for each configured layer.
        """
        resolved_layers = self.config.resolve_layers()
        accumulators: Dict[str, List[np.ndarray]] = {layer: [] for layer in resolved_layers}

        sample_list = list(samples)
        if not sample_list:
            sample_list = [TexGradSample.blank()]

        for layer in resolved_layers:
            for sample in sample_list:
                accumulators[layer].append(self._sample_vector(layer, rank, sample))

        updates: Dict[str, np.ndarray] = {}
        for layer, vectors in accumulators.items():
            stacked = np.vstack(vectors)
            updates[layer] = stacked.mean(axis=0)
        return updates

    @staticmethod
    def delta_hash(vector: np.ndarray) -> str:
        as_bytes = vector.astype(np.float32).tobytes()
        return hashlib.sha256(as_bytes).hexdigest()
