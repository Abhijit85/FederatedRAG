from __future__ import annotations

import enum
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from synapse.hyfical.aggregator import AggregatedLayerResult, AggregationConfig, HyFICALAggregator
from synapse.hyfical.contracts import AdapterUpdate, LayerUpdate


class AggregationMode(enum.Enum):
    ROBUST = "robust"
    SUM_ONLY = "sum_only"

    @classmethod
    def from_string(cls, value: str | None) -> "AggregationMode":
        if not value:
            return cls.ROBUST
        normalized = value.strip().lower()
        for mode in cls:
            if mode.value == normalized:
                return mode
        return cls.ROBUST


class AggregatorFacade:
    """
    Unified interface consumed by SynapseServer regardless of aggregation strategy.
    """

    def aggregate(
        self,
        updates: Sequence[AdapterUpdate],
        decode_fn: Callable[[LayerUpdate], np.ndarray],
    ) -> Dict[str, AggregatedLayerResult]:
        raise NotImplementedError

    @property
    def poisoning_flags(self) -> Dict[str, List[str]]:
        return {}

    @property
    def adapter_norm_zscores(self) -> Dict[str, Dict[str, float]]:
        return {}

    @property
    def quarantine_queue(self) -> List[Dict[str, str]]:
        return []

    @property
    def trust_scores(self) -> Dict[str, float]:
        return {}

    @property
    def config(self) -> AggregationConfig | None:
        return None


class SecAggSumAggregator(AggregatorFacade):
    """
    Implements a SecAgg-style sum-only aggregation where the server only relies
    on aggregated vectors. Robustness is limited to norm statistics, but privacy
    mirrors the pure secure aggregation setting.
    """

    def __init__(self) -> None:
        self._adapter_norm_zscores: Dict[str, Dict[str, float]] = {}
        self._poisoning_flags: Dict[str, List[str]] = {}
        self._quarantine_queue: List[Dict[str, str]] = []
        self._config = AggregationConfig()

    def aggregate(
        self,
        updates: Sequence[AdapterUpdate],
        decode_fn: Callable[[LayerUpdate], np.ndarray],
    ) -> Dict[str, AggregatedLayerResult]:
        totals: Dict[str, np.ndarray] = {}
        counts: Dict[str, int] = {}
        contributors: Dict[str, List[str]] = {}
        telemetry_rows: Dict[str, List[Dict[str, float]]] = {}
        norms: Dict[str, List[Tuple[str, float]]] = {}

        for update in updates:
            for layer_update in update.layer_updates:
                vector = decode_fn(layer_update)
                layer = layer_update.layer
                totals.setdefault(layer, np.zeros_like(vector))
                totals[layer] += vector
                counts[layer] = counts.get(layer, 0) + 1
                contributors.setdefault(layer, []).append(update.client_id)
                telemetry_rows.setdefault(layer, []).append(
                    {
                        "loss_lm": update.telemetry.loss_lm,
                        "entailment": update.telemetry.texgrad.entailment,
                        "citation": update.telemetry.texgrad.citation_coverage,
                        "contrastive": update.telemetry.texgrad.contrastive_margin,
                        "retrieval_entropy": update.telemetry.texgrad.retrieval_entropy,
                    }
                )
                norms.setdefault(layer, []).append((update.client_id, layer_update.norm))

        aggregated: Dict[str, AggregatedLayerResult] = {}
        self._adapter_norm_zscores = {}
        self._poisoning_flags = {}
        self._quarantine_queue = []

        for layer, vector_sum in totals.items():
            count = max(counts.get(layer, 1), 1)
            mean_vec = vector_sum / count

            telem = telemetry_rows.get(layer, [])
            if telem:
                matrix = np.array(
                    [
                        [
                            row.get("loss_lm", 0.0),
                            row.get("entailment", 0.0),
                            row.get("citation", 0.0),
                            row.get("contrastive", 0.0),
                            row.get("retrieval_entropy", 0.0),
                        ]
                        for row in telem
                    ],
                    dtype=np.float64,
                )
                summary = {
                    "loss_lm_avg": float(matrix[:, 0].mean()),
                    "entailment_avg": float(matrix[:, 1].mean()),
                    "citation_avg": float(matrix[:, 2].mean()),
                    "contrastive_avg": float(matrix[:, 3].mean()),
                    "retrieval_entropy_avg": float(matrix[:, 4].mean()),
                }
            else:
                summary = {
                    "loss_lm_avg": 0.0,
                    "entailment_avg": 0.0,
                    "citation_avg": 0.0,
                    "contrastive_avg": 0.0,
                    "retrieval_entropy_avg": 0.0,
                }

            contrib_clients = contributors.get(layer, [])
            weights = [1.0 / count] * len(contrib_clients)
            residuals = {client_id: 0.0 for client_id in contrib_clients}

            aggregated[layer] = AggregatedLayerResult(
                layer=layer,
                vector=mean_vec,
                contributors=contrib_clients,
                weights=weights,
                residual_norms=residuals,
                telemetry_summary=summary,
            )

            # Observability: compute norm z-scores for awareness.
            norm_entries = norms.get(layer, [])
            if norm_entries:
                norm_values = np.array([value for _, value in norm_entries], dtype=np.float64)
                mean = float(norm_values.mean())
                std = float(norm_values.std() + 1e-6)
                zscores = (norm_values - mean) / std
                self._adapter_norm_zscores[layer] = {
                    client_id: float(abs(z))
                    for (client_id, _), z in zip(norm_entries, zscores)
                }

        return aggregated

    @property
    def adapter_norm_zscores(self) -> Dict[str, Dict[str, float]]:
        return self._adapter_norm_zscores

    @property
    def poisoning_flags(self) -> Dict[str, List[str]]:
        return self._poisoning_flags

    @property
    def quarantine_queue(self) -> List[Dict[str, str]]:
        return self._quarantine_queue

    @property
    def config(self) -> AggregationConfig:
        return self._config


class TEEAggregator(AggregatorFacade):
    """
    Wraps the HyFICAL robust aggregator and conceptually represents execution inside
    a trusted enclave. Updates are processed using the full spectral + median pipeline.
    """

    def __init__(self, hyfical: HyFICALAggregator) -> None:
        self._hyfical = hyfical

    def aggregate(
        self,
        updates: Sequence[AdapterUpdate],
        decode_fn: Callable[[LayerUpdate], np.ndarray],
    ) -> Dict[str, AggregatedLayerResult]:
        # In a production setting, decryption and aggregation would happen inside the enclave.
        return self._hyfical.aggregate_updates(updates, decode_fn)

    @property
    def poisoning_flags(self) -> Dict[str, List[str]]:
        return self._hyfical.poisoning_flags

    @property
    def adapter_norm_zscores(self) -> Dict[str, Dict[str, float]]:
        return self._hyfical.adapter_norm_zscores

    @property
    def quarantine_queue(self) -> List[Dict[str, str]]:
        return self._hyfical.quarantine_queue

    @property
    def trust_scores(self) -> Dict[str, float]:
        return self._hyfical.trust_scores

    @property
    def config(self) -> AggregationConfig:
        return self._hyfical.config

    @property
    def core(self) -> HyFICALAggregator:
        return self._hyfical
