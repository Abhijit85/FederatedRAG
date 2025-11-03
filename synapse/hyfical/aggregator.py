from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .contracts import AdapterUpdate, LayerUpdate, TexGradMetrics


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class AggregationConfig:
    """
    Tunable parameters for HyFICAL aggregation.
    """

    spectral_k: int = 20
    anomaly_tau_cos: float = 0.65
    median_iters: int = 15
    freshness_half_life_min: float = 30.0
    trust_alpha: float = 0.9
    z_threshold: float = 3.0
    decay_floor: float = 0.1


@dataclass
class AggregatedLayerResult:
    """
    Final aggregation output for a single layer.
    """

    layer: str
    vector: np.ndarray
    contributors: List[str]
    weights: List[float]
    residual_norms: Dict[str, float]
    telemetry_summary: Dict[str, float]


class HyFICALAggregator:
    """
    Implements Spectral Robust Aggregation + geometric median with trust weighting.
    """

    def __init__(self, config: Optional[AggregationConfig] = None) -> None:
        self.config = config or AggregationConfig()
        self._trust_scores: Dict[str, float] = {}
        self._last_seen: Dict[str, datetime] = {}
        self._quarantine_queue: List[Dict[str, str]] = []
        self._poisoning_flags: Dict[str, List[str]] = {}
        self._adapter_norm_zscores: Dict[str, Dict[str, float]] = {}
        self._norm_statistics: Dict[str, Dict[str, float]] = {}

    def _decode_layer_update(
        self,
        layer_update: LayerUpdate,
        decode_fn: Callable[[LayerUpdate], np.ndarray],
    ) -> np.ndarray:
        vector = decode_fn(layer_update)
        if vector.ndim != 1:
            vector = vector.reshape(-1)
        return vector.astype(np.float64)

    def _freshness_weight(self, client_id: str, timestamp: int, now: datetime) -> float:
        if timestamp <= 0:
            return 1.0
        seen = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        self._last_seen[client_id] = seen
        delta_minutes = max((now - seen).total_seconds() / 60.0, 0.0)
        half_life = max(self.config.freshness_half_life_min, 1e-3)
        decay = 0.5 ** (delta_minutes / half_life)
        return max(decay, self.config.decay_floor)

    def _texgrad_consistency(
        self,
        vector: np.ndarray,
        metrics: TexGradMetrics,
    ) -> float:
        fingerprint = np.asarray(metrics.semantic_fingerprint, dtype=np.float64)
        if fingerprint.size == 0 or fingerprint.size != vector.size:
            return 1.0

        denom = np.linalg.norm(vector) * np.linalg.norm(fingerprint)
        if denom <= 0:
            return 1.0
        cosine = float(np.dot(vector, fingerprint) / denom)
        return cosine

    def _spectral_filter(
        self,
        matrix: np.ndarray,
        keep_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Remove rows with extreme leverage scores based on top-k spectral components.
        """
        # Short circuit when we lack enough samples to run PCA.
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            return keep_mask

        filtered = matrix[keep_mask]
        if filtered.shape[0] < 2:
            return keep_mask

        k = min(self.config.spectral_k, filtered.shape[0], filtered.shape[1])
        if k < 1:
            return keep_mask

        try:
            u, s, _ = np.linalg.svd(filtered, full_matrices=False)
        except np.linalg.LinAlgError:
            return keep_mask

        leverage = np.sum((u[:, :k] ** 2) * (s[:k] ** 2), axis=1)
        mean = leverage.mean()
        std = leverage.std() + 1e-6
        z_scores = np.abs((leverage - mean) / std)
        retained = z_scores < self.config.z_threshold

        updated_mask = keep_mask.copy()
        keep_indices = np.where(keep_mask)[0]
        for idx, keep in zip(keep_indices, retained):
            if not keep:
                updated_mask[idx] = False
        return updated_mask

    def _geometric_median(self, vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Compute (weighted) geometric median using Weiszfeld's algorithm.
        """
        if len(vectors) == 1:
            return vectors[0]

        # Initialize with weighted arithmetic mean.
        median = np.average(vectors, axis=0, weights=weights)
        for _ in range(self.config.median_iters):
            distances = np.linalg.norm(vectors - median, axis=1)
            distances = np.maximum(distances, 1e-8)
            inv_dist = weights / distances
            new_median = np.average(vectors, axis=0, weights=inv_dist)
            if np.linalg.norm(new_median - median) < 1e-6:
                median = new_median
                break
            median = new_median
        return median

    def _update_trust(self, client_id: str, agreement: float) -> float:
        prior = self._trust_scores.get(client_id, 1.0)
        new_score = (self.config.trust_alpha * prior) + ((1 - self.config.trust_alpha) * agreement)
        self._trust_scores[client_id] = max(new_score, 0.05)
        return self._trust_scores[client_id]

    def aggregate_updates(
        self,
        updates: Sequence[AdapterUpdate],
        decode_fn: Callable[[LayerUpdate], np.ndarray],
        now: Optional[datetime] = None,
    ) -> Dict[str, AggregatedLayerResult]:
        """
        Aggregate masked layer updates into a global delta per layer.
        """
        if not updates:
            return {}

        now = now or _now_utc()

        layer_vectors: Dict[str, List[np.ndarray]] = {}
        layer_clients: Dict[str, List[str]] = {}
        layer_weights: Dict[str, List[float]] = {}
        layer_metrics: Dict[str, List[TexGradMetrics]] = {}
        layer_telemetry: Dict[str, List[Dict[str, float]]] = {}
        layer_norms: Dict[str, List[Tuple[str, float]]] = {}

        self._quarantine_queue = []

        # Decode all layer updates.
        for update in updates:
            for layer_update in update.layer_updates:
                vector = self._decode_layer_update(layer_update, decode_fn)
                layer_vectors.setdefault(layer_update.layer, []).append(vector)
                layer_clients.setdefault(layer_update.layer, []).append(update.client_id)
                layer_metrics.setdefault(layer_update.layer, []).append(update.telemetry.texgrad)
                layer_norms.setdefault(layer_update.layer, []).append((update.client_id, layer_update.norm))

                freshness = self._freshness_weight(
                    update.client_id,
                    update.telemetry.freshness_ts,
                    now,
                )
                steps = max(update.telemetry.steps, 1)
                weight = freshness * math.log1p(steps)
                layer_weights.setdefault(layer_update.layer, []).append(weight)

                layer_telemetry.setdefault(layer_update.layer, []).append(
                    {
                        "loss_lm": update.telemetry.loss_lm,
                        "entailment": update.telemetry.texgrad.entailment,
                        "citation": update.telemetry.texgrad.citation_coverage,
                        "contrastive": update.telemetry.texgrad.contrastive_margin,
                        "entropy": update.telemetry.texgrad.retrieval_entropy,
                    }
                )

        aggregated: Dict[str, AggregatedLayerResult] = {}

        for layer, vectors in layer_vectors.items():
            matrix = np.vstack(vectors)
            clients = layer_clients[layer]
            metrics = layer_metrics[layer]
            weights = np.asarray(layer_weights[layer], dtype=np.float64)

            keep_mask = np.ones(len(clients), dtype=bool)

            # TexGrad consistency check.
            for idx, (vector, metric) in enumerate(zip(vectors, metrics)):
                cosine = self._texgrad_consistency(vector, metric)
                if cosine < self.config.anomaly_tau_cos:
                    keep_mask[idx] = False
                    self._quarantine_queue.append(
                        {
                            "layer": layer,
                            "client_id": clients[idx],
                            "reason": "texgrad_cosine_below_tau",
                        }
                    )

            keep_mask = self._spectral_filter(matrix, keep_mask)

            for idx, keep in enumerate(keep_mask):
                if not keep:
                    self._quarantine_queue.append(
                        {
                            "layer": layer,
                            "client_id": clients[idx],
                            "reason": "spectral_outlier",
                        }
                    )

            retained_indices = np.where(keep_mask)[0]
            if retained_indices.size == 0:
                # Fall back to arithmetic mean to avoid empty aggregation.
                retained_indices = np.arange(matrix.shape[0])

            retained_vectors = matrix[retained_indices]
            retained_weights = weights[retained_indices]
            retained_clients = [clients[i] for i in retained_indices]

            # Normalize weights and apply trust scores.
            trust_weights = []
            for client_id, vector in zip(retained_clients, retained_vectors):
                agreement = 1.0
                trust = self._trust_scores.get(client_id, 1.0)
                trust_weights.append(trust)
                # We'll update trust after computing the median.

            retained_weights = retained_weights * np.asarray(trust_weights)
            weight_sum = retained_weights.sum()
            if weight_sum <= 0:
                retained_weights = np.ones_like(retained_weights)
            else:
                retained_weights = retained_weights / (weight_sum + 1e-12)

            median = self._geometric_median(retained_vectors, retained_weights)

            # Update trust using agreement with the median.
            residuals: Dict[str, float] = {}
            for client_id, vector in zip(retained_clients, retained_vectors):
                delta = vector - median
                residual = float(np.linalg.norm(delta))
                residuals[client_id] = residual
                norm_median = float(np.linalg.norm(median)) + 1e-8
                agreement = 1.0 / (1.0 + (residual / norm_median))
                self._update_trust(client_id, agreement)

            aggregated[layer] = AggregatedLayerResult(
                layer=layer,
                vector=median,
                contributors=retained_clients,
                weights=retained_weights.tolist(),
                residual_norms=residuals,
                telemetry_summary=self._summarize_telemetry(layer_telemetry[layer], retained_indices),
            )

            self._update_norm_observability(layer, layer_norms.get(layer, []))

        # Collapse quarantine queue into poisoning flags per layer.
        poisoning_map: Dict[str, List[str]] = defaultdict(list)
        for entry in self._quarantine_queue:
            poisoning_map[entry["layer"]].append(entry["reason"])
        self._poisoning_flags = {layer: reasons for layer, reasons in poisoning_map.items()}

        return aggregated

    def _summarize_telemetry(
        self,
        telemetry_rows: Sequence[Dict[str, float]],
        indices: Sequence[int],
    ) -> Dict[str, float]:
        if not telemetry_rows:
            return {
                "loss_lm_avg": 0.0,
                "entailment_avg": 0.0,
                "citation_avg": 0.0,
                "contrastive_avg": 0.0,
                "retrieval_entropy_avg": 0.0,
            }

        matrix = np.array(
            [
                [
                    row.get("loss_lm", 0.0),
                    row.get("entailment", 0.0),
                    row.get("citation", 0.0),
                    row.get("contrastive", 0.0),
                    row.get("entropy", 0.0),
                ]
                for row in telemetry_rows
            ],
            dtype=np.float64,
        )
        if matrix.ndim != 2 or matrix.shape[1] != 5:
            return {
                "loss_lm_avg": 0.0,
                "entailment_avg": 0.0,
                "citation_avg": 0.0,
                "contrastive_avg": 0.0,
                "retrieval_entropy_avg": 0.0,
            }

        if not indices:
            selected = matrix
        else:
            idx = np.array(indices, dtype=int)
            selected = matrix[idx]

        return {
            "loss_lm_avg": float(selected[:, 0].mean()),
            "entailment_avg": float(selected[:, 1].mean()),
            "citation_avg": float(selected[:, 2].mean()),
            "contrastive_avg": float(selected[:, 3].mean()),
            "retrieval_entropy_avg": float(selected[:, 4].mean()),
        }

    @property
    def trust_scores(self) -> Dict[str, float]:
        return dict(self._trust_scores)

    def _update_norm_observability(self, layer: str, client_norms: Sequence[Tuple[str, float]]) -> None:
        if not client_norms:
            return
        norms = np.array([norm for _, norm in client_norms], dtype=np.float64)
        mean = float(norms.mean())
        std = float(norms.std() + 1e-6)
        zscores = (norms - mean) / std
        self._adapter_norm_zscores[layer] = {
            client_id: float(abs(z))
            for (client_id, _), z in zip(client_norms, zscores)
        }
        self._norm_statistics[layer] = {"mean": mean, "std": std}
        threshold = max(self.config.z_threshold, 1.0)
        for (client_id, _), z in zip(client_norms, zscores):
            if abs(z) >= threshold:
                self._quarantine_queue.append(
                    {
                        "layer": layer,
                        "client_id": client_id,
                        "reason": "adapter_norm_zscore",
                    }
                )

    @property
    def adapter_norm_zscores(self) -> Dict[str, Dict[str, float]]:
        return {layer: dict(scores) for layer, scores in self._adapter_norm_zscores.items()}

    @property
    def norm_statistics(self) -> Dict[str, Dict[str, float]]:
        return {layer: dict(stats) for layer, stats in self._norm_statistics.items()}

    @property
    def quarantine_queue(self) -> List[Dict[str, str]]:
        return list(self._quarantine_queue)

    @property
    def poisoning_flags(self) -> Dict[str, List[str]]:
        return {layer: list(reasons) for layer, reasons in self._poisoning_flags.items()}
