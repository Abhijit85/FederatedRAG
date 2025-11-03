from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from synapse.central.config import (
    CentralModelConfig,
    CentralPrivacyConfig,
    CentralRobustnessConfig,
    CentralRouterConfig,
    CentralTexGradConfig,
    CentralTrainingConfig,
)
from synapse.central.robust import GradientAuditor, GradientAuditRecord
from synapse.compliance import ComplianceLedger, LedgerEntry
from synapse.hyfical import AggregatedLayerResult, AdapterRouter, HyFICALAggregator
from synapse.hyfical.contracts import (
    AdapterExpert,
    AdapterUpdate,
    LayerUpdate,
    PrivacyBudget,
    TexGradMetrics,
    UpdateTelemetry,
)
from synapse.privacy.accountant import RDPAccountant, RDPConfig
from synapse.training import (
    DPConfig,
    DifferentialPrivacyGuard,
    LoRAUpdatePlanner,
    PEFTTexGradTrainer,
    SecAggAdapter,
    SecAggConfig,
    TexGradConfig,
    TexGradHead,
    TexGradSample,
    TexGradLoRATrainer,
)
from synapse.utils import env_bool


def _load_json_records(path: Path) -> Sequence[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Try commonly used keys.
        for key in ("examples", "data", "Usage_Scenarios", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def _materialize_samples(records: Sequence[Dict[str, object]]) -> List[TexGradSample]:
    samples: List[TexGradSample] = []
    total = len(records)
    for idx, record in enumerate(records):
        question = str(
            record.get("question")
            or record.get("Problem")
            or record.get("prompt")
            or record.get("query")
            or f"synthetic-question-{idx}"
        )
        answer = str(
            record.get("answer")
            or record.get("Answer")
            or record.get("solution")
            or record.get("response")
            or ""
        )
        positives: List[str] = []
        negatives: List[str] = []

        lecture = record.get("lecture") or record.get("context") or record.get("explanation")
        if lecture:
            positives.append(str(lecture))

        supporting = record.get("support") or record.get("supporting_facts")
        if isinstance(supporting, list):
            positives.extend(str(item) for item in supporting if item)

        distractor = record.get("distractor") or record.get("negative")
        if distractor:
            negatives.append(str(distractor))

        choices = record.get("choices") or record.get("options")
        if isinstance(choices, list):
            positives.append(" ".join(str(choice) for choice in choices))

        # Use neighbour element as hard negative when available.
        if total > 1:
            neighbour = records[(idx + 1) % total]
            negatives.append(str(neighbour.get("lecture") or neighbour.get("context") or ""))

        samples.append(
            TexGradSample.from_strings(
                question=question,
                answer=answer,
                positives=[text for text in positives if text],
                negatives=[text for text in negatives if text],
            )
        )
    return samples


@dataclass
class CentralTrainingSummary:
    """
    Snapshot of the centralized training run.
    """

    epochs_completed: int
    steps_executed: int
    adapter_versions: List[str]
    privacy_budget: Optional[PrivacyBudget]
    poisoning_flags: Dict[str, List[str]]
    audit_records: List[GradientAuditRecord]
    ledger_entries: List[Dict[str, object]]


class _NullDPGuard:
    """
    No-op DP guard used when centralized DP is disabled.
    """

    def sanitize(self, vector: np.ndarray) -> np.ndarray:
        return vector

    def clip_norm(self) -> float:
        return 0.0

    def sigma(self) -> float:
        return 0.0

    def estimate_local_epsilon(self, steps: int) -> float:
        return 0.0


class CentralTexGradTrainer:
    """
    Centralized TexGrad-LoRA trainer with robustness and observability hooks.
    """

    def __init__(
        self,
        config: CentralTrainingConfig | None = None,
        *,
        ledger: Optional[ComplianceLedger] = None,
    ) -> None:
        self.config = config or CentralTrainingConfig.from_env()
        self.model_config: CentralModelConfig = config.model
        self.texgrad_config: CentralTexGradConfig = config.texgrad
        self.robustness: CentralRobustnessConfig = config.robustness
        self.privacy_config: CentralPrivacyConfig = config.privacy
        self.router_config: CentralRouterConfig = config.router

        self.lora_planner = LoRAUpdatePlanner(self.model_config.adapter_layers)
        self.backprop_trainer = None
        self.use_peft = env_bool("SYNAPSE_CENTRAL_USE_PEFT", False)
        if self.use_peft:
            try:
                self.backprop_trainer = PEFTTexGradTrainer(
                    model_id=self.model_config.base_model,
                    quantization=self.model_config.quantization,
                    lora_config=self.model_config.adapter_layers,
                    dp_config=self.privacy_config.dp,
                )
            except RuntimeError:
                self.backprop_trainer = None
        if self.backprop_trainer is None:
            try:
                self.backprop_trainer = TexGradLoRATrainer(self.model_config.adapter_layers)
            except RuntimeError:
                self.backprop_trainer = None
        self.texgrad_head = TexGradHead(TexGradConfig(lambdas=self.texgrad_config.weights))

        if self.privacy_config.enabled:
            dp_cfg: DPConfig = self.privacy_config.dp
            self.dp_guard: DifferentialPrivacyGuard | _NullDPGuard = DifferentialPrivacyGuard(dp_cfg)
            self._rdp_accountant = RDPAccountant(RDPConfig(delta=self.privacy_config.accountant_delta))
        else:
            self.dp_guard = _NullDPGuard()
            self._rdp_accountant = None

        self.secagg = SecAggAdapter(SecAggConfig.from_env("SYNAPSE_CENTRAL"))

        self.aggregator = HyFICALAggregator(self.config.aggregation)
        self.router = AdapterRouter(self.router_config.router)
        self.auditor = GradientAuditor(
            trim_percent=self.robustness.batch_trim_percent,
            cosine_tau=self.robustness.cosine_tau,
        )
        self.ledger = ledger or ComplianceLedger()
        self._adapter_version = 0
        self._audit_history: List[GradientAuditRecord] = []
        self._adapter_versions: List[str] = []
        self._privacy_budget: Optional[PrivacyBudget] = None
        self._poisoning_flags: Dict[str, List[str]] = {}

        self._samples = self._load_samples(config.training_corpora)

    def _load_samples(self, corpus_paths: Sequence[Path]) -> List[TexGradSample]:
        samples: List[TexGradSample] = []
        for path in corpus_paths:
            if not path.exists():
                continue
            samples.extend(_materialize_samples(_load_json_records(path)))
        if not samples:
            # Fall back to a handful of blanks to keep the trainer operational.
            samples = [TexGradSample.blank() for _ in range(max(self.config.batch_size, 4))]
        return samples

    def _sample_batch(self, step: int) -> List[TexGradSample]:
        rng = np.random.default_rng(seed=step + 1)
        indices = rng.choice(len(self._samples), size=self.config.batch_size, replace=True)
        return [self._samples[idx] for idx in indices]

    def _choose_rank(self, telemetry: Dict[str, float]) -> int:
        rank_set = list(self.model_config.adapter_layers.rank_choices)
        chosen = rank_set[0]
        entropy = telemetry.get("retrieval_entropy", 0.0)
        citation = telemetry.get("citation_cov", 1.0)
        if citation < 0.7 or entropy > 0.6:
            chosen = rank_set[-1]
        return chosen

    def _build_worker_update(
        self,
        worker_id: str,
        batch_samples: Sequence[TexGradSample],
        rank: int,
        timestamp: int,
    ) -> Tuple[AdapterUpdate, np.ndarray]:
        """
        Construct an AdapterUpdate for a synthetic worker along with a representative gradient vector.
        """
        if self.backprop_trainer is not None:
            try:
                layer_vectors, metrics, steps = self.backprop_trainer.train_on_batch(batch_samples, rank)
                step_count = steps
            except RuntimeError:
                layer_vectors = self.lora_planner.build_layer_updates(batch_samples, rank=rank)
                metrics = self.texgrad_head.aggregate_metrics(batch_samples)
                step_count = len(batch_samples)
        else:
            layer_vectors = self.lora_planner.build_layer_updates(batch_samples, rank=rank)
            metrics = self.texgrad_head.aggregate_metrics(batch_samples)
            step_count = len(batch_samples)

        representative = []
        layer_updates: List[LayerUpdate] = []

        for layer, vector in layer_vectors.items():
            sanitized = self.dp_guard.sanitize(vector)
            representative.append(sanitized)
            masked_bytes, metadata, norm = self.secagg.mask(
                client_id=worker_id,
                round_hint=self._adapter_version + 1,
                layer=layer,
                vector=sanitized,
            )
            layer_updates.append(
                LayerUpdate(
                    layer=layer,
                    format="LoRA",
                    rank=rank,
                    delta_hash=self.lora_planner.delta_hash(sanitized),
                    masked_delta=masked_bytes,
                    norm=norm,
                    mask_metadata=metadata,
                )
            )

        combined = np.concatenate(representative) if representative else np.zeros(1, dtype=np.float64)
        steps = max(step_count, 1)

        telemetry = UpdateTelemetry(
            freshness_ts=timestamp,
            steps=steps,
            loss_lm=max(0.5, 2.0 - metrics.entailment - metrics.citation_coverage),
            texgrad=TexGradMetrics(
                entailment=metrics.entailment,
                citation_coverage=metrics.citation_coverage,
                contrastive_margin=metrics.contrastive_margin,
                retrieval_entropy=metrics.retrieval_entropy,
                semantic_fingerprint=metrics.semantic_fingerprint,
            ),
        )

        dp_budget = PrivacyBudget(
            clipping=self.dp_guard.clip_norm(),
            sigma=self.dp_guard.sigma(),
            epsilon_local=self.dp_guard.estimate_local_epsilon(steps),
        )

        update = AdapterUpdate(
            client_id=worker_id,
            round_hint=self._adapter_version + 1,
            layer_updates=layer_updates,
            telemetry=telemetry,
            dp_local=dp_budget,
        )
        return update, combined

    def _decode_layer(self, layer_update: LayerUpdate) -> np.ndarray:
        return self.secagg.unmask(layer_update)

    def _update_privacy_accountant(self, participant_count: int, sigma: float) -> float:
        if not self._rdp_accountant:
            return 0.0
        total_clients = max(participant_count, 1)
        participation_rate = min(1.0, total_clients / max(self.config.batch_size, 1))
        self._rdp_accountant.accumulate(participation_rate=participation_rate, sigma=sigma)
        epsilon = min(self._rdp_accountant.epsilon(), self.privacy_config.epsilon_cap)
        return float(epsilon)

    def run(self) -> CentralTrainingSummary:
        """
        Execute the centralized training loop for the configured number of epochs.
        """
        steps_executed = 0

        for epoch in range(1, self.config.epochs + 1):
            for step in range(1, self.config.steps_per_epoch + 1):
                batch_samples = self._sample_batch(steps_executed)
                timestamp = int(time.time())
                telemetry_hint = {
                    "retrieval_entropy": np.mean([sample.retrieval_entropy for sample in batch_samples]),
                    "citation_cov": np.mean([sample.citation_coverage for sample in batch_samples]),
                }
                rank = self._choose_rank(telemetry_hint)

                workers = max(1, len(batch_samples) // max(rank, 1))
                worker_updates: List[AdapterUpdate] = []
                representative_vectors: List[Tuple[str, np.ndarray]] = []

                for worker_idx in range(workers):
                    self.secagg.next_round()
                    worker_samples = batch_samples[worker_idx::workers] or batch_samples
                    worker_id = f"central-worker-{worker_idx}"
                    update, vector = self._build_worker_update(worker_id, worker_samples, rank, timestamp)
                    worker_updates.append(update)
                    representative_vectors.append((worker_id, vector))

                kept_vectors, records = self.auditor.audit([(cid, vec) for cid, vec in representative_vectors])
                kept_ids = {cid for cid, _ in kept_vectors}
                filtered_updates = [update for update in worker_updates if update.client_id in kept_ids]

                self._audit_history.extend(records)

                if not filtered_updates:
                    continue

                aggregated = self.aggregator.aggregate_updates(filtered_updates, self._decode_layer)
                if not aggregated:
                    continue

                layer_metrics = {
                    layer: result.telemetry_summary
                    for layer, result in aggregated.items()
                }

                plan = self.router.plan_bundle(layer_metrics)
                adapters = {
                    layer: experts
                    for layer, (experts, _) in plan.items()
                }

                average_sigma = np.mean([update.dp_local.sigma for update in filtered_updates])
                epsilon_global = self._update_privacy_accountant(len(filtered_updates), average_sigma)

                self._adapter_version += 1
                version = f"central-v{self._adapter_version}"
                self._adapter_versions.append(version)

                self._poisoning_flags = self.aggregator.poisoning_flags
                self._privacy_budget = PrivacyBudget(
                    clipping=self.dp_guard.clip_norm(),
                    sigma=average_sigma,
                    epsilon_local=epsilon_global,
                )

                if step % self.config.record_every_steps == 0 or step == self.config.steps_per_epoch:
                    self._record_ledger_entry(
                        version=version,
                        updates=filtered_updates,
                        aggregated=aggregated,
                        epsilon=epsilon_global,
                        sigma=average_sigma,
                    )

                steps_executed += 1

        summary = CentralTrainingSummary(
            epochs_completed=self.config.epochs,
            steps_executed=steps_executed,
            adapter_versions=list(self._adapter_versions),
            privacy_budget=self._privacy_budget,
            poisoning_flags=dict(self._poisoning_flags),
            audit_records=list(self._audit_history),
            ledger_entries=self.ledger.to_list(),
        )
        return summary

    def _record_ledger_entry(
        self,
        version: str,
        updates: Sequence[AdapterUpdate],
        aggregated: Dict[str, AggregatedLayerResult],
        epsilon: float,
        sigma: float,
    ) -> None:
        telemetry = {
            f"{layer}.entailment": result.telemetry_summary.get("entailment_avg", 0.0)
            for layer, result in aggregated.items()
        }
        for layer, result in aggregated.items():
            telemetry[f"{layer}.citation"] = result.telemetry_summary.get("citation_avg", 0.0)
            telemetry[f"{layer}.contrastive"] = result.telemetry_summary.get("contrastive_avg", 0.0)
            telemetry[f"{layer}.retrieval_entropy"] = result.telemetry_summary.get("retrieval_entropy_avg", 0.0)

        entry = LedgerEntry(
            round_id=self._adapter_version,
            timestamp=datetime.utcnow(),
            epsilon=epsilon,
            delta=self.privacy_config.accountant_delta,
            participant_count=len({update.client_id for update in updates}),
            layers_updated=list(aggregated.keys()),
            spectral_k=self.config.aggregation.spectral_k,
            dp_sigma=sigma,
            release_notes=f"centralized adapter release {version}",
            telemetry_snapshot=telemetry,
        )
        self.ledger.record(entry)
