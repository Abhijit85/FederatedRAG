from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from synapse.compliance import ComplianceLedger, LedgerEntry
from synapse.hyfical import (
    AdapterRouter,
    AggregationConfig,
    AsyncWindowScheduler,
    AggregatedLayerResult,
    GlobalAdapterBundle,
    HyFICALAggregator,
    RouterConfig,
    SchedulerConfig,
)
from synapse.hyfical.contracts import AdapterExpert, AdapterUpdate, LayerUpdate, PrivacyBudget
from synapse.knowledge.compendium import KnowledgePackage, SynapseCompendium
from synapse.privacy.accountant import RDPAccountant, RDPConfig
from synapse.secure import build_secure_provider
from synapse.server.aggregation import (
    AggregationMode,
    AggregatorFacade,
    SecAggSumAggregator,
    TEEAggregator,
)


@dataclass
class ServerConfig:
    """Configuration object for the central SYNAPSE server."""

    server_id: str = "synapse-central"
    enable_versioning: bool = True
    enable_hyfical: bool = True
    spectral_k: int = 20
    anomaly_tau_cos: float = 0.65
    median_iters: int = 15
    freshness_half_life_min: float = 30.0
    trust_alpha: float = 0.9
    z_threshold: float = 3.0
    decay_floor: float = 0.1
    scheduler_window_seconds: int = 900
    router_experts_per_layer: int = 2
    router_rank_choices: tuple[int, ...] = field(default_factory=lambda: (4, 8, 16))
    privacy_delta: float = 1e-6
    epsilon_cap: float = 8.0
    expected_clients: int = 4
    aggregation_mode: AggregationMode = AggregationMode.ROBUST
    secagg_provider: str = "simple"
    secagg_secret: str = "synapse-shared-secret"
    secagg_attestation: str = ""


class SynapseServer:
    """
    Central coordinator that receives consolidated knowledge packages
    from edge aggregators, updates the global compendium, and distributes
    new versions.
    """

    def __init__(self, config: Optional[ServerConfig] = None) -> None:
        self.config = config or ServerConfig()
        self.compendium = SynapseCompendium()
        self._versions: List[Dict[str, str]] = []
        self._adapter_version: int = 0
        self._latest_bundle: Optional[GlobalAdapterBundle] = None
        self._aggregator_facade: Optional[AggregatorFacade] = None
        self._hyfical_core: Optional[HyFICALAggregator] = None
        self._secagg_provider = None

        self._aggregation_mode = self.config.aggregation_mode

        if self.config.enable_hyfical:
            agg_config = AggregationConfig(
                spectral_k=self.config.spectral_k,
                anomaly_tau_cos=self.config.anomaly_tau_cos,
                median_iters=self.config.median_iters,
                freshness_half_life_min=self.config.freshness_half_life_min,
                trust_alpha=self.config.trust_alpha,
                z_threshold=self.config.z_threshold,
                decay_floor=self.config.decay_floor,
            )
            self._hyfical_core: Optional[HyFICALAggregator]
            if self._aggregation_mode == AggregationMode.ROBUST:
                self._hyfical_core = HyFICALAggregator(agg_config)
                self._aggregator_facade: AggregatorFacade = TEEAggregator(self._hyfical_core)
            else:
                self._hyfical_core = None
                self._aggregator_facade = SecAggSumAggregator()
            self._scheduler = AsyncWindowScheduler(
                SchedulerConfig(window_seconds=self.config.scheduler_window_seconds)
            )
            self._router = AdapterRouter(
                RouterConfig(
                    experts_per_layer=self.config.router_experts_per_layer,
                    rank_choices=self.config.router_rank_choices,
                )
            )
            self._secagg_provider = build_secure_provider(
                self.config.secagg_provider,
                self.config.secagg_secret,
                attestation_key=self.config.secagg_attestation,
            )
            self._ledger = ComplianceLedger()
            self._accountant = RDPAccountant(RDPConfig(delta=self.config.privacy_delta))
            self._adapter_bank: Dict[str, AggregatedLayerResult] = {}
        else:
            self._aggregation_mode = AggregationMode.ROBUST
            self._hyfical_core = None
            self._aggregator_facade = None
            self._scheduler = None
            self._router = None
            self._secagg_provider = None
            self._ledger = ComplianceLedger()
            self._accountant = RDPAccountant(RDPConfig(delta=self.config.privacy_delta))
            self._adapter_bank = {}

    def ingest_from_edge(self, package: KnowledgePackage) -> None:
        """
        Update the global compendium with artifacts from an edge package.
        """
        self.compendium.ingest(package)

        if self.config.enable_versioning:
            version_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "edge_id": package.source_id,
                "artifact_count": str(len(package.artifacts)),
            }
            self._versions.append(version_record)

    def ingest_adapter_update(self, update: AdapterUpdate) -> None:
        if not self.config.enable_hyfical or self._scheduler is None:
            return
        self._scheduler.offer(update)

    def distribute_snapshot(self) -> KnowledgePackage:
        """
        Produce a snapshot package that can be sent to edges or clients.
        """
        snapshot = self.compendium.build_snapshot()
        snapshot.source_id = self.config.server_id
        return snapshot

    def process_pending_windows(self) -> Optional[GlobalAdapterBundle]:
        if not self.config.enable_hyfical or self._scheduler is None:
            return None

        ready_updates = self._scheduler.collect_ready()
        return self._aggregate_updates(ready_updates)

    def flush_pending_updates(self) -> Optional[GlobalAdapterBundle]:
        if not self.config.enable_hyfical or self._scheduler is None:
            return None
        drained = self._scheduler.drain()
        return self._aggregate_updates(drained)

    def _aggregate_updates(self, updates: List[AdapterUpdate]) -> Optional[GlobalAdapterBundle]:
        if not updates:
            return None

        if self._aggregator_facade is None or self._secagg_provider is None:
            return None

        layer_rank_hint: Dict[str, int] = {}
        for update in updates:
            for layer_update in update.layer_updates:
                layer_rank_hint.setdefault(layer_update.layer, layer_update.rank)

        aggregated = self._aggregator_facade.aggregate(updates, self._decode_layer)
        if not aggregated:
            return None

        for layer, result in aggregated.items():
            self._adapter_bank[layer] = result

        layer_metrics = {
            layer: result.telemetry_summary
            for layer, result in aggregated.items()
        }

        plan = self._router.plan_bundle(layer_metrics) if self._router else {}
        if not plan:
            plan = {
                layer: (
                    [
                        AdapterExpert(
                            id=f"default::{layer}",
                            rank=layer_rank_hint.get(layer, 4),
                        )
                    ],
                    "general",
                )
                for layer in aggregated
            }

        adapters: Dict[str, List] = {}
        router_hints: Dict[str, str] = {}
        for layer, (experts, hint) in plan.items():
            adapters[layer] = experts
            router_hints[layer] = hint

        average_sigma = self._avg_sigma(updates)
        avg_clip = self._avg_clip(updates)
        epsilon_global = self._update_accountant(updates, average_sigma)

        self._adapter_version += 1
        version = f"v{self._adapter_version}"
        budget = PrivacyBudget(
            clipping=avg_clip,
            sigma=average_sigma,
            epsilon_local=epsilon_global,
        )
        if self._aggregation_mode == AggregationMode.ROBUST:
            release_notes = f"robust-agg: spectral_k={self.config.spectral_k}, dp_sigma={average_sigma:.2f}"
        else:
            release_notes = f"secagg-sum: contributors={len(filtered_updates)}"

        bundle = GlobalAdapterBundle(
            version=version,
            adapters=adapters,
            router_hints=router_hints,
            privacy_budget_remaining=budget,
            release_notes=release_notes,
        )

        self._record_ledger_entry(version, updates, aggregated, epsilon_global, average_sigma)
        self._latest_bundle = bundle
        return bundle

    def latest_bundle(self) -> Optional[GlobalAdapterBundle]:
        return self._latest_bundle

    def _avg_sigma(self, updates: List[AdapterUpdate]) -> float:
        if not updates:
            return 0.0
        return sum(update.dp_local.sigma for update in updates) / len(updates)

    def _avg_clip(self, updates: List[AdapterUpdate]) -> float:
        if not updates:
            return 0.0
        return sum(update.dp_local.clipping for update in updates) / len(updates)

    def _update_accountant(self, updates: List[AdapterUpdate], sigma: float) -> float:
        if not updates:
            return 0.0
        distinct_clients = {update.client_id for update in updates}
        participation_rate = min(len(distinct_clients) / max(self.config.expected_clients, 1), 1.0)
        if self._accountant:
            self._accountant.accumulate(participation_rate=participation_rate, sigma=sigma)
            epsilon = min(self._accountant.epsilon(), self.config.epsilon_cap)
        else:
            epsilon = 0.0
        return epsilon

    def _decode_layer(self, layer_update: LayerUpdate) -> np.ndarray:
        if not self._secagg_provider:
            raise ValueError("Secure aggregation provider unavailable")
        return self._secagg_provider.unmask(layer_update).astype(np.float64)

    def _record_ledger_entry(
        self,
        version: str,
        updates: List[AdapterUpdate],
        aggregated: Dict[str, AggregatedLayerResult],
        epsilon: float,
        sigma: float,
    ) -> None:
        if not self._ledger:
            return
        layers = list(aggregated.keys())
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
            delta=self.config.privacy_delta,
            participant_count=len({update.client_id for update in updates}),
            layers_updated=layers,
            spectral_k=self.config.spectral_k,
            dp_sigma=sigma,
            release_notes=f"version {version}",
            telemetry_snapshot=telemetry,
        )
        self._ledger.record(entry)

    @property
    def version_history(self) -> List[Dict[str, str]]:
        """Lightweight access to recorded ingestion events."""
        return list(self._versions)

    @property
    def ledger(self) -> ComplianceLedger:
        return self._ledger

    @property
    def trust_scores(self) -> Dict[str, float]:
        if not self.config.enable_hyfical or self._aggregator_facade is None:
            return {}
        return self._aggregator_facade.trust_scores

    @property
    def aggregator(self) -> Optional[HyFICALAggregator]:
        return self._hyfical_core

    @property
    def aggregator_config(self) -> Optional[AggregationConfig]:
        if self._aggregator_facade is None:
            return None
        return self._aggregator_facade.config

    @property
    def scheduler_config(self) -> Optional[SchedulerConfig]:
        return self._scheduler.config if self._scheduler else None

    @property
    def router_config(self) -> Optional[RouterConfig]:
        return self._router.config if self._router else None

    @property
    def secagg_enabled(self) -> bool:
        return self._secagg_provider is not None

    @property
    def rdp_accountant(self) -> Optional[RDPAccountant]:
        return self._accountant

    @property
    def aggregator_facade(self) -> Optional[AggregatorFacade]:
        return self._aggregator_facade

    @property
    def aggregation_mode(self) -> AggregationMode:
        return self._aggregation_mode

    @property
    def secagg_attestation(self) -> Dict[str, str]:
        if not self._secagg_provider:
            return {}
        return getattr(self._secagg_provider, "attest", lambda: {})()
