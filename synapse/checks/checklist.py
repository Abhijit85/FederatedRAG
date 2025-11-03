from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from synapse.clients.unified_client import UnifiedQAClient
    from synapse.runtime import SynapseRuntime
    from synapse.training.lora import LoRALayerConfig

from synapse.server import AggregationMode


@dataclass
class ChecklistResult:
    passed: bool
    failures: List[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if not self.passed:
            message = "Federation checklist violations detected:\n" + "\n".join(f"- {failure}" for failure in self.failures)
            raise ValueError(message)


class FederationChecklist:
    """
    Validates that the critical HyFICAL safeguards remain intact.
    """

    REQUIRED_RANKS: Set[int] = {4, 8, 16}
    REQUIRED_TARGET_KEYS: Set[str] = {"q_proj", "k_proj", "v_proj"}

    @classmethod
    def ensure_runtime(cls, runtime: "SynapseRuntime") -> None:
        result = cls.validate_runtime(runtime)
        result.raise_for_errors()

    @classmethod
    def validate_runtime(cls, runtime: "SynapseRuntime") -> ChecklistResult:
        failures: List[str] = []

        # Client-side guarantees
        for client_id, client in runtime.clients.items():
            if hasattr(client, "lora_planner"):
                config: LoRALayerConfig = client.lora_planner.config
                ranks = set(config.rank_choices)
                if cls.REQUIRED_RANKS - ranks:
                    failures.append(f"{client_id}: LoRA rank choices must include {sorted(cls.REQUIRED_RANKS)}")

                target_modules = set(config.resolve_layers())
                if not cls.REQUIRED_TARGET_KEYS.issubset({token.split(".")[1] if "." in token else token for token in target_modules}):
                    failures.append(f"{client_id}: LoRA target modules must cover q_proj/k_proj/v_proj attention paths")

            if not getattr(client, "base_model_quantized", False):
                failures.append(f"{client_id}: base model must remain frozen & quantized per checklist")

            dp_guard = getattr(client, "dp_guard", None)
            if not dp_guard or dp_guard.config.clip_norm <= 0 or dp_guard.config.noise_multiplier <= 0:
                failures.append(f"{client_id}: differential privacy guard requires positive clip norm and noise multiplier")
            if not dp_guard or dp_guard.config.sample_rate <= 0:
                failures.append(f"{client_id}: differential privacy guard requires a valid client subsampling rate")

            secagg = getattr(client, "secagg", None)
            if not secagg or not getattr(secagg.config, "protocol", ""):
                failures.append(f"{client_id}: secure aggregation protocol must be specified")

        # Server-side guarantees
        server = runtime.server
        if server.config.enable_hyfical:
            facade = server.aggregator_facade
            if not facade:
                failures.append("Server: aggregation facade must be present when HyFICAL is enabled")
            else:
                if server.aggregation_mode == AggregationMode.ROBUST:
                    core = server.aggregator
                    if not core:
                        failures.append("Server: robust mode requires HyFICAL core aggregator")
                    else:
                        agg_config = core.config
                        if agg_config.spectral_k <= 0:
                            failures.append("Server: spectral_k must be positive")
                        if agg_config.anomaly_tau_cos <= 0 or agg_config.anomaly_tau_cos >= 1:
                            failures.append("Server: cosine divergence threshold τ must be within (0,1)")
                        if agg_config.median_iters <= 0:
                            failures.append("Server: geometric median iterations must be positive")
                        if agg_config.freshness_half_life_min <= 0:
                            failures.append("Server: freshness half-life must be positive")
                        if not isinstance(core.quarantine_queue, list):
                            failures.append("Server: quarantine queue should record anomalous updates")
                        if not isinstance(core.adapter_norm_zscores, dict):
                            failures.append("Server: adapter norm z-scores must be tracked for observability")
                else:  # SUM_ONLY
                    # Ensure observability still provides norm statistics
                    if not isinstance(facade.adapter_norm_zscores, dict):
                        failures.append("Server: sum-only mode must expose adapter norm statistics")

            scheduler_config = server.scheduler_config
            if scheduler_config and scheduler_config.window_seconds <= 0:
                failures.append("Server: aggregation window W must be positive")

            router_config = server.router_config
            if router_config:
                rank_set = set(router_config.rank_choices)
                if cls.REQUIRED_RANKS - rank_set:
                    failures.append("Server: router must support ranks {4,8,16}")

            if not server.secagg_enabled:
                failures.append("Server: secure aggregation must remain active")

            if not server.rdp_accountant:
                failures.append("Server: RDP accountant must be active to track ε/δ")
            else:
                if server.rdp_accountant.config.delta <= 0:
                    failures.append("Server: RDP accountant delta must be positive")

            if not getattr(server, "ledger", None):
                failures.append("Server: compliance ledger must be instantiated for rollback tracking")

        return ChecklistResult(passed=not failures, failures=failures)
