from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from synapse.hyfical import AggregationConfig, RouterConfig, SchedulerConfig
from synapse.training import DPConfig, LoRALayerConfig, TexGradConfig
from synapse.utils import (
    env_bool,
    env_float,
    env_int,
    env_list,
    env_path_list,
    env_str,
)


@dataclass
class CentralModelConfig:
    """
    Describes the base LLM and adapter layout used during centralized tuning.
    """

    base_model: str = "Llama-3-8B-Instruct"
    quantization: str = "4bit"
    adapter_layers: LoRALayerConfig = field(default_factory=LoRALayerConfig)


@dataclass
class CentralTexGradConfig:
    """
    TexGrad loss steering configuration.
    """

    weights: Dict[str, float] = field(default_factory=lambda: {"ent": 0.5, "attr": 0.5, "ctr": 0.3})
    cosine_projection: bool = True


@dataclass
class CentralRobustnessConfig:
    """
    Controls gradient hygiene inside the centralized trainer.
    """

    batch_trim_percent: float = 0.05
    cosine_tau: float = 0.65
    spectral_k: int = 20
    median_merge_every: int = 200
    freshness_half_life_min: float = 30.0


@dataclass
class CentralPrivacyConfig:
    """
    Optional differential privacy safeguards.
    """

    enabled: bool = True
    dp: DPConfig = field(default_factory=DPConfig)
    accountant_delta: float = 1e-6
    epsilon_cap: float = 8.0


@dataclass
class CentralRouterConfig:
    """
    ARR/MoA routing hints for centralized training.
    """

    router: RouterConfig = field(default_factory=lambda: RouterConfig(rank_choices=(4, 8, 16)))
    features: Sequence[str] = ("retrieval_entropy", "citation_cov", "nli_score", "domain_tag")


@dataclass
class CentralTrainingConfig:
    """
    Consolidated configuration for the centralized TexGrad-LoRA trainer.
    """

    model: CentralModelConfig = field(default_factory=CentralModelConfig)
    texgrad: CentralTexGradConfig = field(default_factory=CentralTexGradConfig)
    privacy: CentralPrivacyConfig = field(default_factory=CentralPrivacyConfig)
    robustness: CentralRobustnessConfig = field(default_factory=CentralRobustnessConfig)
    router: CentralRouterConfig = field(default_factory=CentralRouterConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig(window_seconds=900))
    epochs: int = 1
    steps_per_epoch: int = 200
    batch_size: int = 8
    record_every_steps: int = 20
    training_corpora: Sequence[Path] = field(default_factory=lambda: ())
    domain_tags: Sequence[str] = field(default_factory=lambda: ("general",))

    @classmethod
    def from_env(cls) -> "CentralTrainingConfig":
        config = cls()

        vlm_default = env_str("VLM_MODEL", config.model.base_model) or config.model.base_model
        config.model.base_model = env_str("SYNAPSE_CENTRAL_BASE_MODEL", vlm_default) or vlm_default
        config.model.quantization = env_str("SYNAPSE_CENTRAL_QUANTIZATION", config.model.quantization) or config.model.quantization
        config.model.adapter_layers = LoRALayerConfig.from_env(prefix="SYNAPSE_CENTRAL")

        weights_raw = env_str("SYNAPSE_CENTRAL_TEXGRAD_WEIGHTS", "")
        if weights_raw:
            parsed: Dict[str, float] = {}
            for token in weights_raw.split(","):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                try:
                    parsed[key.strip()] = float(value)
                except ValueError:
                    continue
            if parsed:
                config.texgrad.weights = parsed
        config.texgrad.cosine_projection = env_bool(
            "SYNAPSE_CENTRAL_TEXGRAD_COSINE",
            config.texgrad.cosine_projection,
        )

        config.robustness.batch_trim_percent = env_float(
            "SYNAPSE_CENTRAL_ROBUST_TRIM",
            config.robustness.batch_trim_percent,
        )
        config.robustness.cosine_tau = env_float(
            "SYNAPSE_CENTRAL_ROBUST_COSINE_TAU",
            config.robustness.cosine_tau,
        )
        config.robustness.spectral_k = env_int(
            "SYNAPSE_CENTRAL_ROBUST_SPECTRAL_K",
            config.robustness.spectral_k,
        )
        config.robustness.median_merge_every = env_int(
            "SYNAPSE_CENTRAL_ROBUST_MEDIAN_EVERY",
            config.robustness.median_merge_every,
        )
        config.robustness.freshness_half_life_min = env_float(
            "SYNAPSE_CENTRAL_ROBUST_HALF_LIFE",
            config.robustness.freshness_half_life_min,
        )

        config.privacy.enabled = env_bool(
            "SYNAPSE_CENTRAL_DP_ENABLED",
            config.privacy.enabled,
        )
        config.privacy.accountant_delta = env_float(
            "SYNAPSE_CENTRAL_DP_DELTA",
            config.privacy.accountant_delta,
        )
        config.privacy.epsilon_cap = env_float(
            "SYNAPSE_CENTRAL_DP_EPS_CAP",
            config.privacy.epsilon_cap,
        )
        if config.privacy.enabled:
            config.privacy.dp = DPConfig.from_env(prefix="SYNAPSE_CENTRAL")

        router_cfg = config.router.router
        router_cfg.experts_per_layer = env_int(
            "SYNAPSE_CENTRAL_ROUTER_EXPERTS",
            router_cfg.experts_per_layer,
        )
        router_cfg.rank_policy = env_str(
            "SYNAPSE_CENTRAL_ROUTER_POLICY",
            router_cfg.rank_policy,
        ) or router_cfg.rank_policy
        router_cfg.entropy_high_threshold = env_float(
            "SYNAPSE_CENTRAL_ROUTER_ENTROPY_HIGH",
            router_cfg.entropy_high_threshold,
        )
        router_cfg.citation_low_threshold = env_float(
            "SYNAPSE_CENTRAL_ROUTER_CITATION_LOW",
            router_cfg.citation_low_threshold,
        )
        router_cfg.contrastive_low_threshold = env_float(
            "SYNAPSE_CENTRAL_ROUTER_CONTRASTIVE_LOW",
            router_cfg.contrastive_low_threshold,
        )
        rank_override = env_list(
            "SYNAPSE_CENTRAL_LORA_RANKS",
            [str(rank) for rank in router_cfg.rank_choices],
        )
        try:
            router_cfg.rank_choices = tuple(sorted({int(value) for value in rank_override}))
        except ValueError:
            pass
        config.router.features = tuple(
            env_list("SYNAPSE_CENTRAL_ROUTER_FEATURES", config.router.features)
        )

        config.aggregation.spectral_k = env_int(
            "SYNAPSE_CENTRAL_AGG_SPECTRAL_K",
            config.aggregation.spectral_k,
        )
        config.aggregation.anomaly_tau_cos = env_float(
            "SYNAPSE_CENTRAL_AGG_TAU",
            config.aggregation.anomaly_tau_cos,
        )
        config.aggregation.median_iters = env_int(
            "SYNAPSE_CENTRAL_AGG_MEDIAN_ITERS",
            config.aggregation.median_iters,
        )
        config.aggregation.freshness_half_life_min = env_float(
            "SYNAPSE_CENTRAL_AGG_HALF_LIFE",
            config.aggregation.freshness_half_life_min,
        )
        config.aggregation.trust_alpha = env_float(
            "SYNAPSE_CENTRAL_AGG_TRUST_ALPHA",
            config.aggregation.trust_alpha,
        )
        config.aggregation.z_threshold = env_float(
            "SYNAPSE_CENTRAL_AGG_Z_THRESHOLD",
            config.aggregation.z_threshold,
        )
        config.aggregation.decay_floor = env_float(
            "SYNAPSE_CENTRAL_AGG_DECAY_FLOOR",
            config.aggregation.decay_floor,
        )

        config.scheduler.window_seconds = env_int(
            "SYNAPSE_CENTRAL_WINDOW_SECONDS",
            config.scheduler.window_seconds,
        )

        config.epochs = env_int("SYNAPSE_CENTRAL_EPOCHS", config.epochs)
        config.steps_per_epoch = env_int(
            "SYNAPSE_CENTRAL_STEPS_PER_EPOCH",
            config.steps_per_epoch,
        )
        config.batch_size = env_int("SYNAPSE_CENTRAL_BATCH_SIZE", config.batch_size)
        config.record_every_steps = env_int(
            "SYNAPSE_CENTRAL_RECORD_EVERY",
            config.record_every_steps,
        )

        config.training_corpora = env_path_list(
            "SYNAPSE_CENTRAL_TRAIN_CORPORA",
            config.training_corpora,
        )
        config.domain_tags = tuple(
            env_list("SYNAPSE_CENTRAL_DOMAIN_TAGS", config.domain_tags)
        )

        return config
