from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Sequence


@dataclass
class TexGradMetrics:
    """
    Faithfulness-oriented telemetry produced by the TexGrad head.
    """

    entailment: float
    citation_coverage: float
    contrastive_margin: float
    retrieval_entropy: float = 0.0
    semantic_fingerprint: Sequence[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "entailment": self.entailment,
            "citation_coverage": self.citation_coverage,
            "contrastive_margin": self.contrastive_margin,
            "retrieval_entropy": self.retrieval_entropy,
            "semantic_fingerprint": list(self.semantic_fingerprint),
        }


@dataclass
class UpdateTelemetry:
    """
    Scalar telemetry bundled with each adapter update.
    """

    freshness_ts: int
    steps: int
    loss_lm: float
    texgrad: TexGradMetrics

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["texgrad"] = self.texgrad.to_dict()
        return data


@dataclass
class PrivacyBudget:
    """
    Local privacy budget accounting metadata.
    """

    clipping: float
    sigma: float
    epsilon_local: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "clipping": float(self.clipping),
            "sigma": float(self.sigma),
            "epsilon_local": float(self.epsilon_local),
        }


@dataclass
class LayerUpdate:
    """
    Per-layer update metadata and SecAgg-masked payload reference.
    """

    layer: str
    format: str
    rank: int
    delta_hash: str
    masked_delta: bytes
    norm: float
    mask_metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "layer": self.layer,
            "format": self.format,
            "rank": int(self.rank),
            "delta_hash": self.delta_hash,
            "masked_delta": self.masked_delta.hex(),
            "norm": float(self.norm),
            "mask_metadata": dict(self.mask_metadata),
        }


@dataclass
class AdapterUpdate:
    """
    Payload emitted by a client after secure aggregation masking.
    """

    client_id: str
    round_hint: int
    layer_updates: List[LayerUpdate]
    telemetry: UpdateTelemetry
    dp_local: PrivacyBudget

    def to_dict(self) -> Dict[str, object]:
        return {
            "client_id": self.client_id,
            "round_hint": int(self.round_hint),
            "layer_updates": [update.to_dict() for update in self.layer_updates],
            "telemetry": self.telemetry.to_dict(),
            "dp_local": self.dp_local.to_dict(),
        }


@dataclass
class AdapterExpert:
    """
    Metadata for a single adapter expert selection.
    """

    id: str
    rank: int

    def to_dict(self) -> Dict[str, object]:
        return {"id": self.id, "rank": int(self.rank)}


@dataclass
class GlobalAdapterBundle:
    """
    Control payload broadcast from the server to clients.
    """

    version: str
    adapters: Dict[str, List[AdapterExpert]]
    router_hints: Dict[str, str]
    privacy_budget_remaining: PrivacyBudget
    release_notes: str
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "adapters": {
                layer: [expert.to_dict() for expert in experts]
                for layer, experts in self.adapters.items()
            },
            "router_hints": dict(self.router_hints),
            "privacy_budget_remaining": self.privacy_budget_remaining.to_dict(),
            "release_notes": self.release_notes,
            "generated_at": self.generated_at.isoformat(),
        }
