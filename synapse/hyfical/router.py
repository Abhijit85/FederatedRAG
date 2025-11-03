from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from .contracts import AdapterExpert


@dataclass
class RouterConfig:
    """
    Configuration for the adaptive routing policy.
    """

    experts_per_layer: int = 2
    rank_choices: Sequence[int] = (4, 8, 16)
    rank_policy: str = "entropy↓ ⇒ rank↑ ; citation↓ ⇒ rank↑"
    entropy_high_threshold: float = 0.75
    citation_low_threshold: float = 0.65
    contrastive_low_threshold: float = 0.25


class AdapterRouter:
    """
    Maintains the global adapter bank and produces ARR/MoA selections.
    """

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()
        self._adapter_bank: Dict[str, Dict[str, AdapterExpert]] = {}
        self._round_counter: int = 0

    def register_experts(self, layer: str, experts: Iterable[AdapterExpert]) -> None:
        bank = self._adapter_bank.setdefault(layer, {})
        for expert in experts:
            bank[expert.id] = expert

    def _allocate_new_experts(self, layer: str, rank: int) -> List[AdapterExpert]:
        """
        Allocate synthetic expert identifiers when bank capacity is insufficient.
        """
        bank = self._adapter_bank.setdefault(layer, {})
        needed = max(self.config.experts_per_layer - len(bank), 0)
        allocated: List[AdapterExpert] = []
        for idx in range(needed):
            expert_id = f"{layer}.expert.{self._round_counter}.{idx}"
            expert = AdapterExpert(id=expert_id, rank=rank)
            bank[expert_id] = expert
            allocated.append(expert)
        return allocated

    def _choose_rank(self, telemetry: Dict[str, float]) -> int:
        rank_options = list(self.config.rank_choices)
        if not rank_options:
            return 4

        base_rank = rank_options[0]
        citation = telemetry.get("citation_avg", 1.0)
        entropy = telemetry.get("retrieval_entropy_avg", 0.0)
        contrastive = telemetry.get("contrastive_avg", 1.0)

        if citation < self.config.citation_low_threshold:
            base_rank = rank_options[min(2, len(rank_options) - 1)]
        if entropy > self.config.entropy_high_threshold or contrastive < self.config.contrastive_low_threshold:
            base_rank = rank_options[-1]
        return base_rank

    def plan_layer(self, layer: str, telemetry: Dict[str, float]) -> Tuple[List[AdapterExpert], str]:
        """
        Pick experts and router hint for the provided layer.
        """
        bank = self._adapter_bank.setdefault(layer, {})

        chosen_rank = self._choose_rank(telemetry)
        available = list(bank.values())
        if len(available) < self.config.experts_per_layer:
            available.extend(self._allocate_new_experts(layer, chosen_rank))

        # Select experts with closest rank to demand.
        available.sort(key=lambda expert: abs(expert.rank - chosen_rank))
        selected = available[: self.config.experts_per_layer]

        hint = "general"
        if telemetry.get("citation_avg", 1.0) < self.config.citation_low_threshold:
            hint = "grounding"
        elif telemetry.get("retrieval_entropy_avg", 0.0) > self.config.entropy_high_threshold:
            hint = "diversify"

        return selected, hint

    def plan_bundle(self, layer_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Tuple[List[AdapterExpert], str]]:
        """
        Produce adapter selections for each layer.
        """
        self._round_counter += 1
        plan: Dict[str, Tuple[List[AdapterExpert], str]] = {}
        for layer, telemetry in layer_metrics.items():
            selected, hint = self.plan_layer(layer, telemetry)
            plan[layer] = (selected, hint)
        return plan

    @property
    def adapter_bank(self) -> Dict[str, Dict[str, AdapterExpert]]:
        return {
            layer: dict(experts)
            for layer, experts in self._adapter_bank.items()
        }
