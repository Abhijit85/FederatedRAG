from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from third_party.textgrad import Variable
from third_party.textgrad.autograd.functional import aggregate as tg_aggregate
from third_party.textgrad_utils.prompt_complexity import calculate_text_complexity
from third_party.textgrad_utils.prompt_template import (
    FORMATTING_INSTRUCTION,
    SUMMARIZATION_TEMPLATE,
    UID_TEMPLATE,
)

from synapse.knowledge.compendium import KnowledgeArtifact, KnowledgePackage
from synapse.textgrad_support import TextGradSettings


@dataclass
class EdgeConfig:
    """Configuration for an edge aggregator."""

    edge_id: str
    domains: List[str] = field(default_factory=list)
    retain_history: bool = True


class EdgeAggregator:
    """
    Consolidates knowledge packages from a cluster of clients before
    forwarding them to the central SYNAPSE server.
    """

    def __init__(self, config: EdgeConfig, textgrad_settings: Optional[TextGradSettings] = None) -> None:
        self.config = config
        self._history: List[KnowledgePackage] = []
        self._domain_cache: Dict[str, Dict[str, KnowledgeArtifact]] = {}
        self.textgrad_settings = textgrad_settings

    def _deduplicate_artifacts(self, artifacts: Iterable[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        """
        Remove straightforward duplicates using artifact signatures.

        Placeholder implementation that keeps the first artifact for each
        signature. Future versions will incorporate semantic similarity checks.
        """
        seen: Dict[str, KnowledgeArtifact] = {}
        for artifact in artifacts:
            if artifact.signature in seen:
                continue
            seen[artifact.signature] = artifact
        return list(seen.values())

    def _update_domain_cache(self, artifacts: Iterable[KnowledgeArtifact]) -> None:
        for artifact in artifacts:
            domain = artifact.metadata.get("domain") or artifact.metadata.get("tool") or "general"
            domain_cache = self._domain_cache.setdefault(domain, {})
            domain_cache[artifact.signature] = artifact

            # Trim cache to avoid unbounded growth
            if len(domain_cache) > 100:
                # Drop oldest inserted artifacts
                keys = list(domain_cache.keys())[: len(domain_cache) - 100]
                for key in keys:
                    domain_cache.pop(key, None)

    def merge_packages(self, packages: Iterable[KnowledgePackage]) -> Optional[KnowledgePackage]:
        """
        Combine a batch of client packages into a single consolidated package.
        """
        artifacts: List[KnowledgeArtifact] = []
        metadata: Dict[str, List[str]] = {"sources": []}

        for package in packages:
            artifacts.extend(package.artifacts)
            metadata["sources"].append(package.source_id)

        if not artifacts:
            return None

        if self.textgrad_settings and self.textgrad_settings.enabled:
            self._apply_textgrad_aggregation(artifacts)

        merged_package = KnowledgePackage(
            source_id=self.config.edge_id,
            artifacts=self._deduplicate_artifacts(artifacts),
            created_at=datetime.utcnow(),
            metadata=metadata,
        )

        if self.config.retain_history:
            self._history.append(merged_package)
        self._update_domain_cache(merged_package.artifacts)

        return merged_package

    @property
    def history(self) -> List[KnowledgePackage]:
        """Return the list of retained merged packages for auditing."""
        return list(self._history)

    def get_domain_view(self, domain: str) -> List[KnowledgeArtifact]:
        """
        Retrieve the cached artifacts for a given domain.
        """
        return list(self._domain_cache.get(domain, {}).values())

    def _apply_textgrad_aggregation(self, artifacts: List[KnowledgeArtifact]) -> None:
        """
        Aggregate TextGrad variables contributed by clients before deduplication.
        """
        if not self.textgrad_settings:
            return

        self.textgrad_settings.ensure_engines()
        method = (self.textgrad_settings.aggregate_method or "summarization").lower()
        eval_engine = self.textgrad_settings.evaluation_engine

        grouped: Dict[str, List[KnowledgeArtifact]] = {}
        for artifact in artifacts:
            variable = getattr(artifact, "textgrad_variable", None)
            if variable is None:
                continue
            group_key = (
                artifact.metadata.get("textgrad_group")
                or artifact.metadata.get("scenario")
                or artifact.metadata.get("tool")
                or artifact.signature
            )
            grouped.setdefault(group_key, []).append(artifact)

        for group_key, group_artifacts in grouped.items():
            variables = [art.textgrad_variable for art in group_artifacts if art.textgrad_variable is not None]
            if not variables:
                continue

            concatenated_variable = tg_aggregate(variables)
            concatenated_text = concatenated_variable.get_value()

            if method == "concat" or eval_engine is None:
                aggregated_text = concatenated_text
                aggregated_variable = concatenated_variable
            elif method in {"summarization", "sum_uid"}:
                template = SUMMARIZATION_TEMPLATE if method == "summarization" else UID_TEMPLATE
                instruction = template.substitute(prompt=concatenated_text)
                aggregated_text = eval_engine(instruction)
                aggregated_text += FORMATTING_INSTRUCTION
                aggregated_variable = Variable(
                    aggregated_text,
                    requires_grad=True,
                    role_description="aggregated system prompt produced by the edge aggregator",
                )
            else:
                raise ValueError(f"Unsupported TextGrad aggregation method: {method}")

            complexity_score = calculate_text_complexity(aggregated_text)
            primary_artifact = group_artifacts[0]
            primary_artifact.text = aggregated_text
            primary_artifact.textgrad_variable = aggregated_variable

            payload = dict(primary_artifact.structured_payload or {})
            textgrad_payload = dict(payload.get("textgrad", {}))
            textgrad_payload.update(
                {
                    "group": group_key,
                    "aggregate_method": method,
                    "complexity": complexity_score,
                }
            )
            payload["textgrad"] = textgrad_payload
            primary_artifact.structured_payload = payload
            primary_artifact.metadata["textgrad_complexity"] = complexity_score
            primary_artifact.metadata["textgrad_group"] = group_key
            primary_artifact.metadata["textgrad_aggregate_method"] = method

            for extra in group_artifacts[1:]:
                if extra.textgrad_variable:
                    extra.textgrad_variable.set_value(aggregated_text)
                extra.metadata["textgrad_complexity"] = complexity_score
                extra.metadata["textgrad_group"] = group_key
                extra.metadata["textgrad_aggregate_method"] = method
