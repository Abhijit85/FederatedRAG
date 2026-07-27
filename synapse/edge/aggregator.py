from __future__ import annotations

import math
import re
from collections import Counter
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
    similarity_threshold: float = 0.85


class EdgeAggregator:
    """
    Consolidates knowledge packages from a cluster of clients before
    forwarding them to the central SYNAPSE server.
    """

    def __init__(self, config: EdgeConfig, textgrad_settings: Optional[TextGradSettings] = None) -> None:
        self.config = config
        self._history: List[KnowledgePackage] = []
        self._domain_cache: Dict[str, Dict[str, KnowledgeArtifact]] = {}
        self._conflict_log: List[Dict[str, object]] = []
        self.textgrad_settings = textgrad_settings

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def _similarity_view(self, artifact: KnowledgeArtifact) -> str:
        payload = artifact.structured_payload or {}
        if isinstance(payload, dict) and payload.get("type") == "usage_scenario":
            parts: List[str] = []
            scenario = artifact.metadata.get("scenario")
            if isinstance(scenario, str) and scenario.strip():
                parts.append(scenario.strip())
            scenario_context = payload.get("scenario_context")
            if isinstance(scenario_context, str) and scenario_context.strip():
                parts.append(scenario_context.strip())
            scenario_notes = payload.get("scenario_notes")
            if isinstance(scenario_notes, list) and scenario_notes:
                parts.extend(str(item).strip() for item in scenario_notes[:2] if str(item).strip())
            if parts:
                return "\n".join(parts)
        return artifact.text

    def _text_vector(self, artifact: KnowledgeArtifact) -> Counter[str]:
        return Counter(self._tokenize(self._similarity_view(artifact)))

    def _cosine_similarity(self, left: KnowledgeArtifact, right: KnowledgeArtifact) -> float:
        left_vec = self._text_vector(left)
        right_vec = self._text_vector(right)
        if not left_vec or not right_vec:
            return 0.0
        overlap = set(left_vec) & set(right_vec)
        numerator = sum(left_vec[token] * right_vec[token] for token in overlap)
        left_norm = math.sqrt(sum(value * value for value in left_vec.values()))
        right_norm = math.sqrt(sum(value * value for value in right_vec.values()))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _artifact_domain(self, artifact: KnowledgeArtifact) -> str:
        return str(artifact.metadata.get("domain") or artifact.metadata.get("tool") or "general")

    def _artifact_payload_repr(self, artifact: KnowledgeArtifact) -> str:
        payload = artifact.structured_payload
        if payload is None:
            return ""
        return repr(payload)

    def _representative_artifact(self, cluster: List[KnowledgeArtifact]) -> KnowledgeArtifact:
        def score(artifact: KnowledgeArtifact) -> tuple[int, int, int]:
            payload_width = len(self._artifact_payload_repr(artifact))
            text_width = len((artifact.text or "").strip())
            meta_width = len(artifact.metadata)
            return (text_width, payload_width, meta_width)

        return max(cluster, key=score)

    def _record_cluster_conflict(self, cluster: List[KnowledgeArtifact], similarity_scores: List[float]) -> None:
        signatures = [artifact.signature for artifact in cluster]
        payloads = {self._artifact_payload_repr(artifact) for artifact in cluster}
        texts = {artifact.text.strip() for artifact in cluster if artifact.text.strip()}
        distinct_sources = sorted(
            {
                str(artifact.metadata.get("source_id"))
                for artifact in cluster
                if artifact.metadata.get("source_id") is not None
            }
        )
        self._conflict_log.append(
            {
                "type": "cosine_cluster_conflict",
                "domain": self._artifact_domain(cluster[0]),
                "cluster_size": len(cluster),
                "signatures": signatures,
                "similarity_threshold": self.config.similarity_threshold,
                "max_cosine_similarity": max(similarity_scores) if similarity_scores else 1.0,
                "min_cosine_similarity": min(similarity_scores) if similarity_scores else 1.0,
                "distinct_text_count": len(texts),
                "distinct_payload_count": len(payloads),
                "source_ids": distinct_sources,
            }
        )

    def _cluster_artifacts(self, artifacts: Iterable[KnowledgeArtifact]) -> tuple[List[KnowledgeArtifact], List[Dict[str, object]]]:
        threshold = self.config.similarity_threshold
        clustered: List[List[KnowledgeArtifact]] = []
        cluster_scores: List[List[float]] = []
        artifacts_list = list(artifacts)

        for artifact in artifacts_list:
            assigned = False
            for idx, cluster in enumerate(clustered):
                representative = self._representative_artifact(cluster)
                if self._artifact_domain(representative) != self._artifact_domain(artifact):
                    continue
                similarity = self._cosine_similarity(representative, artifact)
                if similarity >= threshold:
                    cluster.append(artifact)
                    cluster_scores[idx].append(similarity)
                    assigned = True
                    break
            if not assigned:
                clustered.append([artifact])
                cluster_scores.append([])

        merged_artifacts: List[KnowledgeArtifact] = []
        cluster_metadata: List[Dict[str, object]] = []
        for cluster, similarities in zip(clustered, cluster_scores):
            representative = self._representative_artifact(cluster)
            cluster_info = {
                "representative_signature": representative.signature,
                "domain": self._artifact_domain(representative),
                "cluster_size": len(cluster),
                "member_signatures": [artifact.signature for artifact in cluster],
                "similarity_threshold": threshold,
                "max_cosine_similarity": max(similarities) if similarities else 1.0,
            }
            cluster_metadata.append(cluster_info)

            merged_artifact = KnowledgeArtifact(
                signature=representative.signature,
                text=representative.text,
                structured_payload=representative.structured_payload,
                metadata=dict(representative.metadata or {}),
                textgrad_variable=representative.textgrad_variable,
            )
            merged_artifact.metadata["cluster_size"] = len(cluster)
            merged_artifact.metadata["cluster_member_signatures"] = cluster_info["member_signatures"]
            merged_artifact.metadata["cluster_similarity_threshold"] = threshold
            merged_artifact.metadata["cluster_max_cosine_similarity"] = cluster_info["max_cosine_similarity"]
            merged_artifacts.append(merged_artifact)

            if len(cluster) > 1:
                self._record_cluster_conflict(cluster, similarities)

        return merged_artifacts, cluster_metadata

    def _deduplicate_artifacts(self, artifacts: Iterable[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        merged_artifacts, _ = self._cluster_artifacts(artifacts)
        return merged_artifacts

    def _update_domain_cache(self, artifacts: Iterable[KnowledgeArtifact]) -> None:
        for artifact in artifacts:
            domain = artifact.metadata.get("domain") or artifact.metadata.get("tool") or "general"
            domain_cache = self._domain_cache.setdefault(domain, {})
            domain_cache[artifact.signature] = artifact

            if len(domain_cache) > 100:
                keys = list(domain_cache.keys())[: len(domain_cache) - 100]
                for key in keys:
                    domain_cache.pop(key, None)

    def merge_packages(self, packages: Iterable[KnowledgePackage]) -> Optional[KnowledgePackage]:
        """
        Combine a batch of client packages into a single consolidated package.
        """
        artifacts: List[KnowledgeArtifact] = []
        metadata: Dict[str, object] = {"sources": []}

        for package in packages:
            source_id = package.source_id
            metadata["sources"].append(source_id)
            for artifact in package.artifacts:
                annotated_metadata = dict(artifact.metadata or {})
                annotated_metadata.setdefault("source_id", source_id)
                artifacts.append(
                    KnowledgeArtifact(
                        signature=artifact.signature,
                        text=artifact.text,
                        structured_payload=artifact.structured_payload,
                        metadata=annotated_metadata,
                        textgrad_variable=artifact.textgrad_variable,
                    )
                )

        if not artifacts:
            return None

        if self.textgrad_settings and self.textgrad_settings.enabled:
            self._apply_textgrad_aggregation(artifacts)

        merged_artifacts, cluster_metadata = self._cluster_artifacts(artifacts)
        conflict_slice = self._conflict_log[-sum(1 for item in cluster_metadata if item["cluster_size"] > 1) :]
        metadata["similarity_threshold"] = self.config.similarity_threshold
        metadata["cluster_count"] = len(cluster_metadata)
        metadata["clusters"] = cluster_metadata
        metadata["conflict_log"] = conflict_slice

        merged_package = KnowledgePackage(
            source_id=self.config.edge_id,
            artifacts=merged_artifacts,
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

    @property
    def conflict_log(self) -> List[Dict[str, object]]:
        """Return cosine-cluster conflicts recorded across merge operations."""
        return list(self._conflict_log)

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
