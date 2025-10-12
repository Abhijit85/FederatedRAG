from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set

from synapse.knowledge.compendium import KnowledgeArtifact, KnowledgePackage
from synapse.privacy.policies import PrivacyPolicy


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

    def __init__(self, config: EdgeConfig, privacy_policy: Optional[PrivacyPolicy] = None) -> None:
        self.config = config
        self._history: List[KnowledgePackage] = []
        self._domain_cache: Dict[str, Dict[str, KnowledgeArtifact]] = {}
        self.privacy_policy = privacy_policy or PrivacyPolicy()

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
            decrypted = self.privacy_policy.decrypt_artifacts(package.artifacts)
            artifacts.extend(decrypted)
            metadata["sources"].append(package.source_id)

        if not artifacts:
            return None

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
