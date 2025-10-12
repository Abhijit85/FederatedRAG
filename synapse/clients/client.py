from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set

from synapse.knowledge.compendium import KnowledgeArtifact, KnowledgePackage
from synapse.privacy.policies import PrivacyPolicy


@dataclass
class ClientMetadata:
    """Metadata describing the client and its local environment."""

    client_id: str
    domain_tags: List[str] = field(default_factory=list)
    capabilities: Dict[str, str] = field(default_factory=dict)


class SynapseClient:
    """
    Base implementation for a SYNAPSE client.

    Responsibilities:
      * Harvest local knowledge into KnowledgeArtifact structures.
      * Apply local filters (novelty, sensitivity) before sharing.
      * Communicate filtered packages to an edge aggregator.
    """

    def __init__(self, metadata: ClientMetadata) -> None:
        self.metadata = metadata
        self._last_package: Optional[KnowledgePackage] = None
        self._shared_signatures: Set[str] = set()
        self.privacy_policy = PrivacyPolicy()

    def collect_local_artifacts(self) -> List[KnowledgeArtifact]:
        """
        Gather raw artifacts from the local environment.

        Subclasses should override to pull data from logs, transcripts,
        or other task-specific sources.
        """
        return []

    def build_knowledge_package(self) -> KnowledgePackage:
        """
        Construct a KnowledgePackage ready for transmission to the edge tier.
        """
        artifacts = self.collect_local_artifacts()

        # Ensure all artifacts have unique signatures; derive one if missing.
        normalized: List[KnowledgeArtifact] = []
        for artifact in artifacts:
            signature = artifact.signature or self._derive_signature(artifact)
            normalized.append(
                KnowledgeArtifact(
                    signature=signature,
                    text=artifact.text,
                    structured_payload=artifact.structured_payload,
                    metadata=artifact.metadata,
                )
            )

        package = KnowledgePackage(
            source_id=self.metadata.client_id,
            artifacts=normalized,
            created_at=datetime.utcnow(),
            metadata={
                "domain_tags": self.metadata.domain_tags,
                "capabilities": self.metadata.capabilities,
            },
        )
        self._last_package = package
        return package

    def _derive_signature(self, artifact: KnowledgeArtifact) -> str:
        hasher = hashlib.sha256()
        hasher.update(artifact.text.encode("utf-8"))
        if artifact.structured_payload:
            hasher.update(str(sorted(artifact.structured_payload.items())).encode("utf-8"))
        return hasher.hexdigest()

    def _filter_novel_artifacts(self, artifacts: Iterable[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        """
        Keep only artifacts that have not yet been shared by this client.
        """
        novel: List[KnowledgeArtifact] = []
        for artifact in artifacts:
            if artifact.signature in self._shared_signatures:
                continue
            novel.append(artifact)
        return novel

    def _score_artifact(self, artifact: KnowledgeArtifact) -> float:
        """
        Heuristic novelty score combining text length and metadata hints.
        """
        length_score = len(artifact.text.split()) / 50.0
        metadata_bonus = 0.0
        difficulty = artifact.metadata.get("difficulty")
        if isinstance(difficulty, (int, float)):
            metadata_bonus += difficulty / 5.0
        elif isinstance(difficulty, str) and difficulty.lower() in {"hard", "advanced"}:
            metadata_bonus += 1.0
        return length_score + metadata_bonus

    def _prioritize_artifacts(self, artifacts: Iterable[KnowledgeArtifact], budget: int = 10) -> List[KnowledgeArtifact]:
        scored = sorted(
            artifacts,
            key=lambda art: self._score_artifact(art),
            reverse=True,
        )
        return scored[:budget]

    def select_for_sharing(self, package: KnowledgePackage) -> KnowledgePackage:
        """
        Apply local filtering to remove redundant or sensitive artifacts.

        The default implementation keeps novel artifacts, prioritizes them,
        and applies a privacy policy before sending.
        """
        novel = self._filter_novel_artifacts(package.artifacts)
        prioritized = self._prioritize_artifacts(novel)
        sanitized = self.privacy_policy.enforce(prioritized)

        for artifact in sanitized:
            self._shared_signatures.add(artifact.signature)

        return KnowledgePackage(
            source_id=package.source_id,
            artifacts=sanitized,
            created_at=package.created_at,
            metadata=package.metadata,
        )

    def prepare_for_edge(self) -> KnowledgePackage:
        """
        Public entry point used by orchestrators to request an update.
        """
        raw_package = self.build_knowledge_package()
        return self.select_for_sharing(raw_package)
