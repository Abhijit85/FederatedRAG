from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from synapse.clients import ClientMetadata, SynapseClient, UnifiedQAClient
from synapse.config import ApiCredentials, FederationTopology, SynapseConfig
from synapse.edge import EdgeAggregator, EdgeConfig
from synapse.knowledge import KnowledgeArtifact, SynapseCompendium
from synapse.privacy.policies import PrivacyPolicy
from synapse.retrieval import RetrievalPlanner, RetrievalConfig
from synapse.server import AggregationMode, SynapseServer, ServerConfig


class SynapseRuntime:
    """
    High-level orchestrator for the SYNAPSE hierarchy.
    """

    def __init__(
        self,
        config: SynapseConfig,
        clients: Dict[str, SynapseClient],
        edges: Dict[str, EdgeAggregator],
        server: SynapseServer,
        retrieval_planner: Optional[RetrievalPlanner] = None,
    ) -> None:
        self.config = config
        self.clients = clients
        self.edges = edges
        self.server = server
        self.retrieval_planner = retrieval_planner or RetrievalPlanner()
        self._last_snapshot: Optional[SynapseCompendium] = None
        self._last_adapter_bundle = None
        # Enforce the governance checklist after components are wired.
        try:
            from synapse.checks import FederationChecklist

            FederationChecklist.ensure_runtime(self)
        except ImportError:
            pass

    @staticmethod
    def _dp_enabled_from_env(default: bool = True) -> bool:
        toggle = os.environ.get("SYNAPSE_ENABLE_DP")
        if toggle is None:
            return default
        return toggle.strip().lower() not in {"0", "false", "no", "off"}

    @classmethod
    def _resolve_dp_epsilon(cls, default: Optional[float]) -> Optional[float]:
        epsilon = default
        override = os.environ.get("SYNAPSE_DP_EPSILON")
        if override:
            try:
                epsilon = float(override)
            except ValueError:
                pass
        if not cls._dp_enabled_from_env(True):
            return None
        return epsilon

    @classmethod
    def build_local_runtime(
        cls,
        base_path: Path,
        credentials: ApiCredentials,
        client_count: Optional[int] = None,
    ) -> "SynapseRuntime":
        """
        Construct a runtime instance using repository data for MathQA and ScienceQA.
        """
        if client_count is None:
            env_client_count = os.environ.get("SYNAPSE_CLIENT_COUNT")
            if env_client_count:
                try:
                    client_count = max(1, int(env_client_count))
                except ValueError:
                    client_count = None
        if client_count is None:
            client_count = 2

        client_ids = [f"general-client-{i+1}" for i in range(client_count)]
        edge_clusters = {"edge-general": client_ids}

        topology = FederationTopology(
            client_ids=client_ids,
            edge_clusters=edge_clusters,
            central_server_id="synapse-central",
        )
        config = SynapseConfig(topology=topology, credentials=credentials)

        dp_epsilon: Optional[float]
        if config.enable_privacy:
            dp_epsilon = cls._resolve_dp_epsilon(1.0)
        else:
            dp_epsilon = None

        clients: Dict[str, SynapseClient] = {}

        for client_id in topology.client_ids:
            clients[client_id] = UnifiedQAClient(
                metadata=ClientMetadata(
                    client_id=client_id,
                    domain_tags=["math", "science", "multimodal"],
                    capabilities={"modality": "image+text"},
                ),
                math_compendium_path=base_path / "mathqa_tools_compendium.json",
                math_training_path=base_path / "train_new.json",
                science_compendium_path=base_path / "scienceqa_tools_compendium.json",
                science_dataset_path=base_path / "scienceqa_dataset.json",
                privacy_policy=PrivacyPolicy(dp_epsilon=dp_epsilon),
            )

        edges: Dict[str, EdgeAggregator] = {
            "edge-general": EdgeAggregator(EdgeConfig(edge_id="edge-general", domains=["math", "science"])),
        }

        agg_mode = AggregationMode.from_string(os.environ.get("SYNAPSE_SERVER_AGG_MODE"))
        secagg_provider = os.environ.get("SYNAPSE_SECAGG_PROVIDER", "simple")
        secagg_secret = os.environ.get("SYNAPSE_SECAGG_SECRET", "synapse-shared-secret")
        secagg_attestation = os.environ.get("SYNAPSE_SECAGG_ATTESTATION", "")
        server = SynapseServer(
            ServerConfig(
                server_id="synapse-central",
                expected_clients=client_count,
                aggregation_mode=agg_mode,
                secagg_provider=secagg_provider,
                secagg_secret=secagg_secret,
                secagg_attestation=secagg_attestation,
            )
        )
        retrieval = RetrievalPlanner(RetrievalConfig(max_artifacts=6))

        return cls(config=config, clients=clients, edges=edges, server=server, retrieval_planner=retrieval)

    def run_round(self) -> None:
        """
        Execute a full round of client -> edge -> server knowledge propagation.
        """
        for edge_id, client_ids in self.config.topology.edge_clusters.items():
            edge = self.edges[edge_id]
            packages = []
            for client_id in client_ids:
                client = self.clients[client_id]
                package = client.prepare_for_edge()
                if package.artifacts:
                    packages.append(package)
                if hasattr(client, "prepare_adapter_update"):
                    adapter_update = client.prepare_adapter_update()
                    self.server.ingest_adapter_update(adapter_update)
            if not packages:
                continue
            merged = edge.merge_packages(packages)
            if merged:
                self.server.ingest_from_edge(merged)

        adapter_bundle = self.server.flush_pending_updates()
        if adapter_bundle:
            for client in self.clients.values():
                apply_fn = getattr(client, "apply_global_bundle", None)
                if callable(apply_fn):
                    apply_fn(adapter_bundle)
        self._last_adapter_bundle = adapter_bundle
        self._last_snapshot = self.server.compendium

    def get_context_for_query(self, query: str, max_items: int = 5) -> List[KnowledgeArtifact]:
        """
        Retrieve the most relevant knowledge artifacts for a query using
        the latest global snapshot.
        """
        compendium = self.server.compendium
        artifacts = compendium.build_snapshot().artifacts
        planner = self.retrieval_planner or RetrievalPlanner(RetrievalConfig(max_artifacts=max_items))
        planner.config.max_artifacts = max_items
        return planner.select(query, artifacts)

    def export_snapshot(self, path: Path) -> None:
        """
        Persist the current compendium snapshot to disk.
        """
        snapshot = self.server.distribute_snapshot()
        payload = {
            "metadata": snapshot.metadata,
            "artifacts": [
                {
                    "signature": artifact.signature,
                    "text": artifact.text,
                    "structured_payload": artifact.structured_payload,
                    "metadata": artifact.metadata,
                }
                for artifact in snapshot.artifacts
            ],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def summarize_round(self) -> Dict[str, object]:
        """
        Provide a lightweight summary of the current federation state.
        """
        snapshot = self.server.distribute_snapshot()
        summary = {
            "artifact_count": len(snapshot.artifacts),
            "version_history": self.server.version_history,
        }
        if self._last_adapter_bundle:
            summary["adapter_version"] = self._last_adapter_bundle.version
            summary["privacy_budget"] = self._last_adapter_bundle.privacy_budget_remaining.to_dict()
        try:
            trust_scores = self.server.trust_scores
        except AttributeError:
            trust_scores = {}
        if trust_scores:
            summary["trust_scores"] = trust_scores
        aggregator_facade = getattr(self.server, "aggregator_facade", None)
        if aggregator_facade:
            if aggregator_facade.poisoning_flags:
                summary["poisoning_flags"] = aggregator_facade.poisoning_flags
            if aggregator_facade.adapter_norm_zscores:
                summary["adapter_norm_zscores"] = aggregator_facade.adapter_norm_zscores
            if aggregator_facade.quarantine_queue:
                summary["quarantine_queue"] = aggregator_facade.quarantine_queue
            summary["aggregation_mode"] = getattr(self.server, "aggregation_mode", AggregationMode.ROBUST).value
            attestation = getattr(self.server, "secagg_attestation", lambda: {})()
            if attestation:
                summary["secagg_attestation"] = attestation
        return summary

    def plan_context_for_tool(self, query: str, tool_name: str) -> List[str]:
        """
        Generate textual snippets to augment a downstream tool prompt.
        """
        relevant_artifacts = self.get_context_for_query(query, max_items=5)
        snippets: List[str] = []
        for artifact in relevant_artifacts:
            if artifact.metadata.get("tool") and artifact.metadata["tool"] != tool_name:
                continue
            snippets.append(artifact.text)
        return snippets
