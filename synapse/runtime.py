from __future__ import annotations

import json
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from synapse.clients import ClientMetadata, MathQAClient, ScienceQAClient, SynapseClient
from synapse.config import ApiCredentials, FederationTopology, SynapseConfig
from synapse.edge import EdgeAggregator, EdgeConfig
from synapse.knowledge import KnowledgeArtifact, KnowledgePackage, SynapseCompendium
from synapse.network import NetworkSimulator
from synapse.privacy.policies import PrivacyPolicy
from synapse.retrieval import HashedVectorStore, RetrievalPlanner, RetrievalConfig
from synapse.server import SynapseServer, ServerConfig


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
        network: Optional[NetworkSimulator] = None,
    ) -> None:
        self.config = config
        self.clients = clients
        self.edges = edges
        self.server = server
        self.retrieval_planner = retrieval_planner or RetrievalPlanner()
        self.network = network
        self._last_snapshot: Optional[SynapseCompendium] = None
        self._refresh_retrieval_snapshot()

    @classmethod
    def build_local_runtime(
        cls,
        base_path: Path,
        credentials: ApiCredentials,
    ) -> "SynapseRuntime":
        """
        Construct a runtime instance using repository data for MathQA and ScienceQA.
        """
        topology = FederationTopology(
            client_ids=["mathqa-client", "scienceqa-client"],
            edge_clusters={
                "edge-math": ["mathqa-client"],
                "edge-science": ["scienceqa-client"],
            },
            central_server_id="synapse-central",
        )
        config = SynapseConfig(topology=topology, credentials=credentials)

        clients: Dict[str, SynapseClient] = {}

        privacy_kwargs = {
            "redact_sensitive_metadata": config.privacy.redact_sensitive_metadata,
            "drop_pii_text": config.privacy.drop_pii_text,
            "dp_epsilon": config.privacy.dp_epsilon if config.enable_privacy else None,
            "encryption_secret": credentials.synapse_secret,
        }

        math_client = MathQAClient(
            metadata=ClientMetadata(
                client_id="mathqa-client",
                domain_tags=["math", "numerical_reasoning"],
                capabilities={"modality": "text"},
            ),
            compendium_path=base_path / "mathqa_tools_compendium.json",
            training_data_path=base_path / "train_new.json",
            privacy_policy=PrivacyPolicy(**privacy_kwargs),
        )
        clients["mathqa-client"] = math_client

        science_client = ScienceQAClient(
            metadata=ClientMetadata(
                client_id="scienceqa-client",
                domain_tags=["science", "multimodal"],
                capabilities={"modality": "image+text"},
            ),
            compendium_path=base_path / "scienceqa_tools_compendium.json",
            dataset_path=base_path / "scienceqa_dataset.json",
            privacy_policy=PrivacyPolicy(**privacy_kwargs),
        )
        clients["scienceqa-client"] = science_client

        edges: Dict[str, EdgeAggregator] = {
            "edge-math": EdgeAggregator(
                EdgeConfig(edge_id="edge-math", domains=["math"]),
                privacy_policy=PrivacyPolicy(
                    redact_sensitive_metadata=False,
                    drop_pii_text=False,
                    dp_epsilon=None,
                    encryption_secret=credentials.synapse_secret,
                ),
            ),
            "edge-science": EdgeAggregator(
                EdgeConfig(edge_id="edge-science", domains=["science"]),
                privacy_policy=PrivacyPolicy(
                    redact_sensitive_metadata=False,
                    drop_pii_text=False,
                    dp_epsilon=None,
                    encryption_secret=credentials.synapse_secret,
                ),
            ),
        }

        server = SynapseServer(ServerConfig(server_id="synapse-central"))
        retrieval = RetrievalPlanner(RetrievalConfig(max_artifacts=6), vector_store=HashedVectorStore())

        network = NetworkSimulator(secret=credentials.synapse_secret or "synapse-default-secret")
        return cls(
            config=config,
            clients=clients,
            edges=edges,
            server=server,
            retrieval_planner=retrieval,
            network=network,
        )

    def _run_round_sync(self) -> None:
        """Execute a synchronous round of knowledge propagation."""
        for edge_id, client_ids in self.config.topology.edge_clusters.items():
            edge = self.edges[edge_id]
            packages = []
            for client_id in client_ids:
                client = self.clients[client_id]
                package = client.prepare_for_edge()
                if package.artifacts:
                    packages.append(package)
            if not packages:
                continue
            merged = edge.merge_packages(packages)
            if merged:
                self.server.ingest_from_edge(merged)

        self._last_snapshot = self.server.compendium
        self._refresh_retrieval_snapshot()

    async def run_round_async(self) -> None:
        """Execute an asynchronous round with simulated network latency."""
        edge_batches: Dict[str, List[KnowledgePackage]] = defaultdict(list)

        async def process_client(edge_id: str, client_id: str) -> None:
            client = self.clients[client_id]
            package = client.prepare_for_edge()
            if not package.artifacts:
                return
            if self.network and self.config.network.enable_async:
                package = await self.network.transmit_client_to_edge(package)
            edge_batches[edge_id].append(package)

        tasks = []
        for edge_id, client_ids in self.config.topology.edge_clusters.items():
            for client_id in client_ids:
                tasks.append(process_client(edge_id, client_id))

        if tasks:
            await asyncio.gather(*tasks)

        async def process_edge(edge_id: str, packages: List[KnowledgePackage]) -> None:
            if not packages:
                return
            edge = self.edges[edge_id]
            merged = edge.merge_packages(packages)
            if not merged:
                return
            if self.network and self.config.network.enable_async:
                merged = await self.network.transmit_edge_to_server(merged)
            self.server.ingest_from_edge(merged)

        if edge_batches:
            await asyncio.gather(
                *(process_edge(edge_id, packages) for edge_id, packages in edge_batches.items())
            )

        self._last_snapshot = self.server.compendium
        self._refresh_retrieval_snapshot()

    def run_round(self) -> None:
        if self.config.network.enable_async:
            try:
                asyncio.run(self.run_round_async())
            except RuntimeError:
                loop = asyncio.get_event_loop_policy().new_event_loop()
                try:
                    loop.run_until_complete(self.run_round_async())
                finally:
                    loop.close()
        else:
            self._run_round_sync()

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

    def _refresh_retrieval_snapshot(self) -> None:
        snapshot = self.server.distribute_snapshot()
        self.retrieval_planner.update_artifacts(snapshot.artifacts)

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
        return {
            "artifact_count": len(snapshot.artifacts),
            "version_history": self.server.version_history,
        }

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
