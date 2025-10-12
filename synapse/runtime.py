from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from synapse.clients import ClientMetadata, MathQAClient, ScienceQAClient, SynapseClient
from synapse.config import ApiCredentials, FederationTopology, SynapseConfig
from synapse.edge import EdgeAggregator, EdgeConfig
from synapse.knowledge import KnowledgeArtifact, SynapseCompendium
from synapse.retrieval import RetrievalPlanner, RetrievalConfig
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
    ) -> None:
        self.config = config
        self.clients = clients
        self.edges = edges
        self.server = server
        self.retrieval_planner = retrieval_planner or RetrievalPlanner()
        self._last_snapshot: Optional[SynapseCompendium] = None

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

        math_client = MathQAClient(
            metadata=ClientMetadata(
                client_id="mathqa-client",
                domain_tags=["math", "numerical_reasoning"],
                capabilities={"modality": "text"},
            ),
            compendium_path=base_path / "mathqa_tools_compendium.json",
            training_data_path=base_path / "train_new.json",
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
        )
        clients["scienceqa-client"] = science_client

        edges: Dict[str, EdgeAggregator] = {
            "edge-math": EdgeAggregator(EdgeConfig(edge_id="edge-math", domains=["math"])),
            "edge-science": EdgeAggregator(EdgeConfig(edge_id="edge-science", domains=["science"])),
        }

        server = SynapseServer(ServerConfig(server_id="synapse-central"))
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
            if not packages:
                continue
            merged = edge.merge_packages(packages)
            if merged:
                self.server.ingest_from_edge(merged)

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
