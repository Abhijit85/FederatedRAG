import unittest

from synapse.clients.client import ClientMetadata, SynapseClient
from synapse.edge.aggregator import EdgeAggregator, EdgeConfig
from synapse.knowledge.compendium import KnowledgeArtifact
from synapse.server.orchestrator import SynapseServer
from synapse.retrieval import RetrievalPlanner, RetrievalConfig


class DemoClient(SynapseClient):
    """Minimal client producing a single deterministic artifact."""

    def collect_local_artifacts(self):
        return [
            KnowledgeArtifact(
                signature="demo-artifact",
                text="Example explanation sourced from client logs.",
                structured_payload={"type": "demo", "value": 42},
            )
        ]


class SynapseRoundtripTest(unittest.TestCase):
    def test_client_edge_server_roundtrip(self):
        metadata = ClientMetadata(client_id="client-1", domain_tags=["demo"])
        client = DemoClient(metadata)

        package = client.prepare_for_edge()
        self.assertEqual(package.source_id, "client-1")
        self.assertEqual(len(package.artifacts), 1)

        edge = EdgeAggregator(EdgeConfig(edge_id="edge-1"))
        merged = edge.merge_packages([package])

        self.assertIsNotNone(merged)
        assert merged is not None  # help type checkers
        self.assertEqual(merged.source_id, "edge-1")
        self.assertEqual(len(merged.artifacts), 1)

        server = SynapseServer()
        server.ingest_from_edge(merged)

        snapshot = server.distribute_snapshot()
        self.assertEqual(snapshot.source_id, "synapse-central")
        self.assertEqual(len(snapshot.artifacts), 1)
        self.assertEqual(server.version_history[-1]["edge_id"], "edge-1")

    def test_retrieval_prioritizes_relevant_artifacts(self):
        artifacts = [
            KnowledgeArtifact(
                signature="a1",
                text="Use a calculator to solve ratio problems involving fractions.",
                structured_payload={"type": "usage_scenario", "skills": ["ratios"]},
                metadata={"tool": "mathqa"},
            ),
            KnowledgeArtifact(
                signature="a2",
                text="Identify parts of a plant cell from a labeled diagram.",
                structured_payload={"type": "usage_scenario", "skills": ["biology"]},
                metadata={"tool": "scienceqa"},
            ),
        ]

        planner = RetrievalPlanner(RetrievalConfig(max_artifacts=1))
        selected = planner.select("How do I calculate the ratio of two numbers?", artifacts)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].signature, "a1")


if __name__ == "__main__":
    unittest.main()
