import unittest

from synapse.clients.client import ClientMetadata, SynapseClient
from synapse.edge.aggregator import EdgeAggregator, EdgeConfig
from synapse.knowledge.compendium import KnowledgeArtifact
from synapse.privacy.policies import PrivacyPolicy
from synapse.retrieval import RetrievalPlanner, RetrievalConfig, HashedVectorStore
from synapse.server.orchestrator import SynapseServer
from synapse.config import ApiCredentials, FederationTopology, SynapseConfig
from synapse.network import NetworkSimulator
from synapse.runtime import SynapseRuntime


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


class AsyncDemoClient(SynapseClient):
    """Client used to validate asynchronous SYNAPSE rounds."""

    def collect_local_artifacts(self):
        return [
            KnowledgeArtifact(
                signature="async-artifact",
                text="Async knowledge about experimental physics.",
                structured_payload={"type": "demo", "value": 3.14},
                metadata={"tool": "scienceqa", "value": 1.0},
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

    def test_privacy_policy_encrypts_and_decrypts(self):
        policy = PrivacyPolicy(encryption_secret="test-secret")
        artifact = KnowledgeArtifact(signature="sig", text="Sensitive text", structured_payload=None, metadata={})
        encrypted = policy.encrypt_artifacts([artifact])[0]
        self.assertNotEqual(encrypted.text, artifact.text)
        decrypted = policy.decrypt_artifacts([encrypted])[0]
        self.assertEqual(decrypted.text, artifact.text)

    def test_privacy_policy_dp_noise(self):
        policy = PrivacyPolicy(dp_epsilon=0.5)
        artifact = KnowledgeArtifact(signature="sig", text="", structured_payload={"score": 10.0}, metadata={"value": 5.0})
        noisy = policy.apply_dp_noise([artifact])[0]
        self.assertNotEqual(noisy.metadata["value"], 5.0)

    def test_vector_store_with_image_embedding(self):
        planner = RetrievalPlanner(RetrievalConfig(max_artifacts=1), vector_store=HashedVectorStore(dim=32))
        artifacts = [
            KnowledgeArtifact(
                signature="image-artifact",
                text="Analyze diagram data.",
                structured_payload={"image_embedding": [1.0] * 10},
                metadata={"tool": "scienceqa"},
            ),
            KnowledgeArtifact(
                signature="text-artifact",
                text="Pure textual hint about chemistry experiments.",
                structured_payload={},
                metadata={"tool": "scienceqa"},
            ),
        ]
        selected = planner.select("diagram analysis", artifacts)
        self.assertEqual(selected[0].signature, "image-artifact")

    def test_async_runtime_round(self):
        credentials = ApiCredentials(lambda_api_key="", jina_api_key="", mongo_uri="mongodb://localhost:27017", synapse_secret="unit-secret")
        topology = FederationTopology(
            client_ids=["client-async"],
            edge_clusters={"edge-async": ["client-async"]},
            central_server_id="synapse-central",
        )
        config = SynapseConfig(topology=topology, credentials=credentials)

        client_privacy = PrivacyPolicy(dp_epsilon=1.0, encryption_secret="unit-secret")
        clients = {"client-async": AsyncDemoClient(ClientMetadata(client_id="client-async"), privacy_policy=client_privacy)}

        edge_privacy = PrivacyPolicy(redact_sensitive_metadata=False, drop_pii_text=False, dp_epsilon=None, encryption_secret="unit-secret")
        edges = {
            "edge-async": EdgeAggregator(EdgeConfig(edge_id="edge-async"), privacy_policy=edge_privacy)
        }

        server = SynapseServer()
        runtime = SynapseRuntime(
            config=config,
            clients=clients,
            edges=edges,
            server=server,
            retrieval_planner=RetrievalPlanner(RetrievalConfig(max_artifacts=5), vector_store=HashedVectorStore(dim=32)),
            network=NetworkSimulator(secret="unit-secret"),
        )

        runtime.run_round()
        snapshot = runtime.server.distribute_snapshot()
        self.assertGreater(len(snapshot.artifacts), 0)


if __name__ == "__main__":
    unittest.main()
