import os
import unittest
from unittest.mock import patch

from synapse.clients.client import ClientMetadata, SynapseClient
from synapse.edge.aggregator import EdgeAggregator, EdgeConfig
from synapse.knowledge.compendium import KnowledgeArtifact, KnowledgePackage
from synapse.server.orchestrator import SynapseServer
from synapse.retrieval import RetrievalPlanner, RetrievalConfig
from synapse.runtime import SynapseRuntime
from synapse.privacy.policies import PrivacyPolicy


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


    def test_paperlike_retrieval_penalizes_training_examples(self):
        artifacts = [
            KnowledgeArtifact(
                signature="scenario",
                text="role: structured system prompt\nscenario: Percentage and Proportion Solver\nscenario_context: Handles percentages and proportions.",
                structured_payload={"type": "usage_scenario", "scenario_context": "Handles percentages and proportions."},
                metadata={"tool": "mathqa", "scenario": "Percentage and Proportion Solver"},
            ),
            KnowledgeArtifact(
                signature="example",
                text="Problem: percentage percentage percentage of 360 equals 108. Solution: compute the percent.",
                structured_payload={"type": "training_example"},
                metadata={"tool": "mathqa"},
            ),
        ]

        planner = RetrievalPlanner(RetrievalConfig(max_artifacts=1, retrieval_profile="paperlike"))
        selected = planner.select("What percent of 360 is 108?", artifacts)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].signature, "scenario")

    def test_untyped_structured_mode_strips_type_field(self):
        with patch.dict(os.environ, {"SYNAPSE_STRUCTURED_PAYLOAD_MODE": "untyped"}, clear=False):
            metadata = ClientMetadata(client_id="client-1", domain_tags=["demo"])
            client = DemoClient(metadata)

            package = client.prepare_for_edge()

        self.assertEqual(len(package.artifacts), 1)
        self.assertEqual(package.artifacts[0].structured_payload, {"value": 42})


    def test_merge_up_structured_mode_merges_scenario_and_precautions(self):
        class ScenarioClient(SynapseClient):
            def collect_local_artifacts(self):
                return [
                    KnowledgeArtifact(
                        signature="scenario-artifact",
                        text="demo",
                        structured_payload={
                            "type": "usage_scenario",
                            "scenario_context": "Solve ratio problems.",
                            "precautions": ["Watch unit consistency.", "Validate formulas."],
                            "annex_entities": ["Calculator"],
                        },
                    )
                ]

        with patch.dict(os.environ, {"SYNAPSE_STRUCTURED_PAYLOAD_MODE": "merge_up"}, clear=False):
            metadata = ClientMetadata(client_id="client-1", domain_tags=["demo"])
            client = ScenarioClient(metadata)
            package = client.prepare_for_edge()

        payload = package.artifacts[0].structured_payload
        self.assertEqual(payload["type"], "usage_scenario")
        self.assertEqual(payload["scenario_notes"], ["Solve ratio problems.", "Watch unit consistency.", "Validate formulas."])
        self.assertNotIn("scenario_context", payload)
        self.assertNotIn("precautions", payload)


    def test_drop_annex_structured_mode_removes_annex_fields(self):
        class AnnexClient(SynapseClient):
            def collect_local_artifacts(self):
                return [
                    KnowledgeArtifact(
                        signature="annex-artifact",
                        text="demo",
                        structured_payload={
                            "type": "usage_scenario",
                            "scenario_context": "Solve ratio problems.",
                            "annex_entities": ["Calculator"],
                            "annex_relations": [{"source": "Calculator", "link": "computes", "target": "Answer"}],
                            "annex_summary": "entities: Calculator",
                        },
                    )
                ]

        with patch.dict(os.environ, {"SYNAPSE_STRUCTURED_PAYLOAD_MODE": "drop_annex"}, clear=False):
            metadata = ClientMetadata(client_id="client-1", domain_tags=["demo"])
            client = AnnexClient(metadata)
            package = client.prepare_for_edge()

        payload = package.artifacts[0].structured_payload
        self.assertEqual(payload["scenario_context"], "Solve ratio problems.")
        self.assertNotIn("annex_entities", payload)
        self.assertNotIn("annex_relations", payload)
        self.assertNotIn("annex_summary", payload)


    def test_disabled_structured_mode_removes_payload(self):
        with patch.dict(os.environ, {"SYNAPSE_STRUCTURED_PAYLOAD_MODE": "off"}, clear=False):
            metadata = ClientMetadata(client_id="client-1", domain_tags=["demo"])
            client = DemoClient(metadata)

            package = client.prepare_for_edge()

        self.assertEqual(len(package.artifacts), 1)
        self.assertIsNone(package.artifacts[0].structured_payload)


    def test_dp_env_toggle(self):
        try:
            os.environ["SYNAPSE_ENABLE_DP"] = "0"
            self.assertIsNone(SynapseRuntime._resolve_dp_epsilon(1.0))

            os.environ["SYNAPSE_ENABLE_DP"] = "1"
            os.environ["SYNAPSE_DP_EPSILON"] = "0.25"
            self.assertAlmostEqual(SynapseRuntime._resolve_dp_epsilon(1.0), 0.25)
        finally:
            os.environ.pop("SYNAPSE_ENABLE_DP", None)
            os.environ.pop("SYNAPSE_DP_EPSILON", None)


    def test_dp_clips_numeric_fields_before_noise(self):
        artifact = KnowledgeArtifact(
            signature="dp-artifact",
            text="safe text",
            structured_payload={"value": 9.5, "weights": [5.0, -7.0], "label": "keep"},
            metadata={"score": 4.25, "count": -3.5, "note": "keep"},
        )
        policy = PrivacyPolicy(dp_epsilon=1.0, dp_clip_abs=1.0, adaptive_text_noise=False)

        with patch("synapse.privacy.policies._laplace", return_value=0.0):
            privatized = policy.enforce([artifact])

        self.assertEqual(len(privatized), 1)
        result = privatized[0]
        self.assertEqual(result.metadata["score"], 1.0)
        self.assertEqual(result.metadata["count"], -1.0)
        self.assertEqual(result.metadata["note"], "keep")
        self.assertEqual(result.structured_payload["value"], 1.0)
        self.assertEqual(result.structured_payload["weights"], [1.0, -1.0])
        self.assertEqual(result.structured_payload["label"], "keep")


    def test_dp_noise_scale_uses_clip_bound(self):
        artifact = KnowledgeArtifact(
            signature="dp-scale",
            text="safe text",
            structured_payload={"value": 7.5, "weights": [3.5, -9.0]},
            metadata={"score": 12.0, "count": -4.0},
        )
        policy = PrivacyPolicy(dp_epsilon=0.5, dp_clip_abs=2.0, adaptive_text_noise=False)

        with patch("synapse.privacy.policies._laplace", side_effect=lambda scale: scale):
            privatized = policy.enforce([artifact])

        self.assertEqual(len(privatized), 1)
        result = privatized[0]
        self.assertEqual(result.metadata["score"], 6.0)
        self.assertEqual(result.metadata["count"], 2.0)
        self.assertEqual(result.structured_payload["value"], 6.0)
        self.assertEqual(result.structured_payload["weights"], [6.0, 2.0])


    def test_edge_does_not_overmerge_distinct_scenarios_with_shared_schema(self):
        shared_payload = {
            "type": "usage_scenario",
            "tool_description": "Shared tool description",
            "precautions": ["Shared precaution"],
            "annex_summary": "Shared annex",
        }
        artifact_a = KnowledgeArtifact(
            signature="ratio-scenario",
            text="rich schema text A",
            structured_payload={**shared_payload, "scenario_context": "Handles percentages and proportions."},
            metadata={"tool": "mathqa", "scenario": "Percentage and Proportion Solver"},
        )
        artifact_b = KnowledgeArtifact(
            signature="finance-scenario",
            text="rich schema text B",
            structured_payload={**shared_payload, "scenario_context": "Handles interest rates and financial calculations."},
            metadata={"tool": "mathqa", "scenario": "Financial and Banking Calculator"},
        )

        edge = EdgeAggregator(EdgeConfig(edge_id="edge-1", similarity_threshold=0.85))
        merged = edge.merge_packages([
            KnowledgePackage(source_id="client-1", artifacts=[artifact_a]),
            KnowledgePackage(source_id="client-2", artifacts=[artifact_b]),
        ])

        self.assertIsNotNone(merged)
        assert merged is not None
        self.assertEqual(len(merged.artifacts), 2)


    def test_edge_clusters_similar_artifacts_and_logs_conflicts(self):
        artifact_a = KnowledgeArtifact(
            signature="logic-a",
            text="Solve the algebra word problem by isolating the variable and checking the final answer.",
            structured_payload={"solver": "algebra", "steps": 3},
            metadata={"domain": "math"},
        )
        artifact_b = KnowledgeArtifact(
            signature="logic-b",
            text="Solve this algebra word problem by isolating the variable and checking the answer carefully.",
            structured_payload={"solver": "algebra", "steps": 4},
            metadata={"domain": "math"},
        )
        artifact_c = KnowledgeArtifact(
            signature="science-a",
            text="Identify the organelle responsible for photosynthesis in plant cells.",
            structured_payload={"solver": "biology"},
            metadata={"domain": "science"},
        )

        edge = EdgeAggregator(EdgeConfig(edge_id="edge-1", similarity_threshold=0.75))
        merged = edge.merge_packages(
            [
                KnowledgePackage(source_id="client-1", artifacts=[artifact_a]),
                KnowledgePackage(source_id="client-2", artifacts=[artifact_b]),
                KnowledgePackage(source_id="client-3", artifacts=[artifact_c]),
            ]
        )

        self.assertIsNotNone(merged)
        assert merged is not None
        self.assertEqual(len(merged.artifacts), 2)
        self.assertEqual(merged.metadata["cluster_count"], 2)
        self.assertEqual(len(merged.metadata["conflict_log"]), 1)
        self.assertEqual(len(edge.conflict_log), 1)

        conflict = merged.metadata["conflict_log"][0]
        self.assertEqual(conflict["type"], "cosine_cluster_conflict")
        self.assertEqual(conflict["domain"], "math")
        self.assertEqual(conflict["cluster_size"], 2)
        self.assertEqual(sorted(conflict["signatures"]), ["logic-a", "logic-b"])
        self.assertGreaterEqual(conflict["max_cosine_similarity"], 0.75)

        clustered_artifact = next(artifact for artifact in merged.artifacts if artifact.metadata.get("domain") == "math")
        self.assertEqual(clustered_artifact.metadata["cluster_size"], 2)
        self.assertEqual(
            sorted(clustered_artifact.metadata["cluster_member_signatures"]),
            ["logic-a", "logic-b"],
        )


if __name__ == "__main__":
    unittest.main()
