import unittest

from scripts.run_toolbench_retrieval_at_scale import QueryRecord, select_catalog_tools, select_query_records


class ToolbenchRetrievalAtScaleTest(unittest.TestCase):
    def test_select_query_records_uses_explicit_ids_in_order(self):
        records = [
            QueryRecord(query_id="q1", query_text="one", gold_parent_tools=["a"], provenance="test"),
            QueryRecord(query_id="q2", query_text="two", gold_parent_tools=["b"], provenance="test"),
            QueryRecord(query_id="q3", query_text="three", gold_parent_tools=["c"], provenance="test"),
        ]

        selected, metadata = select_query_records(records, query_count=2, query_seed=42, query_ids=["q3", "q1"])

        self.assertEqual([item.query_id for item in selected], ["q3", "q1"])
        self.assertEqual(metadata["selection_mode"], "explicit_query_ids")

    def test_select_query_records_rejects_mismatched_count(self):
        records = [QueryRecord(query_id="q1", query_text="one", gold_parent_tools=["a"], provenance="test")]

        with self.assertRaises(ValueError):
            select_query_records(records, query_count=2, query_seed=42, query_ids=["q1"])

    def test_select_catalog_tools_honors_explicit_base_manifest(self):
        selected = select_catalog_tools(
            size=32,
            tool_order=[f"tool_{idx}" for idx in range(40)],
            query_tool_priority=[f"priority_{idx}" for idx in range(40)],
            subset_seed=None,
            base_tool_manifest=[f"manifest_{idx}" for idx in range(32)],
        )

        self.assertEqual(selected, [f"manifest_{idx}" for idx in range(32)])


if __name__ == "__main__":
    unittest.main()
