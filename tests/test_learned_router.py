import unittest

from synapse.retrieval.learned_router import LearnedTextRouter, RoutingExample, cross_validated_predictions


class LearnedRouterTest(unittest.TestCase):
    def test_fit_and_predict(self):
        examples = [
            RoutingExample(query_id="q1", sample_id="s1", query_text="profit from selling 3 items", label="Finance"),
            RoutingExample(query_id="q2", sample_id="s2", query_text="discount on a store purchase", label="Finance"),
            RoutingExample(query_id="q3", sample_id="s3", query_text="work rate for two workers", label="Rate"),
            RoutingExample(query_id="q4", sample_id="s4", query_text="time needed to finish the job", label="Rate"),
        ]
        router = LearnedTextRouter().fit(examples)
        self.assertEqual(router.predict_one("calculate profit after a discount"), "Finance")
        self.assertEqual(router.predict_one("workers finish a task in less time"), "Rate")

    def test_cross_validated_predictions_cover_all_rows(self):
        examples = [
            RoutingExample(query_id=f"q{i}", sample_id=f"s{i}", query_text=text, label=label)
            for i, (text, label) in enumerate(
                [
                    ("profit and revenue", "Finance"),
                    ("store discount and cost", "Finance"),
                    ("bank interest and spending", "Finance"),
                    ("workers and hours", "Rate"),
                    ("speed over time", "Rate"),
                    ("distance per hour", "Rate"),
                ],
                start=1,
            )
        ]
        rows = cross_validated_predictions(examples, n_splits=3, random_state=0)
        self.assertEqual(len(rows), len(examples))
        self.assertEqual(sorted(row["query_id"] for row in rows), sorted(example.query_id for example in examples))
        self.assertTrue(all("predicted_domain" in row for row in rows))


if __name__ == "__main__":
    unittest.main()
