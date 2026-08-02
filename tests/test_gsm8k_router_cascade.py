import unittest

from scripts.run_gsm8k_router_cascade import effective_cost_ratio, parse_threshold_list


class Gsm8kRouterCascadeTest(unittest.TestCase):
    def test_parse_threshold_list(self):
        self.assertEqual(parse_threshold_list("0.5, 1.0,1.5"), [0.5, 1.0, 1.5])

    def test_effective_cost_ratio(self):
        ratio = effective_cost_ratio(0.45, small_model_billions=1.0, large_model_billions=8.0)
        self.assertAlmostEqual(ratio, 0.575)


if __name__ == "__main__":
    unittest.main()
