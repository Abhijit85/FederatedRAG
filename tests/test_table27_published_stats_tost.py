import unittest

from scripts.run_table27_published_stats_tost import reconstruct_from_published_stats


class Table27PublishedStatsTostTest(unittest.TestCase):
    def test_reconstructs_expected_table27_numbers(self):
        summary = reconstruct_from_published_stats(
            mean_left=0.92,
            sd_left=0.02,
            mean_right=0.92,
            sd_right=0.02,
            paired_p=0.31,
            n_pairs=5,
            effect_size_d=0.2,
            mean_diff=None,
            d_mode="reported_sd",
            margin=0.03,
            alpha=0.05,
        )
        recon = summary["reconstruction"]
        tost = summary["tost"]

        self.assertAlmostEqual(recon["mean_diff"], 0.004, places=6)
        self.assertAlmostEqual(recon["se_diff"], 0.0034436189, places=6)
        self.assertAlmostEqual(tost["ci_lo"], -0.0033412680, places=6)
        self.assertAlmostEqual(tost["ci_hi"], 0.0113412680, places=6)
        self.assertLess(tost["p_lower"], 0.0012)
        self.assertLess(tost["p_upper"], 0.0010)
        self.assertTrue(tost["equivalent"])


if __name__ == "__main__":
    unittest.main()
