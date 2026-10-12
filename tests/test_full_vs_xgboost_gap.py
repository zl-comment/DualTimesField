"""Tests for the read-only gap decomposition (standard library only; loaded by path
because ``forecasting/__init__`` imports torch)."""
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gap", ROOT / "forecasting/analysis/full_vs_xgboost_gap.py")
gap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gap)
REPORT = ROOT / gap.DEFAULT_REPORT


def cell(n, mae, crps):
    return {"n_observations": n, "mae": mae, "crps_quantile_approx": crps}


def region(shift):
    bands = {"negative": cell(10, 5 + shift, 4 + shift), "0_to_100": cell(60, 10 + shift, 8 + shift),
             "100_to_300": cell(25, 30 + shift, 20 + shift), "spike": cell(5, 500 - shift, 400 - shift)}
    n = 100
    overall = {"n_observations": n}
    for m in ("mae", "crps_quantile_approx"):
        overall[m] = sum(b["n_observations"] * b[m] for b in bands.values()) / n
    horizons = {str(h): {"n_observations": 1, "mae": 1 + shift, "crps_quantile_approx": 1 + shift} for h in range(1, 25)}
    months = {"2023-01": {"n_observations": 1, "mae": 1 + shift, "crps_quantile_approx": 1 + shift}}
    return {"overall": overall, "price_bands": bands, "horizons": horizons, "months_aest": months}


class SliceTableTest(unittest.TestCase):
    def test_band_contributions_sum_to_regional_gap(self):
        full, xgb = region(1.0), region(0.0)
        for metric in gap.METRICS:
            rows = gap.slice_table(full, xgb, "price_bands", gap.BANDS, metric)
            total = sum(r["contribution"] for r in rows.values())
            self.assertAlmostEqual(total, full["overall"][metric] - xgb["overall"][metric], places=12)
            self.assertAlmostEqual(sum(r["weight"] for r in rows.values()), 1.0)

    def test_sign_convention_positive_means_full_worse(self):
        rows = gap.slice_table(region(2.0), region(0.0), "price_bands", ("0_to_100",), "mae")
        self.assertGreater(rows["0_to_100"]["contribution"], 0)
        rows = gap.slice_table(region(2.0), region(0.0), "price_bands", ("spike",), "mae")
        self.assertLess(rows["spike"]["contribution"], 0)


@unittest.skipUnless(REPORT.is_file(), "frozen exploration report not present")
class RealReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.load(gzip.open(REPORT, "rt"))
        cls.result = gap.analyse(cls.report)

    def test_macro_gaps_match_published_static_table(self):
        # report_static.md: full 38.923 / 25.936, xgboost 37.968 / 25.527
        self.assertAlmostEqual(self.result["macro"]["mae"]["full"], 38.923, places=3)
        self.assertAlmostEqual(self.result["macro"]["mae"]["xgboost"], 37.968, places=3)
        self.assertAlmostEqual(self.result["macro"]["crps_quantile_approx"]["full"], 25.936, places=3)
        self.assertAlmostEqual(self.result["macro"]["crps_quantile_approx"]["xgboost"], 25.527, places=3)

    def test_decompositions_are_exact(self):
        self.assertLess(max(abs(v) for v in self.result["checks"].values()), 1e-9)

    def test_gap_matches_bootstrap_point_estimate(self):
        c = self.report["protocols"]["static"]["comparisons"]["168"]["xgboost"]["metrics"]
        for metric in gap.METRICS:
            self.assertAlmostEqual(-c[metric]["macro_regions"]["difference"], self.result["macro"][metric]["gap"], places=9)

    def test_deterministic_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            again = gap.analyse(self.report)
            self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(self.result, sort_keys=True))
            self.assertTrue(Path(tmp).exists())


if __name__ == "__main__":
    unittest.main()
