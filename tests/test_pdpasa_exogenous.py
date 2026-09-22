import tempfile
import unittest
from pathlib import Path

import numpy as np

from forecasting.build_pdpasa_exogenous import _write_region


class PdpasaExogenousTest(unittest.TestCase):
    def test_stpasa_fills_only_missing_pdpasa_tail_and_hourly_min_is_used(self):
        origin = 1_700_000_000
        pd_values = np.arange(48, dtype=np.float32)
        pd_values[46:] = np.nan
        pd_runs = {
            origin: {
                "half_hour": pd_values,
                "last_changed": origin - 60,
            }
        }
        st_runs = {
            origin - 7200: {
                "values": {
                    origin + 47 * 1800: 100.0,
                    origin + 48 * 1800: 90.0,
                },
                "last_changed": origin - 30,
            },
            origin: {
                "values": {
                    origin + 47 * 1800: -100.0,
                    origin + 48 * 1800: -100.0,
                },
                "last_changed": origin + 1,
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "region.npz"
            summary = _write_region("NSW1", pd_runs, st_runs, output)
            with np.load(output) as archive:
                values = archive["max_spare_capacity_mw"]

        self.assertEqual(summary["complete_origins"], 1)
        self.assertEqual(summary["stpasa_filled_half_hours"], 2)
        self.assertEqual(values.shape, (1, 24))
        self.assertEqual(values[0, 0], 0.0)
        self.assertEqual(values[0, -1], 90.0)


if __name__ == "__main__":
    unittest.main()
