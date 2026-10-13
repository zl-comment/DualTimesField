import tempfile
import unittest
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.build_pdpasa_exogenous import FIELDS, _write_region
from forecasting.datasets import AEMOForecastDataset, load_forecast_config


class PdpasaExogenousTest(unittest.TestCase):
    def test_stpasa_fills_only_missing_pdpasa_tail_and_hourly_min_is_used(self):
        origin = 1_700_000_000
        # The builder now fills six independent features, not scalar spare capacity.
        pd_values = np.repeat(np.arange(48, dtype=np.float32)[:, None], len(FIELDS), axis=1)
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
                    origin + 47 * 1800: np.full(len(FIELDS), 100.0, dtype=np.float32),
                    origin + 48 * 1800: np.full(len(FIELDS), 90.0, dtype=np.float32),
                },
                "last_changed": origin - 30,
            },
            origin: {
                "values": {
                    origin + 47 * 1800: np.full(len(FIELDS), -100.0, dtype=np.float32),
                    origin + 48 * 1800: np.full(len(FIELDS), -100.0, dtype=np.float32),
                },
                "last_changed": origin + 1,
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "region.npz"
            summary = _write_region("NSW1", pd_runs, st_runs, output)
            with np.load(output) as archive:
                values = archive["max_spare_capacity_mw"]
                demand = archive["demand50_mw"]
                self.assertTrue((archive["source_last_changed_unix"] <= origin).all())
                self.assertTrue((archive["source_run_unix"] <= origin).all())
                self.assertEqual(archive["source_stpasa_run_unix"][0], origin - 7200)

        self.assertEqual(summary["complete_origins"], 1)
        self.assertEqual(summary["stpasa_filled_half_hours"], 2)
        self.assertEqual(values.shape, (1, 24))
        self.assertEqual(values[0, 0], 0.0)
        self.assertEqual(values[0, -1], 90.0)
        self.assertEqual(demand[0, 0], 0.5)
        self.assertEqual(demand[0, -1], 95.0)

    def test_zero_values_masks_only_requested_source_after_derived_transforms(self):
        root = Path(__file__).resolve().parents[1]
        config = load_forecast_config(root / "configs/legacy/aemo_forecast_ic_calibrator.yaml")
        config["forecast_protocol"].update(input_hours=4, output_hours=2)
        config["origin_context"]["enabled"] = False
        stamps = pd.date_range("2020-12-31 12:00", periods=32, freq="h", tz="Australia/Brisbane")
        frame = pd.DataFrame({
            "delivery_start_aest": stamps,
            "available_at_aest": stamps,
            "region": "NSW1",
            "split": ["train"] * 20 + ["validation"] * 6 + ["test"] * 6,
            "rrp_aud_per_mwh": np.arange(32, dtype=float),
            "total_demand_mw": np.arange(32, dtype=float) + 100,
        })
        origins = stamps.tz_convert("UTC").as_unit("ns").asi8 // 1_000_000_000
        blocks = ("future_exogenous", "ctf_exogenous", "quantile_exogenous",
                  "predispatch_exogenous", "interconnector_exogenous")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            names = {name for block in blocks for name in config[block].get(
                "features", [config[block].get("name")])}
            values = np.arange(64, dtype=np.float32).reshape(32, 2)
            arrays = {name: values + index for index, name in enumerate(sorted(names))}
            np.savez(path, forecast_origin_unix=origins, source_run_unix=origins - 60,
                     source_last_changed_unix=origins - 30, **arrays)
            for block in blocks:
                config[block]["region_files"] = {"NSW1": str(path)}
            full = AEMOForecastDataset(frame, config, "NSW1", "test")[0]
            sizes = [1, len(config["predispatch_exogenous"]["features"]),
                     len(config["interconnector_exogenous"]["features"])]
            slices = (slice(0, sizes[0]), slice(sizes[0], sum(sizes[:2])),
                      slice(sum(sizes[:2]), sum(sizes)))
            for masked, expected_slice in (("future_exogenous", slices[0]),
                                           ("predispatch_exogenous", slices[1]),
                                           ("interconnector_exogenous", slices[2])):
                with self.subTest(masked=masked):
                    ablated = copy.deepcopy(config)
                    ablated[masked]["zero_values"] = True
                    if masked == "future_exogenous":
                        # A scarcity ablation must also mask net load and derived shortfalls.
                        ablated["ctf_exogenous"]["zero_values"] = True
                        ablated["quantile_exogenous"]["zero_values"] = True
                    sample = AEMOForecastDataset(frame, ablated, "NSW1", "test")[0]
                    self.assertEqual(sample["future_exogenous"].shape, full["future_exogenous"].shape)
                    self.assertTrue((sample["future_exogenous"][:, expected_slice] == 0).all())
                    for part in slices:
                        if part != expected_slice:
                            np.testing.assert_array_equal(sample["future_exogenous"][:, part],
                                                          full["future_exogenous"][:, part])
                            self.assertTrue((sample["future_exogenous"][:, part] != 0).any())
                    for key in ("ctf_exogenous", "quantile_exogenous"):
                        if masked == "future_exogenous":
                            self.assertTrue((sample[key] == 0).all())
                        else:
                            np.testing.assert_array_equal(sample[key], full[key])


if __name__ == "__main__":
    unittest.main()
