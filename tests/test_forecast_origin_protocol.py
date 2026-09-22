import unittest

import numpy as np
import pandas as pd

from forecasting.datasets import _valid_origins


class ForecastOriginProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        delivery = pd.Series(
            pd.date_range("2020-01-01", periods=24 * 14, freq="h", tz="Australia/Brisbane")
        )
        self.frame = pd.DataFrame(
            {
                "delivery": delivery,
                "available": delivery + pd.Timedelta(hours=1),
                "split": np.where(delivery < pd.Timestamp("2020-01-08", tz=delivery.dt.tz), "train", "test"),
            }
        )
        self.base_config = {
            "data": {
                "delivery_column": "delivery",
                "available_column": "available",
                "split_column": "split",
            },
            "forecast_protocol": {
                "input_hours": 72,
                "output_hours": 24,
                "stride_hours": 1,
            },
        }

    def test_default_protocol_keeps_original_stride_behavior(self) -> None:
        origins = _valid_origins(self.frame, self.base_config, "train")
        expected = np.arange(72, 24 * 7 - 24 + 1)
        np.testing.assert_array_equal(origins, expected)

    def test_fixed_origin_has_one_consistent_local_hour_per_day(self) -> None:
        config = {
            **self.base_config,
            "forecast_protocol": {
                **self.base_config["forecast_protocol"],
                "origin_hour": 6,
            },
        }
        for split in ("train", "test"):
            origins = _valid_origins(self.frame, config, split)
            origin_times = self.frame.loc[origins, "delivery"]
            self.assertTrue(origin_times.dt.hour.eq(6).all())
            if len(origins) > 1:
                np.testing.assert_array_equal(np.diff(origins), np.full(len(origins) - 1, 24))

    def test_fixed_origin_rejects_overlapping_daily_targets(self) -> None:
        config = {
            **self.base_config,
            "forecast_protocol": {
                **self.base_config["forecast_protocol"],
                "output_hours": 25,
                "origin_hour": 0,
            },
        }
        with self.assertRaisesRegex(ValueError, "overlapping targets"):
            _valid_origins(self.frame, config, "train")


if __name__ == "__main__":
    unittest.main()
