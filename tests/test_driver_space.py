import unittest

import torch

from forecasting.models import DriverEventField


class DriverEventFieldTests(unittest.TestCase):
    def test_events_only_in_listed_channels(self):
        torch.manual_seed(0)
        field = DriverEventField(num_variables=5, event_channels=[2, 4], events_per_channel=3)
        x = torch.randn(6, 72, 5)
        x[:, 10, 2] += 8.0
        x[:, 50, 4] -= 8.0
        signal, amplitude, gate = field.extract_events(x, torch.linspace(0, 1, 72))
        self.assertEqual(tuple(signal.shape), (6, 72, 5))
        self.assertEqual(tuple(amplitude.shape), (6, 6, 5))
        for channel in (0, 1, 3):
            self.assertEqual(float(signal[..., channel].abs().sum()), 0.0)
        self.assertGreater(float(signal[..., 2].abs().sum()), 0.0)
        # the injected departures are the strongest events, with their sign
        self.assertGreater(float(amplitude[:, :3, 2].max()), 4.0)
        self.assertLess(float(amplitude[:, 3:, 4].min()), -4.0)

    def test_thresholds_learn_per_channel(self):
        field = DriverEventField(num_variables=4, event_channels=[1, 2], events_per_channel=2)
        x = torch.randn(3, 72, 4)
        signal, _, _ = field.extract_events(x, torch.linspace(0, 1, 72))
        signal.sum().backward()
        self.assertEqual(tuple(field.raw_threshold.shape), (2,))
        self.assertIsNotNone(field.raw_threshold.grad)

    def test_rejects_unknown_channel(self):
        with self.assertRaises(ValueError):
            DriverEventField(num_variables=3, event_channels=[3])


if __name__ == "__main__":
    unittest.main()
