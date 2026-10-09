"""Check figure data and native-vector exports without retraining a model."""
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


class FrameworkHierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(ROOT / "figure_data/framework_hierarchy.npz", allow_pickle=False) as data:
            cls.data = {key: data[key] for key in data.files}

    def test_history_and_forecast_lengths(self):
        for name in ("history", "demand", "trend", "events", "lowpass"):
            self.assertEqual(self.data[name].shape, (72,))
        for name in ("point", "actual", "q05", "q95"):
            self.assertEqual(self.data[name].shape, (24,))
        self.assertTrue(np.all(self.data["q05"] <= self.data["q95"]))

    def test_detector_and_atoms(self):
        centres = self.data["centres"]
        self.assertEqual(len(centres), 8)
        self.assertTrue(np.all(np.diff(np.sort(centres)) >= 3))
        hist = self.data["history"]
        # torch.median uses the lower middle value for an even-length vector.
        departure = hist - np.sort(hist)[35]
        np.testing.assert_allclose(departure[centres], self.data["departures"])
        remaining = np.abs(departure).copy()
        for centre in centres:
            self.assertEqual(centre, np.argmax(remaining))
            remaining[np.abs(np.arange(72)-centre) < 3] = 0
        amplitude = np.sign(departure[centres]) * np.maximum(
            np.abs(departure[centres])-self.data["threshold"], 0)
        atoms = amplitude[:, None] * np.exp(
            -(np.arange(72)[None]-centres[:, None])**2 / (2*self.data["width"]**2))
        np.testing.assert_allclose(atoms, self.data["atoms"], atol=2e-6)
        np.testing.assert_allclose(atoms.sum(axis=0), self.data["events"], atol=2e-6)

    def test_lowpass_reads_full_history(self):
        fft = np.fft.rfft(self.data["history"])
        cutoff = max(1, len(fft)//8)
        mask = np.zeros(len(fft))
        mask[:cutoff] = 1
        mask[cutoff:2*cutoff] = np.linspace(1, 0, cutoff)
        np.testing.assert_allclose(np.fft.irfft(fft*mask, n=72),
                                   self.data["lowpass"], atol=2e-6)

    def test_exports_are_vectors_with_text(self):
        svg = ET.parse(ROOT / "figures/framework_hierarchy.svg")
        self.assertEqual(len(svg.findall(".//{http://www.w3.org/2000/svg}image")), 0)
        self.assertGreater(len(svg.findall(".//{http://www.w3.org/2000/svg}text")), 40)
        pdf = (ROOT / "figures/framework_hierarchy.pdf").read_bytes()
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertNotIn(b"/Subtype /Image", pdf)
        self.assertIn(b"/FontFile", pdf)


if __name__ == "__main__":
    unittest.main()
