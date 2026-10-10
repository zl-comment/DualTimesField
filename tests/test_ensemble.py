import unittest

import numpy as np

from forecasting.ensemble import fit_weights, simplex_grid


class EnsembleWeightTests(unittest.TestCase):
    def test_simplex_grid_sums_to_one(self):
        grid = simplex_grid(3, 0.1)
        self.assertEqual(len(grid), 66)
        for weights in grid:
            self.assertAlmostEqual(float(weights.sum()), 1.0)
            self.assertTrue((weights >= 0).all())

    def test_weights_follow_the_better_member_and_are_shrunk(self):
        rng = np.random.default_rng(0)
        actual = rng.normal(size=(500, 24))
        good = actual + 0.1 * rng.normal(size=actual.shape)
        bad = actual + 2.0 * rng.normal(size=actual.shape)
        weights = fit_weights(np.stack([good, bad]), actual, 0.05)
        self.assertGreater(weights[0], 0.5)
        # shrinking halfway toward equal weights keeps both members in the mix
        self.assertGreater(weights[1], 0.2)
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_equal_members_get_equal_weights(self):
        rng = np.random.default_rng(1)
        actual = rng.normal(size=(400, 24))
        member = actual + rng.normal(size=actual.shape)
        weights = fit_weights(np.stack([member, member.copy()]), actual, 0.1)
        self.assertTrue(np.allclose(weights, [0.5, 0.5], atol=0.26))


if __name__ == "__main__":
    unittest.main()
