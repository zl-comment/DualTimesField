import numpy as np
import pytest

from forecasting.baseline_audit.moirai_replay import comparison, expected_start_ordinals


def test_comparison_does_not_hide_small_changes():
    assert comparison([1., 2.], [1., 2.])["array_equal"]
    assert not comparison([1., 2. + 1e-12], [1., 2.])["array_equal"]


@pytest.mark.parametrize("a,b", [([1.], [1., 2.]), ([np.nan], [1.]), ([1.], [np.inf])])
def test_comparison_rejects_invalid_arrays(a, b):
    with pytest.raises(ValueError):
        comparison(a, b)


def test_native_start_uses_local_hour_and_preserves_72h_offset():
    import pandas as pd
    origin = int(pd.Timestamp("2023-01-01", tz="Australia/Brisbane").timestamp())
    expected = pd.Period("2023-01-01 00:00", freq="h").ordinal
    np.testing.assert_array_equal(expected_start_ordinals(np.array([origin, origin+72*3600])),
                                  [expected, expected+72])
