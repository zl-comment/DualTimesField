import numpy as np
import pytest
from forecasting.full_v5_joint.final_test import combine_memory
from forecasting.full_v5_joint.protocol import timestamp


def test_final_test_preserves_all_origins_including_last_boundary_row():
    history={'origin':np.array([timestamp('2022-12-30')]),'actual':np.zeros((1,24))}
    test={'origin':np.array([timestamp('2023-01-01'),timestamp('2024-12-31')]),'actual':np.ones((2,24))}
    merged=combine_memory(history,test)
    np.testing.assert_array_equal(merged['origin'][1:],test['origin'])
    np.testing.assert_array_equal(merged['actual'][1:],test['actual'])


def test_final_memory_rejects_overlapping_and_wrong_periods():
    history={'origin':np.array([timestamp('2023-01-02')])}
    with pytest.raises(ValueError,match='overlap'):
        combine_memory(history,{'origin':np.array([timestamp('2023-01-01')])})
    with pytest.raises(ValueError,match='interval'):
        combine_memory(history,{'origin':np.array([timestamp('2025-01-01')])})
