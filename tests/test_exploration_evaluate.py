import json
from pathlib import Path

import numpy as np
import pytest

from forecasting.exploration_evaluate import (
    Forecast, LEVELS, _block_sums, check_sources, compare_sources, holm_adjust,
    load_forecast, load_source, origin_loss_parts, score, summarize_source,
)


def make_forecast(actual=None, error=0.0, n=400, h=2):
    if actual is None:
        actual = np.tile(np.array([-20.0, 400.0]), (n, 1))[:, :h]
    actual = np.asarray(actual, dtype=float)
    point = actual + error
    quantile = np.repeat(point[..., None], 5, axis=-1)
    return Forecast(point, quantile, actual, np.arange(len(actual), dtype=np.int64) * 3600, 0.0)


def make_source(error=0.0):
    return {region: {seed: make_forecast(error=error) for seed in (2026, 2027, 2028)}
            for region in ("NSW1", "QLD1", "TAS1")}


def save_forecast(path, forecast, **override):
    arrays = {key: getattr(forecast, key) for key in ("point", "quantile", "actual", "origin_unix")}
    arrays.update(override)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


@pytest.mark.parametrize("bad", ["nan", "shape", "ensemble", "origins", "levels"])
def test_rejects_invalid_archives(tmp_path, bad):
    f = make_forecast()
    overrides = {}
    if bad == "nan":
        f.point[0, 0] = np.nan
    elif bad == "shape":
        overrides["quantile"] = f.quantile[..., :4]
    elif bad == "ensemble":
        overrides["quantile"] = f.quantile[None]
    elif bad == "origins":
        f.origin_unix[20:] += 3600
    else:
        overrides["quantile_levels"] = [0.01, 0.1, 0.5, 0.9, 0.99]
    path = tmp_path / "invalid.npz"
    save_forecast(path, f, **overrides)
    with pytest.raises(ValueError):
        load_forecast(path)


def test_level_contract_and_rearrangement(tmp_path):
    f = make_forecast()
    f.quantile[..., 0] += 100
    path = tmp_path / "valid.npz"
    save_forecast(path, f, levels=LEVELS)
    loaded = load_forecast(path)
    assert loaded.crossing_rate == 1.0
    assert np.all(np.diff(loaded.quantile, axis=-1) >= 0)


def test_loading_requires_three_distinct_seed_runs_and_alignment(tmp_path):
    pattern = str(tmp_path / "{region}" / "{seed}.npz")
    for region in ("NSW1", "QLD1", "TAS1"):
        for seed in (2026, 2027, 2028):
            save_forecast(Path(pattern.format(region=region, seed=seed)), make_forecast())
    assert len(load_source(pattern)) == 3
    f = make_forecast()
    f.actual[0, 0] += 1
    save_forecast(Path(pattern.format(region="QLD1", seed=2028)), f)
    with pytest.raises(ValueError, match="Actual prices"):
        load_source(pattern)
    with pytest.raises(ValueError, match="both"):
        load_source(str(tmp_path / "{region}.npz"))


def test_cross_variant_actual_and_origin_mismatch_rejected():
    ref, candidate = make_source(), make_source()
    candidate["TAS1"][2026].actual[0, 0] += 1
    with pytest.raises(ValueError, match="Actual prices"):
        check_sources(ref, candidate)
    candidate = make_source()
    candidate["QLD1"][2026].origin_unix += 3600
    with pytest.raises(ValueError, match="aligned"):
        compare_sources(ref, candidate, replicates=10)


def test_scores_distinguish_window_and_pooled_rmse_and_empty_tails():
    f = make_forecast(actual=[[50, 50], [50, 50]])
    f.point[1] += 4
    scores = score(f)
    assert scores["rmse_pooled"] == pytest.approx(np.sqrt(8))
    assert scores["rmse_window_mean"] == 2
    assert scores["crps_negative"] is None
    assert scores["pinball_q95_spike_hours"] is None
    assert scores["n_negative"] == 0
    assert scores["n_spike"] == 0
    json.dumps(scores, allow_nan=False)


def test_seed_summary_averages_losses_not_forecasts():
    # Opposite forecast errors cancel under an ensemble but not across runs.
    forecasts = {2026: make_forecast(error=-10, n=2), 2027: make_forecast(error=10, n=2),
                 2028: make_forecast(error=0, n=2)}
    result = summarize_source({"NSW1": forecasts})
    mae = result["region_seed_summary"]["NSW1"]["overall"]["mae"]
    assert mae["mean"] == pytest.approx(20 / 3)
    assert mae["std"] == pytest.approx(np.std([10, 10, 0], ddof=1))
    assert mae["n"] == 3
    assert "1" in result["per_seed"]["NSW1"]["2026"]["horizons"]
    json.dumps(result, allow_nan=False)


def test_tail_parts_keep_empty_origins_and_observation_weighting():
    f = make_forecast(actual=[[50, 50], [400, 50], [400, 450]], error=10)
    sums, counts = origin_loss_parts(f, "pinball_q95_spike_hours")
    assert len(sums) == 3
    assert counts.tolist() == [0, 1, 2]
    assert sums.tolist() == pytest.approx([0, 0.5, 1.0])


def test_prefix_block_sums_equal_explicit_resampling():
    values = np.arange(30, dtype=float).reshape(10, 3)
    starts = np.array([[0, 3, 4], [2, 0, 5]])
    lengths = np.array([4, 4, 2])
    expected = np.stack([sum((values[s:s + size].sum(axis=0) for s, size in zip(row, lengths)))
                         for row in starts])
    np.testing.assert_allclose(_block_sums(values, starts, lengths), expected)


def test_paired_bootstrap_direction_and_joint_region_dependence():
    ref, candidate = make_source(error=5), make_source(error=3)
    result = compare_sources(ref, candidate, replicates=99)
    for region in ("NSW1", "QLD1", "TAS1", "macro_regions"):
        row = result["metrics"]["mae"][region]
        assert row["difference"] == -2
        assert row["ci_low"] == -2
        assert row["ci_high"] == -2
        assert row["p_value"] == 0.01
    assert result["bootstrap"]["block_hours_used"] == 168
    # Identical series across markets must have the same bootstrap CI, including
    # the macro CI. Independent region resampling would narrow the macro CI.
    for region in ref:
        for seed in ref[region]:
            ref[region][seed].point += np.arange(400)[:, None] % 31
    result = compare_sources(ref, candidate, replicates=199, metrics=("mae",))
    for key in ("difference", "ci_low", "ci_high"):
        assert result["metrics"]["mae"]["macro_regions"][key] == pytest.approx(result["metrics"]["mae"]["NSW1"][key])


def test_empty_tail_bootstrap_is_null_and_json_safe():
    ref, candidate = make_source(), make_source()
    for source in (ref, candidate):
        for region in source:
            source[region] = {seed: make_forecast(actual=np.ones((400, 2)) * 50) for seed in source[region]}
    result = compare_sources(ref, candidate, replicates=10)
    assert result["metrics"]["crps_spike"]["NSW1"]["p_value"] is None
    assert result["metrics"]["crps_spike"]["macro_regions"]["difference"] is None
    holm_adjust({"candidate": result})
    json.dumps(result, allow_nan=False)


def test_holm_uses_one_family_across_comparisons():
    comparisons = {
        "a": {"metrics": {"mae": {"NSW1": {"p_value": 0.01}, "QLD1": {"p_value": 0.04}}}},
        "b": {"metrics": {"mae": {"NSW1": {"p_value": 0.03}, "QLD1": {"p_value": None}}}},
    }
    holm_adjust(comparisons)
    assert comparisons["a"]["metrics"]["mae"]["NSW1"]["p_value_holm"] == pytest.approx(0.03)
    assert comparisons["b"]["metrics"]["mae"]["NSW1"]["p_value_holm"] == pytest.approx(0.06)
    assert comparisons["a"]["metrics"]["mae"]["QLD1"]["p_value_holm"] == pytest.approx(0.06)
    assert comparisons["b"]["metrics"]["mae"]["QLD1"]["p_value_holm"] is None


def test_month_bins_use_delivery_time_in_aest_not_origin_or_utc():
    from forecasting.exploration_evaluate import summarize_forecast
    f = make_forecast(actual=[[50, 50]])
    f.origin_unix[:] = np.datetime64("2023-01-31T13:00:00", "s").astype(np.int64)
    months = summarize_forecast(f)["months_aest"]
    assert months["2023-01"]["n_observations"] == 1
    assert months["2023-02"]["n_observations"] == 1


def test_delivery_support_deduplicates_overlapping_windows_and_masks():
    from forecasting.exploration_evaluate import summarize_forecast
    # Four physical hours represented by six rolling origin/horizon cells.
    f = make_forecast(actual=[[-20, 400], [400, 50], [50, -10]])
    result = summarize_forecast(f)
    overall = result["overall"]
    assert overall["n_observations"] == 6
    assert overall["n_unique_delivery_hours"] == 4
    assert overall["n_spike"] == 2
    assert overall["n_unique_spike_delivery_hours"] == 1
    assert overall["n_unique_negative_delivery_hours"] == 2
    band = result["price_bands"]["spike"]
    assert band["n_observations"] == 2
    assert band["n_unique_delivery_hours"] == 1
    assert band["n_unique_spike_delivery_hours"] == 1
    assert band["n_unique_negative_delivery_hours"] == 0
    # Selecting only the second horizon must not count cells from the first.
    assert result["horizons"]["2"]["n_unique_delivery_hours"] == 3
    mask = np.array([[True, False], [False, False], [False, True]])
    assert score(f, mask)["n_unique_delivery_hours"] == 2
    empty = score(f, np.zeros(f.actual.shape, dtype=bool))
    assert empty["n_unique_delivery_hours"] == 0
