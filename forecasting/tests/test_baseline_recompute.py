"""Regression guards for the historical R25/R27 compatibility entry points."""
import sys

import numpy as np
import pytest

from forecasting.aligned_scoring import per_origin_scores
from scripts import analyze_r25_ceiling as r25
from scripts import run_r27_confirmation as r27


def test_both_analysis_entries_use_common_sorted_losses():
    actual = np.array([[0., 10.], [-5., 2.]])
    quantiles = actual[..., None] + np.array([20., -10., 2., 8., -3.])
    expected = per_origin_scores(actual, quantiles)
    qs, mae = r25.per_origin_scores(actual, quantiles)
    np.testing.assert_array_equal(qs, expected["quantile_score_proxy"])
    np.testing.assert_array_equal(mae, expected["mae"])
    np.testing.assert_array_equal(r27.per_origin_qs(actual, quantiles), qs)
    unsorted = actual[..., None] - quantiles
    assert not np.allclose(qs, 2 * np.maximum(r27.Q * unsorted, (r27.Q - 1) * unsorted).mean((1, 2)))


def test_r25_rejects_candidate_label_misalignment(tmp_path):
    y = np.zeros((3, 2))
    q = np.broadcast_to(np.arange(5.), (3, 2, 5))
    for seed in (2026, 2027):
        folder = tmp_path / "A/NSW1" / f"lightgbm_seed{seed}"
        folder.mkdir(parents=True)
        np.savez(folder / "selection.npz", actual=y + (seed == 2027), quantiles=q,
                 origin_unix=np.arange(3))
    with pytest.raises(ValueError, match="sample/label mismatch"):
        r25.load_scores(tmp_path)


@pytest.fixture
def r27_selection_archives(tmp_path, monkeypatch):
    actual = np.array([[0., 10.], [-5., 2.], [7., 8.]])
    origins = np.array([100, 200, 300], dtype=np.int64)
    paths = {}
    for index, key in enumerate(r27.MEMBERS):
        path = tmp_path / f"{key.replace(':', '_')}.npz"
        np.savez(path, actual=actual, origin_unix=origins,
                 quantiles=actual[..., None] + np.array([20., -10., 2., 8., -3.]) + index)
        paths[key] = path
    monkeypatch.setattr(r27, "r25_selection_path", lambda key, region: paths[key])
    return paths


def test_r27_selection_guard_preserves_valid_candidate_losses(r27_selection_archives):
    scores = r27.load_selection_scores("NSW1")
    assert tuple(scores) == r27.MEMBERS
    for key, path in r27_selection_archives.items():
        with np.load(path) as z:
            assert list(scores[key]) == list(z["origin_unix"])
            np.testing.assert_array_equal(list(scores[key].values()),
                                          r27.per_origin_qs(z["actual"], z["quantiles"]))


@pytest.mark.parametrize("origins", [
    np.array([100, 100, 300]),  # Duplicate origins must not be collapsed by dict.
    np.array([100, 300, 200]),
    np.array([100, 200]),  # zip must not silently truncate losses.
    np.array([[100], [200], [300]]),
    np.array([100., 200., 300.]),
])
@pytest.mark.parametrize("key", [r27.MEMBERS[0], r27.MEMBERS[-1]])
def test_r27_selection_rejects_invalid_origins(r27_selection_archives, origins, key):
    path = r27_selection_archives[key]
    with np.load(path) as z:
        arrays = {name: z[name] for name in z.files}
    arrays["origin_unix"] = origins
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="strictly increasing integer timestamps"):
        r27.load_selection_scores("NSW1")


@pytest.mark.parametrize("field", ["origin_unix", "actual"])
def test_r27_selection_rejects_candidate_misalignment(r27_selection_archives, field):
    path = r27_selection_archives[r27.MEMBERS[-1]]
    with np.load(path) as z:
        arrays = {name: z[name] for name in z.files}
    arrays[field] = arrays[field] + 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="candidate sample/label mismatch"):
        r27.load_selection_scores("NSW1")


def test_r25_refuses_to_overwrite_old_analysis(tmp_path, monkeypatch):
    report = tmp_path / "original.json"
    report.write_text("original")
    monkeypatch.setattr(sys, "argv", ["analyze_r25_ceiling.py", "--output", str(report)])
    with pytest.raises(FileExistsError):
        r25.main()
    assert report.read_text() == "original"


def test_r27_refuses_to_overwrite_old_run(tmp_path, monkeypatch):
    monkeypatch.setattr(r27, "OUT", tmp_path)
    with pytest.raises(FileExistsError):
        r27.main(["--output-dir", str(tmp_path)])


def test_dm_extreme_tail_remains_positive_instead_of_cancellation_to_zero():
    diff = np.random.default_rng(42).normal(-0.055, 0.1, 1000)
    _, _, p = r27.dm_hln_one_sided(diff)
    assert 0 < p < 1e-20


def test_bootstrap_wrappers_are_repeatable_and_share_calendar_blocks():
    import pandas as pd
    origins = pd.date_range("2022-01-01", periods=35, tz="Australia/Brisbane").as_unit("s").asi8
    differences = np.linspace(-1., 2., len(origins))
    r25_input = {region: pd.DataFrame({"origin": origins, "diff_qs": differences})
                 for region in r25.REGIONS}
    r27_input = {region: pd.Series(-differences, index=origins) for region in r25.REGIONS}
    expected = r25.bootstrap_and_jackknife(r25_input)
    tail, ci = r27.block_bootstrap(r27_input)
    assert tail == expected["bootstrap_tail_probability"]
    np.testing.assert_allclose(ci, [-expected["ci_high"], -expected["ci_low"]])
    assert r27.block_bootstrap(r27_input) == (tail, ci)
