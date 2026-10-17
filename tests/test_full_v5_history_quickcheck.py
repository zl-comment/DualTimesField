"""Selection provenance must prevent silent changes after inner validation."""
from copy import deepcopy

import pytest

from forecasting.full_v5 import data, schema
from forecasting.full_v5.history_quickcheck import IMPLEMENTATION_FILES, validate_selection


def selection_fixture(root):
    for relative in IMPLEMENTATION_FILES:
        p = root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixed candidate")
    return dict(parent_identity_sha="parent", selected_policy="recent_consistent",
                fit_cutoff=schema.unix_timestamp("2022-11-01"),
                score_end_exclusive=schema.unix_timestamp("2022-12-01"),
                implementation_hashes={p: data.sha256_file(root / p) for p in IMPLEMENTATION_FILES},
                macro={"C_roll": dict(mae=10., quantile_score_proxy=8.),
                       "fresh28": dict(mae=9., quantile_score_proxy=7.),
                       "recent_consistent": dict(mae=8., quantile_score_proxy=7.5)})


def test_selection_binds_implementation_and_fixed_protocol(tmp_path):
    s = selection_fixture(tmp_path)
    assert validate_selection(s, tmp_path, "parent") == "recent_consistent"
    for key, value in (("selected_policy", "fresh28"), ("fit_cutoff", 0),
                       ("score_end_exclusive", 0), ("parent_identity_sha", "other")):
        altered = deepcopy(s)
        altered[key] = value
        with pytest.raises(ValueError):
            validate_selection(altered, tmp_path, "parent")
    (tmp_path / IMPLEMENTATION_FILES[0]).write_text("different candidate")
    with pytest.raises(ValueError, match="implementation changed"):
        validate_selection(s, tmp_path, "parent")


def test_failed_selection_cannot_authorize_followup(tmp_path):
    s = selection_fixture(tmp_path)
    for p in ("fresh28", "recent_consistent"):
        s["macro"][p]["quantile_score_proxy"] = 20.
    with pytest.raises(ValueError, match="did not qualify"):
        validate_selection(s, tmp_path, "parent")
    s["selected_policy"] = None
    with pytest.raises(ValueError, match="did not qualify"):
        validate_selection(s, tmp_path, "parent")
