"""Integration regressions for the actual development execution path."""
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from forecasting.agentic_dev_eval import r9


def test_failed_decision_is_not_silent_success_or_missing_forecast():
    baseline = np.arange(120, dtype=float).reshape(24, 5)
    failed = {"action": "", "exit_code": 1, "failure_reason": "INVALID_JSON"}
    with patch.object(r9.arms, "decide_c", return_value=failed) as decide:
        result, values = r9.decision_result({"history_prices": np.arange(72)}, baseline,
                                           "C", None, 2028, object())
    assert result["exit_code"] == 1
    assert values.tobytes() == baseline.tobytes()
    assert decide.call_args.kwargs["seed"] == 2028


def test_experiment_publication_is_complete_and_cannot_overwrite(tmp_path):
    final = tmp_path / "2027_NSW1_D"
    r9._publish(final, lambda stage: (stage / "forecast.bin").write_bytes(b"unchanged"))
    r9.verify_directory(final)
    with pytest.raises(FileExistsError):
        r9._publish(final, lambda stage: (stage / "forecast.bin").write_bytes(b"replacement"))
    assert (final / "forecast.bin").read_bytes() == b"unchanged"
    (final / "forecast.bin").write_bytes(b"modified")
    with pytest.raises(ValueError, match="artifact changed"):
        r9.verify_directory(final)


def test_score_includes_failed_decisions_in_full_grid(tmp_path):
    cfg = {"output_root": str(tmp_path), "seeds": [2026, 2027, 2028], "regions": ["NSW1"]}
    identity = {"origin_unix": np.array([1, 2]), "target_time_unix": np.zeros((2, 24), dtype=int)}
    truth = np.zeros((2, 24))
    baseline = np.ones((2, 24, 5)) * 10
    r9._publish(tmp_path / "labels", lambda stage: r9._npz(stage / "NSW1.npz", actual=truth, **identity))
    for seed in cfg["seeds"]:
        for arm in ("A", "B", "C", "D"):
            values = baseline.copy()
            exit_codes = np.zeros(2, dtype=int)
            if arm == "D":
                values[0] = 0
                exit_codes[1] = 1  # second (failed) cell must contribute baseline error 10
            r9._publish(tmp_path / str(seed) / "NSW1" / arm,
                lambda stage, q=values, exits=exit_codes: r9._npz(
                    stage / "predictions.npz", quantiles=q, exit_codes=exits, **identity))
    with patch.object(r9, "_execution", return_value={"execution_commit": "test"}):
        r9.score(cfg)
    summary = json.loads((tmp_path / "results" / "summary.json").read_text())
    for result in summary["results"]:
        assert result["arms"]["D"]["mae"] == 5.0
        assert result["arms"]["D"]["decision_failures"] == 1
        assert result["arms"]["D"]["n_origins"] == 2


def test_publication_preparation_uses_real_deadline(tmp_path):
    final = tmp_path / "blocked"
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        r9._publish(final, lambda stage: time.sleep(1), seconds=0.03)
    assert time.monotonic() - started < 0.5
    assert not final.exists()
