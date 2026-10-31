"""Small synthetic checks for the R10 structural source audit; no real data."""
from datetime import datetime
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from forecasting.agent_forecasting import totaldemand_information as td


ORIGIN = int(datetime(2022, 1, 2, tzinfo=td.AEST).timestamp())


def rows(seqno=2022010139, runno=1, *, target_only=False):
    ends = ORIGIN - 1800 + np.arange(50, dtype=np.int64) * 1800
    if target_only:
        ends = ends[2:]
    return pd.DataFrame({
        "seqno": seqno, "runno": runno, "region": "NSW1",
        "periodid": np.arange(1, len(ends) + 1), "intervention": 0,
        "totaldemand": 1000.0 + np.arange(len(ends)),
        "lastchanged_unix": ORIGIN - 1200, "period_end_unix": ends,
        "schedule_label_unix": td.vintage_label_unix(seqno),
    })


def test_schedule_label_market_day_and_origin_boundary():
    assert td.vintage_label_unix(2022010101) == int(datetime(2022, 1, 1, 4, 30, tzinfo=td.AEST).timestamp())
    assert td.vintage_label_unix(2022010140) == ORIGIN
    assert td.vintage_label_unix(2022010148) == ORIGIN + 4 * 3600
    with pytest.raises(ValueError):
        td.vintage_label_unix(2022010149)
    with pytest.raises(ValueError):
        td.coverage(rows(), [ORIGIN + 3600], ["NSW1"])


def test_chunked_month_reader_keeps_runno_and_only_demand_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(td, "CHUNK_ROWS", 2)
    fields = ",".join(td.REQUIRED_COLUMNS)
    text = (
        "C,SETP.WORLD,DVD_PREDISPATCHREGIONSUM\n"
        f"I,PREDISPATCH,REGION_SOLUTION,5,{fields}\n"
        "D,PREDISPATCH,REGION_SOLUTION,5,2022010101,1,NSW1,1,0,1000,2022/01/01 04:01:00,2022/01/01 04:30:00\n"
        "D,PREDISPATCH,REGION_SOLUTION,5,2022010101,2,NSW1,1,0,1100,2022/01/01 04:02:00,2022/01/01 04:30:00\n"
        "D,PREDISPATCH,REGION_SOLUTION,5,2022010101,1,QLD1,1,0,2000,2022/01/01 04:01:00,2022/01/01 04:30:00\n"
        "D,PREDISPATCH,REGION_SOLUTION,5,2022010101,1,NSW1,1,1,9999,2022/01/01 04:01:00,2022/01/01 04:30:00\n"
        "C,END OF REPORT,7\n"
    )
    path = tmp_path / "month.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("month.csv", text)
    actual = td.read_month(path, ["NSW1"])
    assert list(actual.columns) == list(td.COLUMNS)
    assert actual["runno"].tolist() == [1, 2]
    assert actual["totaldemand"].tolist() == [1000.0, 1100.0]
    assert actual["period_end_unix"].tolist() == [td.vintage_label_unix(2022010101)] * 2


def test_full_bridge_and_empty_region_are_reported_as_json():
    result = td.coverage(rows(), [ORIGIN], ["NSW1", "TAS1"])
    assert result[0]["covered"] and result[0]["target24_covered"]
    assert not result[0]["target24_only"]
    assert result[0]["run_age_seconds"] == 1800
    assert result[0]["lastchanged_max"] == ORIGIN - 1200
    assert not result[1]["covered"] and result[1]["candidate_count"] == 0
    assert result[1]["selected_seqno"] is None
    assert result[0]["publication_basis"] == "archive_proxy"
    assert result[0]["strict_asof_verified"] is False
    json.dumps(result, allow_nan=False)


def test_target_only_does_not_fill_bridge_or_splice_vintages():
    target = rows(target_only=True)
    gap = rows(seqno=2022010138).iloc[:2]
    result = td.coverage(pd.concat([target, gap]), [ORIGIN], ["NSW1"])[0]
    assert not result["covered"] and result["target24_only"]
    assert result["target24_selected_seqno"] == 2022010139
    assert result["selected_seqno"] is None


def test_newest_complete_runno_and_late_rows_cannot_rewrite_prefix():
    first = rows(runno=1)
    second = rows(runno=2)
    second.loc[second.index[-1], "lastchanged_unix"] = ORIGIN - 599
    late_conflict = first.iloc[[0]].copy()
    late_conflict["totaldemand"] += 200
    late_conflict["lastchanged_unix"] = ORIGIN - 599
    out = td.coverage(pd.concat([first, second, late_conflict]), [ORIGIN], ["NSW1"])[0]
    assert out["selected_runno"] == 1
    assert out["candidate_conflict_count"] == 0
    second["lastchanged_unix"] = ORIGIN - 600
    out = td.coverage(pd.concat([first, second]), [ORIGIN], ["NSW1"])[0]
    assert out["selected_runno"] == 2  # cutoff equality is visible


def test_visible_conflict_and_nonfinite_reject_candidate_not_older_complete():
    older = rows(seqno=2022010138)
    newer = rows()
    conflicting = newer.iloc[[0]].copy()
    conflicting["totaldemand"] += 1
    out = td.coverage(pd.concat([older, newer, conflicting]), [ORIGIN], ["NSW1"])[0]
    assert out["selected_seqno"] == 2022010138
    assert out["candidate_conflict_count"] == 1
    newer.loc[newer.index[-1], "totaldemand"] = np.nan
    out = td.coverage(pd.concat([older, newer]), [ORIGIN], ["NSW1"])[0]
    assert out["selected_seqno"] == 2022010138
    assert out["candidate_nonfinite_count"] == 1


def test_consistent_duplicates_do_not_change_coverage_and_future_label_is_excluded():
    past = rows()
    future = rows(seqno=2022010140)
    result = td.coverage(pd.concat([past, past.iloc[[0]], future]), [ORIGIN], ["NSW1"])[0]
    assert result["covered"] and result["selected_seqno"] == 2022010139
    assert result["candidate_count"] == 1
    assert result["candidate_conflict_count"] == 0
