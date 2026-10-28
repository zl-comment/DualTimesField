"""Synthetic G0 boundary tests; no files, labels, models or network are loaded."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from forecasting.agent_forecasting.asof import (
    AsOfSnapshot, ForecastVintage, SourceVersion, build_snapshot, parse_utc,
    same_target_revision,
)
from forecasting.agent_forecasting.contracts import FrozenForecastInput
from forecasting.agent_forecasting.trace import AppendOnlyTrace


ORIGIN = 1641218400
DIGEST = "a" * 64


def input_payload():
    return dict(origin_unix=ORIGIN,
                target_time_unix=[ORIGIN + h * 3600 for h in range(24)],
                price_history=[-20.0 + h for h in range(72)],
                past_covariates={"cal_0": [0.5] * 72, "demand_std": [1.0] * 72},
                future_covariates={"cal_0": [0.25] * 24},
                metadata={"region": "NSW1", "input_profile": "g0_base_v1",
                          "source_manifest_sha256": DIGEST})


def source(version="v1", **changes):
    values = dict(chain_id="notice-1", version_id=version,
                  source_url="https://example.org/notice/1", source_sha256=DIGEST,
                  published_unix=ORIGIN - 7200, first_seen_unix=ORIGIN - 3600,
                  valid_from_unix=ORIGIN, valid_to_unix=ORIGIN + 24 * 3600,
                  publication_basis="verified")
    values.update(changes)
    return SourceVersion(**values)


def vintage(version="run-1", **changes):
    values = dict(source_id="predispatch", vintage_id=version,
                  variable="predispatch_price", region="NSW1", unit="AUD_per_MWh",
                  target_time_unix=ORIGIN + 5 * 3600, issued_unix=ORIGIN - 7200,
                  first_seen_unix=ORIGIN - 7100, value=-30.0)
    values.update(changes)
    return ForecastVintage(**values)


class InputContractTests(unittest.TestCase):
    def test_copies_all_nested_input_and_serialization(self):
        payload = input_payload()
        frozen = FrozenForecastInput.from_dict(payload)
        digest = frozen.input_hash()
        payload["price_history"][0] = 999999
        payload["past_covariates"]["cal_0"][0] = 999999
        payload["metadata"]["region"] = "QLD1"
        exported = frozen.to_dict()
        exported["future_covariates"]["cal_0"][0] = 999999
        self.assertEqual(frozen.input_hash(), digest)
        self.assertEqual(frozen.price_history[0], -20.0)
        with self.assertRaises(TypeError):
            frozen.metadata["region"] = "QLD1"
        with self.assertRaises(FrozenInstanceError):
            frozen.origin_unix = 0
        self.assertFalse(hasattr(frozen, "__dict__"))
        self.assertEqual(FrozenForecastInput.from_dict(frozen.to_dict()).input_hash(), digest)

    def test_denies_labels_at_all_public_input_boundaries(self):
        for key in ("actual", "target", "labels", "future_price", "context", "score"):
            payload = input_payload()
            payload[key] = [1.0] * 24
            with self.subTest(key=key), self.assertRaises(ValueError):
                FrozenForecastInput.from_dict(payload)
            for container in ("metadata", "past_covariates", "future_covariates"):
                payload = input_payload()
                payload[container][key] = [1.0] * (72 if container == "past_covariates" else 24)
                with self.subTest(key=key, container=container), self.assertRaises(ValueError):
                    FrozenForecastInput.from_dict(payload)
        for payload_value in ({"actual": [100]}, "[100, 200]", "x\nactual=100"):
            payload = input_payload()
            payload["metadata"]["snapshot_id"] = payload_value
            with self.assertRaises(ValueError):
                FrozenForecastInput.from_dict(payload)

    def test_time_lengths_finiteness_and_negative_prices(self):
        self.assertLess(FrozenForecastInput.from_dict(input_payload()).price_history[0], 0)
        changes = [dict(origin_unix=True), dict(origin_unix=float(ORIGIN)),
                   dict(origin_unix=ORIGIN + 1), dict(price_history=[0.0] * 71),
                   dict(price_history=[float("nan")] * 72),
                   dict(price_history=[float("inf")] * 72),
                   dict(price_history=[False] * 72),
                   dict(target_time_unix=[ORIGIN + (h + 1) * 3600 for h in range(24)]),
                   dict(target_time_unix=[ORIGIN] * 24)]
        for change in changes:
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                FrozenForecastInput.from_dict({**input_payload(), **change})
        payload = input_payload()
        payload["origin_unix"] = np.int64(ORIGIN)
        payload["price_history"] = np.asarray(payload["price_history"], dtype=np.float32)
        self.assertEqual(FrozenForecastInput.from_dict(payload).origin_unix, ORIGIN)

    def test_future_requires_same_named_past_and_approved_semantics(self):
        for future in ({"cal_1": [0.0] * 24}, {"demand_std": [0.0] * 24},
                       {"predispatch_price": [0.0] * 24}, {"cal_0": [0.0] * 23}):
            with self.subTest(future=list(future)), self.assertRaises(ValueError):
                FrozenForecastInput.from_dict({**input_payload(), "future_covariates": future})
        # A mutable source's later history changes cannot change a prior request.
        payload = input_payload()
        first = FrozenForecastInput.from_dict(payload)
        payload["price_history"][-1] += 1
        second = FrozenForecastInput.from_dict(payload)
        self.assertNotEqual(first.input_hash(), second.input_hash())


class AvailabilityTests(unittest.TestCase):
    def test_explicit_utc_only(self):
        self.assertEqual(parse_utc("2022-01-03T14:00:00Z"), ORIGIN)
        self.assertEqual(parse_utc(datetime(2022, 1, 3, 14, tzinfo=timezone.utc)), ORIGIN)
        for value in ("2022-01-03T14:00:00", "2022-01-04T00:00:00+10:00",
                      "2022-01-03T14:00:00.1Z", "unknown", ORIGIN,
                      datetime(2022, 1, 3, 14)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_utc(value)
        for value in ("2022-01-03T14:00:00", datetime(2022, 1, 3, 14), True):
            with self.assertRaises(ValueError):
                source(published_unix=value)

    def test_published_is_insufficient_without_actual_acquisition(self):
        original = source()
        hidden = [source("late", first_seen_unix=ORIGIN + 1),
                  source("missing", first_seen_unix=None),
                  source("proxy", publication_basis="archive_proxy"),
                  source("unknown", publication_basis="unknown", published_unix=None)]
        snapshot = build_snapshot(ORIGIN, [original, *hidden])
        self.assertEqual(snapshot.visible_versions, (original,))
        self.assertEqual(snapshot.active_for(ORIGIN), (original,))
        self.assertEqual(snapshot.snapshot_id, build_snapshot(ORIGIN, [original]).snapshot_id)
        with self.assertRaises(ValueError):
            AsOfSnapshot(ORIGIN, (hidden[0],), (hidden[0],))

    def test_later_retraction_does_not_rewrite_prior_snapshot(self):
        original = source()
        revision = source("v2", supersedes_id="v1", published_unix=ORIGIN - 60,
                          first_seen_unix=ORIGIN + 3600, withdrawn=True,
                          source_sha256="b" * 64)
        prior = build_snapshot(ORIGIN, [original, revision])
        newer = build_snapshot(ORIGIN + 3600, [revision, original])
        self.assertEqual(prior.active_for(ORIGIN), (original,))
        self.assertEqual(newer.selected_versions, (revision,))
        self.assertEqual(newer.active_for(ORIGIN + 3600), ())
        self.assertEqual(prior.snapshot_id, build_snapshot(ORIGIN, [original]).snapshot_id)
        self.assertNotEqual(prior.snapshot_id, newer.snapshot_id)

    def test_future_effective_withdrawal_preserves_predecessor_until_start(self):
        original = source()
        withdrawal = source("v2", supersedes_id="v1", withdrawn=True,
                            published_unix=ORIGIN - 1800, first_seen_unix=ORIGIN - 900,
                            valid_from_unix=ORIGIN + 6 * 3600,
                            valid_to_unix=ORIGIN + 7 * 3600)
        snapshot = build_snapshot(ORIGIN, [original, withdrawal])
        self.assertEqual(snapshot.selected_versions, (withdrawal,))
        for hour in range(24):
            with self.subTest(hour=hour):
                self.assertEqual(snapshot.active_for(ORIGIN + hour * 3600),
                                 (original,) if hour < 6 else ())
        # The notice's valid_to does not revive a withdrawn source chain.
        self.assertEqual(snapshot.active_for(ORIGIN + 7 * 3600), ())

    def test_withdrawal_backtracking_obeys_each_explicit_predecessor(self):
        original = source()
        first = source("v2", supersedes_id="v1", withdrawn=True,
                       valid_from_unix=ORIGIN + 4 * 3600)
        second = source("v3", supersedes_id="v2", withdrawn=True,
                        valid_from_unix=ORIGIN + 6 * 3600)
        snapshot = build_snapshot(ORIGIN, [second, original, first])
        self.assertEqual(snapshot.active_for(ORIGIN + 3 * 3600), (original,))
        self.assertEqual(snapshot.active_for(ORIGIN + 4 * 3600), ())
        self.assertEqual(snapshot.active_for(ORIGIN + 6 * 3600), ())
        # Backtracking never ignores a predecessor's actual validity interval.
        expired = replace(original, valid_to_unix=ORIGIN + 2 * 3600)
        self.assertEqual(build_snapshot(ORIGIN, [expired, first]).active_for(ORIGIN + 3 * 3600), ())

    def test_normal_postponement_remains_whole_version_replacement(self):
        original = source()
        postponed = source("v2", supersedes_id="v1", valid_from_unix=ORIGIN + 6 * 3600)
        snapshot = build_snapshot(ORIGIN, [original, postponed])
        self.assertEqual(snapshot.active_for(ORIGIN + 5 * 3600), ())
        self.assertEqual(snapshot.active_for(ORIGIN + 6 * 3600), (postponed,))

    def test_effective_interval_is_half_open_and_conflicts_fail_closed(self):
        original = source(valid_from_unix=ORIGIN + 3600, valid_to_unix=ORIGIN + 7200)
        snapshot = build_snapshot(ORIGIN, [original, original])
        self.assertEqual(snapshot.active_for(ORIGIN), ())
        self.assertEqual(snapshot.active_for(ORIGIN + 3600), (original,))
        self.assertEqual(snapshot.active_for(ORIGIN + 7200), ())
        with self.assertRaises(ValueError):
            snapshot.active_for(ORIGIN + 24 * 3600)
        cases = [(original, replace(original, source_sha256="b" * 64)),
                 (original, source("v2", published_unix=ORIGIN - 30, first_seen_unix=ORIGIN)),
                 (source("v2", supersedes_id="v1"),)]
        for records in cases:
            with self.subTest(records=records), self.assertRaises(ValueError):
                build_snapshot(ORIGIN, records)

    def test_same_target_revision_not_same_horizon_or_imputed_values(self):
        old = vintage()
        new = vintage("run-2", issued_unix=ORIGIN - 3600,
                      first_seen_unix=ORIGIN - 3500, value=15)
        self.assertEqual(same_target_revision(old, new, ORIGIN), 45)
        cases = [replace(new, target_time_unix=new.target_time_unix + 3600),
                 replace(new, variable="predispatch_demand"), replace(new, unit="MW"),
                 replace(new, region="QLD1"), replace(new, covered=False),
                 replace(new, first_seen_unix=ORIGIN + 1),
                 replace(new, vintage_id=old.vintage_id)]
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                same_target_revision(old, candidate, ORIGIN)

    def test_explicit_version_relationships_resolve_equal_clocks(self):
        first = source("z-first")
        next_version = source("a-second", supersedes_id="z-first", source_sha256="b" * 64)
        snapshot = build_snapshot(ORIGIN, [next_version, first])
        self.assertEqual(snapshot.selected_versions, (next_version,))
        self.assertEqual(snapshot.snapshot_id, build_snapshot(ORIGIN, [first, next_version]).snapshot_id)
        fork = source("third", supersedes_id="z-first", source_sha256="c" * 64)
        with self.assertRaises(ValueError):
            build_snapshot(ORIGIN, [first, next_version, fork])


class TraceTests(unittest.TestCase):
    def test_exact_repeat_is_noop_and_conflicts_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            trace = AppendOnlyTrace(path)
            first = trace.append("request-1", "requested", DIGEST)
            before = path.read_bytes()
            self.assertEqual(first, trace.append("request-1", "requested", DIGEST))
            self.assertEqual(before, path.read_bytes())
            with self.assertRaises(ValueError):
                trace.append("request-1", "requested", "b" * 64)
            with self.assertRaises(ValueError):
                trace.append("request-1", "completed", "b" * 64)
            self.assertEqual(before, path.read_bytes())
            done = trace.append("request-1", "completed", DIGEST, output_hash="c" * 64)
            self.assertEqual(done["previous_hash"], first["record_hash"])
            self.assertEqual(len(trace.read()), 2)

    def test_fixed_failure_reason_and_no_arbitrary_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            trace = AppendOnlyTrace(path)
            with self.assertRaises(ValueError):
                trace.append("request", "failed", DIGEST, reason="actual=[100,200]")
            self.assertFalse(path.exists())
            record = trace.append("request", "failed", DIGEST, reason="timeout")
            self.assertEqual(record["reason"], "timeout")

    def test_corruption_or_partial_write_refuses_append(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            trace = AppendOnlyTrace(path)
            trace.append("request-1", "requested", DIGEST)
            record = json.loads(path.read_text())
            record["input_hash"] = "b" * 64
            path.write_text(json.dumps(record) + "\n")
            corrupted = path.read_bytes()
            with self.assertRaises(ValueError):
                trace.append("request-2", "requested", DIGEST)
            self.assertEqual(corrupted, path.read_bytes())
            path.write_bytes(b'{"request_id":')
            with self.assertRaises(ValueError):
                trace.append("request-2", "requested", DIGEST)


if __name__ == "__main__":
    unittest.main()
