"""Source-bound candidate-2 M0 tests: receipt, parser, as-of, typed contract.

RED-first: at the frozen base these tests fail because the candidate module
and the typed contract extension do not exist.  Every value used here comes
from synthetic PREDISPATCHIS-format ZIP bytes built by the fixture builder,
never from hand-fed rows.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "fixtures" / "agent_forecasting"))

from forecasting.agent_forecasting.contracts import FrozenForecastInput

try:  # RED granularity: fail per test while the candidate module is absent
    from predispatchis_td_v2_fixture import (  # type: ignore
        FIXTURE_FILENAME, FIXTURE_ORIGIN_UNIX, FIXTURE_RECEIPT_UNIX, FIXTURE_RUNNO,
        FIXTURE_SEQNO, build_body_bytes, build_fixture_receipt, fixture_values,
    )
    _FIXTURE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only during the RED commit
    _FIXTURE_IMPORT_ERROR = exc


def _require_fixture():
    if _FIXTURE_IMPORT_ERROR is not None:
        raise AssertionError(f"candidate fixture chain unavailable: {_FIXTURE_IMPORT_ERROR}")


def _input(**overrides):
    payload = {
        "origin_unix": 1641218400,
        "target_time_unix": tuple(1641218400 + 3600 * h for h in range(24)),
        "price_history": tuple(float(v) for v in range(1, 73)),
    }
    payload.update(overrides)
    return FrozenForecastInput(**payload)


class TypedFutureContractTests(unittest.TestCase):
    COVARIATE = "predispatchis_totaldemand_mw"

    def test_typed_future_accepted_without_same_named_past(self):
        values = tuple(6000.0 + h for h in range(24))
        request = _input(future_forecasts={self.COVARIATE: values})
        self.assertEqual(request.future_forecasts[self.COVARIATE], values)
        round_trip = FrozenForecastInput.from_dict(request.to_dict())
        self.assertEqual(round_trip.input_hash(), request.input_hash())

    def test_typed_future_input_hash_differs_from_base(self):
        base = _input()
        candidate = _input(future_forecasts={self.COVARIATE: tuple(6000.0 + h for h in range(24))})
        self.assertNotEqual(base.input_hash(), candidate.input_hash())

    def test_typed_future_rejects_unknown_name_length_and_nonfinite(self):
        with self.assertRaisesRegex(ValueError, "unapproved"):
            _input(future_forecasts={"rrp_future": tuple(1.0 for _ in range(24))})
        with self.assertRaisesRegex(ValueError, "length"):
            _input(future_forecasts={self.COVARIATE: tuple(1.0 for _ in range(23))})
        values = [1.0] * 24
        values[3] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            _input(future_forecasts={self.COVARIATE: tuple(values)})

    def test_calendar_future_still_requires_same_named_past(self):
        with self.assertRaisesRegex(ValueError, "same-semantic past"):
            _input(future_covariates={"cal_0": tuple(1.0 for _ in range(24))})

    def test_typed_name_cannot_fabricate_a_past(self):
        with self.assertRaisesRegex(ValueError, "unapproved"):
            _input(past_covariates={self.COVARIATE: tuple(1.0 for _ in range(72))})


class RawBodyParserTests(unittest.TestCase):
    def setUp(self):
        _require_fixture()

    def _module(self):
        from forecasting.agent_forecasting import predispatch_td_abs as module
        return module

    def _parse(self, body):
        return self._module().parse_body(body)

    def test_accepts_well_formed_body(self):
        parsed = self._parse(build_body_bytes())
        values = fixture_values(70)
        self.assertEqual(parsed.seqno, FIXTURE_SEQNO)
        self.assertEqual(parsed.runno, FIXTURE_RUNNO)
        self.assertEqual(len(parsed.rows), 70)
        self.assertEqual(parsed.rows[0], (FIXTURE_ORIGIN_UNIX + 1800, values[0]))
        self.assertEqual(parsed.rows[-1], (FIXTURE_ORIGIN_UNIX + 70 * 1800, values[-1]))
        again = self._parse(build_body_bytes())
        self.assertEqual(parsed.body_sha256, again.body_sha256)

    def test_rejects_non_zip_and_wrong_member_count(self):
        with self.assertRaises(ValueError):
            self._parse(b"not a zip")
        import io
        import zipfile
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as archive:
            archive.writestr("a.csv", "x")
            archive.writestr("b.csv", "y")
        with self.assertRaises(ValueError):
            self._parse(out.getvalue())

    def test_rejects_header_column_drift(self):
        module = self._module()
        body = build_body_bytes()
        import io
        import zipfile
        archive = zipfile.ZipFile(io.BytesIO(body))
        text = archive.read(archive.namelist()[0]).decode()
        archive.close()
        drifted = text.replace(",DEMANDFORECAST,", ",DEMANDFORECAST2,", 1)
        self.assertNotEqual(drifted, text)
        buffer = io.BytesIO()
        info = zipfile.ZipInfo("SYNTHETIC_PREDISPATCHIS_SECTION.csv", date_time=(2022, 1, 3, 23, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        with zipfile.ZipFile(buffer, "w") as out_zip:
            out_zip.writestr(info, drifted.encode())
        with self.assertRaisesRegex(ValueError, "drifted"):
            module.parse_body(buffer.getvalue())

    def test_rejects_table_version_drift(self):
        module = self._module()
        body = build_body_bytes()
        import io
        import zipfile
        archive = zipfile.ZipFile(io.BytesIO(body))
        text = archive.read(archive.namelist()[0]).decode()
        archive.close()
        bumped = text.replace("I,PREDISPATCH,REGION_SOLUTION,9,", "I,PREDISPATCH,REGION_SOLUTION,10,", 1)
        buffer = io.BytesIO()
        info = zipfile.ZipInfo("x.csv", date_time=(2022, 1, 3, 23, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        with zipfile.ZipFile(buffer, "w") as out_zip:
            out_zip.writestr(info, bumped.encode())
        with self.assertRaises(ValueError):
            module.parse_body(buffer.getvalue())

    def test_rejects_row_width_mismatch_duplicate_pk_and_two_vintages(self):
        module = self._module()
        import io
        import zipfile

        def rezip(mutated_lines):
            # keep the frozen trailer invariant: the count equals total rows
            mutated_lines = list(mutated_lines)
            mutated_lines[-1] = f"C,END OF REPORT,{len(mutated_lines)}"
            text = "\n".join(mutated_lines)
            buffer = io.BytesIO()
            info = zipfile.ZipInfo("x.csv", date_time=(2022, 1, 3, 23, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            with zipfile.ZipFile(buffer, "w") as out_zip:
                out_zip.writestr(info, text.encode())
            return buffer.getvalue()

        body = build_body_bytes()
        archive = zipfile.ZipFile(io.BytesIO(body))
        text = archive.read(archive.namelist()[0]).decode()
        archive.close()
        lines = text.splitlines()
        target_first = next(i for i, line in enumerate(lines)
                            if line.startswith("I,PREDISPATCH,REGION_SOLUTION,9"))
        data_line = lines[target_first + 1]
        truncated = ",".join(data_line.split(",")[:-1])
        with self.assertRaisesRegex(ValueError, "width"):
            module.parse_body(rezip(
                lines[:target_first + 1] + [truncated] + lines[target_first + 2:]))
        duplicated = lines[:target_first + 2] + [data_line] + lines[target_first + 2:]
        with self.assertRaisesRegex(ValueError, "duplicate primary key"):
            module.parse_body(rezip(duplicated))
        other_vintage = data_line.replace(FIXTURE_SEQNO, "2022010399")
        with self.assertRaisesRegex(ValueError, "exactly one vintage"):
            module.parse_body(rezip(
                lines[:target_first + 1] + [data_line, other_vintage] + lines[target_first + 2:]))

    def test_builder_duplicate_and_drop_mutations(self):
        module = self._module()
        with self.assertRaisesRegex(ValueError, "duplicate primary key"):
            module.parse_body(build_body_bytes(
                duplicate_row_end=FIXTURE_ORIGIN_UNIX + 5 * 1800))
        parsed = module.parse_body(build_body_bytes(
            drop_half_hour_end=FIXTURE_ORIGIN_UNIX + 5 * 1800))
        self.assertEqual(len(parsed.rows), 69)

    def test_rejects_missing_trailer_count_and_bad_preamble(self):
        module = self._module()
        import io
        import zipfile

        def rezip(text):
            buffer = io.BytesIO()
            info = zipfile.ZipInfo("x.csv", date_time=(2022, 1, 3, 23, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            with zipfile.ZipFile(buffer, "w") as out_zip:
                out_zip.writestr(info, text.encode())
            return buffer.getvalue()

        body = build_body_bytes()
        archive = zipfile.ZipFile(io.BytesIO(body))
        text = archive.read(archive.namelist()[0]).decode()
        archive.close()
        lines = text.splitlines()
        broken_count = lines[:-1] + [",".join(["C", "END OF REPORT", "999999999"])]
        with self.assertRaisesRegex(ValueError, "total row"):
            module.parse_body(rezip("\n".join(broken_count)))
        broken_preamble = ["C", "NEMP.WORLD", "SOMETHINGELSE", "AEMO", "PUBLIC",
                           "2022/01/03", "23:00:00", "0", "PREDISPATCHIS", "0"] + lines[1:]
        with self.assertRaisesRegex(ValueError, "preamble"):
            module.parse_body(rezip("\n".join(broken_preamble)))

    def test_rejects_scopeless_body(self):
        module = self._module()
        body = build_body_bytes()
        import io
        import zipfile
        archive = zipfile.ZipFile(io.BytesIO(body))
        text = archive.read(archive.namelist()[0]).decode()
        archive.close()
        lines = [line for line in text.splitlines()
                 if not (line.startswith("D,PREDISPATCH,REGION_SOLUTION,9")
                         and ",NSW1," in line)]
        lines[-1] = f"C,END OF REPORT,{len(lines)}"  # retotal the single trailer
        buffer = io.BytesIO()
        info = zipfile.ZipInfo("x.csv", date_time=(2022, 1, 3, 23, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        with zipfile.ZipFile(buffer, "w") as out_zip:
            out_zip.writestr(info, "\n".join(lines).encode())
        with self.assertRaisesRegex(ValueError, "no NSW1"):
            module.parse_body(buffer.getvalue())


class ReceiptTrustTests(unittest.TestCase):
    def setUp(self):
        _require_fixture()

    def _module(self):
        from forecasting.agent_forecasting import predispatch_td_abs as module
        return module

    def _receipt(self):
        return build_fixture_receipt()

    def test_receipt_from_raw_body_grants_first_seen(self):
        receipt = self._receipt()
        self.assertEqual(receipt.basis, "synthetic_fixture")  # fixture builder path
        module = self._module()
        real = module.receipt_from_raw_body(
            build_body_bytes(), FIXTURE_RECEIPT_UNIX, FIXTURE_FILENAME)
        self.assertEqual(real.basis, module.REAL_BASIS)
        self.assertEqual(real.seqno, FIXTURE_SEQNO)
        self.assertTrue(real.visible_at(FIXTURE_ORIGIN_UNIX))
        self.assertFalse(real.visible_at(FIXTURE_RECEIPT_UNIX - 1))

    def test_direct_construction_and_replace_promotion_rejected(self):
        module = self._module()
        receipt = self._receipt()
        with self.assertRaisesRegex(ValueError, "builders"):
            module.PredispatchTdReceipt(
                seqno=receipt.seqno, runno=receipt.runno,
                receipt_complete_unix=receipt.receipt_complete_unix,
                body_sha256=receipt.body_sha256,
                header_line_sha256=receipt.header_line_sha256,
                filename=receipt.filename, table_id=receipt.table_id,
                region=receipt.region, intervention=receipt.intervention,
                unit=receipt.unit, basis=module.REAL_BASIS, rows=receipt.rows,
                raw_body=receipt.raw_body)
        with self.assertRaises(ValueError):
            dataclasses.replace(receipt, basis=module.REAL_BASIS)

    def test_value_tamper_with_same_body_hash_fails_closed(self):
        receipt = self._receipt()
        tampered = [(end, value + 1.0) for end, value in receipt.rows]
        with self.assertRaisesRegex(ValueError, "do not match the verified raw body"):
            dataclasses.replace(receipt, rows=tampered)
        with self.assertRaises(ValueError):
            dataclasses.replace(receipt, body_sha256="0" * 64)

    def test_synthetic_never_claims_strict_verification(self):
        module = self._module()
        receipt = self._receipt()
        series = module.build_future_series(FIXTURE_ORIGIN_UNIX, [receipt])
        self.assertEqual(series.publication_basis, module.SYNTHETIC_BASIS)
        self.assertEqual(series.strict_asof, module.SYNTHETIC_STATUS)


class DedupeSelectionTests(unittest.TestCase):
    def setUp(self):
        _require_fixture()

    def _module(self):
        from forecasting.agent_forecasting import predispatch_td_abs as module
        return module

    def test_duplicate_body_folds_to_earliest_receipt(self):
        module = self._module()
        body = build_body_bytes()
        early = module.synthetic_receipt_from_raw_body(body, FIXTURE_ORIGIN_UNIX - 600, "a.zip")
        late = module.synthetic_receipt_from_raw_body(body, FIXTURE_ORIGIN_UNIX - 60, "b.zip")
        deduped = module._dedupe([late, early])
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].receipt_complete_unix, early.receipt_complete_unix)
        # visibility at an instant only the early receipt precedes
        origin_probe = FIXTURE_ORIGIN_UNIX - 300
        self.assertTrue(deduped[0].visible_at(origin_probe))

    def test_same_vintage_two_bodies_is_a_fork(self):
        module = self._module()
        first = module.synthetic_receipt_from_raw_body(
            build_body_bytes(), FIXTURE_ORIGIN_UNIX - 600, "a.zip")
        second = module.synthetic_receipt_from_raw_body(
            build_body_bytes(shift_seconds=1800), FIXTURE_ORIGIN_UNIX - 600, "b.zip")
        self.assertNotEqual(first.body_sha256, second.body_sha256)
        with self.assertRaisesRegex(ValueError, "fork"):
            module._dedupe([first, second])

    def test_newest_visible_vintage_selected_without_fallback(self):
        module = self._module()
        origin = FIXTURE_ORIGIN_UNIX
        older = module.synthetic_receipt_from_raw_body(
            build_body_bytes(origin_unix=origin, seqno="2022010342"),
            origin - 3600, "older.zip")
        # newer vintage covers a shifted grid: it cannot serve this origin
        newer = module.synthetic_receipt_from_raw_body(
            build_body_bytes(origin_unix=origin, seqno="2022010344", shift_seconds=3600),
            origin - 3500, "newer.zip")
        selected = module.select_vintage(origin, [older, newer])
        self.assertEqual(selected.seqno, "2022010344")
        with self.assertRaisesRegex(ValueError, "no fill"):
            module.build_future_series(origin, [older, newer])

    def test_invisible_newer_vintage_excluded_by_clock(self):
        module = self._module()
        origin = FIXTURE_ORIGIN_UNIX
        older = module.synthetic_receipt_from_raw_body(
            build_body_bytes(origin_unix=origin, seqno="2022010342"),
            origin - 3600, "older.zip")
        newer_unseen = module.synthetic_receipt_from_raw_body(
            build_body_bytes(origin_unix=origin, seqno="2022010344"),
            origin + 1, "newer.zip")  # body-complete AFTER origin
        selected = module.select_vintage(origin, [older, newer_unseen])
        self.assertEqual(selected.seqno, "2022010342")

    def test_uncovered_origin_without_visible_receipt(self):
        module = self._module()
        receipt = self._receipt_late(module)
        with self.assertRaisesRegex(ValueError, "uncovered"):
            module.select_vintage(FIXTURE_ORIGIN_UNIX, [receipt])

    def _receipt_late(self, module):
        return module.synthetic_receipt_from_raw_body(
            build_body_bytes(), FIXTURE_ORIGIN_UNIX + 1800, "late.zip")


class SeriesMappingTests(unittest.TestCase):
    def setUp(self):
        _require_fixture()

    def _module(self):
        from forecasting.agent_forecasting import predispatch_td_abs as module
        return module

    def test_series_is_exact_pair_means_over_48_half_hours(self):
        module = self._module()
        receipt = build_fixture_receipt()
        series = module.build_future_series(FIXTURE_ORIGIN_UNIX, [receipt])
        values = fixture_values(70)
        expected = [(values[2 * h] + values[2 * h + 1]) / 2.0 for h in range(24)]
        self.assertEqual(series.future_values, tuple(expected))
        self.assertEqual(len(series.per_point), 24)
        for hour, point in enumerate(series.per_point):
            self.assertEqual(point["delivery_hour_start_unix"],
                             FIXTURE_ORIGIN_UNIX + hour * 3600)
            self.assertEqual(point["half_hour_end_unix"],
                             [FIXTURE_ORIGIN_UNIX + (2 * hour + 1) * 1800,
                              FIXTURE_ORIGIN_UNIX + (2 * hour + 2) * 1800])
        rebuilt = module.build_future_series(FIXTURE_ORIGIN_UNIX, [build_fixture_receipt()])
        self.assertEqual(rebuilt.series_hash, series.series_hash)

    def test_missing_half_hour_refuses_with_no_fill(self):
        module = self._module()
        receipt = module.synthetic_receipt_from_raw_body(
            build_body_bytes(drop_half_hour_end=FIXTURE_ORIGIN_UNIX + 7 * 1800),
            FIXTURE_ORIGIN_UNIX - 600, "gap.zip")
        with self.assertRaisesRegex(ValueError, "no fill"):
            module.build_future_series(FIXTURE_ORIGIN_UNIX, [receipt])

    def test_shifted_grid_refuses(self):
        module = self._module()
        receipt = module.synthetic_receipt_from_raw_body(
            build_body_bytes(shift_seconds=3600), FIXTURE_ORIGIN_UNIX - 600, "shift.zip")
        with self.assertRaisesRegex(ValueError, "no fill"):
            module.build_future_series(FIXTURE_ORIGIN_UNIX, [receipt])

    def test_origin_must_be_hour_aligned(self):
        module = self._module()
        receipt = build_fixture_receipt()
        with self.assertRaisesRegex(ValueError, "aligned"):
            module.build_future_series(FIXTURE_ORIGIN_UNIX + 1800, [receipt])


class AttachAdapterTests(unittest.TestCase):
    def setUp(self):
        _require_fixture()

    def _module(self):
        from forecasting.agent_forecasting import predispatch_td_abs as module
        return module

    def _base(self):
        return _input(metadata={"region": "NSW1", "input_profile": "legacy_price_demand_calendar"})

    def test_attach_adds_only_the_typed_future_field(self):
        module = self._module()
        series = module.build_future_series(FIXTURE_ORIGIN_UNIX, [build_fixture_receipt()])
        base = self._base()
        candidate = module.attach_future_forecast(base, series)
        self.assertEqual(set(candidate.future_forecasts), {module.COVARIATE_NAME})
        self.assertEqual(candidate.past_covariates, base.past_covariates)
        self.assertEqual(candidate.future_covariates, base.future_covariates)
        self.assertEqual(candidate.price_history, base.price_history)
        self.assertNotEqual(candidate.input_hash(), base.input_hash())
        self.assertEqual(candidate.metadata["source_manifest_sha256"], series.series_hash)
        self.assertTrue(candidate.metadata["input_profile"].endswith(module.CANDIDATE_ID))

    def test_attach_refuses_origin_mismatch_and_second_manifest(self):
        module = self._module()
        series = module.build_future_series(FIXTURE_ORIGIN_UNIX, [build_fixture_receipt()])
        base = self._base()
        wrong_origin = _input(origin_unix=FIXTURE_ORIGIN_UNIX + 3600,
                              target_time_unix=tuple(FIXTURE_ORIGIN_UNIX + 3600 + 3600 * h for h in range(24)),
                              metadata={"region": "NSW1"})
        with self.assertRaisesRegex(ValueError, "origin"):
            module.attach_future_forecast(wrong_origin, series)
        attached = module.attach_future_forecast(base, series)
        with self.assertRaisesRegex(ValueError, "second manifest"):
            module.attach_future_forecast(attached, series)


class ToolMappingTests(unittest.TestCase):
    def test_model_input_encodes_honest_absence_for_typed_future(self):
        from forecasting.agent_forecasting.frozen_chronos import build_model_input
        values = tuple(6000.0 + h for h in range(24))
        request = _input(future_forecasts={"predispatchis_totaldemand_mw": values})
        model_input = build_model_input(request)
        past, future = model_input["past_covariates"], model_input["future_covariates"]
        self.assertEqual(set(past), set(future))  # chronos-2 requires future keys ⊆ past keys
        import math
        self.assertTrue(all(math.isnan(v) for v in past["predispatchis_totaldemand_mw"]))
        self.assertEqual(len(past["predispatchis_totaldemand_mw"]), 72)
        self.assertEqual(list(future["predispatchis_totaldemand_mw"]), list(values))

    def test_model_input_unchanged_without_typed_future(self):
        from forecasting.agent_forecasting.frozen_chronos import build_model_input
        request = _input()
        model_input = build_model_input(request)
        self.assertEqual(len(model_input["past_covariates"]), 0)
        self.assertEqual(len(model_input["future_covariates"]), 0)
        self.assertEqual(model_input["target"].shape, (72,))


class ReceiptStoreTests(unittest.TestCase):
    def setUp(self):
        _require_fixture()

    def test_store_reload_reverifies_and_rejects_tamper(self):
        import tempfile

        module = self._module_for_store()
        body = build_body_bytes()
        import hashlib
        digest = hashlib.sha256(body).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bodies").mkdir()
            (root / "bodies" / f"{digest}.zip").write_bytes(body)
            event = {"event": "body_complete", "filename": FIXTURE_FILENAME,
                     "body_sha256": digest, "body_path": str(root / "bodies" / f"{digest}.zip"),
                     "receipt_complete_unix_ceil": FIXTURE_ORIGIN_UNIX - 600}
            (root / "events.jsonl").write_text(json.dumps(event) + "\n")
            receipts, events = module.load_receipt_store(root / "events.jsonl")
            self.assertEqual(len(receipts), 1)
            self.assertEqual(receipts[0].basis, module.REAL_BASIS)
            tampered = bytearray(body)
            tampered[-40] ^= 0x01
            (root / "bodies" / f"{digest}.zip").write_bytes(bytes(tampered))
            with self.assertRaises(ValueError):
                module.load_receipt_store(root / "events.jsonl")

    def _module_for_store(self):
        from forecasting.agent_forecasting import predispatch_td_abs as module
        return module


if __name__ == "__main__":
    unittest.main()
