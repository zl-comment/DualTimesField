"""Synthetic G0 acquisition/review tests; never access a network or model."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from scripts import audit_agent_evidence_g0 as audit


def frozen_config():
    return dict(status="frozen_g0", execution_scope="g0_only", branch=audit.BRANCH,
        evidence=dict(directory_display_start="2026-10-21", directory_display_end_exclusive="2026-10-28",
            display_timezone="unverified", max_primary_candidates=120, max_predecessors=36,
            max_chain_members=6, lookback_start="2026-09-21",
            class_priority=["withdrawal", "postponement", "conflict", "revision", "cross_day", "initial"],
            per_class=2, max_chains=12, max_http_requests=160, max_total_bytes=16777216,
            max_response_bytes=262144, max_wall_seconds=600, per_request_timeout_seconds=15,
            request_interval_seconds=1, max_attempts_per_url=2))


class Response:
    def __init__(self, body, length=None):
        self.buffer = io.BytesIO(body)
        self.headers = {} if length is None else {"Content-Length": str(length)}
        self.url, self.status, self.read_sizes = audit.CURRENT, 200, []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read1(self, size):
        self.read_sizes.append(size)
        return self.buffer.read(size)


class SlowlyStreamingResponse(Response):
    def read1(self, size):
        time.sleep(0.008)
        self.read_sizes.append(size)
        return b"x"


class EvidenceIndexTests(unittest.TestCase):
    def test_literal_window_boundaries_and_id_order_without_timezone_inference(self):
        fragments = []
        for literal, notice_id, filedate in (
            ("Wednesday, October 28, 2026 12:00 AM", 4, "20261027"),
            ("Wednesday, October 21, 2026 12:00 AM", 3, "20261020"),
            ("Tuesday, October 20, 2026 11:59 PM", 2, "20261021"),
            ("Tuesday, October 27, 2026 11:59 PM", 1, "20261028"),
        ):
            fragments.append(f'{literal} 14 <A HREF="/Reports/CURRENT/Market_Notice/NEMITWEB1_MKTNOTICE_{filedate}.R{notice_id}">file</A>')
        rows = audit.parse_index("<br>".join(fragments).encode(), audit.CURRENT)
        selected = audit.window_candidates(rows)
        self.assertEqual([row["notice_id"] for row in selected], [1, 3])
        self.assertTrue(all(row["clock_status"] == "display_clock_unverified" for row in rows))
        self.assertEqual(selected[1]["display_time_naive"], "2026-10-21T00:00:00")
        self.assertEqual(selected[1]["filename_date"], "20261020")

    def test_routes_reject_other_products_traversal_credentials_and_queries(self):
        forbidden = [
            "http://nemweb.com.au/Reports/Current/Market_Notice/",
            "https://evil.example/Reports/Current/Market_Notice/",
            "https://nemweb.com.au/Reports/Current/Dispatch_Reports/",
            "https://nemweb.com.au/Reports/Current/Market_Notice/../Dispatch_Reports/",
            "https://nemweb.com.au/Reports/Current/Market_Notice/%2e%2e/x.R1",
            "https://user@nemweb.com.au/Reports/Current/Market_Notice/",
            "https://nemweb.com.au:443/Reports/Current/Market_Notice/",
            audit.CURRENT + "?other=prices", audit.CURRENT + "#fragment",
        ]
        with tempfile.TemporaryDirectory() as directory:
            fetcher = audit.Fetcher(Path(directory))
            fetcher.opener = Mock()
            for url in forbidden:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    fetcher.get(url, "test")
            self.assertEqual(fetcher.calls, [])
            fetcher.opener.open.assert_not_called()
        audit.validate_url(audit.CURRENT + "NEMITWEB1_MKTNOTICE_20261021.R123456")

    def test_config_and_branch_are_part_of_freeze_contract(self):
        audit.validate_config(frozen_config())
        for section, key, value in (("evidence", "display_timezone", "AEST"),
                                    ("evidence", "per_request_timeout_seconds", 20),
                                    ("evidence", "max_wall_seconds", 601)):
            changed = frozen_config()
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.validate_config(changed)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, config, script = root / "protocol.txt", root / "config.json", root / "script.py"
            protocol.write_text("frozen protocol")
            config.write_text(json.dumps(frozen_config()))
            script.write_text("frozen source")
            commit = "a" * 40
            def git(*args):
                return commit if args == ("rev-parse", "HEAD") else audit.BRANCH
            with patch.multiple(audit, ROOT=root, CONFIG=config, __file__=str(script)), \
                    patch.object(audit, "git", side_effect=git), patch.object(audit.subprocess, "run") as run:
                result = audit.verify_freeze(commit, protocol)
                self.assertEqual(result["config_sha256"], audit.sha(config.read_bytes()))
                self.assertTrue(any(f"{commit}:config.json" in call.args[0] for call in run.call_args_list))
                with patch.object(audit, "git", side_effect=lambda *args: commit if args == ("rev-parse", "HEAD") else "wrong"):
                    with self.assertRaisesRegex(ValueError, "branch"):
                        audit.verify_freeze(commit, protocol)


class AcquisitionBudgetTests(unittest.TestCase):
    def get_fake(self, response, limits=None, prior_bytes=0):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fetcher = audit.Fetcher(Path(temporary.name))
        fetcher.bytes = prior_bytes
        fetcher.opener = Mock()
        fetcher.opener.open.return_value = response
        self.enterContext(patch.dict(audit.LIMITS, limits or {}))
        self.enterContext(patch.object(audit, "REQUEST_INTERVAL", 0))
        return fetcher

    def test_exact_cap_without_length_is_partial_without_extra_byte(self):
        response = Response(b"0123456789EXCESS")
        fetcher = self.get_fake(response, dict(response_bytes=10, total_bytes=10))
        with self.assertRaises(audit.BudgetExceeded):
            fetcher.get(audit.CURRENT, "cap")
        self.assertEqual(response.read_sizes, [10])
        self.assertEqual(fetcher.bytes, 10)
        entry = fetcher.calls[0]
        self.assertFalse(entry["accepted"])
        self.assertEqual((fetcher.output / entry["partial_path"]).read_bytes(), b"0123456789")

    def test_exact_cap_with_declared_complete_body_is_accepted(self):
        response = Response(b"0123456789", length=10)
        fetcher = self.get_fake(response, dict(response_bytes=10, total_bytes=10))
        body, entry = fetcher.get(audit.CURRENT, "complete")
        self.assertEqual(body, b"0123456789")
        self.assertTrue(entry["complete"])
        self.assertEqual(response.read_sizes, [10])
        self.assertEqual(fetcher.bytes, 10)

    def test_remaining_global_budget_is_not_exceeded(self):
        response = Response(b"0123456789", length=10)
        fetcher = self.get_fake(response, dict(response_bytes=8, total_bytes=10), prior_bytes=7)
        with self.assertRaises(audit.BudgetExceeded):
            fetcher.get(audit.CURRENT, "remaining")
        self.assertEqual(response.read_sizes, [3])
        self.assertEqual(fetcher.bytes, 10)
        self.assertEqual(fetcher.calls[0]["bytes_received"], 3)

    def test_absolute_deadline_stops_continuous_stream_and_restores_alarm(self):
        previous = signal.getsignal(signal.SIGALRM)
        response = SlowlyStreamingResponse(b"")
        fetcher = self.get_fake(response, dict(response_bytes=1000, total_bytes=1000, wall_seconds=1))
        with patch.object(audit, "REQUEST_TIMEOUT", 0.025), patch.object(audit, "MAX_ATTEMPTS", 1):
            started = time.monotonic()
            body, entry = fetcher.get(audit.CURRENT, "slow")
            elapsed = time.monotonic() - started
        self.assertIsNone(body)
        self.assertLess(elapsed, 0.3)
        self.assertIn("AcquisitionTimeout", entry["error"])
        self.assertGreaterEqual(entry["partial_bytes"], 1)
        self.assertLessEqual(entry["bytes_charged"], 1000)
        self.assertGreaterEqual(entry["bytes_charged"], entry["partial_bytes"])
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_global_deadline_prevents_a_retry(self):
        response = SlowlyStreamingResponse(b"")
        fetcher = self.get_fake(response, dict(response_bytes=1000, total_bytes=10000, wall_seconds=0.025))
        with patch.object(audit, "REQUEST_TIMEOUT", 1), self.assertRaises(audit.BudgetExceeded):
            fetcher.get(audit.CURRENT, "global")
        self.assertEqual(fetcher.opener.open.call_count, 1)
        self.assertEqual(len(fetcher.calls), 1)

    def test_failed_url_attempt_limit_is_shared_across_calls(self):
        fetcher = self.get_fake(Response(b""))
        fetcher.opener.open.side_effect = audit.urllib.error.URLError("synthetic failure")
        first_body, first_record = fetcher.get(audit.CURRENT, "first_seed")
        second_body, second_record = fetcher.get(audit.CURRENT, "second_seed")
        self.assertIsNone(first_body)
        self.assertIsNone(second_body)
        self.assertIs(first_record, second_record)
        self.assertEqual(fetcher.opener.open.call_count, 2)
        self.assertEqual([call["attempt"] for call in fetcher.calls], [1, 2])
        self.assertEqual(len(list((fetcher.output / "raw").iterdir())), 2)
        self.assertFalse(any((fetcher.output / "raw").glob("second_seed*")))


class FinalizeTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        self.enterContext(patch.object(audit, "OUTPUT", output))
        (output / "raw").mkdir()
        records, manifest, chains = [], [], []
        for class_index, category in enumerate(audit.PRIORITY):
            for item in range(3):
                key = 100000 + class_index * 10 + item
                text = f"Notice {key}: {category}; exact documentary evidence."
                path = f"raw/{key}.bin"
                raw = text.encode()
                (output / path).write_bytes(raw)
                records.append(dict(notice_id=key, candidate=True, raw_path=path, raw_sha256=audit.sha(raw), text=text))
                manifest.append(dict(accepted=True, path=path, sha256=audit.sha(raw)))
                chains.append(dict(notice_ids=[key], evidence=[dict(category=category, notice_id=key,
                    quote=category, reason="Synthetic semantic adjudication")], edges=[], missing_notice_ids=[],
                    source_clock_review="unverified", event_time_review="not available in synthetic fixture"))
        # An unresolved early chain must keep its quota position, not be replaced.
        chains[0]["missing_notice_ids"] = [99999]
        chains[0]["evidence"].append(dict(category="revision", notice_id=100000,
                                         quote="exact documentary evidence", reason="Secondary category"))
        audit.write_json(output / "notice_records.json", records)
        audit.write_json(output / "fetch_manifest.json", manifest)
        audit.write_json(output / "candidate_index.json", {"synthetic": True})
        artifact_names = ("notice_records.json", "fetch_manifest.json", "candidate_index.json")
        audit.write_json(output / "collection.json", dict(status="partial",
            frozen=dict(script_sha256=audit.sha(Path(audit.__file__).read_bytes())),
            artifact_sha256={name: audit.sha((output / name).read_bytes()) for name in artifact_names}))
        review = dict(chains=list(reversed(chains)), excluded=[])
        review_path = output / "manual_review.json"
        review_path.write_text(json.dumps(review))
        return output, review_path, review

    def test_priority_and_two_per_class_quota_are_order_independent(self):
        _, review_path, _ = self.fixture()
        result = audit.finalize(review_path)
        self.assertEqual(result["selected_chain_count"], 12)
        self.assertEqual(result["category_counts"], {category: 2 for category in audit.PRIORITY})
        self.assertEqual(result["selected_anchor_ids"], [100000 + group * 10 + item for group in range(6) for item in (0, 1)])
        first = next(chain for chain in result["chains"] if chain["anchor_notice_id"] == 100000)
        self.assertEqual(first["primary_category"], "withdrawal")
        self.assertFalse(first["chain_complete"])
        self.assertIn(100000, result["selected_anchor_ids"])

    def test_fabricated_quote_and_incomplete_review_are_rejected(self):
        output, review_path, review = self.fixture()
        changed = copy.deepcopy(review)
        changed["chains"][0]["evidence"][0]["quote"] = "fabricated quote"
        review_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "exact source quote"):
            audit.finalize(review_path)
        review["chains"].pop()
        review_path.write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError, "every successfully fetched"):
            audit.finalize(review_path)
        self.assertFalse((output / "chain_audit.json").exists())

    def test_raw_source_tampering_is_rejected(self):
        output, review_path, _ = self.fixture()
        (output / "raw/100000.bin").write_bytes(b"changed documentary evidence")
        with self.assertRaisesRegex(ValueError, "Raw source hash"):
            audit.finalize(review_path)


if __name__ == "__main__":
    unittest.main()
