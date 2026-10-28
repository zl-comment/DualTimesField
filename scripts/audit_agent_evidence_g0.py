#!/usr/bin/env python3
"""Bounded AEMO notice acquisition; no forecasts, labels, or fitted models.

Collect only after the protocol and this script have been committed.  Directory
dates are unverified display-clock values, not asserted publication timestamps.
Finalization requires explicit, source-backed human chain adjudication; regex
hints never determine the final semantic category.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import html
import json
import re
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/agent_forecasting/g0_20261028.json"
BRANCH = "research/agent-evidence-contract-g0-20261028"
OUTPUT = ROOT / "logs/forecasting/agent_evidence_contract_g0_20261028/evidence"
CURRENT = "https://nemweb.com.au/Reports/Current/Market_Notice/"
ARCHIVE = "https://nemweb.com.au/Reports/Archive/Market_Notice/"
START, END = datetime(2026, 10, 21), datetime(2026, 10, 28)
PRIORITY = ("withdrawal", "postponement", "conflict", "revision", "cross_day", "initial")
LIMITS = dict(requests=160, total_bytes=16 * 1024**2, response_bytes=256 * 1024,
              candidates=120, predecessors=36, chain_members=6, lookback_days=30,
              wall_seconds=600)
REQUEST_TIMEOUT = 15
REQUEST_INTERVAL = 1
MAX_ATTEMPTS = 2
REFERENCE = re.compile(r"\b(?:market\s+notice|notice|MN)\s*(?:number|no\.?|id|#|:)?\s*(\d{5,8})\b", re.I)
EVENT_ID = re.compile(r"\bCHG\d{5,12}\b", re.I)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_freeze(commit: str, protocol: Path) -> dict:
    protocol = protocol.resolve()
    relative = protocol.relative_to(ROOT).as_posix()
    if git("rev-parse", "HEAD") != commit:
        raise ValueError("HEAD must equal the explicitly supplied frozen commit")
    if git("branch", "--show-current") != BRANCH:
        raise ValueError("The frozen evidence-audit branch is required")
    config_relative = CONFIG.resolve().relative_to(ROOT).as_posix()
    for name in (relative, config_relative, Path(__file__).resolve().relative_to(ROOT).as_posix()):
        subprocess.run(["git", "cat-file", "-e", f"{commit}:{name}"], cwd=ROOT, check=True)
        subprocess.run(["git", "diff", "--exit-code", commit, "--", name], cwd=ROOT, check=True)
    validate_config(json.loads(CONFIG.read_text(encoding="utf-8")))
    return dict(commit=commit, protocol=relative, protocol_sha256=sha(protocol.read_bytes()),
                config=config_relative, config_sha256=sha(CONFIG.read_bytes()), branch=BRANCH,
                script_sha256=sha(Path(__file__).read_bytes()))


def validate_config(config: dict) -> None:
    expected = dict(directory_display_start=START.date().isoformat(),
                    directory_display_end_exclusive=END.date().isoformat(), display_timezone="unverified",
                    max_primary_candidates=LIMITS["candidates"], max_predecessors=LIMITS["predecessors"],
                    max_chain_members=LIMITS["chain_members"],
                    lookback_start=(START - timedelta(days=LIMITS["lookback_days"])).date().isoformat(),
                    class_priority=list(PRIORITY), per_class=2, max_chains=12,
                    max_http_requests=LIMITS["requests"], max_total_bytes=LIMITS["total_bytes"],
                    max_response_bytes=LIMITS["response_bytes"], max_wall_seconds=LIMITS["wall_seconds"],
                    per_request_timeout_seconds=REQUEST_TIMEOUT,
                    request_interval_seconds=REQUEST_INTERVAL, max_attempts_per_url=MAX_ATTEMPTS)
    if config.get("branch") != BRANCH or config.get("status") != "frozen_g0" or config.get("execution_scope") != "g0_only":
        raise ValueError("Config does not identify the frozen G0 branch and scope")
    actual = config.get("evidence", {})
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Frozen evidence config mismatch: {key}")


def validate_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in {"nemweb.com.au", "www.nemweb.com.au"}
            or parsed.username is not None or parsed.password is not None or parsed.port is not None
            or not re.fullmatch(r"/reports/(current|archive)/market_notice/(?:[A-Za-z0-9_-]+\.R\d+)?", parsed.path, re.I)
            or parsed.query or parsed.fragment):
        raise ValueError("Only the frozen official Market_Notice routes are allowed")


def parse_index(raw: bytes, source_url: str) -> list[dict]:
    """Parse directory metadata only; no inferred clock or publication date."""
    rows = []
    for fragment in re.split(r"<br\s*/?>|\n", raw.decode("utf-8", "replace"), flags=re.I):
        link = re.search(r'<a\s+href="([^"]+)"[^>]*>([^<]+)</a>', fragment, re.I)
        if not link:
            continue
        notice = re.search(r"\.R(\d+)$", link.group(1), re.I)
        if not notice:
            continue
        prefix = html.unescape(re.sub(r"<[^>]+>", "", fragment[:link.start()])).strip()
        match = re.fullmatch(r"(.+?\b[AP]M)\s+(\d+)", prefix, re.I)
        if not match:
            raise ValueError(f"Unrecognized index timestamp/size format for {notice.group(1)}")
        literal = " ".join(match.group(1).split())
        displayed = datetime.strptime(literal, "%A, %B %d, %Y %I:%M %p")
        url = urllib.parse.urljoin(source_url, html.unescape(link.group(1)))
        validate_url(url)
        filename_date = re.search(r"_(\d{8})\.R\d+$", url, re.I)
        rows.append(dict(notice_id=int(notice.group(1)), url=url,
                         display_time_literal=literal, display_time_naive=displayed.isoformat(),
                         clock_status="display_clock_unverified", listed_bytes=int(match.group(2)),
                         filename_date=filename_date.group(1) if filename_date else None,
                         source_index=source_url))
    return sorted(rows, key=lambda row: (row["notice_id"], row["url"]))


def window_candidates(rows: list[dict]) -> list[dict]:
    return sorted((row for row in rows
                   if START <= datetime.fromisoformat(row["display_time_naive"]) < END),
                  key=lambda row: row["notice_id"])


class BudgetExceeded(RuntimeError):
    pass


class AcquisitionTimeout(TimeoutError):
    pass


@contextmanager
def hard_deadline(seconds: float):
    """Absolute main-thread deadline, including a continuously streaming body."""
    def expired(_signum, _frame):
        raise AcquisitionTimeout("Absolute acquisition deadline reached")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise RuntimeError("Run acquisition in a standalone process without an existing alarm")
    if seconds <= 0:
        raise BudgetExceeded("Global acquisition deadline reached")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Fetcher:
    def __init__(self, output: Path):
        self.output, self.calls, self.bytes = output, [], 0
        self.last_request = 0.0
        self.started = time.monotonic()
        self.opener = urllib.request.build_opener(NoRedirect())
        (output / "raw").mkdir()

    def get(self, url: str, key: str) -> tuple[bytes | None, dict]:
        validate_url(url)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError("Invalid capture key")
        prior_calls = [call for call in self.calls if call["url"] == url]
        if len(prior_calls) >= MAX_ATTEMPTS:
            return None, prior_calls[-1]
        for attempt in range(len(prior_calls) + 1, MAX_ATTEMPTS + 1):
            if (len(self.calls) >= LIMITS["requests"] or self.bytes >= LIMITS["total_bytes"]
                    or time.monotonic() - self.started >= LIMITS["wall_seconds"]):
                raise BudgetExceeded("Global acquisition budget reached")
            remaining_time = LIMITS["wall_seconds"] - (time.monotonic() - self.started)
            time.sleep(min(remaining_time, max(0.0, REQUEST_INTERVAL - (time.monotonic() - self.last_request))))
            remaining_time = LIMITS["wall_seconds"] - (time.monotonic() - self.started)
            if remaining_time <= 0:
                raise BudgetExceeded("Global acquisition deadline reached")
            self.last_request = time.monotonic()
            record = dict(url=url, attempt=attempt,
                          fetched_at_utc=datetime.now(timezone.utc).isoformat(),
                          publication_time_verified=False, status=None, accepted=False)
            raw = bytearray()
            charged_before = self.bytes
            started = time.monotonic()
            try:
                request = urllib.request.Request(url, headers={"Accept-Encoding": "identity",
                    "User-Agent": "AEMO-public-notice-path-audit/1.0"})
                deadline = min(REQUEST_TIMEOUT, remaining_time)
                with hard_deadline(deadline), self.opener.open(request, timeout=deadline) as response:
                    validate_url(response.url)
                    record.update(status=response.status, final_url=response.url, deadline_seconds=deadline,
                                  headers={k: response.headers.get(k) for k in ("Date", "Last-Modified", "ETag",
                                           "Content-Type", "Content-Length", "Transfer-Encoding")})
                    cap = min(LIMITS["response_bytes"], LIMITS["total_bytes"] - self.bytes)
                    length = response.headers.get("Content-Length", "")
                    declared = int(length) if str(length).isdigit() and not response.headers.get("Transfer-Encoding") else None
                    complete = declared == 0
                    while len(raw) < cap and not complete:
                        request_size = min(8192, cap - len(raw))
                        # Charge the complete in-flight allowance first. If interrupted,
                        # conservatively retain it so retries cannot exceed the budget.
                        self.bytes += request_size
                        block = response.read1(request_size)
                        self.bytes -= request_size - len(block)
                        raw.extend(block)
                        complete = not block or (declared is not None and len(raw) == declared)
                    if declared is not None and len(raw) != declared:
                        complete = False
                    if not complete:
                        raise BudgetExceeded("Body reached its byte limit or ended incomplete; no extra byte read")
                filename = f"raw/{key}_attempt{attempt}.bin"
                with (self.output / filename).open("xb") as handle:
                    handle.write(raw)
                record.update(path=filename, bytes=len(raw), sha256=sha(raw), accepted=True, complete=True)
            except BudgetExceeded as error:
                record.update(accepted=False, error=str(error), complete=False)
                raise
            except urllib.error.HTTPError as error:
                record.update(accepted=False, status=error.code,
                              error=f"HTTPError: {error.code}; redirects disabled",
                              headers={key: error.headers.get(key) for key in
                                       ("Date", "Last-Modified", "ETag", "Content-Type", "Content-Length")})
            except (OSError, ValueError) as error:
                record.update(accepted=False, error=f"{type(error).__name__}: {error}")
            finally:
                record.update(bytes_received=len(raw), bytes_charged=self.bytes - charged_before,
                              elapsed_seconds=time.monotonic() - started)
                if not record.get("accepted", False):
                    filename = f"raw/{key}_attempt{attempt}.partial.bin"
                    with (self.output / filename).open("xb") as handle:
                        handle.write(raw)
                    record.update(partial_path=filename, partial_bytes=len(raw), partial_sha256=sha(raw))
                self.calls.append(record)
            if record["accepted"]:
                return bytes(raw), record
        return None, record


def notice_record(row: dict, raw: bytes, provenance: dict, candidate: bool) -> dict:
    text = raw.decode("utf-8", "replace")
    return dict(**row, candidate=candidate, raw_path=provenance["path"], raw_sha256=sha(raw),
                text=text, explicit_reference_hints=sorted(set(map(int, REFERENCE.findall(text)))),
                stable_event_id_hints=sorted(set(x.upper() for x in EVENT_ID.findall(text))),
                source_clock_review="pending_manual_review", event_start=None, event_end=None,
                semantic_category="unadjudicated")


def collect(protocol: Path, commit: str) -> dict:
    frozen = verify_freeze(commit, protocol)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    fetcher, records, missing = Fetcher(OUTPUT), {}, []
    status, all_rows, candidates, archive_rows = "complete_within_budget", [], [], []
    try:
        raw, _ = fetcher.get(CURRENT, "index_current")
        if raw is None:
            raise RuntimeError("Current notice directory unavailable")
        all_rows = parse_index(raw, CURRENT)
        if not all_rows:
            raise RuntimeError("Current directory has no parseable notice files")
        archive, _ = fetcher.get(ARCHIVE, "index_archive")
        if archive is not None:
            archive_rows = parse_index(archive, ARCHIVE)
        indexed = {}
        for row in all_rows:
            if row["notice_id"] in indexed:
                raise ValueError("Duplicate notice ID in index; do not choose a version silently")
            indexed[row["notice_id"]] = row
        candidates = window_candidates(all_rows)
        write_json(OUTPUT / "candidate_index.json", dict(
            window_start_display=START.isoformat(), window_end_display_exclusive=END.isoformat(),
            clock_status="display_clock_unverified", all_index_rows=all_rows,
            archive_index_rows=archive_rows, candidates=candidates,
            inspect_candidate_ids=[r["notice_id"] for r in candidates[:LIMITS["candidates"]]]))
        for row in candidates[:LIMITS["candidates"]]:
            body, provenance = fetcher.get(row["url"], f"notice_{row['notice_id']}")
            if body is None:
                missing.append(dict(notice_id=row["notice_id"], reason="candidate_fetch_failed"))
                continue
            records[row["notice_id"]] = notice_record(row, body, provenance, True)
        # Only backward, explicitly numbered references are fetched. No future notices.
        extra = 0
        for seed in [r["notice_id"] for r in candidates[:LIMITS["candidates"]]]:
            queue, visited = [seed], set()
            while queue:
                current_id = queue.pop(0)
                if current_id in visited or current_id not in records:
                    continue
                visited.add(current_id)
                for prior_id in records[current_id]["explicit_reference_hints"]:
                    if prior_id >= current_id or prior_id in visited or prior_id in queue:
                        continue
                    prior = indexed.get(prior_id)
                    reason = None
                    if len(visited) + len(queue) >= LIMITS["chain_members"]:
                        reason = "chain_member_cap"
                    elif prior is None:
                        reason = "not_in_current_index_archive_not_used_as_history_proof"
                    elif datetime.fromisoformat(prior["display_time_naive"]) < START - timedelta(days=LIMITS["lookback_days"]):
                        reason = "lookback_cap"
                    elif datetime.fromisoformat(prior["display_time_naive"]) >= END:
                        reason = "outside_frozen_window_end"
                    elif prior_id not in records and extra >= LIMITS["predecessors"]:
                        reason = "predecessor_cap"
                    if reason:
                        missing.append(dict(from_notice_id=current_id, notice_id=prior_id, reason=reason))
                        continue
                    if prior_id not in records:
                        body, provenance = fetcher.get(prior["url"], f"notice_{prior_id}")
                        extra += 1
                        if body is None:
                            missing.append(dict(from_notice_id=current_id, notice_id=prior_id,
                                                reason="predecessor_fetch_failed"))
                            continue
                        records[prior_id] = notice_record(prior, body, provenance, False)
                    if prior_id not in visited and prior_id not in queue:
                        queue.append(prior_id)
        if len(candidates) > LIMITS["candidates"] or missing:
            status = "partial"
    except (OSError, ValueError, RuntimeError) as error:
        status = "partial"
        missing.append(dict(reason=f"{type(error).__name__}: {error}"))
    finally:
        write_json(OUTPUT / "fetch_manifest.json", fetcher.calls)
        write_json(OUTPUT / "notice_records.json", sorted(records.values(), key=lambda r: r["notice_id"]))
        report = dict(status=status, frozen=frozen, limits=LIMITS, requests=len(fetcher.calls),
                      bytes_charged=fetcher.bytes,
                      bytes_received=sum(call["bytes_received"] for call in fetcher.calls),
                      candidate_count=len(candidates),
                      fetched_candidate_count=sum(r["candidate"] for r in records.values()),
                      predecessor_count=sum(not r["candidate"] for r in records.values()), missing=missing,
                      archive_file_count=len(archive_rows), clock_status="display_clock_unverified",
                      historical_publication_availability_proven=False,
                      sampling_frame_complete=False, predictions_run=False, labels_loaded=False,
                      selection_status="awaiting_source_backed_manual_review")
        report["artifact_sha256"] = {name: sha((OUTPUT / name).read_bytes()) for name in
                                    ("candidate_index.json", "fetch_manifest.json", "notice_records.json")
                                    if (OUTPUT / name).exists()}
        write_json(OUTPUT / "collection.json", report)
        write_json(OUTPUT / "review_template.json", dict(chains=[], excluded=[],
            instructions="Partition every fetched candidate into one chain or an evidenced exclusion. "
            "Each chain has notice_ids, evidence=[{category,notice_id,quote,reason}], "
            "edges=[{from_notice_id,to_notice_id,quote,reason}], missing_notice_ids, "
            "source_clock_review, event_time_review. Categories use frozen priority; "
            "conflict means unresolved simultaneous incompatible claims, not a normal revision. "
            "Initial means observed initial only. Never infer absence from missing history."))
    return report


def finalize(review_path: Path) -> dict:
    """Source-verified manual coding followed by deterministic quota selection."""
    review_path = review_path.resolve()
    review_path.relative_to(OUTPUT.resolve())
    collection = json.loads((OUTPUT / "collection.json").read_text())
    if sha(Path(__file__).read_bytes()) != collection["frozen"]["script_sha256"]:
        raise ValueError("Acquisition script changed after freeze")
    for name, expected in collection["artifact_sha256"].items():
        if sha((OUTPUT / name).read_bytes()) != expected:
            raise ValueError(f"Collection artifact changed: {name}")
    records = {r["notice_id"]: r for r in json.loads((OUTPUT / "notice_records.json").read_text())}
    for entry in json.loads((OUTPUT / "fetch_manifest.json").read_text()):
        if entry["accepted"] and sha((OUTPUT / entry["path"]).read_bytes()) != entry["sha256"]:
            raise ValueError("Raw source hash mismatch")
        if "partial_path" in entry and sha((OUTPUT / entry["partial_path"]).read_bytes()) != entry["partial_sha256"]:
            raise ValueError("Partial source hash mismatch")
    for record in records.values():
        raw = (OUTPUT / record["raw_path"]).read_bytes()
        if sha(raw) != record["raw_sha256"] or raw.decode("utf-8", "replace") != record["text"]:
            raise ValueError("Decoded notice is not the archived raw source")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    candidates = {key for key, record in records.items() if record["candidate"]}
    accounted, chains, all_members = set(), [], set()
    for chain in review["chains"]:
        members = set(chain["notice_ids"])
        if not members or not members <= records.keys() or members & all_members:
            raise ValueError("Invalid, overlapping, or unavailable chain members")
        all_members.update(members)
        inside = members & candidates
        if not inside:
            raise ValueError("Every reviewed chain must contain a fetched candidate")
        categories = set()
        for evidence in chain["evidence"]:
            if (evidence["category"] not in PRIORITY or evidence["notice_id"] not in members
                    or not evidence["quote"] or evidence["quote"] not in records[evidence["notice_id"]]["text"]
                    or not evidence["reason"].strip()):
                raise ValueError("Each category needs an exact source quote and explicit reason")
            categories.add(evidence["category"])
        if not categories:
            raise ValueError("Chain has no substantiated category")
        adjacent = {key: set() for key in members}
        for edge in chain["edges"]:
            newer, older = edge["from_notice_id"], edge["to_notice_id"]
            if (newer not in members or older not in members or newer == older
                    or not edge["quote"] or edge["quote"] not in records[newer]["text"]
                    or not edge["reason"].strip()):
                raise ValueError("Every chain link needs exact source evidence")
            adjacent[newer].add(older)
            adjacent[older].add(newer)
        connected, queue = set(), [min(members)]
        while queue:
            key = queue.pop()
            if key not in connected:
                connected.add(key)
                queue.extend(adjacent[key] - connected)
        if connected != members:
            raise ValueError("Chain is not connected by explicitly reviewed links")
        for required in ("missing_notice_ids", "source_clock_review", "event_time_review"):
            if required not in chain:
                raise ValueError(f"Missing required review field: {required}")
        accounted.update(inside)
        primary = next(category for category in PRIORITY if category in categories)
        chains.append(dict(**chain, primary_category=primary, anchor_notice_id=min(inside),
                           over_member_cap=len(members) > LIMITS["chain_members"],
                           chain_complete=False if chain["missing_notice_ids"] else None))
    for exclusion in review["excluded"]:
        key = exclusion["notice_id"]
        if (key not in candidates or key in accounted or not exclusion["reason"].strip()
                or not exclusion["quote"] or exclusion["quote"] not in records[key]["text"]):
            raise ValueError("Invalid or duplicate candidate exclusion")
        accounted.add(key)
    if accounted != candidates:
        raise ValueError("Review must account for every successfully fetched candidate")
    selected, counts = [], {category: 0 for category in PRIORITY}
    for chain in sorted(chains, key=lambda item: item["anchor_notice_id"]):
        category = chain["primary_category"]
        # A long/incomplete early chain is retained; do not replace it with an
        # easier later chain. The member cap limits backward acquisition per seed.
        chain["selected"] = counts[category] < 2
        if chain["selected"]:
            counts[category] += 1
            selected.append(chain["anchor_notice_id"])
    result = dict(priority=list(PRIORITY), category_counts=counts, selected_anchor_ids=selected,
                  selected_chain_count=len(selected), chains=chains, excluded=review["excluded"],
                  manual_review_sha256=sha(review_path.read_bytes()),
                  clock_status="display_clock_unverified", historical_publication_availability_proven=False,
                  zero_count_interpretation="not_observed_under_cap_or_current_index; not proof of absence",
                  collection_status=collection["status"], predictions_run=False, labels_loaded=False)
    write_json(OUTPUT / "chain_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("collect")
    capture.add_argument("--protocol", type=Path, required=True)
    capture.add_argument("--frozen-commit", required=True)
    finish = commands.add_parser("finalize")
    finish.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.protocol, args.frozen_commit) if args.command == "collect" else finalize(args.review)
    print(json.dumps({key: value for key, value in result.items() if key != "chains"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
