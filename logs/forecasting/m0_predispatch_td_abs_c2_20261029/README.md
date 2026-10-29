# M0 candidate 2 evidence — predispatchis_absolute_totaldemand_future_v1

Branch `research/agentic-asof-abs-predispatch-c2-20261029-r4` (local only).
Commits: RED `5039cdb` → GREEN `97d7aa7` → EVIDENCE (this commit).
Protocol: `docs/research/agent_forecasting/m0_predispatch_td_abs_protocol_20261029.txt`.
Config: `configs/agent_forecasting/m0_predispatch_td_abs_c2_20261029.json`.

## What was delivered

1. **Real raw-body receipt capture** (`capture/`): 1 directory listing + 2 exact
   ZIP GETs from the official CURRENT PredispatchIS_Reports source (zero
   retries, 15 s timeouts, 2,451,206 body bytes total), each body stored
   write-once under its SHA-256 with an in-process body-complete receipt
   (UTC wall clock at EOF ceiled to the next second, monotonic clock, clock
   health).  Bodies:
   - `PUBLIC_PREDISPATCHIS_202610291800_20261029173204.zip` —
     `8cf00421fa44afd031b3d186a1c544bf6b5f0ccc3ee5257182829ba25bdf925f`,
     receipt_complete ceil `1793260190`.
   - `PUBLIC_PREDISPATCHIS_202610291730_20261029170208.zip` —
     `bee5e79a1dfc374d6df053569f586941be2b2cacfbce66275519ec1688cac65b`
     (byte-identical to the upstream research node's independent download of
     the same publication), receipt_complete ceil `1793260200`.
2. **Frozen source identity** discovered from these authenticated bytes:
   table `PREDISPATCH,REGION_SOLUTION,9`, 131-field I header, 127 frozen
   column names, header-line SHA-256
   `2f013ba453d6ef05959d74895fa6db6f9cd2522b6c0f7d4356af09fadaaa529a` —
   the same hash the upstream node derived from a separate download.
3. **Raw-bound parser + typed adapter**
   (`forecasting/agent_forecasting/predispatch_td_abs.py`): receipts exist
   only via `receipt_from_raw_body` (first_seen) or
   `synthetic_receipt_from_raw_body` (fixture); both re-parse the immutable
   bytes on every (re)construction; `dataclasses.replace` cannot flip basis,
   rows or hashes; duplicates fold to the earliest receipt; same-vintage
   different-body forks fail closed; newest-visible-only selection with no
   fallback; exact 48 half-hour ends per origin with deterministic pair
   means and zero fill.
4. **Verified real series** (`verify/verify.json`): origin `1793260800`
   (2026-10-29T08:00Z == 18:00 AEST), selected vintage `2026102928/1`,
   receipt 610 s strictly before origin, 48/48 exact NSW1 half-hours,
   `basis=first_seen_receipt`, `strict_asof=STRICT_ASOF_VERIFIED`,
   series SHA-256 `a92d99cfe3e97590baad318daa2ecdd62156c8316d5658daf07bebe2ab608bb4`.
5. **Typed contract extension** (`contracts.py`): `future_forecasts` allowlisted
   to `predispatchis_totaldemand_mw` (24 finite MW), exempt from the
   same-named-past rule; the name cannot appear in `past_covariates`.
   Tool encoding (`frozen_chronos.build_model_input`): Chronos-2 rejects a
   future-only covariate name (probe recorded in
   `response_check/response_check.json`) and masks NaN context natively, so
   the honest absence is an all-NaN 72-step past — no fabricated history.
6. **Paired deterministic response** (`response_check/`): base x2 + candidate x2
   in fresh child processes, frozen settings, CPU-only, offline.
   - input hashes differ: `6d7885d5…` vs `9f2ce552…`
   - both inputs reproduce within atol=rtol=1e-6
   - max|candidate − base| = **10.9917 MW > 1e-6** on the 24x5 native quantiles
   - model state hash unchanged before/after; checkpoint hashes stable;
     training calls 0; 4 calls, 45.66 s total.
   - candidate tensor basis: `synthetic_fixture` (PREDISPATCHIS-format
     fixture bytes, parser-verified; receipt at origin−600 s).

## RED → GREEN record (`red_green/`)

- `01_baseline_agent_suite.txt` — base HEAD: 39 passed + 89 subtests, EXIT 0.
- `03_red_suite.txt` — RED commit `5039cdb`: 29 failed / 41 passed, EXIT 1.
- `04_green_suite.txt` — GREEN attempt 1: 3 failed (test-side expectation
  bugs: first-scope-row end and trailer retotaling after mutation), EXIT 1.
- `04_green_suite_attempt2.txt` — after fixing the TESTS only (module
  unchanged): 70 passed + 89 subtests, EXIT 0.  Both attempt logs are
  preserved at unique paths; no log path was ever overwritten.

## Honest disclosures

1. **Capture 02 display bug (corrected offline):** the first capture run's
   `discovery_*.json` derived display-only origin numbers with an inverted
   AEST sign (+10 h instead of −10 h).  Bodies, hashes, receipts and the
   event ledger were never affected.  `02b_rediscover_offline.txt`
   regenerated the discovery files from the stored bodies with the correct
   conversion, using no network.  All authoritative values come from the
   frozen parser (`05_verify_run.txt`), not from discovery output.
2. **Input-hash schema change:** `to_dict()` now includes the (possibly
   empty) `future_forecasts` mapping, so the base input hash here
   (`6d7885d5…`) differs from pre-extension G0-era records even though the
   base input VALUES are unchanged (same pinned row 72 files and decode).
3. **Model-facing candidate uses the synthetic fixture by design:** the real
   captured vintage (2026-10-29) cannot cover the frozen 2022 development
   origin; `attach_future_forecast` refuses origin mismatch, and no
   fabricated 2022 vintage was created to bridge them.  The real-body path
   is fully exercised by the capture + verify evidence.
4. RED commit message contains a typo ("TOTALDIs"); left unamended to keep
   strictly linear history under the no-rebase constraint.
5. **Response-check script placement:** `scripts/check_m0_predispatch_td_response.py`
   gained its CPU/offline environment guards (lines setting CUDA_VISIBLE_DEVICES/
   HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE for child inheritance) after the GREEN
   commit; run `06_response_check` executed exactly this version (argv recorded
   in its log), and it is committed with this EVIDENCE commit.
6. **Manifest snapshot boundary:** `artifact_manifest.json` (built by run 08b)
   hashes every file present at build time; the only post-snapshot write is the
   single `commands.jsonl` line appended by run 08b's own logger invocation.
   Failed manifest attempt 07 (f-string syntax error) is preserved at its own
   unique log path alongside the successful 07b/08b rebuilds.

## Budget accounting

- Network: 1 listing + 2 ZIP GETs (≤ 16 MiB cap; 2.45 MiB used), 15 s
  timeouts, ≥ 1 s interval, 0 retries.  The 02b re-discovery was offline.
- Model: 4 local CPU calls (≤ 300 s each; 45.7 s used total), 2 CPU
  threads, offline, CUDA hidden; 0 GPU, 0 training, 0 downloads, 0 external
  LLM calls, 0 labels, 0 scores.
- Tests: repo interpreter with pytest; no model import anywhere in tests.

## Prohibitions honored

No labels or score computation, no fitting/calibration, no model download,
no GPU, no G1/G2, no collector started, no push/merge/rebase/reset, no
branch switch, no writes outside this worktree.  The main checkout was
never modified.
