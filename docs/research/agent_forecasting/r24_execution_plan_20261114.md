# R24 execution plan: iterative loop with staged attribution

Base: `273a4da` (R23 development result recorded). Branch: `research/r24-iterative-loop`.
Date frozen: 2026-11-14. This document is the pre-registered execution plan. The
development gate and evaluation protocol below must not be adjusted after results
are observed.

## Objective

Let Kev's next action depend on the previous round's evaluation inside one
prediction, and let matured cross-time evidence update later predictions, then
verify that these changes improve accuracy. The claim under test is not "more
rounds" but "the previous round's evaluation changes the next action and this
raises accuracy." Frozen LightGBM predictors; no retraining.

## What is reused, not rebuilt

- The R20 `IterativeToolHost` action loop: `retrieve_analogs:` cohort locking,
  `diagnose:` paired retrospective recompute, charged caching, budget accounting.
- `FeedbackLedger` cross-time maturation (`available_at = origin + 86400`).
- Budget limits, trace/manifest fingerprints, resume prefix checks, skills
  versioning (new `skills_r24`, version `8.0.0`), paired analysis loaders.

R23 deliberately hardened the free loop into a fixed protocol. R24 reopens
exactly three degrees of freedom and keeps every hard boundary.

## Design

### R24a — inner loop only (evidence machinery unchanged)

Freedoms returned to Kev, per round:

1. Which already-materialized candidate to diagnose next, and in what order.
2. Whether to supplement retrieval by appending episodes to the locked cohort
   (incremental recompute; comparisons only on the common successful episode
   intersection; a cohort replacement re-diagnoses everything and is recorded).
3. When to stop.

Kept hard and outside Kev's authority:

- Materialization of all operation-admitted candidates stays protocol-required.
  Candidate generation is cheap local inference; the scarce resources are Kev
  calls and historical-evaluation units. On-demand generation is deferred until
  the operation space is parameterized, and then implemented as a
  code-offered menu with budget, never free-form requests.
- Deterministic structured evidence admission (the R23 contract) and hard risk
  vetoes. The independent Stage C risk review stays at the endpoint only; risk
  signals do not enter the loop feedback, or the reviewer's independence is
  void.
- Loop feedback to Kev contains historical gains, support, and current
  candidate disagreement. Nothing else.

Frozen stop conditions: at most 3 rounds per origin; budget exhausted; one
round with no newly diagnosed candidate; Stage A already chose identity.
"Qualified candidate" at stop time = passes structured evidence admission AND
risk review approves. If none qualifies, identity is an explicit terminal
decision, never an error masquerading as a forecast.

### R24b — ledger extension (after R24a is evaluated)

Extend `FeedbackLedger` to record every diagnosed candidate per origin
(operation, forecast, market-state bucket), not only the committed one;
mature all of them at `available_at`. Two mandatory guards:

- Per (operation, state-bucket) support counts are reported in the evidence
  payload so exploration-density bias in self-generated evidence stays visible.
- Same-day partial maturation stays out of the evidence; only fully matured
  (+86400) records are visible. This is stated in the admission contract.

### Error semantics change

Infra failures (API transport, timeouts) are retried at the run level and the
origin is re-run. Protocol errors fail the origin loudly; there is no
`protocol_error -> identity` fallback path. Failed origins are removed from all
systems simultaneously in paired evaluation and reported as a count. Identity
appears only as an explicit decision (Stage A choice, risk revert, or no
qualified candidate at stop).

## Step 2 diagnosis and the pre-registered branch rule

Using only the saved R23 development run (materialized candidates, traces,
actuals), per region and region-equal, with QS as the primary metric:

- G (generation headroom) = QS(identity) − QS(oracle over all materialized candidates)
- A (admission loss)     = QS(oracle materialized) − QS(oracle over evidence-eligible)
- S (selection+risk loss) = QS(oracle eligible) − QS(committed)

Oracle selection is per-origin post-hoc minimization. It is a diagnostic upper
bound with selection bias, not a deployable score, and must never enter any
decision input. Branch rule, on region-equal values:

- G < 0.05 → the current operation space has no headroom; parameterized
  operations come first, and R24a runs only after the space is expanded.
- G ≥ 0.05 and S > A → decision quality is the bottleneck; proceed with R24a
  as specified.
- G ≥ 0.05 and A ≥ S → admission filtering is the bottleneck; re-examine the
  admission contract before building the loop.

## Evaluation protocol (pre-registered)

Development split 2022H2, identical origins/axes/frozen models/admission
artifact as R23. Systems compared:

| System | Question |
|---|---|
| identity | net benefit of the agent at all |
| R23 | does iteration beat the single-round protocol |
| R24-single-round | same new machinery, loop capped at one round (isolates iteration) |
| greedy fixed search | same op space and budget, similarity-ranked diagnostic order (isolates adaptive value) |
| random-order fixed search | same budget, random diagnostic order (isolates order effects) |
| R24a (and later R24b) | the actual candidates |

Variance control: the Kev client's sampling controls (or, if deterministic,
the run seed affecting tie-breaking) are run at three settings; the headline is
mean ± spread across the three. A single-seed improvement below the R23-era
magnitude (~0.01–0.03 QS) is not evidence.

Gate (frozen, applied to R24a and R24b separately, per region): QS and MAE
improve against identity and R23, RMSE not worse, coverage90 loss ≤ 0.005 —
the existing gate implementation in `analyze_r23_development.py` extended to
the R24 systems. Block-level deltas and a paired block bootstrap are reported
as supporting statistics, not gates.

Reported cost columns: rounds per origin, API calls per origin (P50/P95),
historical-evaluation units, identity-selection rate, wall time.

## Non-negotiable red lines

1. R24a and R24b are evaluated separately, each against the pre-registered
   gate; no combined-attribution jump.
2. Three sampling settings for the headline number.
3. Gate and protocol are frozen in this document before any R24 run; results
   never adjust them afterwards.
4. The formal 2023–2024 test split stays closed until the development gate
   passes and the full freeze manifest (models, operations and parameters,
   admission contract, risk thresholds, skills versions, budgets, stop rules,
   evaluation protocol) is recorded.

## Non-goals

No generic plugin framework, no new abstraction layers, no multi-agent review
stack, no silent error swallowing, no expansion of the operation space inside
R24a, no partial-maturation evidence, no claims about traditional methods
before the formal test.

## Sequence

1. Commit R23 leftovers (done: `273a4da`); branch `research/r24-iterative-loop` (done).
2. Opportunity-loss diagnosis script + tests; run on the saved R23 development
   run; record the G/A/S table and the branch decision. Diagnostic only.
3. Execute the branch rule above.
4. R24a implementation (`skills_r24`, loop policy, stop conditions, error
   semantics), regression tests, engineering smoke on 2022H1.
5. R24a development evaluation per the protocol above.
6. R24b ledger extension with bias guards; repeat the evaluation.
7. If the gate passes: freeze manifest, one pseudo-holdout pass on an unused
   development-adjacent period (e.g. part of 2023H1), then formal test.
