# R16: use Kev only for conflicting evidence

Experiment branch: `research/kev-agent-gated-routing-20261102-r16`.
Base commit: `0ae728e7e8eb478a7dc463c8b87dfbb0890660ad`.

R15's 72-origin development pilot completed with zero API fallbacks. AGENT QS was
55.29735 versus BASE 56.94635 and FIXED 55.27689 (equal region weight). It improved
on BASE but not FIXED. The Agent made 144 API calls, and its selected profile differed
from FIXED on only three origins. These observations do not demonstrate a predictive
contribution from the LLM.

R16 targets redundant decisions, not a newly designed baseline or new model training.
The existing numerical profiles and FIXED policy stay unchanged. A deterministic
gate skips the LLM if there are zero or one alternatives passing the existing >2%
recent-fit guard, or the fixed winner is no worse in both observed 12-hour halves.
Only conflicting eligible alternatives go to a single Kev call. The new versioned
skill resolves that trade-off using current-origin evidence. An uncertain response
(selected probability below .55) falls back to the strong FIXED default. No per-date
rules or outcomes from the evaluated future are supplied to this gate or skill.

Before results, the objectives are: fewer API calls than R15, no missing origins or
API failures hidden from scoring, and comparison of QS/MAE/coverage to BASE, FIXED,
and R15. A reduction in API calls does not itself prove forecasting superiority.

The paired pilot repeats the SAME 72 development origins and frozen numerical
profiles, with source/model/input fingerprints. BASE and FIXED predictions must
match the saved R15 predictions before attributing differences to workflow changes.
These 2022 H1 samples were already used in development; this is not independent
confirmation or a main-table ranking. This run does not open the 2023–2024 formal
test cache; prior R12/R13 historical-test exposure remains disclosed.

Execution: `python -m forecasting.api_agent.pilot --workflow r16
--origins-per-region 24 --out outputs/forecasting/kev_r16_pilot_20261102`, using the
reserved GPU 7 and the unchanged local Kev service on GPU 6.

## Completed results

The live service remains running in tmux `kev-4b`, with a separate `logs` window.
Its endpoint is `http://127.0.0.1:8009/v1/systemone`; GPU 6 model-process memory
was 9,328 MiB. GPU 7 is released when the prediction experiment finishes.
All eleven unit tests and the real API smoke test passed.

| Method | Development QS | MAE | 90% interval coverage |
|---|---:|---:|---:|
| Frozen BASE | 56.94635 | 91.87910 | 85.36% |
| Strong FIXED | 55.27689 | 90.61962 | 84.72% |
| R15 Agent | 55.29735 | 90.66253 | 84.90% |
| R16 Agent | 55.27689 | 90.61962 | 84.72% |

R16 reduced API calls from 144 to 1 (99.31%) and input tokens from 96,585 to
1,020. Both pilots scored all 72 origins, with no API fallbacks. Recorded runtime
after predictor loading was 60.89s for R15 and 10.20s for R16; this is a paired
development observation, not a repeated latency benchmark.

BASE and FIXED arrays, origin axes, and target axes matched R15 exactly in all
three regions. R16 AGENT also matched FIXED. Thus the 2.93% QS improvement over
BASE comes from the profile-selection workflow; an additional LLM forecasting
benefit has NOT been demonstrated. Coverage declined by 0.637 percentage points
relative to BASE. No main-table rank is reported, and formal test data was not
used to select this workflow.

The architecture objective is implemented: frozen predictor, local decision API,
versioned skills, bounded actions, isolated per-origin evidence, reproducible
artifacts, and honest deterministic controls. The stronger research objective of
demonstrating useful LLM-specific forecasting gains remains open. Further work
should introduce evidence requiring semantic judgment with a validated numerical
effect, rather than treating agreement with a numerical ranking as an LLM gain.

Artifacts: `logs/forecasting/kev_r16_20261102/comparison.csv`,
`deployment_and_optimization_result.json`, `pilot_manifest.json`, and
`pilot_report.json`. Per-origin full request/response traces and predictions remain
under `outputs/forecasting/kev_r16_pilot_20261102`.
