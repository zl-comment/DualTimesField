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
confirmation or a main-table ranking. The 2023–2024 formal test remains untouched.

Execution: `python -m forecasting.api_agent.pilot --workflow r16
--origins-per-region 24 --out outputs/forecasting/kev_r16_pilot_20261102`, using the
reserved GPU 7 and the unchanged local Kev service on GPU 6.
