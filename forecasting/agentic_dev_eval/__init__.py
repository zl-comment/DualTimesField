"""r6 minimal A/B/C/D 2022 development evaluation (goal AGENTIC_M2M3_REAL_DEV_CAUSAL_EVAL_C2).

Arms share one frozen base numeric stream (A: Chronos-2 zero-shot, CPU, offline)
and one frozen action allowlist; B applies predeclared fixed rules, C one fixed
one-shot LLM call, D a bounded allowlisted agent over the same as-of-visible data.
"""
