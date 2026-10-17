"""V5 Calibrated History: frozen LightGBM + bounded location/scale correction
+ state-matched matured residual memory.

Implements the round-2 frozen specification (reports/round2_codex_review.md,
sections 3-11).  All reuse of existing artefacts is read-only; nothing in
full_v3/full_v4 or the public scorer is modified.
"""
