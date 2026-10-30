R9 archive

Read results_cn.txt for the conclusions, all nine region/seed results and resources.
all_region_seed_metrics.csv contains all 36 arm/region/seed metric rows.
summary.json and independent_audit.json are byte copies of the published outputs.

Prediction/scoring execution commit: 04293abe114a8adee096910bb6ff82bf3f02a9db.
The implementation was tested at 83d5a4c9102a7390f876718e64ad1ea9990b509a:
263 tests and 89 subtests passed; the next freeze commit added the independent
offline auditor, its synthetic regression, and acceptance evidence.

raw_artifacts.tar.gz contains all 310 files from the new output root, including
inputs, separate 2022 labels, predictions, raw request/response wire, attempt
journals, command records, scores and independent audit. raw_output_manifest.json
records each original byte count/hash and the archive hash. Every archived file
was decompressed and checked against its original hash before committing.
Archive SHA256: 55aaea394a861b7cb7e8262f66c1a86964785e2ad2dfe4fac7c39f1d8d71fc4c.

For offline inspection, extract the archive under outputs/forecasting in a
checkout with the frozen source files. The contained directory is named
agent_repair_three_seed_r9_20261030. The independent auditor can recheck these
artifacts without any model or network call; give it a fresh --output path:
  /root/workspace/Code/DualTimesField/.venv/bin/python -B scripts/audit_r9_three_seed.py --output /tmp/r9-recheck-unique.json
Do not rerun the prediction launcher over existing outputs. This archive records
one frozen experiment; no result-dependent retries or parameter changes occurred.

M1 engineering evidence is separately archived in
logs/forecasting/m1_agent_contract_c2_20261030_r9.
m1_closure_verification.txt records the default-recompute verifier exit 0.
That proof is synthetic and label-free; it is not a claim that the real M1
TOTALDEMAND acquisition mechanism or real-time news inputs were forecast-tested.
The actual R9 experiment evaluates the inherited r8 finite forecast adjustments.

No 2023-2024 final-period scoring, training, package installation, checkpoint
download, old-output replacement, formal-main-table changes or GitHub push.
The local numerical model used CPU. The existing LLM service's GPU use,
energy consumption and monetary cost were not measured.
