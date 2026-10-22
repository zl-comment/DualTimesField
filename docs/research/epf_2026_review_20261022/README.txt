2026 electricity-price forecasting literature review — 2026-10-22

Deliverables
  2026电价论文_市场结果与基线结构总表.xlsx
  建议补充的基线论文_实施清单.xlsx
  2026电价论文_对比结论与结构建议.docx
  2026电价论文_对比结论与结构建议.html

Scope and interpretation
  The 109 records are the supplied collection, not a proven exhaustive census.
  Three reviewers recorded paper-specific sources, results, protocols and limits.
  Full-text access means relevant sections were inspected, not that every table
  was independently reproduced. Old-version and fragment evidence is separated.
  No external paper's published score is inserted into the local ranking.
  Local rankings describe matched samples and scoring, not matched information
  or tuning budgets. The newer Anchor candidates have development results only.

Rebuild documents from the repository root (Python 3.10+)
  python -m pip install -r docs/research/epf_2026_review_20261022/requirements.txt
  python docs/research/epf_2026_review_20261022/build_deliverables.py

  The build reads the archived JSON inputs plus tracked baseline/development CSV
  files under logs/forecasting. It does not require the original attachment or
  external network access. input_metadata.json records the original attachment
  hash and verifies the extracted input_rows.json hash during each rebuild.
  review_validation.json records document hashes and structural checks.
  Rebuilding changes paths/timestamps; byte-identical Office ZIPs are not promised.

Optional rescoring of the original frozen predictions
  python docs/research/epf_2026_review_20261022/audit_local.py

  This requires the forecasting project's NumPy environment and all original NPZ
  files at the paths recorded in the experiment manifests. Those prediction files
  are not included in this review archive. The script verifies archive hashes,
  aligns origins/target times/actuals and recomputes all 51 region/model outputs.
  It writes local_audit.json, local_rankings.json and local_regional.json; rebuild
  the deliverables afterwards. It does not train or select a forecasting model.

Contents
  input_rows.json / input_notes.json / input_metadata.json: extracted source data.
  papers_*_review.json: three reviewers' evidence-backed records.
  synthesis.py: project-specific recommendations and protocol comparisons.
  all_reviews.json / baseline_recommendations.json: generated combined records.
  local_*.json: frozen-prediction audit and descriptive rankings.
  build_deliverables.py / audit_local.py: generation and rescoring scripts.

Downloaded third-party PDFs/HTML, temporary files and model weights are excluded.
The existing Anchor implementation and experiment commits are preserved unchanged.
