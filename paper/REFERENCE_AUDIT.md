# Reference audit and manuscript alignment

Checked: 8 October 2026. Branch: `feature/paper-revision-audit`.

## Scope and remaining uncertainty

This audit covers all 33 entries in `manuscript/references.bib` (32 retained,
one AEMO procedure added). It distinguishes bibliographic identity, support for
the statement actually cited, and full experimental reproducibility. Finding a
publisher record is not a full-text or independent replication check. No paper
is labelled fabricated merely because a direct request was blocked.

**RE-Price full-text check completed using the user-supplied 13-page PDF**:
Chen et al., *Reasoning-enhanced probabilistic electricity price forecasting
using parameter-efficient large language models*, DOI
[10.1016/j.apenergy.2026.128712](https://doi.org/10.1016/j.apenergy.2026.128712).
The title, DOI, four author names, volume **426**, publication year **2026**
and article number **128712** all agree with the existing BibTeX entry. Page 1
records online publication on **28 August 2026**, so the browser's December
2026 issue date does not imply that the paper was unavailable on the audit
date. The full experimental setup and result tables can now be inspected;
the previous request for this PDF and uncertainty about the volume are closed.
The manuscript now describes the method and distinguishes its input/evaluation
protocol from ours. This is a document audit, not an independent replication;
remaining reproducibility questions are recorded below.

For historical market-rule provenance, the newly cited AEMO procedure is the
accessible **2026 version 19**, not a claimed archived 2024 version. Its section
2.1 documents the timing described in the text. Exact historical rule-version
matching across 2015–2024 would require archived procedures; point-in-time data
selection in this study instead uses the archived run/publication timestamps.
An older `wa.aemo.com.au` search result redirected to the homepage and was not
retained as a stable source.

No other identity mismatch remains identified in this audit. This does not
mean that all cited papers were read end to end. Author details, affiliations,
the AI-use declaration and the repository-release statement in the manuscript
are still submission placeholders, not facts verified by this audit.

## Source-by-source checks

“Confirmed” below means that an official publisher/conference record, the
authors' primary manuscript, or an official operator resource supports the
identity and the scope of the revised citation. It does not certify every
experimental claim in the cited work. Conference papers retain their
conference year rather than their earlier arXiv submission year.

| BibTeX key | Primary record consulted | Outcome and citation scope |
|---|---|---|
| `lago2021forecasting` | [Applied Energy / DOI](https://doi.org/10.1016/j.apenergy.2021.116983) | Confirmed: forecasting review and open benchmark. Frequent recalibration motivates our schedule; the paper does not prescribe our quarterly schedule. |
| `weron2014electricity` | [International Journal of Forecasting / DOI](https://doi.org/10.1016/j.ijforecast.2014.08.008) | Confirmed: general electricity-price forecasting review. |
| `nowotarski2018recent` | [Publisher](https://www.sciencedirect.com/science/article/pii/S1364032117308808) | Confirmed: probabilistic forecasting review; retain final publication year 2018 despite the 2017 DOI suffix. |
| `zeng2023transformers` | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/26317) | Confirmed: DLinear source, 37(9), 11121–11128, 2023. |
| `zhou2021informer` | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/17325) | Confirmed: Informer source and conference metadata. |
| `salinas2020deepar` | [Publisher](https://www.sciencedirect.com/science/article/pii/S0169207019301888) | Confirmed: DeepAR, final journal publication 2020. |
| `chen2016xgboost` | [Authors' manuscript](https://arxiv.org/abs/1603.02754) | Confirmed: XGBoost method; no universal boosted-tree range-preservation claim is attributed to it. |
| `cho2014learning` | [ACL Anthology](https://aclanthology.org/D14-1179/) | Confirmed: GRU source. Corrected van Merriënboer accent; pages 1724–1734. |
| `diebold1995comparing` | [NBER original working-paper record](https://www.nber.org/papers/t0169) | Confirmed: predictive-accuracy test; retain the final 1995 journal citation, not the 1994 working-paper date. |
| `newey1987simple` | [Original paper PDF](https://users.ssc.wisc.edu/~behansen/718/NeweyWest1987.pdf) | Confirmed: HAC covariance estimator. Corrected incomplete page field to 703–708. Our lag choice is not prescribed by this citation. |
| `gneiting2007strictly` | [Authors' paper PDF](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) | Confirmed: proper scoring rules. Our five-quantile CRPS is explicitly an approximation. |
| `uniejewski2018variance` | [Author's publication list](https://p.wz.pwr.edu.pl/~weron.rafal/Publ) | Confirmed final record: IEEE TPWRS 33(2), 2219–2229, 2018. Do not substitute early-access 2017 metadata. |
| `koenker1978regression` | [Original paper PDF](https://www.maths.usyd.edu.au/u/jchan/GLM/Koenker%26Bassett1978QuantileReg.pdf) | Confirmed: quantile regression. Corrected pages to 33–50 and encoded Gilbert Bassett Jr. correctly. |
| `hochreiter1997long` | [Publisher](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory) | Confirmed: LSTM, Neural Computation 9(8), 1735–1780. |
| `hong2020energy` | [IEEE / DOI](https://doi.org/10.1109/OAJPE.2020.3029979) | Confirmed: energy forecasting review, not evidence of realized trading profit from our model. |
| `marcjasz2023distributional` | [Authors' manuscript and journal reference](https://arxiv.org/abs/2207.02832) | Confirmed: distributional neural networks, Energy Economics 125, 106843, 2023. |
| `ziel2018day` | [Publisher](https://www.sciencedirect.com/science/article/pii/S014098831730436X) | Confirmed: high-dimensional linear price forecasting. |
| `tashman2000out` | [Publisher](https://www.sciencedirect.com/science/article/pii/S0169207000000650) | Confirmed: out-of-sample evaluation principles, not proof that our exact split or tail-test weighting is optimal. |
| `chen2026reprice` | User-supplied full publisher PDF; [publisher record](https://www.sciencedirect.com/science/article/pii/S0306261926013681) | Full-text check completed: Applied Energy 426 (2026), 128712; existing authors/DOI correct. Setup and tables checked; not a same-protocol comparison or independent replication. See detailed audit below. |
| `zhang2003time` | [Publisher](https://www.sciencedirect.com/science/article/pii/S0925231201007020) | Confirmed: hybrid ARIMA/neural forecasting precedent. Normalized author given name to G. Peter. |
| `gneiting2011making` | [Authors' manuscript](https://arxiv.org/abs/0912.0902) | Confirmed: point-forecast evaluation and loss-dependent targets. |
| `zhang2026dualtimesfield` | [Official PMLR record](https://proceedings.mlr.press/v306/zhang26co.html) | Confirmed: ICML 2026, PMLR 306, 156544–156566. Added official volume/pages/URL; removed unverified venue location. Original reconstruction/interpolation setting is distinguished from our forecasting adaptation. |
| `nie2023time` | [Authors' manuscript](https://arxiv.org/abs/2211.14730) | Confirmed: PatchTST, accepted ICLR 2023. Original `Nguyen, Nam H.` author entry was correct and is retained. |
| `liu2024itransformer` | [Official ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html) | Confirmed; replaced preprint link with conference record. |
| `oreshkin2020nbeats` | [Official ICLR 2020 record](https://iclr.cc/virtual_2020/poster_r1ecqn4YwB.html) | Confirmed; linked conference record. |
| `wu2021autoformer` | [Official NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html) | Confirmed; linked proceedings. |
| `sitzmann2020implicit` | [Official NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html) | Confirmed: periodic-activation implicit representations, a methodological precedent rather than a same-data baseline. |
| `tancik2020fourier` | [Official NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html) | Confirmed: Fourier-feature motivation; our learned bounded frequencies are implementation-specific. |
| `grinsztajn2022tree` | [Official NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html) | Confirmed: retain conference title including “typical”; added proceedings URL and DOI. Supports including tree baselines, not a claim that they must win electricity-price tasks. |
| `aemo2025priceanddemand` | [AEMO aggregated data](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data) | Official source identified. Corrected URL; removed unsupported publication year and recorded access date. |
| `aemo2025mmsdm` | [Official NEMWeb archive](https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/) | Confirmed archive. Removed unsupported publication year; listed relevant tables and access date. |
| `aemo2025dwgm` | [AEMO Victorian gas data](https://www.aemo.com.au/energy-systems/gas/declared-wholesale-gas-market-dwgm/data-dwgm/vic-wholesale-price-withdrawals) | Confirmed official resource. Removed unsupported publication year; recorded access date. |
| `aemo2026predispatch` | [AEMO SO_OP_3704](https://www.aemo.com.au/-/media/files/electricity/nem/security_and_reliability/power_system_ops/procedures/so_op_3704-predispatch.pdf) | Added verified version 19, effective 1 April 2026, section 2.1, to support predispatch timing. Historical-version limitation noted above. |

The `aemo2025...` BibTeX keys are retained as internal identifiers only: their
digits no longer assert a publication year in the rendered bibliography.
Undated operator pages explicitly use `n.d.` instead of a fabricated year.
Seven conference entries retain no page field: N-BEATS, PatchTST, iTransformer,
Autoformer, SIREN, Fourier Features and the tabular-tree benchmark. Their
identities and official links are confirmed; primary proceedings pagination
was not retrieved in this audit. BibTeX reports these as non-fatal empty-page
warnings. No page range was invented to suppress a warning.

## RE-Price full-text follow-up

Evidence: the user-supplied file
`Reasoning-enhanced probabilistic electricity price forecasting using parameter-efficient large language models.pdf`,
13 pages, SHA-256
`a0f4aceafb3cb09778c027b7248e17da2ae9354564cbff8b4f70cd040e40d91d`.
Page references below are printed PDF pages. The copyrighted full PDF is not
copied into the Git repository.

### Confirmed method and protocol

- Pages 1 and 12: Haoxuan Chen, Yinliang Xu, Wenchuan Wu and Hongbin Sun;
  accepted 20 August 2026, online 28 August 2026. No correction to the existing
  BibTeX author, year, volume, article number or DOI fields is needed.
- Sections 3–4, pages 3–7: news is distilled into impact categories and event
  attributes, then combined with numerical features using cross-modal
  attention. A frozen reasoning branch is distinguished from the adapted
  forecasting backbone. Probabilistic trajectories come from Monte Carlo
  dropout under standard and tail-aware prompts, not ordinary token sampling;
  adaptive Gamma-kernel density estimation follows a positive-space shift.
  Section 6 explicitly cautions that these are not exact samples from the
  true conditional price distribution.
- Section 5.1 and Table 1, page 8: hourly NSW, QLD and TAS data, 1 January
  2015–31 December 2024; chronological 70/10/20 train/validation/test split;
  72-hour history and 24-hour forecast horizon. Inputs include price, load
  forecasts, weather proxies and more than 2,000 WattClarity articles.
- Table 1: Qwen3-4B with 4-bit NF4, LoRA rank 8 and alpha 16; one LoRA-stage
  epoch plus 50 main-stage epochs, Adam, learning rate 2e-4, batch size 32,
  MSE loss, 100 generated scenarios and 10% tail-aware scenarios. Table 7
  reports 11.54 million **trainable** parameters, not total backbone size.
- Section 5.2, pages 8–9: the paper says all baselines receive its numerical
  and extracted text inputs, with MiniLM sentence embeddings adapted to each
  architecture. Comparators are GPT4TS, Informer, DeepAR, GRU and quantile
  XGBoost. This is an internal same-input comparison, not the same inputs as
  the present repository's benchmark.

Tables 2–3 (page 9) report the following RE-Price results. They are transcribed
as evidence only and must not be merged into our benchmark tables:

| Region | Reported MAE | Reported CRPS |
|---|---:|---:|
| NSW | 23.48 | 17.36 |
| QLD | 25.85 | 20.58 |
| TAS | 18.96 | 13.13 |

### Remaining interpretation and reproducibility limits

1. **Weather availability needs care.** Section 5.1 constructs its day-ahead
   temperature proxy by adding Gaussian noise (standard deviation 1.5 degrees
   Celsius) to observed daily temperatures. This is not an archived forecast
   issued before the origin. Our assessment: adding noise alone does not
   establish ex-ante availability, so the paper's stated intention to avoid
   look-ahead bias is not independent evidence that this weather input is
   point-in-time safe. Exact alignment would require code/data; this is not
   a finding that every reported experiment is invalid.
2. **News controls are described but not independently verified.** Section
   3.3.2 uses the latest publication/edit timestamp and excludes subsequent
   articles; Section 5.1 additionally removes update components. Section
   3.3.3 explicitly says the extracted labels were not manually verified.
   Auditing these safeguards requires the timestamped corpus and extraction
   pipeline, not merely the paper's prose.
3. **Protocol matching is incomplete.** The PDF reports split proportions,
   not exact sample-index boundaries or an explicit forecast-origin stride.
   Section 5.6 discusses monthly/quarterly updates as deployment options;
   that does not establish that Tables 2–3 used the quarterly expanding-window
   refits used in our main experiment. Price transformations, clipping rules
   and target-window handling also require implementation-level matching.
4. **Training configuration is available, exhaustive tuning provenance is
   not.** Table 1 supplies the retained model settings. The PDF does not
   establish per-baseline search budgets, repeated-seed uncertainty, or the
   validation/test provenance of every selection in the tail-ratio sensitivity
   study. Published point estimates should not be treated as our matched
   Diebold–Mariano comparisons.

Decision: retain and describe the citation more specifically, without importing
its performance gains or claiming our model is better/worse by comparing
unmatched table values. The user does not need to obtain this paper again.
Code, archived forecasts and the timestamped news corpus would be needed only
for a future reproduction or fair direct comparison.

## Alignment changes grounded in repository results

- Preserve numerical results, training code, checkpoints and the Elsevier
  `elsarticle` / `elsarticle-num-names` template. No new training or altered
  significance tests are claimed by this manuscript-only revision.
- Define an origin at the start of delivery hour `t`: history `t-72:t-1`,
  targets `t:t+23`. This is rolling hourly forecasting, not one fixed daily
  auction forecast.
- Describe actual event detection, learned threshold/width, full-history CTF
  input, feature routing and calibrator. The hard activity penalty contributes
  to the recorded objective/checkpoint choice but supplies no training gradient.
- Separate static MAE deterioration from static CRPS improvement; pooled gains
  do not imply gains in every region (raw QLD1 field-only MAE worsens by 0.42).
- Limit the final negative-price superiority claim to main-table comparators:
  removing the final bound yields better negative-price scores, at worse
  overall MAE. The bound is empirical, not a regulatory or physical floor.
- Distinguish averaging seed scores from scoring averaged predictions. Tail
  table point estimates and tests also weight event-containing origins
  differently. Do not silently compare these quantities as identical estimands.
- Identify the case plots as cumulative outputs of one jointly trained model,
  and the field-separation ratios as standardized-asinh diagnostics, not
  monetary shares.
- Disclose that test-error diagnosis motivated the lower-bound design even
  though its numeric value was selected on validation data. An untouched later
  evaluation remains necessary to assess that development choice.
- Replace the overview with editable PDF/SVG/PNG generated by
  `draw_framework.py`; retain checkpoints only for empirical mechanism plots.

## Validation

The preceding result audit checked generated tables against saved results and
recomputed 108 metric sets from saved predictions (maximum observed absolute
discrepancy approximately 4.5e-13). This revision does not change those numbers.
The updated manuscript was rebuilt with `pdflatex`, `bibtex` and two further
`pdflatex` passes (29 pages in the initial revision; 30 pages after the
RE-Price full-text follow-up). Checks passed for all 33 unique/cited BibTeX keys,
the six included figure copies, unchanged numerical tables, Python source
syntax, absence of unresolved citations/references and overfull boxes, and
`git diff --check`. The framework and its manuscript-page rendering were
visually inspected. Non-fatal conference-pagination, author PDF-metadata and
underfull bibliography-line warnings remain. The built-in editor's compilation
attempt failed at environment
startup (`Unable to find standard directories for platform`), so its preview is
not evidence of a successful build.
