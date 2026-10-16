"""End-to-end protocol tests on a small synthetic world.

Covers spec section 9 items 10, 12, 13, 14: the test-read guard (real access,
not a config string), hash-frozen resume/final-test refusal, one-shot final
evaluation, and a complete prepare -> dev -> freeze -> final-test -> report
flow.  Segment dates are patched to a small window purely to exercise the
machinery; SCHEMA_HASH is recomputed from the patched windows (Round-9 fix 2)
so parent and CLI children share one consistent schema identity; the real run
uses the frozen production segments.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from forecasting.aligned_inputs import build_aligned_splits
from forecasting.aligned_scoring import make_prediction_artifact, save_prediction_artifact
from forecasting.full_v5 import data, experiment, schema

TZ = "Australia/Brisbane"
HOUR = 3600

# The mini H2 is long enough that same-hour matched pools exceed K=16 during
# V1/V2, so the matched-history branch can actually discriminate (short pools
# would force v to zero and the G2 contrast would be vacuous).
MINI_SEGMENTS = {
    "dfit": ("2022-07-01", "2022-08-12"),
    "v1": ("2022-08-12", "2022-08-18"),
    "v2": ("2022-08-18", "2022-08-21"),
    "h2": ("2022-07-01", "2022-08-21"),
    "test": ("2022-08-21", "2022-08-25"),
}
MINI_CUTOFFS = {"dfit": "2022-08-12", "h2": "2022-08-21"}
# A cheaper segment set for the resume/tamper/budget machinery tests; still
# large enough for the structured world to pass the development gates.
SMALL_SEGMENTS = {
    "dfit": ("2022-07-01", "2022-07-29"),
    "v1": ("2022-07-29", "2022-08-01"),
    "v2": ("2022-08-01", "2022-08-03"),
    "h2": ("2022-07-01", "2022-08-03"),
    "test": ("2022-08-21", "2022-08-26"),
}
SMALL_CUTOFFS = {"dfit": "2022-07-29", "h2": "2022-08-03"}
DATA_START, DATA_END = "2021-12-25", "2022-08-26"  # two calendar years in train (year_standardized needs variance)
PREDISPATCH_FEATURES = ["predispatch_rrp", "predispatch_total_demand_mw",
                        "predispatch_available_generation_mw", "predispatch_net_interchange_mw",
                        "predispatch_covered"]
IC_FEATURES = ["ic_net_import_mw", "ic_import_room_mw", "ic_export_room_mw",
               "ic_main_import_room_mw", "ic_covered"]


def _unix(series):
    return (series.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[s]")
            .astype(np.int64))


def build_world(tmp_path: Path, structured: bool = True,
                perturb_test_labels_from: str | None = None,
                regions=("NSW1",), segments=None, cutoffs=None) -> Path:
    world = tmp_path / ("world_ok" if structured else "world_noise")
    if perturb_test_labels_from is not None:
        world = tmp_path / "world_perturbed"
    world.mkdir(parents=True)
    csv_files, pdpasa_files, predispatch_files, ic_files = {}, {}, {}, {}
    splits_by_region = {}
    for r_i, region in enumerate(regions):
        rng = np.random.default_rng(2026 + r_i)
        index = pd.date_range(pd.Timestamp(DATA_START, tz=TZ), pd.Timestamp(DATA_END, tz=TZ),
                              freq="h", inclusive="left")
        unix = _unix(index)
        n = len(unix)
        hour = ((unix // 3600 + 10) % 24).astype(float)
        day_index = (unix + 10 * HOUR) // 86400
        day_offset = day_index - day_index[0]  # days since mini-data start (same base as the archives)
        block = (day_offset // 2) % 3                      # 2-day regime blocks
        level = np.array([-1.0, 0.0, 1.0])[block]          # block 1 is the middle regime
        hour_pattern = np.sin(2 * np.pi * hour / 24.0)

        spare_raw = 5000.0 + 600.0 * hour_pattern + 1400.0 * level + rng.normal(0, 25, n)
        b_q50 = 85.0 + 8.0 * np.sin(2 * np.pi * hour / 24.0 + 1.0) + rng.normal(0, 2.0, n)
        # Only the middle regime level carries bias: a non-monotone (bump) function
        # of every z column that a bounded linear calibrator cannot represent, but
        # state-matched same-hour history can (the 42-day Dfit keeps matched pools
        # >= 24 so the middle class holds the matched median).
        bias = (45.0 * (level == 0.0)) if structured else np.zeros(n)
        sigma = 3.0 if structured else 30.0
        y = b_q50 + bias + rng.normal(0, sigma, n)
        if perturb_test_labels_from is not None:
            # Change ONLY labels delivered at/after the given date (no rng draws
            # consumed: every other artefact stays byte-identical to the base
            # world).  Queries that cannot have seen these labels must produce
            # identical predictions.
            late = np.asarray(index >= pd.Timestamp(perturb_test_labels_from, tz=TZ))
            assert late.any()
            y = y.copy()
            y[late] += 777.0
        demand = 8000.0 + 400.0 * np.cos(2 * np.pi * hour / 24.0) + rng.normal(0, 30, n)

        # --- region hourly csv ---------------------------------------------
        frame = pd.DataFrame({
            "delivery_start_aest": index.strftime("%Y-%m-%dT%H:%M:%S+10:00"),
            "available_at_aest": (index + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S+10:00"),
            "region": region,
            "rrp_aud_per_mwh": y,
            "total_demand_mw": demand,
            "split": "unused",  # relabel_splits=True rewrites labels from boundaries
        })
        csv_path = world / f"{region.lower()}_hourly.csv"
        frame.to_csv(csv_path, index=False)
        csv_files[region] = str(csv_path)

        # --- exogenous archives (one row per possible forecast origin) -----
        first_origin, last_origin = 72, n - 24
        origins = unix[first_origin:last_origin + 1]
        m = len(origins)
        lead = np.arange(24)
        delivery = origins[:, None] + lead[None, :] * HOUR
        dhour = ((delivery // 3600 + 10) % 24).astype(float)
        dday = (delivery + 10 * HOUR) // 86400
        doff = (dday - day_index[0]).astype(int)
        dlevel = np.array([-1.0, 0.0, 1.0])[(doff // 2) % 3]
        dpat = np.sin(2 * np.pi * dhour / 24.0)
        uigf = 900.0 + 200.0 * dpat
        demand50 = 7900.0 + 380.0 * np.cos(2 * np.pi * dhour / 24.0)
        spare = 5000.0 + 600.0 * dpat + 1400.0 * dlevel + rng.normal(0, 20, (m, 24))
        vintage_run = origins - HOUR
        common = dict(forecast_origin_unix=origins.astype(np.int64),
                      source_run_unix=vintage_run.astype(np.int64),
                      source_last_changed_unix=vintage_run.astype(np.int64))
        np.savez(world / f"pdpasa_{region}.npz", **common,
                 source_stpasa_run_unix=vintage_run.astype(np.int64),
                 max_spare_capacity_mw=spare.astype(np.float32),
                 demand50_mw=demand50.astype(np.float32),
                 demand10_mw=(demand50 + 250).astype(np.float32),
                 demand90_mw=(demand50 - 250).astype(np.float32),
                 uigf_mw=uigf.astype(np.float32),
                 available_capacity_mw=(12000 + 300 * dpat).astype(np.float32))
        b_del = 85.0 + 8.0 * np.sin(2 * np.pi * dhour / 24.0 + 1.0)
        np.savez(world / f"predispatch_{region}.npz", **common,
                 **{name: value.astype(np.float32) for name, value in dict(
                     predispatch_rrp=b_del + rng.normal(0, 3, (m, 24)),
                     predispatch_total_demand_mw=demand50 + rng.normal(0, 40, (m, 24)),
                     predispatch_available_generation_mw=11500 + 400 * dpat,
                     predispatch_net_interchange_mw=-200 + 150 * dpat,
                     predispatch_covered=(np.sin(2 * np.pi * dhour / 24.0 + 2.0) > -0.9).astype(np.float32),
                 ).items()})
        np.savez(world / f"interconnector_{region}.npz", **common,
                 **{name: value.astype(np.float32) for name, value in dict(
                     ic_net_import_mw=300 + 120 * dpat,
                     ic_import_room_mw=1800 + 250 * dpat,
                     ic_export_room_mw=1500 + 200 * dpat,
                     ic_main_import_room_mw=900 + 150 * dpat,
                     ic_covered=(np.cos(2 * np.pi * dhour / 24.0 + 3.0) > -0.9).astype(np.float32),
                 ).items()})
        pdpasa_files[region] = str(world / f"pdpasa_{region}.npz")
        predispatch_files[region] = str(world / f"predispatch_{region}.npz")
        ic_files[region] = str(world / f"interconnector_{region}.npz")

    # --- data config and shared cache --------------------------------------
    data_config = dict(
        project_root=str(world),
        data=dict(
            timezone=TZ, delivery_column="delivery_start_aest",
            available_column="available_at_aest", region_column="region", split_column="split",
            history_columns=["rrp_aud_per_mwh", "total_demand_mw"],
            target_column="rrp_aud_per_mwh", price_transform="identity",
            quantile_target="transformed",
            split_names=["train", "calibration", "test"],
            split_boundaries=dict(
                train=dict(start=DATA_START, end_exclusive="2022-07-01"),
                calibration=dict(start="2022-07-01", end_exclusive="2022-08-21"),
                test=dict(start="2022-08-21", end_exclusive=DATA_END)),
            relabel_splits=True, load_end_exclusive=DATA_END,
            region_files=dict(csv_files)),
        forecast_protocol=dict(input_hours=72, output_hours=24, stride_hours=1,
                               calendar_features=["hour_sin", "hour_cos", "is_weekend", "day_of_week_sin"]),
        origin_context=dict(enabled=False),
        future_exogenous=dict(enabled=True, name="max_spare_capacity_mw", dimension=1,
                              aggregation="min", standardization="train_zscore",
                              vintage_rule="latest_available_pdpasa_with_stpasa_tail",
                              region_files=dict(pdpasa_files)),
        ctf_exogenous=dict(enabled=True, features=["net_load_mw"], standardization="train_zscore",
                           clip_to_train_quantiles=[0.005, 0.995], out_of_range="log",
                           vintage_rule="latest_available_pdpasa_with_stpasa_tail",
                           region_files=dict(pdpasa_files)),
        quantile_exogenous=dict(enabled=True, features=["max_spare_capacity_mw"],
                                shortfall_below_train_quantiles=[0.01, 0.05, 0.1],
                                standardization="train_zscore",
                                vintage_rule="latest_available_pdpasa_with_stpasa_tail",
                                region_files=dict(pdpasa_files)),
        predispatch_exogenous=dict(enabled=True, features=PREDISPATCH_FEATURES,
                                   price_features=["predispatch_rrp"], standardization="train_zscore",
                                   vintage_rule="latest_predispatch_run_published_by_origin",
                                   region_files=dict(predispatch_files)),
        interconnector_exogenous=dict(enabled=True, features=IC_FEATURES,
                                      standardization="train_zscore",
                                      vintage_rule="latest_predispatch_run_published_by_origin",
                                      region_files=dict(ic_files)),
    )
    (world / "data_config.yaml").write_text(yaml.safe_dump(data_config))
    from forecasting.aligned_experiment import save_cache
    for region in regions:
        splits = build_aligned_splits(data_config, region, split_names=["calibration", "test"])
        save_cache(world / "data_cache" / region, splits)
        splits_by_region[region] = splits

    # --- precomputed baseline and V4 substitutes ---------------------------
    for r_i, region in enumerate(regions):
        rng = np.random.default_rng(5000 + r_i)
        splits = splits_by_region[region]

        def baseline_quantiles(split):
            split_obj = splits[split]
            origins_split = np.asarray(split_obj.origin_unix, dtype=np.int64)
            delivery_hour = (((origins_split[:, None] + np.arange(24)[None, :] * HOUR) // 3600 + 10) % 24)
            # The baseline must share the label's delivery-hour phase, otherwise a
            # systematic origin-vs-delivery phase residual swamps the DGP bias.
            centre = 85.0 + 8.0 * np.sin(2 * np.pi * delivery_hour / 24.0 + 1.0)
            centre = centre + rng.normal(0, 1.5, centre.shape)
            width = 1.0 + 0.15 * np.sin(2 * np.pi * delivery_hour / 24.0)
            spread = np.array([-28.0, -13.0, 0.0, 13.0, 28.0])
            return (centre[..., None] + spread[None, None, :] * width[..., None]).astype(np.float64)

        (world / "baseline").mkdir(exist_ok=True)
        for split in ("calibration", "test"):
            # The test-side file carries the grids and labels so the final
            # elementwise identity check can run in precomputed mode too.
            split_obj = splits[split]
            np.savez(world / "baseline" / f"{region}_{split}.npz",
                     quantiles=baseline_quantiles(split),
                     origin_unix=np.asarray(split_obj.origin_unix, dtype=np.int64),
                     target_time_unix=np.asarray(split_obj.target_time_unix, dtype=np.int64),
                     actual=np.asarray(split_obj.actual))
        (world / "v4").mkdir(exist_ok=True)
        for split in ("calibration", "test"):
            split_obj = splits[split]
            q = baseline_quantiles(split) + rng.normal(0, 6, size=baseline_quantiles(split).shape)
            q = np.sort(q, axis=-1)
            artifact = make_prediction_artifact(
                actual=np.asarray(split_obj.actual), origin_unix=split_obj.origin_unix,
                target_time_unix=split_obj.target_time_unix,
                metadata=dict(protocol_id="mini", region=region, split=split, seed=2026,
                              source="mini_v4", input_hash="mini", sample_hash="mini",
                              preprocessing_hash="mini", prediction_space="raw_rrp",
                              output_type="quantile"),
                quantiles=q, quantile_levels=[0.05, 0.1, 0.5, 0.9, 0.95])
            save_prediction_artifact(world / "v4" / f"{region}_{split}.npz", artifact)
    # Trusted pre-bound test-hash manifests (Round-6 M1): freeze records these
    # values verbatim instead of reading test bytes; final-test verifies.
    import hashlib as _hashlib

    def _file_sha(path):
        return _hashlib.sha256(Path(path).read_bytes()).hexdigest()

    (world / "baseline" / "test_source_hashes.json").write_text(
        json.dumps({region: _file_sha(world / "baseline" / f"{region}_test.npz") for region in regions}))
    (world / "v4" / "test_source_hashes.json").write_text(
        json.dumps({region: _file_sha(world / "v4" / f"{region}_test.npz") for region in regions}))
    (world / "jobs.json").write_text(json.dumps({"data_hashes": {}}))

    # --- fake project layout for identity freezing -------------------------
    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(repo / "forecasting" / "full_v5", world / "forecasting" / "full_v5",
                    ignore=shutil.ignore_patterns("__pycache__"))
    if segments is not None:
        # Round-9 fix 2: tests that spawn CLI children install the SAME small
        # window into the world's own schema.py, so parent (in-memory patch)
        # and children (import from the world) execute identical windows; the
        # child entry verifies the effective dates before any code runs.
        # SEGMENTS is a multi-line literal, FIT_CUTOFFS a single-line one.
        schema_py = world / "forecasting" / "full_v5" / "schema.py"
        text = schema_py.read_text()
        patched, n_seg = re.subn(
            r"SEGMENTS = \{.*?\n\}", "SEGMENTS = " + repr(dict(segments)), text, count=1, flags=re.S)
        patched, n_cut = re.subn(
            r"FIT_CUTOFFS = \{[^\n]*\}", "FIT_CUTOFFS = " + repr(dict(cutoffs)), patched, count=1)
        assert n_seg == 1 and n_cut == 1, f"schema window installation failed ({n_seg}, {n_cut})"
        # Build-time self-check: the installed file must parse and derive the
        # exact SCHEMA_HASH the patched parent will record into the identity.
        namespace = {}
        exec(compile(patched, str(schema_py), "exec"), namespace)  # noqa: S102 - test fixture
        installed_hash = hashlib.sha256(
            schema._canonical(namespace["_schema_payload"]()).encode()).hexdigest()
        assert installed_hash == schema.SCHEMA_HASH, (
            f"installed window schema hash {installed_hash} != parent {schema.SCHEMA_HASH}")
        schema_py.write_text(patched)
    (world / "configs" / "full_v5").mkdir(parents=True)
    shutil.copyfile(repo / "configs" / "full_v5" / "experiment.yaml",
                    world / "configs" / "full_v5" / "experiment.yaml")
    (world / "docs").mkdir()
    for name in ("full_v5_design_20261015.md", "full_v5_baseline_identity.md",
                 "full_v5_literature_comparability.md"):
        shutil.copyfile(repo / "docs" / name, world / "docs" / name)
    # Identity-bound reused sources (the full execution-dependency closure)
    # and the V5 test files must exist in the synthetic project root exactly
    # as in the real repository.
    for relative in experiment.REUSED_SOURCE_FILES:
        target = world / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, target)
    for relative in experiment.V5_TEST_FILES:
        target = world / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, target)

    v5_config = dict(v5_calibrated_history=dict(
        protocol_id="v5-calibrated-history-20261015-r1", seed=2026, threads=4,
        regions=list(regions), timezone=TZ,
        baseline=dict(mode="precomputed", run_root=str(world / "baseline"), seed=2026, threads=4),
        cache=dict(root=str(world / "data_cache"), jobs_json=str(world / "jobs.json")),
        v4=dict(mode="precomputed", run_root=str(world / "v4"), variant="v4_full", seed=2026),
        features=dict(price_scale=300, exogenous_clip=3.0, match_k=16, match_min_candidates=8,
                      lookback_days=365, window_short_days=7, window_long_days=28,
                      window_min_records=24,
                      horizon_groups=[[0, 5], [6, 11], [12, 17], [18, 23]]),
        calibrator=dict(delta_scale=150, s_min=0.5, s_max=1.5, a_l1_max=1.0, c_l1_max=0.5,
                        reg=0.001, extra_q50_weight=0.2, constraint_tolerance=1.0e-6),
        stacker=dict(reg=0.001),
        trees=dict(rounds=200, learning_rate=0.05, num_leaves=15, max_depth=4,
                   min_child_samples=200, max_bin=63, feature_fraction=1.0,
                   bagging_fraction=1.0, seed=2026, threads=4),
        gates=dict(g1_macro_improvement=0.01, g1_region_mae_max_worsening=0.01,
                   g1_region_band0300_max_worsening=0.005, g2_macro_improvement=0.005),
        bootstrap=dict(n=9999, block_hours=168, sensitivity_block_hours=24, seed=2026),
        budget=dict(dev_core_hours=24, final_core_hours=24)))
    config_path = world / "configs" / "full_v5" / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(v5_config, sort_keys=False))
    return world


def _apply_window_schema(monkeypatch, segments, cutoffs):
    """Install the small window in THIS process and recompute SCHEMA_HASH.

    The recomputation uses the same payload the (window-installed) world
    schema.py derives at import, so parent-side identity records and child-side
    verification agree by construction, not by coincidence (Round-9 fix 2).
    """
    monkeypatch.setattr(schema, "SEGMENTS", dict(segments))
    monkeypatch.setattr(schema, "FIT_CUTOFFS", dict(cutoffs))
    monkeypatch.setattr(schema, "SCHEMA_HASH", hashlib.sha256(
        schema._canonical(schema._schema_payload()).encode()).hexdigest())


@pytest.fixture()
def patched_schema(monkeypatch):
    _apply_window_schema(monkeypatch, MINI_SEGMENTS, MINI_CUTOFFS)
    return schema


def _small_schema(monkeypatch):
    """Cheaper segments for machinery tests (applied after patched_schema)."""
    _apply_window_schema(monkeypatch, SMALL_SEGMENTS, SMALL_CUTOFFS)


def _write_tests_passed(run_root: Path):
    path = run_root / "development" / "tests_passed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(files=["simulated"], n_passed=1, n_failed=0, exit_code=0,
                                    note="operator-recorded test status for the synthetic pipeline test")))


def test_full_pipeline_prepare_dev_freeze_final_test_report(tmp_path, monkeypatch, patched_schema):
    world = build_world(tmp_path, structured=True)
    monkeypatch.chdir(world)
    run_root = world / "run"
    config_path = world / "configs" / "full_v5" / "experiment.yaml"

    assert experiment.prepare(str(config_path), str(run_root)) == 0
    assert (run_root / "identity" / "identity.json").exists()
    assert (run_root / "baseline_predictions" / "NSW1_calibration.npz").exists()
    identity = json.loads((run_root / "identity" / "identity.json").read_text())
    # Round-4 H2: tests, reused sources and the prepared baseline (with grids)
    # are bound into the frozen identity.
    assert set(identity["tests"]) == set(experiment.V5_TEST_FILES)
    assert set(identity["reused_sources"]) == set(experiment.REUSED_SOURCE_FILES)
    bound = identity["baseline_predictions"]["NSW1"]
    assert bound["n_origins"] > 0 and len(bound["origins_sha256"]) == 64
    # Round-6 M1/L2: prepare binds the trusted pre-bound test-side hashes from
    # the manifests (never the test bytes) and carries the unfinished
    # declaration; the identity doc records the machine-verified NSW1 hash.
    assert identity["baseline_run"]["test_prediction_hashes_from_jobs"]["NSW1"]
    assert identity["v4"]["test_prediction_hashes"]["NSW1"]
    assert identity["baseline_run"]["test_hash_manifests"]["NSW1"]["sha256"]
    unfinished = json.loads((run_root / "unfinished.json").read_text())
    assert any("force_null" in item for item in unfinished), unfinished
    doc = (world / "docs" / "full_v5_baseline_identity.md").read_text()
    assert "454ac6013d13f351c6c1edf5dde832b66a441639a9f35414c8de0d23a8c8c08f" in doc
    assert "454ac6013d13f351c6e1" not in doc
    _write_tests_passed(run_root)

    # Development may not open the test split: real access guards.  The V4
    # test file is REMOVED first, proving the stage guard fires before any
    # file handle is created (otherwise FileNotFoundError would surface).
    config = yaml.safe_load(config_path.read_text())
    data.set_stage("dev")
    with pytest.raises(PermissionError):
        data.BaselineSource(config, world).test_quantiles("NSW1")
    with pytest.raises(PermissionError):
        data.load_cache_split(world / "data_cache", "NSW1", "test")
    v4_removed = world / "v4" / "NSW1_test.npz"
    v4_backup = v4_removed.read_bytes()
    v4_removed.unlink()
    with pytest.raises(PermissionError):
        data.V4Source(config, world).quantiles("NSW1", "test", np.array([0], dtype=np.int64))
    with pytest.raises(PermissionError):
        data.V4Source(config, world).file_digest("NSW1", "test")
    v4_removed.write_bytes(v4_backup)
    with pytest.raises(PermissionError):
        data.load_prepared_baseline(run_root, "NSW1", "test")
    data.set_stage("prepare")

    # Hash freeze: modifying a V5 source file or the config must make later
    # stages refuse until the exact original bytes are restored.
    source = world / "forecasting" / "full_v5" / "schema.py"
    original_source = source.read_bytes()
    source.write_bytes(original_source + b"\n# tampered\n")
    with pytest.raises(ValueError, match="changed after prepare"):
        experiment.dev(str(run_root))
    source.write_bytes(original_source)
    original_config = config_path.read_bytes()
    config_path.write_bytes(original_config + b"\n# tampered\n")
    with pytest.raises(ValueError, match="changed after prepare"):
        experiment.dev(str(run_root))
    config_path.write_bytes(original_config)
    # A bound V5 test source must be refused as well.
    test_source = world / "tests" / "test_full_v5_history.py"
    original_test_source = test_source.read_bytes()
    test_source.write_bytes(original_test_source + b"\n# tampered\n")
    with pytest.raises(ValueError, match="tests changed after prepare"):
        experiment.dev(str(run_root))
    test_source.write_bytes(original_test_source)
    # A bound reused dependency source must be refused as well.
    reused = world / "forecasting" / "aligned_scoring.py"
    original_reused = reused.read_bytes()
    reused.write_bytes(original_reused + b"\n# tampered\n")
    with pytest.raises(ValueError, match="reused_sources changed after prepare"):
        experiment.dev(str(run_root))
    reused.write_bytes(original_reused)
    # Round-6 H2: the lazily imported execution dependency chain is bound too
    # (predictor -> aligned_baselines -> lasso.model, the package init's
    # datasets/full chain, and the src/dualfield torch modules; all refuse).
    for relative in ("forecasting/aligned_baselines.py", "forecasting/lasso/model.py",
                     "forecasting/datasets.py", "src/dualfield/core.py"):
        dep = world / relative
        original_dep = dep.read_bytes()
        dep.write_bytes(original_dep + b"\n# tampered\n")
        with pytest.raises(ValueError, match="reused_sources changed after prepare"):
            experiment.dev(str(run_root))
        dep.write_bytes(original_dep)
    # A mutated prepared baseline prediction file must be refused.
    baseline_npz = run_root / "baseline_predictions" / "NSW1_calibration.npz"
    baseline_backup = baseline_npz.read_bytes()
    with np.load(baseline_npz, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["quantiles"] = payload["quantiles"] + 1.0
    np.savez_compressed(baseline_npz, **payload)
    with pytest.raises(ValueError, match="Prepared baseline"):
        experiment.dev(str(run_root))
    baseline_npz.write_bytes(baseline_backup)

    assert experiment.dev(str(run_root)) == 0
    gates = json.loads((run_root / "development" / "gates.json").read_text())
    assert gates["passed"], json.dumps({k: gates[k].get("passed") for k in ("G0", "G1", "G2")})
    assert (run_root / "development" / "report.md").exists() or (run_root / "reports" / "dev_report.md").exists()

    assert experiment.freeze(str(run_root)) == 0
    freeze = json.loads((run_root / "freeze.json").read_text())
    assert freeze["model_files"]

    assert experiment.final_test(str(run_root)) == 0
    assert (run_root / "final_test" / "complete.json").exists()
    for model in schema.ALL_MODELS:
        assert (run_root / "final_test" / "predictions" / model / "NSW1_test.npz").exists()

    # One-shot final evaluation: a re-run must refuse, not rescore.
    with pytest.raises(PermissionError):
        experiment.final_test(str(run_root))

    assert experiment.report_stage(str(run_root)) == 0
    report_text = (run_root / "reports" / "final_report.md").read_text()
    assert "2023-2024 已在此前研究中反复查看" in report_text


def test_failed_dev_gates_block_freeze_and_final_test(tmp_path, monkeypatch, patched_schema):
    world = build_world(tmp_path, structured=False)
    monkeypatch.chdir(world)
    run_root = world / "run"
    assert experiment.prepare(str(world / "configs" / "full_v5" / "experiment.yaml"), str(run_root)) == 0
    _write_tests_passed(run_root)
    assert experiment.dev(str(run_root)) == 0
    gates = json.loads((run_root / "development" / "gates.json").read_text())
    assert not gates["passed"]
    with pytest.raises(PermissionError):
        experiment.freeze(str(run_root))
    with pytest.raises(PermissionError):
        experiment.final_test(str(run_root))
    assert not (run_root / "final_test").exists() or not (run_root / "final_test" / "complete.json").exists()
    # The report must state that no final evaluation ran (never claim one did).
    assert experiment.report_stage(str(run_root)) == 0
    report_text = (run_root / "reports" / "final_report.md").read_text()
    assert "本次未运行任何新增最终评价" in report_text
    assert "仅进行一次新增最终评价" not in report_text


def test_dev_resume_and_completed_state_reject_tampering(tmp_path, monkeypatch, patched_schema):
    """Round-4 H2: done markers bind models and predictions; tampering refuses."""
    world = build_world(tmp_path, structured=True)
    monkeypatch.chdir(world)
    run_root = world / "run"
    assert experiment.prepare(str(world / "configs" / "full_v5" / "experiment.yaml"), str(run_root)) == 0
    _write_tests_passed(run_root)
    # Simulated crash after the region finished but before gates evaluation.
    real_evaluate = experiment.evaluate_gates

    def crash(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(experiment, "evaluate_gates", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        experiment.dev(str(run_root))
    marker = run_root / "development" / "done_NSW1.json"
    assert marker.exists()
    outputs = json.loads(marker.read_text())["outputs"]
    assert any(rel.endswith("models/C_roll_NSW1.json") for rel in outputs)
    assert any("/predictions/" in rel for rel in outputs)
    assert any(rel.endswith("models/T_roll/NSW1/q50.txt") for rel in outputs)
    # Tampering a persisted development model must make resume refuse.
    model = run_root / "development" / "models" / "C_roll_NSW1.json"
    original_model = model.read_bytes()
    payload = json.loads(original_model)
    payload["params"]["0"][0][0] = 0.5
    model.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="outputs changed or missing"):
        experiment.dev(str(run_root))
    model.write_bytes(original_model)
    monkeypatch.setattr(experiment, "evaluate_gates", real_evaluate)
    assert experiment.dev(str(run_root)) == 0        # clean resume completes
    # Completed (dev_passed) state: tampered predictions must be refused too.
    prediction = run_root / "development" / "predictions" / "C_roll" / "NSW1_v1.npz"
    backup = prediction.read_bytes()
    with np.load(prediction, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["quantiles"] = payload["quantiles"] + 1.0
    np.savez_compressed(prediction, **payload)
    with pytest.raises(ValueError, match="outputs changed or missing"):
        experiment.dev(str(run_root))
    prediction.write_bytes(backup)
    assert experiment.dev(str(run_root)) == 0


def test_budget_exhaustion_stops_and_persists_usage(tmp_path, monkeypatch, patched_schema):
    """Round-4 M4: CPU core-hour ceilings are enforced; usage survives failure."""
    _small_schema(monkeypatch)
    world = build_world(tmp_path, structured=True)
    monkeypatch.chdir(world)
    config_path = world / "configs" / "full_v5" / "experiment.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["v5_calibrated_history"]["budget"]["dev_core_hours"] = 0.0
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    run_root = world / "run"
    assert experiment.prepare(str(config_path), str(run_root)) == 0
    _write_tests_passed(run_root)
    with pytest.raises(experiment.BudgetExceeded):
        experiment.main(["dev", "--run-root", str(run_root)])
    # Round-9: the region lock owns the budget lifecycle, so the exhausted
    # attempt books into its region record (ledger folded, no live keys).
    budget = json.loads((run_root / "development" / "regions" / "NSW1" / "budget.json").read_text())
    assert "dev" in budget and budget["dev"]["cpu_seconds"] >= 0.0
    assert "live_pid" not in budget["dev"]
    state = json.loads((run_root / "state.json").read_text())
    assert state["stage"] == "prepared" and state["status"] == "interrupted"
    commands = (run_root / "development" / "regions" / "NSW1" / "commands.jsonl").read_text()
    assert "BudgetExceeded" in commands.strip().splitlines()[-1]
    assert not (run_root / "development" / "gates.json").exists()


def test_final_run_lock_lifecycle_held_and_no_owner_race(tmp_path):
    """Round-6 High: the run lock is an OS flock held for the process
    lifetime.  A second acquirer (even in the same process, via a second open
    file description) is refused while the first lives; the lock file is
    NEVER removed merely because owner metadata is absent, stale or garbage —
    a leftover file without a live holder is simply re-acquirable."""
    paths = experiment._run_paths(tmp_path / "run")
    lock = experiment._acquire_final_lock(paths)
    lock_file = paths["final"] / "run.lock"
    assert lock_file.exists()
    with pytest.raises(PermissionError, match="held"):
        experiment._acquire_final_lock(paths)      # second open description
    lock.release()
    # released by us -> immediately re-acquirable
    again = experiment._acquire_final_lock(paths)
    again.release()
    # a leftover lock file with missing/garbage owner metadata is NOT
    # deletion evidence and must not block or be deleted
    lock_file.write_text("garbage-without-owner-metadata")
    recovered = experiment._acquire_final_lock(paths)
    recovered.release()
    assert lock_file.exists()


def test_sequential_final_test_resume_and_future_label_perturbation(
        tmp_path, monkeypatch, patched_schema):
    """Round-4 M2: sequential matured-queue final evaluation with atomic
    per-block persistence, identity-safe resume, block tamper rejection, and
    immunity of already-predicted origins to later-maturing label changes."""
    world = build_world(tmp_path / "base", structured=True)
    perturbed_world = build_world(tmp_path / "perturbed", structured=True,
                                  perturb_test_labels_from="2022-08-24")

    monkeypatch.chdir(world)
    run_root = world / "run"
    config_path = world / "configs" / "full_v5" / "experiment.yaml"
    assert experiment.prepare(str(config_path), str(run_root)) == 0
    _write_tests_passed(run_root)
    assert experiment.dev(str(run_root)) == 0
    assert json.loads((run_root / "development" / "gates.json").read_text())["passed"]
    assert experiment.freeze(str(run_root)) == 0
    freeze_payload = json.loads((run_root / "freeze.json").read_text())
    assert freeze_payload["test_sources"]["cache"]["NSW1"]
    assert freeze_payload["test_sources"]["baseline_test"]["NSW1"]
    assert freeze_payload["test_sources"]["v4_test"]["NSW1"]

    # Pristine copy for the uninterrupted reference run.
    run_clean = world / "run_clean"
    shutil.copytree(run_root, run_clean)

    # Interrupted sequential run: crash while persisting block index 2.
    original_persist = experiment._persist_final_block

    def flaky_persist(*args, **kwargs):
        if args[2] == 2:
            raise RuntimeError("simulated interruption")
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(experiment, "_persist_final_block", flaky_persist)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        experiment.final_test(str(run_root))
    # the lock is kernel-held: the crashed process no longer holds it and a
    # fresh acquirer may proceed (the leftover file itself is not a holder)
    freed = experiment._acquire_final_lock(experiment._run_paths(run_root))
    freed.release()
    manifest = json.loads((run_root / "final_test" / "blocks" / "NSW1" / "manifest.json").read_text())
    assert set(manifest["blocks"]) == {"0", "1"}
    # Tampering a persisted block file must make resume refuse.
    block_file = run_root / "final_test" / "blocks" / "NSW1" / "B" / "block_00000.npz"
    block_backup = block_file.read_bytes()
    with np.load(block_file, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["quantiles"] = payload["quantiles"] + 5.0
    np.savez_compressed(block_file, **payload)
    with pytest.raises(ValueError, match="changed after"):
        experiment.final_test(str(run_root))
    block_file.write_bytes(block_backup)
    monkeypatch.setattr(experiment, "_persist_final_block", original_persist)
    assert experiment.final_test(str(run_root)) == 0     # resumes from block 2
    complete = json.loads((run_root / "final_test" / "complete.json").read_text())
    assert complete["sequential"]["resumed_from"]["NSW1"] == 2
    with pytest.raises(PermissionError):
        experiment.final_test(str(run_root))             # one-shot, no re-scoring

    # Uninterrupted reference run in the pristine copy: identical results.
    assert experiment.final_test(str(run_clean)) == 0
    for model in schema.ALL_MODELS:
        a = np.load(run_root / "final_test" / "predictions" / model / "NSW1_test.npz")
        b = np.load(run_clean / "final_test" / "predictions" / model / "NSW1_test.npz")
        assert np.array_equal(a["quantiles"], b["quantiles"]), model
        assert np.array_equal(a["origin_unix"], b["origin_unix"])
        assert np.array_equal(a["actual"], b["actual"])

    # Future-label perturbation: only labels delivered >= 2022-08-24 differ.
    monkeypatch.chdir(perturbed_world)
    run2 = perturbed_world / "run"
    config2 = perturbed_world / "configs" / "full_v5" / "experiment.yaml"
    assert experiment.prepare(str(config2), str(run2)) == 0
    _write_tests_passed(run2)
    assert experiment.dev(str(run2)) == 0
    assert experiment.freeze(str(run2)) == 0
    assert experiment.final_test(str(run2)) == 0
    cutoff = schema.unix_timestamp("2022-08-24") + 3600
    late_changed = set()
    for model in schema.ALL_MODELS:
        a = np.load(run_root / "final_test" / "predictions" / model / "NSW1_test.npz")
        b = np.load(run2 / "final_test" / "predictions" / model / "NSW1_test.npz")
        early = a["origin_unix"] < cutoff
        assert int(early.sum()) >= 48
        # Every prediction made before the perturbed labels could mature is
        # bit-for-bit unchanged.
        assert np.array_equal(a["quantiles"][early], b["quantiles"][early]), model
        if not np.array_equal(a["quantiles"][~early], b["quantiles"][~early]):
            late_changed.add(model)
    # History-consuming controls must react once the perturbed labels mature.
    assert {"T_roll", "T_history"} & late_changed, late_changed
