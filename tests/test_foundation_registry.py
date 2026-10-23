"""Registry, layout and config contracts for the 2026-10-23 foundation batch."""
import json
from pathlib import Path

import pytest
import yaml

from forecasting.aligned_expansion import validate_expansion_config
from forecasting.common.baseline_registry import EXPANSION_MODELS, expansion_model_ids
from forecasting.common.foundation_registry import (
    FOUNDATION_BATCH, FOUNDATION_MODELS, FOUNDATION_REGISTRY, FOUNDATION_SCHEMA,
    foundation_availability, foundation_model_ids,
)
from forecasting.foundation_experiment import validate_foundation_config

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = ("__init__.py", "model.py", "adapter.py", "source.json",
                  "CITATIONS.bib", "README.md", "requirements.lock.txt", "ENVIRONMENT.md")
FIRST_BATCH = ("lightgbm_cp", "lightgbm_qra", "xgboost_cp", "ddnn_jsu", "nhits_qra")


def test_first_batch_registry_is_untouched():
    assert expansion_model_ids() == FIRST_BATCH
    assert [spec.model_id for spec in EXPANSION_MODELS] == list(FIRST_BATCH)
    # The frozen first-batch matrix config still validates under its own rules.
    config = yaml.safe_load(
        (ROOT / "configs/experiments/aemo_forecast_aligned_expansion_20261022.yaml").read_text())
    assert validate_expansion_config(config) == list(FIRST_BATCH)


def test_foundation_registry_declares_exactly_the_four_models():
    assert foundation_model_ids() == (
        "tabpfn_v3", "chronos2_zeroshot", "moirai2_zeroshot", "timesfm25_xreg")
    for spec in FOUNDATION_MODELS:
        assert spec.module == f"forecasting.{spec.model_id}.adapter"
        assert spec.entry == "Adapter"
        assert spec.output_table in ("Q", "P")
        assert FOUNDATION_SCHEMA == "foundation-expansion-20261023-v1"
        assert FOUNDATION_BATCH == "second_20261023"
        # The new stage/scope declarations the first batch could not express.
        scope = spec.scope_declaration()
        assert isinstance(scope["fit_requires_train_labels"], bool)
        assert scope["weight_training"] is False  # frozen pretrained backbones only
        assert spec.selection_use == "none"
        assert spec.calibration_use is False


def test_expected_capability_groups():
    assert FOUNDATION_REGISTRY["tabpfn_v3"].output_table == "Q"
    assert FOUNDATION_REGISTRY["chronos2_zeroshot"].output_table == "Q"
    assert FOUNDATION_REGISTRY["moirai2_zeroshot"].output_table == "P"
    assert FOUNDATION_REGISTRY["timesfm25_xreg"].output_table == "P"
    assert FOUNDATION_REGISTRY["tabpfn_v3"].fit_requires_train_labels is True
    assert FOUNDATION_REGISTRY["chronos2_zeroshot"].fit_requires_train_labels is False
    assert FOUNDATION_REGISTRY["timesfm25_xreg"].fit_requires_train_labels is False


def test_every_model_directory_has_the_required_files():
    for spec in FOUNDATION_MODELS:
        directory = ROOT / "forecasting" / spec.model_id
        for name in REQUIRED_FILES:
            assert (directory / name).is_file(), f"{spec.model_id}/{name} missing"
        source = json.loads((directory / "source.json").read_text())
        for key in ("implementation_identity", "experiment_model_id", "paper_reference",
                    "paper_url", "repository", "weights", "dependency_versions",
                    "input_mapping", "output_quantiles", "randomness", "context_policy",
                    "differences_from_paper", "verified_date"):
            assert key in source, f"{spec.model_id}/source.json lacks {key}"
        assert (ROOT / "configs" / spec.model_id / "aligned.yaml").is_file()


def test_foundation_config_validation():
    config = yaml.safe_load(
        (ROOT / "configs/experiments/aemo_forecast_foundation_20261023.yaml").read_text())
    models = validate_foundation_config(config)
    assert set(models) == set(foundation_model_ids())
    assert config["foundation"]["schema"] == FOUNDATION_SCHEMA
    single = yaml.safe_load(yaml.safe_dump(config))
    single["experiment"]["models"] = ["chronos2_zeroshot"]
    single["experiment"]["execution_scope"] = "single_model"
    assert validate_foundation_config(single) == ["chronos2_zeroshot"]
    for mutation in (
        {"models": ["tabpfn_v3", "chronos2_zeroshot"]},                     # partial matrix
        {"models": ["lightgbm"]},                                           # outside registry
        {"models": list(foundation_model_ids()) + ["tabpfn_v3"]},           # duplicate
    ):
        broken = yaml.safe_load(yaml.safe_dump(config))
        broken["experiment"].update(mutation)
        with pytest.raises(ValueError):
            validate_foundation_config(broken)
    broken = yaml.safe_load(yaml.safe_dump(config))
    del broken["experiment"]["interpreters"]["moirai2_zeroshot"]
    with pytest.raises(ValueError):
        validate_foundation_config(broken)
    broken = yaml.safe_load(yaml.safe_dump(config))
    broken["foundation"]["schema"] = "aligned-expansion-20261022-v1"
    with pytest.raises(ValueError):
        validate_foundation_config(broken)


def test_availability_reports_missing_dependencies_without_importing_models():
    status = foundation_availability()
    assert set(status) == set(foundation_model_ids())
    for model, row in status.items():
        assert row["status"] in ("available", "missing_dependency")
        assert row["registered_entry"].endswith("adapter::Adapter")
