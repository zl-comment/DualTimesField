"""V4 orchestrates the audited native quantile-tree engine from V3."""
from forecasting.full_v3.model import (fit_residual_model, predict_residual_model,
                                      residual_input_provenance, _load_manifest)


def crossing_rate(model_directory, split, extras, *, threads=4):
    """Diagnostic before common sorting; translation/positive scale preserve crossing."""
    import lightgbm as lgb
    import numpy as np
    from forecasting.aligned_inputs import flatten_inputs
    from forecasting.aligned_scoring import QUANTILE_LEVELS
    directory, manifest = _load_manifest(model_directory)
    x = np.concatenate((flatten_inputs(split), extras), axis=1)
    values = np.empty((len(split), 24, 5), dtype=np.float64)
    for entry in manifest['files']:
        booster = lgb.Booster(model_file=str(directory / entry['path']))
        q = int(np.flatnonzero(QUANTILE_LEVELS == entry['quantile'])[0])
        values[:, entry['horizon'], q] = booster.predict(x, num_iteration=entry['selected_rounds'], num_threads=threads)
    return float((np.diff(values, axis=-1) < 0).any(axis=-1).mean())

__all__ = ['fit_residual_model', 'predict_residual_model', 'residual_input_provenance']
