import warnings

import numpy as np


def zscore(data, axis=0, ddof=0, nan_policy="propagate"):
    """Compute z-scores along an axis, without SciPy."""
    values = np.asarray(data, dtype=float)

    if nan_policy not in {"propagate", "raise", "omit"}:
        raise ValueError("nan_policy must be 'propagate', 'raise', or 'omit'")
    if nan_policy == "raise" and np.isnan(values).any():
        raise ValueError("input contains NaN")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if nan_policy == "omit":
            mean = np.nanmean(values, axis=axis, keepdims=True)
            std = np.nanstd(values, axis=axis, ddof=ddof, keepdims=True)
        else:
            mean = np.mean(values, axis=axis, keepdims=True)
            std = np.std(values, axis=axis, ddof=ddof, keepdims=True)

        # Match SciPy: constant or insufficient data produces NaN z-scores.
        std = np.where(std == 0, np.nan, std)
        return (values - mean) / std