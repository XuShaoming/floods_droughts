"""Regression and hydrological performance metrics."""

from typing import Dict

import numpy as np


METRIC_NAMES = (
    "MSE",
    "RMSE",
    "MAE",
    "R2",
    "MAPE",
    "NSE",
    "KGE",
    "KGE_r",
    "KGE_alpha",
    "KGE_beta",
    "NRMSE_sigma",
    "Bias",
)


def _nan_metrics() -> Dict[str, float]:
    """Return the complete metric schema populated with NaNs."""
    return {name: float("nan") for name in METRIC_NAMES}


def calculate_hydrology_metrics(predictions, observations) -> Dict[str, float]:
    """Calculate regression metrics for paired predictions and observations.

    Arrays are flattened, and pairs containing a non-finite value are omitted.
    KGE follows Gupta et al. (2009), with components ``r`` (Pearson
    correlation), ``alpha`` (standard-deviation ratio), and ``beta`` (mean
    ratio). ``Bias`` uses the signed convention ``mean(prediction -
    observation)``.

    Metrics with a zero denominator or insufficient variability are returned
    as NaN.
    """
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    obs = np.asarray(observations, dtype=np.float64).reshape(-1)
    if pred.shape != obs.shape:
        raise ValueError(
            "Predictions and observations must contain the same number of values "
            f"(got {pred.size} and {obs.size})."
        )

    valid = np.isfinite(pred) & np.isfinite(obs)
    pred = pred[valid]
    obs = obs[valid]
    if obs.size == 0:
        return _nan_metrics()

    residual = pred - obs
    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))

    obs_mean = float(np.mean(obs))
    pred_mean = float(np.mean(pred))
    obs_std = float(np.std(obs))
    pred_std = float(np.std(pred))

    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((obs - obs_mean) ** 2))
    nse = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    nonzero_obs = obs != 0.0
    mape = (
        float(np.mean(np.abs(residual[nonzero_obs] / obs[nonzero_obs])) * 100.0)
        if np.any(nonzero_obs)
        else float("nan")
    )

    nrmse_sigma = float(rmse / obs_std) if obs_std > 0.0 else float("nan")
    alpha = float(pred_std / obs_std) if obs_std > 0.0 else float("nan")
    beta = float(pred_mean / obs_mean) if obs_mean != 0.0 else float("nan")

    if obs.size >= 2 and obs_std > 0.0 and pred_std > 0.0:
        covariance = float(np.mean((obs - obs_mean) * (pred - pred_mean)))
        correlation = float(covariance / (obs_std * pred_std))
        # Protect against tiny floating-point excursions outside [-1, 1].
        correlation = float(np.clip(correlation, -1.0, 1.0))
    else:
        correlation = float("nan")

    if np.all(np.isfinite([correlation, alpha, beta])):
        kge = float(
            1.0
            - np.sqrt(
                (correlation - 1.0) ** 2
                + (alpha - 1.0) ** 2
                + (beta - 1.0) ** 2
            )
        )
    else:
        kge = float("nan")

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        # R2 and NSE share the 1 - SSE/SST definition for this application.
        "R2": nse,
        "MAPE": mape,
        "NSE": nse,
        "KGE": kge,
        "KGE_r": correlation,
        "KGE_alpha": alpha,
        "KGE_beta": beta,
        "NRMSE_sigma": nrmse_sigma,
        "Bias": bias,
    }
