from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    b = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
    return pearson(a, b)


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return 0.0
    a = a[mask] - a[mask].mean()
    b = b[mask] - b[mask].mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        raise ValueError("No finite prediction-target pairs available.")
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": spearman(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "n": int(len(y_true)),
    }
