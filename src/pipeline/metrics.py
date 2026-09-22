#!/usr/bin/env python3
"""Classification/regression metrics with percentile bootstrap 95% CIs."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def enrichment_factor(y_true, y_score, fraction: float) -> float:
    """EF@fraction: hit rate in the top `fraction` of ranked predictions vs. baseline hit rate."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    n_top = max(1, int(np.ceil(n * fraction)))

    order = np.argsort(-y_score)
    top_idx = order[:n_top]

    hit_rate_top = y_true[top_idx].sum() / n_top
    baseline_rate = y_true.sum() / n
    if baseline_rate == 0:
        return 0.0
    return hit_rate_top / baseline_rate


def bootstrap_ci(y_true, y_score, metric_fn, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap 95% CI for a metric_fn(y_true, y_score)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    rng = np.random.RandomState(seed)

    point = metric_fn(y_true, y_score)

    boot_vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            boot_vals.append(metric_fn(yt, ys))
        except ValueError:
            continue

    if not boot_vals:
        return {"value": float(point), "ci_low": float(point), "ci_high": float(point), "n_boot": 0}

    lo = np.percentile(boot_vals, 100 * (alpha / 2))
    hi = np.percentile(boot_vals, 100 * (1 - alpha / 2))
    return {"value": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_boot": len(boot_vals)}


def classification_report_with_ci(y_true, y_score, n_boot: int = 1000, seed: int = 42) -> dict:
    """PR-AUC, ROC-AUC, EF@1%, EF@5% each with a percentile bootstrap 95% CI."""
    return {
        "pr_auc": bootstrap_ci(y_true, y_score, average_precision_score, n_boot, seed),
        "roc_auc": bootstrap_ci(y_true, y_score, roc_auc_score, n_boot, seed),
        "ef_1pct": bootstrap_ci(y_true, y_score, lambda yt, ys: enrichment_factor(yt, ys, 0.01), n_boot, seed),
        "ef_5pct": bootstrap_ci(y_true, y_score, lambda yt, ys: enrichment_factor(yt, ys, 0.05), n_boot, seed),
    }


def regression_report(y_true, y_pred) -> dict:
    """RMSE / MAE / R2 regression report."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}
