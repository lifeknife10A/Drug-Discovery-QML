import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.pipeline.metrics import (
    bootstrap_ci,
    classification_report_with_ci,
    enrichment_factor,
    regression_report,
)


def _synthetic_classification(n=200, seed=42):
    rng = np.random.RandomState(seed)
    y_true = (rng.rand(n) > 0.7).astype(int)
    # score correlated with label
    y_score = y_true * rng.rand(n) * 0.5 + rng.rand(n) * 0.5
    return y_true, y_score


def test_enrichment_factor_perfect_ranking():
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.array([1.0, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    ef = enrichment_factor(y_true, y_score, fraction=0.2)
    assert ef == 5.0  # top 20% captures both actives, baseline rate = 0.2 -> EF = 1/0.2


def test_bootstrap_ci_bounds_contain_point_estimate():
    y_true, y_score = _synthetic_classification()
    from sklearn.metrics import roc_auc_score
    result = bootstrap_ci(y_true, y_score, roc_auc_score, n_boot=200, seed=42)
    assert result["ci_low"] <= result["value"] <= result["ci_high"]
    assert result["n_boot"] > 0


def test_classification_report_with_ci_keys():
    y_true, y_score = _synthetic_classification()
    report = classification_report_with_ci(y_true, y_score, n_boot=100, seed=42)
    for key in ["pr_auc", "roc_auc", "ef_1pct", "ef_5pct"]:
        assert key in report
        assert "value" in report[key] and "ci_low" in report[key] and "ci_high" in report[key]


def test_regression_report_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    report = regression_report(y_true, y_true)
    assert report["rmse"] == 0.0
    assert report["mae"] == 0.0
    assert report["r2"] == 1.0
