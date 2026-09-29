#!/usr/bin/env python3
"""Tier 1: classical 2D ligand-based baseline.

Trains three classifiers (XGBoost, Random Forest, RBF-SVM) to predict activity and
an XGBoost regressor to predict pIC50, using a leakage-free Bemis-Murcko scaffold
split. Each classifier is evaluated with percentile-bootstrap 95% CIs and a
Y-scramble negative control, then serialized to ``{models_dir}/{name}_classifier.joblib``.
A consolidated report is written to ``{artifacts_dir}/tier1_metrics.json`` for the
downstream DSS ranking (``src/pipeline/rank.py``).

The feature columns are detected from the parquet at runtime (7 RDKit descriptors +
every ``fp_*`` Morgan-fingerprint bit present), so the same code path works for the
production 2048-bit dataset and the tiny 32-bit test fixtures alike.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from src.pipeline.metrics import classification_report_with_ci, regression_report  # noqa: E402
from src.pipeline.split import scaffold_split  # noqa: E402

SEED = 42
DESC_COLS = ["mw", "logp", "hbd", "hba", "tpsa", "rotatable_bonds", "aromatic_rings"]


def _feature_columns(df: pd.DataFrame) -> list:
    """Descriptor columns present + every Morgan fingerprint bit (``fp_*``)."""
    desc = [c for c in DESC_COLS if c in df.columns]
    fp = sorted((c for c in df.columns if c.startswith("fp_")), key=lambda c: int(c[3:]))
    return desc + fp


def _build_classifiers(smoke: bool):
    n_estimators = 50 if smoke else 300
    return {
        "xgboost": xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=SEED,
        ),
        # SVC needs scaled inputs; wrap in a pipeline so the persisted estimator can be
        # called with raw feature values by rank.py without a separate scaler artifact.
        "rbf_svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED),
        ),
    }


def _roc_auc_or_nan(y_true, y_score) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def run_tier1(
    feature_path: str,
    models_dir: str,
    artifacts_dir: str,
    n_boot: int = 1000,
    smoke: bool = False,
) -> dict:
    """Train Tier 1 models, persist them, and write ``tier1_metrics.json``.

    Returns the in-memory metrics dict (also serialized to disk).
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    if smoke:
        n_boot = min(n_boot, 50)

    df = pd.read_parquet(feature_path)
    feature_cols = _feature_columns(df)

    train_df, test_df = scaffold_split(df, test_size=0.2, seed=SEED)

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df["is_active"].astype(int).values
    y_test = test_df["is_active"].astype(int).values

    rng = np.random.RandomState(SEED)
    y_train_scrambled = rng.permutation(y_train)

    results = {
        "seed": SEED,
        "split_type": "bemis_murcko_scaffold",
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": len(feature_cols),
        "classifiers": {},
        "regression": {},
    }

    for name, clf in _build_classifiers(smoke).items():
        clf.fit(X_train, y_train)
        p_test = clf.predict_proba(X_test)[:, 1]
        metrics = classification_report_with_ci(y_test, p_test, n_boot=n_boot, seed=SEED)

        # Y-scramble negative control: labels shuffled, real features → AUC should ~= 0.5.
        control_clf = _build_classifiers(smoke)[name]
        control_clf.fit(X_train, y_train_scrambled)
        p_control = control_clf.predict_proba(X_test)[:, 1]

        results["classifiers"][name] = {
            "metrics": metrics,
            "y_scramble_control": {"roc_auc": _roc_auc_or_nan(y_test, p_control)},
        }

        joblib.dump(clf, os.path.join(models_dir, f"{name}_classifier.joblib"))

    # Regression head (pIC50) — XGBoost only.
    if "pIC50" in df.columns:
        reg = xgb.XGBRegressor(
            n_estimators=50 if smoke else 300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            random_state=SEED,
            n_jobs=-1,
        )
        reg.fit(X_train, train_df["pIC50"].values)
        y_pred = reg.predict(X_test)
        results["regression"]["xgboost"] = regression_report(test_df["pIC50"].values, y_pred)
        joblib.dump(reg, os.path.join(models_dir, "xgboost_regressor.joblib"))

    metrics_path = os.path.join(artifacts_dir, "tier1_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    best_name = max(
        results["classifiers"],
        key=lambda n: results["classifiers"][n]["metrics"]["roc_auc"]["value"],
    )
    best_auc = results["classifiers"][best_name]["metrics"]["roc_auc"]["value"]
    print(f"[tier1] {len(feature_cols)} features | best classifier: {best_name} "
          f"(ROC-AUC {best_auc:.4f}) -> {metrics_path}")

    return results


def main():
    feature_path = os.path.join(REPO_ROOT, "data/processed/egfr_features.parquet")
    models_dir = os.path.join(REPO_ROOT, "models")
    artifacts_dir = os.path.join(REPO_ROOT, "artifacts")
    run_tier1(feature_path, models_dir, artifacts_dir)


if __name__ == "__main__":
    main()
