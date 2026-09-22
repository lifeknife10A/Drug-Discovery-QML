#!/usr/bin/env python3
"""Weighted Drug Selection Score (DSS): combines whatever tier outputs are available
into artifacts/top_candidates.csv (Top-3 + full ranking).

Tier 1 (classical ligand model) is required. Tier 2 (docking), Tier 3 (quantum kernel),
and Tier 4 (ADMET) are optional — if their artifact files are absent, rank.py renormalizes
weights over whatever tiers ARE available and prints a notice.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

DEFAULT_WEIGHTS = {
    "tier1_activity": 0.5,
    "tier2_docking": 0.2,
    "tier3_quantum": 0.1,
    "tier4_admet": 0.2,
}


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    return (series - lo) / (hi - lo)


def _load_tier1_scores(artifacts_dir: str, models_dir: str, feature_path: str) -> pd.DataFrame:
    metrics_path = os.path.join(artifacts_dir, "tier1_metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Tier 1 metrics not found at {metrics_path}; run `run.py tier1` first.")

    with open(metrics_path) as f:
        tier1_metrics = json.load(f)

    best_name, best_auc = None, -1.0
    for name, payload in tier1_metrics["classifiers"].items():
        auc = payload["metrics"]["roc_auc"]["value"]
        if auc > best_auc:
            best_name, best_auc = name, auc

    model_path = os.path.join(models_dir, f"{best_name}_classifier.joblib")
    model = joblib.load(model_path)

    df = pd.read_parquet(feature_path)
    desc_cols = ["mw", "logp", "hbd", "hba", "tpsa", "rotatable_bonds", "aromatic_rings"]
    fp_cols = [c for c in df.columns if c.startswith("fp_")]
    feature_cols = desc_cols + fp_cols

    p_active = model.predict_proba(df[feature_cols].values)[:, 1]
    out = df[["molecule_chembl_id", "canonical_smiles"]].copy()
    out["tier1_activity_raw"] = p_active
    out["tier1_model"] = best_name
    out["tier1_roc_auc"] = best_auc
    return out


def _load_optional_tier(artifacts_dir: str, filename: str, score_col: str):
    path = os.path.join(artifacts_dir, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "molecule_chembl_id" not in df.columns or score_col not in df.columns:
        print(f"[rank] {filename} found but missing required columns; skipping.")
        return None
    return df[["molecule_chembl_id", score_col]]


def run_rank(
    artifacts_dir: str,
    models_dir: str = None,
    feature_path: str = None,
    top_n: int = 3,
) -> pd.DataFrame:
    models_dir = models_dir or os.path.join(REPO_ROOT, "models")
    feature_path = feature_path or os.path.join(REPO_ROOT, "data/processed/egfr_features.parquet")

    ranking = _load_tier1_scores(artifacts_dir, models_dir, feature_path)
    ranking["tier1_activity"] = _minmax(ranking["tier1_activity_raw"])

    optional_tiers = {
        "tier2_docking": ("docking_scores.csv", "docking_score"),
        "tier3_quantum": ("quantum_scores.csv", "quantum_score"),
        "tier4_admet": ("admet_scores.csv", "admet_score"),
    }

    active_weight_keys = ["tier1_activity"]
    for key, (filename, score_col) in optional_tiers.items():
        tier_df = _load_optional_tier(artifacts_dir, filename, score_col)
        if tier_df is None:
            print(f"[rank] {key} unavailable ({filename} not found) — excluded from DSS, notice logged.")
            continue
        tier_df = tier_df.rename(columns={score_col: f"{key}_raw"})
        ranking = ranking.merge(tier_df, on="molecule_chembl_id", how="left")
        ranking[f"{key}_raw"] = ranking[f"{key}_raw"].fillna(ranking[f"{key}_raw"].median())
        ranking[key] = _minmax(ranking[f"{key}_raw"])
        active_weight_keys.append(key)

    weight_sum = sum(DEFAULT_WEIGHTS[k] for k in active_weight_keys)
    ranking["dss_score"] = 0.0
    for key in active_weight_keys:
        norm_weight = DEFAULT_WEIGHTS[key] / weight_sum
        ranking["dss_score"] += norm_weight * ranking[key]

    ranking = ranking.sort_values("dss_score", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)

    os.makedirs(artifacts_dir, exist_ok=True)
    out_path = os.path.join(artifacts_dir, "top_candidates.csv")
    ranking.to_csv(out_path, index=False)

    top_n_df = ranking.head(top_n)
    print(f"[✓] Full ranking ({len(ranking)} candidates) written to {out_path}")
    print(f"[✓] Top-{top_n}:\n{top_n_df[['rank', 'molecule_chembl_id', 'dss_score']].to_string(index=False)}")

    return ranking


def main():
    artifacts_dir = os.path.join(REPO_ROOT, "artifacts")
    run_rank(artifacts_dir)


if __name__ == "__main__":
    main()
