#!/usr/bin/env python3
"""
Mutant-vs-wild-type EGFR selectivity analysis.

The clinical goal is a compound that is potent against the resistance mutants
(especially the C797S-bearing triple mutant `L858R/T790M/C797S`) while *sparing*
wild-type EGFR — WT inhibition is what drives the dose-limiting skin/GI toxicity
of EGFR inhibitors.

For every compound with paired measurements this computes:
    selectivity(variant) = pIC50(variant) - pIC50(WT)         # >0  => mutant-preferring
and classifies "mutant-selective hits": active on the mutant (pIC50 >= 6, i.e.
IC50 <= 1 uM) AND not markedly worse than WT (selectivity >= -0.5).

Also trains a scaffold-split classifier on the triple-mutant compounds so a
mutant-activity probability can be assigned to compounds that were only ever
tested against WT.

Outputs:
    artifacts/selectivity.csv          per-compound paired pIC50 + selectivity
    artifacts/selectivity_summary.json counts, and the mutant-model metrics
    artifacts/mutant_activity_scores.csv  molecule_chembl_id, mutant_activity (prob)
    models/mutant_activity_classifier.joblib
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)

from src.pipeline.metrics import classification_report_with_ci  # noqa: E402
from src.pipeline.split import scaffold_split  # noqa: E402

PROC = os.path.join(REPO_ROOT, "data/processed")
ARTIFACTS = os.path.join(REPO_ROOT, "artifacts")
MODELS = os.path.join(REPO_ROOT, "models")

TRIPLE = "L858R/T790M/C797S"
DOUBLE = "L858R/T790M"
C797S_VARIANTS = (TRIPLE, "T790M/C797S")
SEED = 42
ACTIVE_PIC50 = 6.0


def _feature_cols(df: pd.DataFrame):
    desc = ["mw", "logp", "hbd", "hba", "tpsa", "rotatable_bonds", "aromatic_rings"]
    return desc + [c for c in df.columns if c.startswith("fp_")]


def build_selectivity_table(by_variant: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per compound with a pIC50 column per variant + selectivity deltas."""
    piv = by_variant.pivot_table(
        index=["molecule_chembl_id", "canonical_smiles"],
        columns="variant",
        values="pIC50",
        aggfunc="median",
    ).reset_index()
    piv.columns.name = None

    for v in (TRIPLE, DOUBLE, "T790M", "T790M/C797S"):
        if v in piv.columns and "WT" in piv.columns:
            piv[f"sel[{v}-WT]"] = piv[v] - piv["WT"]

    # C797S-family potency = best (max) pIC50 across any C797S-bearing variant tested
    c797s_cols = [c for c in C797S_VARIANTS if c in piv.columns]
    if c797s_cols:
        piv["c797s_pIC50"] = piv[c797s_cols].max(axis=1)
        if "WT" in piv.columns:
            piv["c797s_selectivity"] = piv["c797s_pIC50"] - piv["WT"]
    return piv


def classify_hits(sel: pd.DataFrame) -> pd.DataFrame:
    """Flag mutant-selective hits among compounds with paired C797S + WT data."""
    have = sel.dropna(subset=["c797s_pIC50", "WT"]).copy() if "c797s_pIC50" in sel.columns else pd.DataFrame()
    if have.empty:
        return have
    have["mutant_active"] = have["c797s_pIC50"] >= ACTIVE_PIC50
    have["wt_sparing"] = have["c797s_selectivity"] >= -0.5
    have["mutant_selective_hit"] = have["mutant_active"] & have["wt_sparing"]
    return have.sort_values("c797s_selectivity", ascending=False)


def train_mutant_activity_model(feat_by_variant: pd.DataFrame) -> dict:
    """Scaffold-split classifier: is a compound active against the triple mutant?"""
    trip = feat_by_variant[feat_by_variant["variant"] == TRIPLE].reset_index(drop=True)
    if len(trip) < 80:
        return {"trained": False, "reason": f"only {len(trip)} triple-mutant compounds"}

    trip["is_active"] = (trip["pIC50"] >= ACTIVE_PIC50).astype(int)
    train_df, test_df = scaffold_split(trip, test_size=0.25, seed=SEED)
    cols = _feature_cols(trip)

    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=250, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(train_df[cols].values, train_df["is_active"].values)
    y_prob = model.predict_proba(test_df[cols].values)[:, 1]
    metrics = classification_report_with_ci(test_df["is_active"].values, y_prob, n_boot=1000, seed=SEED)

    # y-scramble control
    rng = np.random.RandomState(SEED)
    y_shuf = train_df["is_active"].values.copy()
    rng.shuffle(y_shuf)
    ctrl = XGBClassifier(
        n_estimators=250, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED, n_jobs=-1, eval_metric="logloss",
    )
    ctrl.fit(train_df[cols].values, y_shuf)
    ctrl_auc = classification_report_with_ci(
        test_df["is_active"].values, ctrl.predict_proba(test_df[cols].values)[:, 1],
        n_boot=1000, seed=SEED,
    )["roc_auc"]

    os.makedirs(MODELS, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS, "mutant_activity_classifier.joblib"))

    return {
        "trained": True,
        "n_triple_mutant_compounds": int(len(trip)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "active_frac": float(trip["is_active"].mean()),
        "metrics": metrics,
        "y_scramble_control": {"roc_auc": ctrl_auc},
        "feature_cols_count": len(cols),
    }


def score_all_compounds(feat_wt: pd.DataFrame) -> pd.DataFrame:
    """Apply the mutant-activity model to every WT-featurized compound."""
    model_path = os.path.join(MODELS, "mutant_activity_classifier.joblib")
    if not os.path.exists(model_path):
        return pd.DataFrame(columns=["molecule_chembl_id", "mutant_activity"])
    model = joblib.load(model_path)
    cols = _feature_cols(feat_wt)
    prob = model.predict_proba(feat_wt[cols].values)[:, 1]
    return pd.DataFrame({
        "molecule_chembl_id": feat_wt["molecule_chembl_id"].values,
        "mutant_activity": prob,
    })


def main() -> None:
    os.makedirs(ARTIFACTS, exist_ok=True)
    by_variant = pd.read_parquet(os.path.join(PROC, "egfr_by_variant.parquet"))

    sel = build_selectivity_table(by_variant)
    sel.to_csv(os.path.join(ARTIFACTS, "selectivity.csv"), index=False)
    hits = classify_hits(sel)

    feat_bv_path = os.path.join(PROC, "egfr_features_by_variant.parquet")
    model_info = {"trained": False, "reason": "features_by_variant not found"}
    if os.path.exists(feat_bv_path):
        model_info = train_mutant_activity_model(pd.read_parquet(feat_bv_path))

    feat_wt_path = os.path.join(PROC, "egfr_features.parquet")
    if os.path.exists(feat_wt_path) and model_info.get("trained"):
        scores = score_all_compounds(pd.read_parquet(feat_wt_path))
        scores.to_csv(os.path.join(ARTIFACTS, "mutant_activity_scores.csv"), index=False)

    summary = {
        "seed": SEED,
        "paired_WT_and_C797S_compounds": int(len(hits)) if not hits.empty else 0,
        "mutant_selective_hits": int(hits["mutant_selective_hit"].sum()) if not hits.empty else 0,
        "median_c797s_selectivity": float(hits["c797s_selectivity"].median()) if not hits.empty else None,
        "mutant_activity_model": model_info,
        "variant_row_counts": by_variant["variant"].value_counts().to_dict(),
    }
    with open(os.path.join(ARTIFACTS, "selectivity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"[OK] selectivity table: {len(sel)} compounds -> artifacts/selectivity.csv")
    if not hits.empty:
        print(f"[OK] {len(hits)} compounds with paired C797S+WT data; "
              f"{int(hits['mutant_selective_hit'].sum())} mutant-selective hits")
        print(hits.head(10)[["molecule_chembl_id", "WT", "c797s_pIC50", "c797s_selectivity",
                             "mutant_selective_hit"]].to_string(index=False))
    if model_info.get("trained"):
        m = model_info["metrics"]
        print(f"[OK] mutant-activity model ROC-AUC {m['roc_auc']['value']:.3f} "
              f"[{m['roc_auc']['ci_low']:.3f},{m['roc_auc']['ci_high']:.3f}]  "
              f"(y-scramble {model_info['y_scramble_control']['roc_auc']['value']:.3f})")


if __name__ == "__main__":
    main()
