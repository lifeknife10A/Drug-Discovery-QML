#!/usr/bin/env python3
"""Tier 3 candidate scoring: applies the fidelity quantum kernel QSVC that
src/quantum/run_benchmark.py benchmarks to a bounded slice of real candidates,
writing artifacts/quantum_scores.csv for src/pipeline/rank.py's DSS fusion.

This is deliberately a SEPARATE concern from run_benchmark.py, which exists to
honestly characterize quantum-vs-classical performance on a held-out scaffold
split (see its "no quantum advantage claimed" disclaimer) -- not to produce
deployable scores. Scoring the full ~13k-compound library the same way the
benchmark evaluates itself would mean fitting SVC on a precomputed kernel
over the full ~10.4k-row scaffold-split training partition, which is not
practical to run (O(n^2) kernel matrix, likely minutes-to-hours on a
simulator). So both the training set and the query set are bounded here:

  - Training: a stratified sample (both classes represented) of the
    scaffold-split train partition, capped at max_train_size.
  - Query: the top_n candidates by Tier 1 activity probability (same
    "prioritize, then only spend expensive tiers on the shortlist"
    principle src/pipeline/dock.py uses for Tier 2 docking).
"""

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pipeline.split import scaffold_split  # noqa: E402
from src.quantum.features import compress_features, extract_feature_matrix  # noqa: E402
from src.quantum.kernels import PennyLaneFidelityKernel, QSVCClassifier  # noqa: E402

DEFAULT_N_QUBITS = 4  # matches the benchmark's negative-control / baseline dimension
DEFAULT_MAX_TRAIN_SIZE = 2000
DEFAULT_TOP_N = 300


def _stratified_train_sample(train_df: pd.DataFrame, max_train_size: int, seed: int) -> pd.DataFrame:
    per_class = max(1, max_train_size // 2)
    parts = [group.sample(n=min(len(group), per_class), random_state=seed) for _, group in train_df.groupby("is_active")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def run_quantum_scores(
    feature_path: str = None,
    artifacts_dir: str = None,
    models_dir: str = None,
    top_n: int = DEFAULT_TOP_N,
    n_qubits: int = DEFAULT_N_QUBITS,
    max_train_size: int = DEFAULT_MAX_TRAIN_SIZE,
    seed: int = 42,
    smoke: bool = False,
) -> pd.DataFrame:
    feature_path = feature_path or os.path.join(REPO_ROOT, "data/processed/egfr_features.parquet")
    artifacts_dir = artifacts_dir or os.path.join(REPO_ROOT, "artifacts")
    models_dir = models_dir or os.path.join(REPO_ROOT, "models")

    if not os.path.exists(feature_path):
        print(f"[quantum] skipping quantum_scores: {feature_path} not found.")
        return None

    if smoke:
        top_n, max_train_size = min(top_n, 5), min(max_train_size, 80)

    df = pd.read_parquet(feature_path)
    train_df, test_df = scaffold_split(df, test_size=0.2, seed=seed)
    train_sample = _stratified_train_sample(train_df, max_train_size, seed)
    y_train = train_sample["is_active"].values.astype(int)

    from src.pipeline.rank import _load_tier1_scores  # local import: avoid a hard dependency when Tier 1 hasn't run

    try:
        ranked = _load_tier1_scores(artifacts_dir, models_dir, feature_path)
    except FileNotFoundError:
        print("[quantum] no Tier 1 ranking available yet (run `run.py tier1` first) — skipping quantum_scores.")
        return None
    top_ids = (
        ranked.sort_values("tier1_activity_raw", ascending=False)
        .head(top_n)["molecule_chembl_id"]
        .tolist()
    )
    order = {cid: i for i, cid in enumerate(top_ids)}
    candidates = df[df["molecule_chembl_id"].isin(top_ids)].copy()
    candidates["_rank_order"] = candidates["molecule_chembl_id"].map(order)
    candidates = candidates.sort_values("_rank_order").drop(columns="_rank_order")
    if candidates.empty:
        print("[quantum] no candidates matched the Tier 1 ranking — skipping quantum_scores.")
        return None

    print(
        f"[quantum] scoring {len(candidates)} candidates with the {n_qubits}-qubit fidelity "
        f"kernel QSVC (trained on {len(train_sample)} stratified samples)..."
    )
    _, _, pca_artifacts = compress_features(train_sample, test_df.head(1), n_components=n_qubits, seed=seed)
    scaler, pca, angle_scaler = pca_artifacts["scaler"], pca_artifacts["pca"], pca_artifacts["angle_scaler"]

    X_train_raw, cols = extract_feature_matrix(train_sample)
    X_train_final = angle_scaler.transform(pca.transform(scaler.transform(X_train_raw)))

    kernel = PennyLaneFidelityKernel(n_qubits=n_qubits, n_layers=2, seed=seed)
    qsvc = QSVCClassifier(quantum_kernel=kernel, C=1.0, random_state=seed)
    qsvc.fit(X_train_final, y_train)

    X_cand_raw, _ = extract_feature_matrix(candidates, feature_cols=cols)
    X_cand_final = angle_scaler.transform(pca.transform(scaler.transform(X_cand_raw)))
    quantum_score = qsvc.predict_proba(X_cand_final)[:, 1]

    out = candidates[["molecule_chembl_id"]].copy()
    out["quantum_score"] = np.round(quantum_score, 4)

    out_path = os.path.join(artifacts_dir, "quantum_scores.csv")
    os.makedirs(artifacts_dir, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[quantum] Tier 3 quantum kernel scores: {len(out)} candidates -> {out_path}")
    return out


def main():
    run_quantum_scores()


if __name__ == "__main__":
    main()
