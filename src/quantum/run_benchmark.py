#!/usr/bin/env python3
"""Phase 3: Controlled Classical vs NISQ Quantum Kernel Benchmark for EGFR Screening.

Executes the Tier 3 benchmark across matched 4, 6, and 8 latent dimensions:
  1. Compresses Bemis-Murcko scaffold-split features via PCA to matched 4/6/8 dims.
  2. Evaluates Classical Kernels (Linear SVM, RBF SVM, Poly SVM, Random Fourier Features).
  3. Evaluates PennyLane Fidelity Quantum Kernel (AngleEmbedding + BasicEntanglerLayers).
  4. Runs Negative Controls (Y-Scrambling, Feature Permutation, Fixed Random Circuit).
  5. Computes Headline Metrics: PR-AUC, ROC-AUC, EF@1%, EF@5% with percentile bootstrap 95% CIs.
  6. Evaluates 20x20 Stratified Hardware Concordance with Zero-Noise Extrapolation (ZNE).
  7. Outputs structured artifacts/tier3_kernel_benchmark.json and summary CSV.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.svm import SVC

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pipeline.metrics import classification_report_with_ci  # noqa: E402
from src.pipeline.split import dataset_hash, scaffold_split  # noqa: E402
from src.quantum.features import compress_features  # noqa: E402
from src.quantum.kernels import (  # noqa: E402
    FixedRandomCircuitKernel,
    PennyLaneFidelityKernel,
    QSVCClassifier,
    build_classical_kernel_models,
    simulate_hardware_concordance_zne,
)
from src.quantum.negative_controls import (  # noqa: E402
    run_feature_permutation_control,
    run_random_circuit_control,
    run_y_scramble_control,
)

DEFAULT_SEED = 42
DEFAULT_DIMENSIONS = (4, 6, 8)


def run_tier3_benchmark(
    feature_parquet_path: str,
    artifacts_dir: str,
    dimensions: Sequence[int] = DEFAULT_DIMENSIONS,
    n_boot: int = 1000,
    seed: int = DEFAULT_SEED,
    smoke: bool = False,
) -> Dict:
    """Run the complete Tier 3 Quantum vs Classical Kernel Benchmark.

    Args:
        feature_parquet_path: Path to egfr_features.parquet.
        artifacts_dir: Directory where benchmark artifacts will be saved.
        dimensions: Sequence of latent dimensions to benchmark (e.g. 4, 6, 8).
        n_boot: Number of bootstrap iterations for 95% CIs.
        seed: Random seed for determinism.
        smoke: If True, runs on small subset and reduced bootstrap for rapid CI smoke testing.

    Returns:
        Structured dictionary containing full benchmark results and negative controls.
    """
    start_total_time = time.time()
    os.makedirs(artifacts_dir, exist_ok=True)

    if not os.path.exists(feature_parquet_path):
        raise FileNotFoundError(f"Feature parquet not found at: {feature_parquet_path}")

    df = pd.read_parquet(feature_parquet_path)
    if smoke:
        df = df.sample(n=min(100, len(df)), random_state=seed).reset_index(drop=True)
        dimensions = (4,)
        n_boot = min(50, n_boot)

    data_hash = dataset_hash(df)
    train_df, test_df = scaffold_split(df, test_size=0.2, seed=seed)

    y_train = train_df["is_active"].values.astype(int)
    y_test = test_df["is_active"].values.astype(int)

    benchmark_results = {
        "metadata": {
            "stage": "Tier 3: Controlled Classical vs NISQ Quantum Kernel Benchmark",
            "target": "EGFR (WT vs T790M/C797S / L858R/T790M/C797S)",
            "seed": seed,
            "dataset_hash": data_hash,
            "split_type": "bemis_murcko_scaffold",
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "train_active_ratio": float(np.mean(y_train)),
            "test_active_ratio": float(np.mean(y_test)),
            "dimensions_evaluated": list(dimensions),
            "n_bootstrap": n_boot,
            "smoke_mode": smoke,
            "disclaimer_rule1": (
                "Controlled NISQ feasibility characterization. No quantum advantage is claimed "
                "over classical RBF baselines; results rigorously document representational "
                "capacity across matched 4-8 latent dimensions."
            ),
        },
        "per_dimension_benchmarks": {},
        "negative_controls": {},
        "hardware_concordance_20x20": {},
    }

    tabular_summary_rows = []

    for dim in dimensions:
        print(f"\n==================================================")
        print(f"  Benchmarking Latent Dimension / Qubits: {dim}")
        print(f"==================================================")

        # 1. Prepare compressed features
        X_train_comp, X_test_comp, pca_meta = compress_features(
            train_df=train_df,
            test_df=test_df,
            n_components=dim,
            seed=seed,
        )

        dim_results = {
            "dimension": dim,
            "pca_explained_variance_ratio": pca_meta["explained_variance_ratio"],
            "pca_total_explained_variance": pca_meta["total_explained_variance"],
            "models": {},
        }

        # 2. Classical Kernel Models
        classical_models = build_classical_kernel_models(seed=seed)
        for model_name, model in classical_models.items():
            t0 = time.time()
            model.fit(X_train_comp, y_train)
            fit_time = time.time() - t0

            t0 = time.time()
            y_prob = model.predict_proba(X_test_comp)[:, 1]
            inf_time = time.time() - t0

            metrics = classification_report_with_ci(y_test, y_prob, n_boot=n_boot, seed=seed)

            dim_results["models"][model_name] = {
                "type": "classical",
                "fit_time_sec": float(round(fit_time, 4)),
                "infer_time_sec": float(round(inf_time, 4)),
                "metrics": metrics,
            }

            tabular_summary_rows.append({
                "Dimension": dim,
                "Model": model_name,
                "Type": "Classical",
                "PR-AUC": f"{metrics['pr_auc']['value']:.4f} [{metrics['pr_auc']['ci_low']:.3f}, {metrics['pr_auc']['ci_high']:.3f}]",
                "ROC-AUC": f"{metrics['roc_auc']['value']:.4f} [{metrics['roc_auc']['ci_low']:.3f}, {metrics['roc_auc']['ci_high']:.3f}]",
                "EF@1%": f"{metrics['ef_1pct']['value']:.2f} [{metrics['ef_1pct']['ci_low']:.2f}, {metrics['ef_1pct']['ci_high']:.2f}]",
                "Fit Time (s)": f"{fit_time:.2f}",
            })

        # 3. PennyLane Fidelity Quantum Kernel (AngleEmbedding + BasicEntanglerLayers)
        print(f"[*] Simulating PennyLane Fidelity Quantum Kernel ({dim} qubits)...")
        q_kernel = PennyLaneFidelityKernel(n_qubits=dim, n_layers=2, seed=seed)
        qsvc = QSVCClassifier(quantum_kernel=q_kernel, C=1.0, random_state=seed)

        t0 = time.time()
        qsvc.fit(X_train_comp, y_train)
        q_fit_time = time.time() - t0

        t0 = time.time()
        y_prob_q = qsvc.predict_proba(X_test_comp)[:, 1]
        q_inf_time = time.time() - t0

        q_metrics = classification_report_with_ci(y_test, y_prob_q, n_boot=n_boot, seed=seed)

        dim_results["models"]["quantum_fidelity_qsvc"] = {
            "type": "quantum_nisq",
            "n_qubits": dim,
            "ansatz": "AngleEmbedding_Y + BasicEntanglerLayers_2L",
            "fit_time_sec": float(round(q_fit_time, 4)),
            "infer_time_sec": float(round(q_inf_time, 4)),
            "metrics": q_metrics,
        }

        tabular_summary_rows.append({
            "Dimension": dim,
            "Model": "quantum_fidelity_qsvc",
            "Type": "Quantum (Fidelity)",
            "PR-AUC": f"{q_metrics['pr_auc']['value']:.4f} [{q_metrics['pr_auc']['ci_low']:.3f}, {q_metrics['pr_auc']['ci_high']:.3f}]",
            "ROC-AUC": f"{q_metrics['roc_auc']['value']:.4f} [{q_metrics['roc_auc']['ci_low']:.3f}, {q_metrics['roc_auc']['ci_high']:.3f}]",
            "EF@1%": f"{q_metrics['ef_1pct']['value']:.2f} [{q_metrics['ef_1pct']['ci_low']:.2f}, {q_metrics['ef_1pct']['ci_high']:.2f}]",
            "Fit Time (s)": f"{q_fit_time:.2f}",
        })

        benchmark_results["per_dimension_benchmarks"][f"{dim}_dim"] = dim_results

    # 4. Mandatory Negative Controls (Evaluated at dim=4 baseline)
    print("\n[*] Running Mandatory Negative Controls (dim=4)...")
    X_train_4, X_test_4, _ = compress_features(train_df, test_df, n_components=4, seed=seed)

    # Control 1: Y-Scrambling on Classical RBF
    rbf_yscramble = run_y_scramble_control(
        model_factory=lambda: SVC(kernel="rbf", probability=True, random_state=seed),
        X_train=X_train_4,
        y_train=y_train,
        X_test=X_test_4,
        y_test=y_test,
        seed=seed,
        n_boot=n_boot,
    )

    # Control 2: Y-Scrambling on Quantum Fidelity QSVC
    q_kernel_4 = PennyLaneFidelityKernel(n_qubits=4, n_layers=2, seed=seed)
    q_yscramble = run_y_scramble_control(
        model_factory=lambda: QSVCClassifier(quantum_kernel=q_kernel_4, random_state=seed),
        X_train=X_train_4,
        y_train=y_train,
        X_test=X_test_4,
        y_test=y_test,
        seed=seed,
        n_boot=n_boot,
    )

    # Control 3: Feature Permutation on Quantum Fidelity QSVC
    q_featperm = run_feature_permutation_control(
        model_factory=lambda: QSVCClassifier(quantum_kernel=q_kernel_4, random_state=seed),
        X_train=X_train_4,
        y_train=y_train,
        X_test=X_test_4,
        y_test=y_test,
        seed=seed,
        n_boot=n_boot,
    )

    # Control 4: Fixed Random Unitary Circuit Kernel
    q_randcircuit = run_random_circuit_control(
        X_train=X_train_4,
        y_train=y_train,
        X_test=X_test_4,
        y_test=y_test,
        n_qubits=4,
        n_layers=2,
        seed=seed,
        n_boot=n_boot,
    )

    benchmark_results["negative_controls"] = {
        "classical_rbf_y_scramble": rbf_yscramble,
        "quantum_qsvc_y_scramble": q_yscramble,
        "quantum_qsvc_feature_permutation": q_featperm,
        "fixed_random_circuit_kernel": q_randcircuit,
    }

    # 5. Simulated Depolarizing-Noise Concordance on Stratified 20x20 Subset with ZNE (No Real QPU)
    print("[*] Evaluating 20x20 Stratified Simulated Depolarizing-Noise Concordance with ZNE (No Real QPU)...")
    # Stratified 10 active + 10 inactive subset
    active_idx = np.where(y_test == 1)[0][:10]
    inactive_idx = np.where(y_test == 0)[0][:10]
    subset_idx = np.concatenate([active_idx, inactive_idx])
    X_subset_20 = X_test_4[subset_idx]

    simulated_noise_concordance = simulate_hardware_concordance_zne(
        X_subset=X_subset_20,
        quantum_kernel=q_kernel_4,
        noise_rates=(0.0, 0.01, 0.03),
        n_shots=1024,
        seed=seed,
    )
    simulated_noise_concordance["description"] = "simulated depolarizing-noise concordance (no real QPU)"
    benchmark_results["simulated_depolarizing_noise_concordance_20x20_no_real_qpu"] = simulated_noise_concordance
    benchmark_results["hardware_concordance_20x20"] = simulated_noise_concordance

    # 6. Save Artifacts
    total_elapsed = time.time() - start_total_time
    benchmark_results["metadata"]["total_elapsed_sec"] = float(round(total_elapsed, 2))

    json_path = os.path.join(artifacts_dir, "tier3_kernel_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n[✓] Tier 3 Benchmark JSON saved to: {json_path}")

    csv_path = os.path.join(artifacts_dir, "tier3_kernel_benchmark.csv")
    df_summary = pd.DataFrame(tabular_summary_rows)
    df_summary.to_csv(csv_path, index=False)
    print(f"[✓] Tier 3 Benchmark CSV saved to: {csv_path}")

    # Print Summary Table
    print("\n==========================================================================================")
    print("                 TIER 3 QUANTUM VS CLASSICAL KERNEL BENCHMARK SUMMARY")
    print("==========================================================================================")
    print(df_summary.to_string(index=False))
    print("------------------------------------------------------------------------------------------")
    print(f"Negative Controls (dim=4):")
    print(f"  • Classical RBF Y-Scramble ROC-AUC    : {rbf_yscramble['metrics']['roc_auc']['value']:.4f} (Expected ~0.50)")
    print(f"  • Quantum QSVC Y-Scramble ROC-AUC     : {q_yscramble['metrics']['roc_auc']['value']:.4f} (Expected ~0.50)")
    print(f"  • Quantum Feature Permutation ROC-AUC : {q_featperm['metrics']['roc_auc']['value']:.4f} (Expected ~0.50)")
    print(f"  • Fixed Random Circuit Kernel ROC-AUC : {q_randcircuit['metrics']['roc_auc']['value']:.4f}")
    print(f"Simulated Depolarizing-Noise Concordance (20x20 Subset with ZNE - No Real QPU):")
    print(f"  • Ideal vs Noisy (3% depolarizing) Pearson r : {simulated_noise_concordance['pearson_ideal_vs_noisy']:.4f}")
    print(f"  • Ideal vs ZNE-Mitigated Pearson r           : {simulated_noise_concordance['pearson_ideal_vs_zne']:.4f}")
    print(f"  • ZNE Error Reduction                       : {simulated_noise_concordance['concordance_improvement_pct']:.1f}%")
    print("==========================================================================================")
    print(f"Total Execution Time: {total_elapsed:.2f}s\n")

    return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="Tier 3 Quantum vs Classical Kernel Benchmark")
    parser.add_argument(
        "--features",
        default=os.path.join(REPO_ROOT, "data/processed/egfr_features.parquet"),
        help="Path to processed features parquet",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=os.path.join(REPO_ROOT, "artifacts"),
        help="Output directory for benchmark artifacts",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke test for CI",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for 95% CIs",
    )
    args = parser.parse_args()

    run_tier3_benchmark(
        feature_parquet_path=args.features,
        artifacts_dir=args.artifacts_dir,
        dimensions=DEFAULT_DIMENSIONS,
        n_boot=args.n_boot,
        seed=args.seed,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
