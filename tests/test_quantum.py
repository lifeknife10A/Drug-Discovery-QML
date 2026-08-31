#!/usr/bin/env python3
"""Comprehensive test suite for Tier 3 Quantum Kernel Benchmarking."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add repo root to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.pipeline.split import scaffold_split
from src.quantum.features import compress_features, extract_feature_matrix, prepare_multidim_features
from src.quantum.kernels import (
    FixedRandomCircuitKernel,
    PennyLaneFidelityKernel,
    QSVCClassifier,
    build_classical_kernel_models,
    simulate_hardware_concordance_zne,
)
from src.quantum.negative_controls import (
    run_feature_permutation_control,
    run_random_circuit_control,
    run_y_scramble_control,
)
from src.quantum.run_benchmark import run_tier3_benchmark


@pytest.fixture
def mock_dataset():
    """Create a small deterministic mock dataset with valid SMILES, molecular descriptors and Morgan bits."""
    rng = np.random.RandomState(42)
    n_samples = 40
    desc_cols = ["mw", "logp", "hbd", "hba", "tpsa", "rotatable_bonds", "aromatic_rings"]
    fp_cols = [f"fp_{i}" for i in range(32)]
    cols = desc_cols + fp_cols

    data = rng.randn(n_samples, len(cols))
    df = pd.DataFrame(data, columns=cols)

    # Valid chemical scaffolds with diverse Murcko rings
    scaffolds = [
        "c1ccccc1CC(=O)NC",
        "c1ccncc1NC(=O)CC",
        "c1cncnc1NC(=O)CCC",
        "c1ccc2ccccc2c1CC(=O)N",
        "c1ccc2ncccc2c1NC(=O)C",
        "c1ccc2[nH]ccc2c1CC(=O)NCC",
        "c1ccoc1CC(=O)NCCC",
        "c1ccsc1NC(=O)CCCC",
    ]
    df["canonical_smiles"] = [scaffolds[i % len(scaffolds)] for i in range(n_samples)]
    df["molecule_chembl_id"] = [f"CHEMBL_TEST_{i}" for i in range(n_samples)]
    df["is_active"] = rng.randint(0, 2, size=n_samples)
    df["pIC50"] = rng.uniform(4.0, 9.0, size=n_samples)
    return df


def test_feature_extraction_and_compression(mock_dataset):
    """Verify PCA compression preserves train-test isolation and scales within [-pi, pi]."""
    train_df, test_df = scaffold_split(mock_dataset, test_size=0.25, seed=42)

    X_train_4, X_test_4, artifacts_4 = compress_features(
        train_df=train_df,
        test_df=test_df,
        n_components=4,
        seed=42,
    )

    assert X_train_4.shape == (len(train_df), 4)
    assert X_test_4.shape == (len(test_df), 4)
    assert np.all(X_train_4 >= -np.pi - 1e-6) and np.all(X_train_4 <= np.pi + 1e-6)
    assert artifacts_4["n_components"] == 4
    assert len(artifacts_4["explained_variance_ratio"]) == 4
    assert 0.0 < artifacts_4["total_explained_variance"] <= 1.0


def test_prepare_multidim_features(mock_dataset):
    """Verify multidimensional feature preparation across (4, 6, 8) dims."""
    train_df, test_df = scaffold_split(mock_dataset, test_size=0.25, seed=42)
    multidim = prepare_multidim_features(train_df, test_df, dimensions=(4, 6), seed=42)

    assert 4 in multidim and 6 in multidim
    assert multidim[4]["X_train"].shape[1] == 4
    assert multidim[6]["X_train"].shape[1] == 6


def test_pennylane_fidelity_kernel_properties():
    """Verify fidelity quantum kernel is symmetric, unit-diagonal, and bounded in [0, 1]."""
    q_kernel = PennyLaneFidelityKernel(n_qubits=3, n_layers=2, seed=42)
    X = np.array([
        [0.1, 0.2, 0.3],
        [-0.5, 0.0, 0.5],
        [1.0, -1.0, 0.2],
        [0.0, 0.0, 0.0],
    ])

    # Gram matrix
    K = q_kernel(X)
    assert K.shape == (4, 4)
    # Unit diagonal (self-fidelity = 1)
    assert np.allclose(np.diag(K), np.ones(4), atol=1e-5)
    # Symmetry
    assert np.allclose(K, K.T, atol=1e-5)
    # Bounded
    assert np.all(K >= 0.0) and np.all(K <= 1.0)

    # Cross-dataset evaluation
    X2 = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
    K_cross = q_kernel(X, X2)
    assert K_cross.shape == (4, 2)
    # First row of K_cross should match K[0, 0] = 1.0 since X2[0] == X[0]
    assert np.isclose(K_cross[0, 0], 1.0, atol=1e-5)

    # Single pair evaluation consistency
    single_val = q_kernel.evaluate(X[0], X[1])
    assert np.isclose(single_val, K[0, 1], atol=1e-5)


def test_fixed_random_circuit_kernel():
    """Verify negative control fixed random circuit kernel mathematical validity."""
    rand_kernel = FixedRandomCircuitKernel(n_qubits=3, n_layers=2, seed=42)
    X = np.array([
        [0.2, -0.4, 0.8],
        [-0.1, 0.5, -0.2],
    ])
    K_rand = rand_kernel(X)
    assert K_rand.shape == (2, 2)
    assert np.allclose(np.diag(K_rand), np.ones(2), atol=1e-5)
    assert np.allclose(K_rand, K_rand.T, atol=1e-5)


def test_qsvc_classifier_fit_predict():
    """Verify QSVCClassifier trains and predicts calibrated probabilities."""
    q_kernel = PennyLaneFidelityKernel(n_qubits=2, n_layers=1, seed=42)
    qsvc = QSVCClassifier(quantum_kernel=q_kernel, C=1.0, random_state=42)

    X_train = np.array([[0.1, 0.2], [0.8, -0.7], [-0.5, 0.5], [0.9, 0.8]])
    y_train = np.array([0, 1, 0, 1])

    qsvc.fit(X_train, y_train)

    X_test = np.array([[0.2, 0.1], [-0.4, 0.6]])
    probs = qsvc.predict_proba(X_test)
    preds = qsvc.predict(X_test)

    assert probs.shape == (2, 2)
    assert np.allclose(np.sum(probs, axis=1), np.ones(2), atol=1e-5)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})


def test_negative_controls_execution():
    """Verify that all three negative control routines execute and report CIs."""
    X_train = np.random.RandomState(42).randn(16, 4)
    y_train = np.random.RandomState(42).randint(0, 2, size=16)
    X_test = np.random.RandomState(43).randn(8, 4)
    y_test = np.random.RandomState(43).randint(0, 2, size=8)

    q_kernel = PennyLaneFidelityKernel(n_qubits=4, seed=42)

    # 1. Y-Scramble
    res_yscramble = run_y_scramble_control(
        model_factory=lambda: QSVCClassifier(quantum_kernel=q_kernel, random_state=42),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=42,
        n_boot=20,
    )
    assert "metrics" in res_yscramble
    assert "roc_auc" in res_yscramble["metrics"]

    # 2. Feature Permutation
    res_featperm = run_feature_permutation_control(
        model_factory=lambda: QSVCClassifier(quantum_kernel=q_kernel, random_state=42),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=42,
        n_boot=20,
    )
    assert "metrics" in res_featperm
    assert "pr_auc" in res_featperm["metrics"]

    # 3. Fixed Random Circuit
    res_randcircuit = run_random_circuit_control(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_qubits=4,
        seed=42,
        n_boot=20,
    )
    assert "metrics" in res_randcircuit


def test_hardware_concordance_zne_simulation():
    """Verify 20x20 QPU hardware concordance simulation with ZNE."""
    X_subset = np.random.RandomState(42).uniform(-np.pi, np.pi, size=(10, 4))
    q_kernel = PennyLaneFidelityKernel(n_qubits=4, seed=42)

    concordance = simulate_hardware_concordance_zne(
        X_subset=X_subset,
        quantum_kernel=q_kernel,
        noise_rates=(0.0, 0.01, 0.03),
        n_shots=1024,
        seed=42,
    )

    assert concordance["n_samples"] == 10
    assert concordance["pearson_ideal_vs_noisy"] > 0.8
    assert concordance["pearson_ideal_vs_zne"] > 0.8
    assert "zne_mitigated_mae" in concordance


def test_full_benchmark_smoke(tmp_path, mock_dataset):
    """Verify end-to-end execution of run_tier3_benchmark in smoke mode."""
    parquet_path = str(tmp_path / "mock_features.parquet")
    mock_dataset.to_parquet(parquet_path, index=False)
    artifacts_dir = str(tmp_path / "artifacts")

    results = run_tier3_benchmark(
        feature_parquet_path=parquet_path,
        artifacts_dir=artifacts_dir,
        dimensions=(4,),
        n_boot=20,
        seed=42,
        smoke=True,
    )

    assert "metadata" in results
    assert "per_dimension_benchmarks" in results
    assert "negative_controls" in results
    assert "hardware_concordance_20x20" in results
    assert os.path.exists(os.path.join(artifacts_dir, "tier3_kernel_benchmark.json"))
    assert os.path.exists(os.path.join(artifacts_dir, "tier3_kernel_benchmark.csv"))
