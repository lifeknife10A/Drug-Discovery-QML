#!/usr/bin/env python3
"""Negative controls for Tier 3 Quantum Kernel Benchmarking.

Implements the three mandatory negative controls to eliminate false discovery:
  1. Y-Scrambling (shuffled training labels)
  2. Feature Permutation (shuffled feature columns destroying correlations)
  3. Fixed Random Circuit Kernel (unstructured quantum Hilbert space mapping)
"""

from typing import Callable, Dict, Optional, Tuple
import numpy as np
from src.pipeline.metrics import classification_report_with_ci
from src.quantum.kernels import FixedRandomCircuitKernel, QSVCClassifier


def run_y_scramble_control(
    model_factory: Callable[[], object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    n_boot: int = 1000,
) -> Dict:
    """Evaluate model under y-scrambled (randomly permuted) training labels.

    Args:
        model_factory: Callable returning a fresh, unfitted model instance.
        X_train: Training features.
        y_train: True training binary labels.
        X_test: Test features.
        y_test: True test binary labels.
        seed: Random seed for label permutation.
        n_boot: Bootstrap iterations for 95% CIs.

    Returns:
        Dict with classification metrics and bootstrap CIs under label scrambling.
    """
    rng = np.random.RandomState(seed)
    y_train_scrambled = y_train.copy()
    rng.shuffle(y_train_scrambled)

    model = model_factory()
    model.fit(X_train, y_train_scrambled)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = classification_report_with_ci(y_test, y_prob, n_boot=n_boot, seed=seed)
    return {
        "control_type": "y_scrambling",
        "seed": seed,
        "metrics": metrics,
    }


def run_feature_permutation_control(
    model_factory: Callable[[], object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    n_boot: int = 1000,
) -> Dict:
    """Evaluate model under feature permutation (columns shuffled independently).

    Args:
        model_factory: Callable returning a fresh, unfitted model instance.
        X_train: Training features.
        y_train: True training binary labels.
        X_test: Test features.
        y_test: True test binary labels.
        seed: Random seed for feature permutation.
        n_boot: Bootstrap iterations for 95% CIs.

    Returns:
        Dict with classification metrics and bootstrap CIs under feature permutation.
    """
    rng = np.random.RandomState(seed)
    X_train_perm = X_train.copy()
    X_test_perm = X_test.copy()

    for col in range(X_train.shape[1]):
        rng.shuffle(X_train_perm[:, col])
        rng.shuffle(X_test_perm[:, col])

    model = model_factory()
    model.fit(X_train_perm, y_train)
    y_prob = model.predict_proba(X_test_perm)[:, 1]

    metrics = classification_report_with_ci(y_test, y_prob, n_boot=n_boot, seed=seed)
    return {
        "control_type": "feature_permutation",
        "seed": seed,
        "metrics": metrics,
    }


def run_random_circuit_control(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_qubits: int = 4,
    n_layers: int = 2,
    seed: int = 42,
    n_boot: int = 1000,
) -> Dict:
    """Evaluate QSVC with a Fixed Random Unitary Circuit kernel.

    Args:
        X_train: Training features.
        y_train: True training binary labels.
        X_test: Test features.
        y_test: True test binary labels.
        n_qubits: Number of qubits (feature dimension).
        n_layers: Number of entangling layers.
        seed: Random seed for circuit parameters.
        n_boot: Bootstrap iterations.

    Returns:
        Dict with classification metrics and bootstrap CIs for the random circuit control.
    """
    random_kernel = FixedRandomCircuitKernel(n_qubits=n_qubits, n_layers=n_layers, seed=seed)
    qsvc = QSVCClassifier(quantum_kernel=random_kernel, random_state=seed)
    qsvc.fit(X_train, y_train)
    y_prob = qsvc.predict_proba(X_test)[:, 1]

    metrics = classification_report_with_ci(y_test, y_prob, n_boot=n_boot, seed=seed)
    return {
        "control_type": "fixed_random_circuit",
        "n_qubits": n_qubits,
        "seed": seed,
        "metrics": metrics,
    }
