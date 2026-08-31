"""Quantum Machine Learning (Tier 3) benchmarking package for EGFR bioactivity screening."""

from src.quantum.features import compress_features, prepare_multidim_features
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

__all__ = [
    "compress_features",
    "prepare_multidim_features",
    "PennyLaneFidelityKernel",
    "FixedRandomCircuitKernel",
    "QSVCClassifier",
    "build_classical_kernel_models",
    "simulate_hardware_concordance_zne",
    "run_y_scramble_control",
    "run_feature_permutation_control",
    "run_random_circuit_control",
    "run_tier3_benchmark",
]
