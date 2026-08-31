#!/usr/bin/env python3
"""Quantum and classical kernel implementations for NISQ bioactivity benchmarking.

Provides:
  - PennyLaneFidelityKernel: Vectorized fidelity quantum kernel K(x_i, x_j) = |<psi(x_i)|psi(x_j)>|^2
  - FixedRandomCircuitKernel: Negative control using fixed random unitary circuits
  - QSVCClassifier: Scikit-learn compatible Support Vector Classifier wrapping precomputed quantum kernels
  - Classical kernel control estimators (Linear, RBF, Polynomial, RFF) on matched compressed feature spaces
  - QPU hardware concordance simulation with Zero-Noise Extrapolation (ZNE)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pennylane as qml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


class PennyLaneFidelityKernel:
    """Vectorized Fidelity Quantum Kernel in PennyLane.

    Computes:
      K_FQ(x_i, x_j) = |<psi(x_i)|psi(x_j)>|^2
    where |psi(x)> = U(x)|0> prepared via AngleEmbedding and BasicEntanglerLayers.

    Vectorization:
      Statevectors Psi_1 (N1 x 2^n) and Psi_2 (N2 x 2^n) are computed via batch
      state simulation, yielding K = |Psi_1 @ Psi_2^H|^2 in O(N1 * N2 * 2^n) BLAS operations.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        rotation: str = "Y",
        weights: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rotation = rotation
        self.seed = seed
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        if weights is None:
            # Deterministic standard weights for reproducible entangling layers
            rng = np.random.RandomState(self.seed)
            self.weights = rng.uniform(0, 2 * np.pi, size=(self.n_layers, self.n_qubits))
        else:
            self.weights = np.asarray(weights, dtype=np.float64)

        @qml.qnode(self.dev, interface="numpy")
        def _circuit(X_batch, layer_weights):
            qml.AngleEmbedding(X_batch, wires=range(self.n_qubits), rotation=self.rotation)
            qml.BasicEntanglerLayers(weights=layer_weights, wires=range(self.n_qubits))
            return qml.state()

        self._state_qnode = _circuit

    def compute_states(self, X: np.ndarray) -> np.ndarray:
        """Compute statevectors for a batch of input points.

        Args:
            X: Input array of shape (N, n_qubits) or (n_qubits,).

        Returns:
            Complex ndarray of shape (N, 2^n_qubits).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if X_arr.shape[1] != self.n_qubits:
            raise ValueError(
                f"Feature dimension {X_arr.shape[1]} does not match n_qubits={self.n_qubits}"
            )

        states = self._state_qnode(X_arr, self.weights)
        return np.asarray(states, dtype=np.complex128)

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Evaluate kernel value between two single samples."""
        s1 = self.compute_states(x1)
        s2 = self.compute_states(x2)
        overlap = np.abs(np.vdot(s2[0], s1[0])) ** 2
        return float(np.clip(overlap, 0.0, 1.0))

    def __call__(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the full Gram kernel matrix between datasets X1 and X2.

        Args:
            X1: First dataset of shape (N1, n_qubits).
            X2: Optional second dataset of shape (N2, n_qubits). If None or X1 is X2,
                computes square symmetric self-kernel matrix.

        Returns:
            Real ndarray of shape (N1, N2) with entries in [0, 1].
        """
        states1 = self.compute_states(X1)
        if X2 is None or X1 is X2:
            inner_prod = states1 @ states1.conj().T
            K = np.abs(inner_prod) ** 2
            K = 0.5 * (K + K.T)
            np.fill_diagonal(K, 1.0)
        else:
            states2 = self.compute_states(X2)
            inner_prod = states1 @ states2.conj().T
            K = np.abs(inner_prod) ** 2

        return np.clip(K, 0.0, 1.0)


class FixedRandomCircuitKernel:
    """Fixed Random Quantum Circuit Kernel (Negative Control).

    Applies fixed random unitaries and random entanglement parameters.
    Used to test if quantum feature map expressivity alone gives rise to spurious
    correlations in the absence of structured inductive bias.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.seed = seed
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        rng = np.random.RandomState(self.seed)
        self.weights = rng.uniform(0, 2 * np.pi, size=(self.n_layers, self.n_qubits, 3))

        @qml.qnode(self.dev, interface="numpy")
        def _circuit(X_batch, layer_weights):
            qml.AngleEmbedding(X_batch, wires=range(self.n_qubits), rotation="X")
            qml.StronglyEntanglingLayers(weights=layer_weights, wires=range(self.n_qubits))
            return qml.state()

        self._state_qnode = _circuit

    def compute_states(self, X: np.ndarray) -> np.ndarray:
        """Compute statevectors from random circuit."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if X_arr.shape[1] != self.n_qubits:
            raise ValueError(
                f"Feature dimension {X_arr.shape[1]} does not match n_qubits={self.n_qubits}"
            )

        states = self._state_qnode(X_arr, self.weights)
        return np.asarray(states, dtype=np.complex128)

    def __call__(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the random circuit kernel matrix."""
        states1 = self.compute_states(X1)
        if X2 is None or X1 is X2:
            inner_prod = states1 @ states1.conj().T
            K = np.abs(inner_prod) ** 2
            K = 0.5 * (K + K.T)
            np.fill_diagonal(K, 1.0)
        else:
            states2 = self.compute_states(X2)
            inner_prod = states1 @ states2.conj().T
            K = np.abs(inner_prod) ** 2

        return np.clip(K, 0.0, 1.0)


class QSVCClassifier(BaseEstimator, ClassifierMixin):
    """Support Vector Classifier using a precomputed quantum kernel."""

    def __init__(
        self,
        quantum_kernel: Union[PennyLaneFidelityKernel, FixedRandomCircuitKernel],
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.quantum_kernel = quantum_kernel
        self.C = C
        self.random_state = random_state
        self.svc_ = SVC(
            kernel="precomputed",
            C=self.C,
            probability=True,
            random_state=self.random_state,
        )
        self.X_train_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QSVCClassifier":
        """Fit the QSVC model on training data."""
        self.X_train_ = np.asarray(X, dtype=np.float64)
        K_train = self.quantum_kernel(self.X_train_)
        self.svc_.fit(K_train, y)
        self.classes_ = self.svc_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        if self.X_train_ is None:
            raise RuntimeError("Model is not fitted yet.")
        K_test = self.quantum_kernel(np.asarray(X, dtype=np.float64), self.X_train_)
        return self.svc_.predict_proba(K_test)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        if self.X_train_ is None:
            raise RuntimeError("Model is not fitted yet.")
        K_test = self.quantum_kernel(np.asarray(X, dtype=np.float64), self.X_train_)
        return self.svc_.predict(K_test)


def build_classical_kernel_models(seed: int = 42) -> Dict[str, BaseEstimator]:
    """Instantiate classical kernel models evaluated on matched compressed feature spaces.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping model name to Scikit-learn estimator.
    """
    return {
        "linear_svm": SVC(kernel="linear", probability=True, random_state=seed),
        "rbf_svm": SVC(kernel="rbf", probability=True, random_state=seed),
        "poly_svm": SVC(kernel="poly", degree=3, probability=True, random_state=seed),
        "random_fourier_features": make_pipeline(
            RBFSampler(n_components=100, random_state=seed),
            LogisticRegression(max_iter=1000, random_state=seed),
        ),
    }


def simulate_hardware_concordance_zne(
    X_subset: np.ndarray,
    quantum_kernel: PennyLaneFidelityKernel,
    noise_rates: Tuple[float, ...] = (0.0, 0.01, 0.03),
    n_shots: int = 1024,
    seed: int = 42,
) -> Dict:
    """Evaluate 20x20 stratified subset on simulated noisy QPU with Zero-Noise Extrapolation (ZNE).

    Demonstrates NISQ hardware budgeting and error mitigation without triggering unapproved cloud spend.

    Args:
        X_subset: 20x20 representative subset (stratified active/inactive).
        quantum_kernel: PennyLane fidelity kernel.
        noise_rates: Noise scale factors for Richardson ZNE extrapolation.
        n_shots: Simulated shot budget per element.
        seed: Random seed.

    Returns:
        Dict of ideal vs noisy vs ZNE-mitigated kernel metrics and concordance correlation.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(X_subset)
    ideal_K = quantum_kernel(X_subset)

    noisy_matrices = []
    for rate in noise_rates:
        if rate == 0.0:
            noisy_matrices.append(ideal_K)
            continue
        # Depolarizing channel noise simulation + shot noise
        depolarized = (1.0 - rate) * ideal_K + rate * (1.0 / (2**quantum_kernel.n_qubits))
        shot_noise = rng.normal(0, np.sqrt(np.clip(depolarized * (1 - depolarized), 0, 1) / n_shots))
        noisy_K = np.clip(depolarized + shot_noise, 0.0, 1.0)
        np.fill_diagonal(noisy_K, 1.0)
        noisy_K = 0.5 * (noisy_K + noisy_K.T)
        noisy_matrices.append(noisy_K)

    # Linear Richardson extrapolation to zero noise (rate -> 0)
    # K_zne = K(rate_1) + rate_1 / (rate_2 - rate_1) * (K(rate_1) - K(rate_2))
    if len(noise_rates) >= 3:
        r1, r2 = noise_rates[1], noise_rates[2]
        K1, K2 = noisy_matrices[1], noisy_matrices[2]
        zne_K = K1 - (r1 / (r2 - r1)) * (K2 - K1)
        zne_K = np.clip(zne_K, 0.0, 1.0)
        np.fill_diagonal(zne_K, 1.0)
        zne_K = 0.5 * (zne_K + zne_K.T)
    else:
        zne_K = noisy_matrices[-1]

    # Metrics vs Ideal
    noisy_mae = float(np.mean(np.abs(noisy_matrices[-1] - ideal_K)))
    zne_mae = float(np.mean(np.abs(zne_K - ideal_K)))
    pearson_noisy = float(np.corrcoef(ideal_K.flatten(), noisy_matrices[-1].flatten())[0, 1])
    pearson_zne = float(np.corrcoef(ideal_K.flatten(), zne_K.flatten())[0, 1])

    return {
        "n_samples": n_samples,
        "n_qubits": quantum_kernel.n_qubits,
        "n_shots": n_shots,
        "noise_rates": list(noise_rates),
        "noisy_mae": noisy_mae,
        "zne_mitigated_mae": zne_mae,
        "pearson_ideal_vs_noisy": pearson_noisy,
        "pearson_ideal_vs_zne": pearson_zne,
        "concordance_improvement_pct": float(max(0.0, (noisy_mae - zne_mae) / (noisy_mae + 1e-9) * 100)),
    }
