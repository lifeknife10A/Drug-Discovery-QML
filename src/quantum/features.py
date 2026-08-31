#!/usr/bin/env python3
"""Feature extraction and PCA compression pipeline for quantum kernel benchmarking.

Compresses high-dimensional Morgan fingerprints (2048-bit) and 7 RDKit molecular descriptors
down to matched 4, 6, and 8 latent dimensions with strict fit-on-train / transform-on-test isolation.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DESC_COLS: List[str] = [
    "mw",
    "logp",
    "hbd",
    "hba",
    "tpsa",
    "rotatable_bonds",
    "aromatic_rings",
]


def extract_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extract molecular descriptor and fingerprint columns from a DataFrame.

    Args:
        df: Input DataFrame containing compound features.
        feature_cols: Optional explicit list of columns to extract.

    Returns:
        Tuple of (feature matrix as ndarray, list of column names).
    """
    if feature_cols is None:
        fp_cols = [c for c in df.columns if c.startswith("fp_")]
        present_desc = [c for c in DESC_COLS if c in df.columns]
        feature_cols = present_desc + fp_cols

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing expected feature columns: {missing[:5]}...")

    X = df[feature_cols].values.astype(np.float64)
    return X, feature_cols


def compress_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_components: int = 4,
    seed: int = 42,
    feature_range: Tuple[float, float] = (-np.pi, np.pi),
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Fit PCA on training data and transform train/test features to target dimensions.

    Pipeline:
      1. StandardScaler (fit on train, transform train and test)
      2. PCA(n_components=n_components) (fit on train, transform train and test)
      3. MinMaxScaler(feature_range) (fit on train, transform train and test)

    Args:
        train_df: Training set DataFrame.
        test_df: Test set DataFrame.
        n_components: Target latent dimensionality (e.g. 4, 6, 8).
        seed: Random seed for PCA solver.
        feature_range: Scaling range for quantum rotation angles (default [-pi, pi]).
        feature_cols: Explicit list of feature columns.

    Returns:
        Tuple of (X_train_reduced, X_test_reduced, pipeline_artifacts_dict).
    """
    X_train_raw, cols = extract_feature_matrix(train_df, feature_cols=feature_cols)
    X_test_raw, _ = extract_feature_matrix(test_df, feature_cols=cols)

    # 1. Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 2. PCA Compression
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # 3. Angle Scaling (suitable for Pauli/Angle rotation gates)
    angle_scaler = MinMaxScaler(feature_range=feature_range)
    X_train_final = angle_scaler.fit_transform(X_train_pca)
    X_test_final = angle_scaler.transform(X_test_pca)

    explained_var = pca.explained_variance_ratio_.tolist()
    total_explained_var = float(np.sum(pca.explained_variance_ratio_))

    artifacts = {
        "n_components": n_components,
        "explained_variance_ratio": explained_var,
        "total_explained_variance": total_explained_var,
        "scaler": scaler,
        "pca": pca,
        "angle_scaler": angle_scaler,
        "feature_cols_count": len(cols),
    }

    return X_train_final, X_test_final, artifacts


def prepare_multidim_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dimensions: Sequence[int] = (4, 6, 8),
    seed: int = 42,
    feature_range: Tuple[float, float] = (-np.pi, np.pi),
) -> Dict[int, Dict]:
    """Generate compressed features for multiple dimensionalities in one pass.

    Args:
        train_df: Training set DataFrame.
        test_df: Test set DataFrame.
        dimensions: Sequence of latent dimensions (e.g. (4, 6, 8)).
        seed: Random seed.
        feature_range: Target angle feature range.

    Returns:
        Dict mapping dimension -> {
            'X_train': np.ndarray,
            'X_test': np.ndarray,
            'artifacts': Dict
        }
    """
    results = {}
    for d in dimensions:
        X_train, X_test, artifacts = compress_features(
            train_df=train_df,
            test_df=test_df,
            n_components=d,
            seed=seed,
            feature_range=feature_range,
        )
        results[d] = {
            "X_train": X_train,
            "X_test": X_test,
            "artifacts": artifacts,
        }
    return results
