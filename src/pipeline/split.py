#!/usr/bin/env python3
"""Bemis-Murcko scaffold split — enforces zero scaffold overlap between train/test."""

import hashlib

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def compute_scaffold(smiles: str) -> str:
    """Return the Bemis-Murcko generic scaffold SMILES for a molecule, or '' if it fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold) if scaffold is not None else ""


def dataset_hash(df: pd.DataFrame, smiles_col: str = "canonical_smiles") -> str:
    """Deterministic content hash of a dataset's SMILES column (order-independent)."""
    smiles_sorted = sorted(df[smiles_col].astype(str).tolist())
    joined = "\n".join(smiles_sorted).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    Split df into train/test by Bemis-Murcko scaffold groups so that no scaffold
    appears in both sets. Returns (train_df, test_df).
    """
    scaffolds = df[smiles_col].apply(compute_scaffold)
    groups = {}
    for idx, scaffold in scaffolds.items():
        groups.setdefault(scaffold, []).append(idx)

    rng = np.random.RandomState(seed)
    scaffold_keys = list(groups.keys())
    rng.shuffle(scaffold_keys)

    n_total = len(df)
    target_test = int(round(n_total * test_size))

    test_indices = []
    train_indices = []
    for key in scaffold_keys:
        group = groups[key]
        if len(test_indices) + len(group) <= target_test:
            test_indices.extend(group)
        else:
            train_indices.extend(group)

    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)

    train_scaffolds = set(train_df[smiles_col].apply(compute_scaffold))
    test_scaffolds = set(test_df[smiles_col].apply(compute_scaffold))
    overlap = train_scaffolds & test_scaffolds
    assert not overlap, f"Scaffold leakage detected: {len(overlap)} scaffolds span train/test"

    return train_df, test_df
