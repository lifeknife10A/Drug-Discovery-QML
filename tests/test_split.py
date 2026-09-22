import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.pipeline.split import compute_scaffold, dataset_hash, scaffold_split


SMILES = [
    "c1ccccc1CCN",       # benzene-scaffold analog A
    "c1ccccc1CCO",       # benzene-scaffold analog B (same scaffold)
    "c1ccncc1CCN",       # pyridine-scaffold
    "C1CCCCC1CCN",       # cyclohexane-scaffold
    "c1ccc2ccccc2c1CCN",  # naphthalene-scaffold
    "CCCCCCCC",          # no ring -> empty scaffold
]


def _df():
    return pd.DataFrame({
        "canonical_smiles": SMILES,
        "is_active": [1, 0, 1, 0, 1, 0],
    })


def test_compute_scaffold_valid_and_invalid():
    assert compute_scaffold("c1ccccc1CCN") != ""
    assert compute_scaffold("not_a_smiles") == ""


def test_scaffold_split_no_leakage():
    df = _df()
    train_df, test_df = scaffold_split(df, test_size=0.34, seed=42)

    train_scaffolds = set(train_df["canonical_smiles"].apply(compute_scaffold))
    test_scaffolds = set(test_df["canonical_smiles"].apply(compute_scaffold))
    assert train_scaffolds.isdisjoint(test_scaffolds)
    assert len(train_df) + len(test_df) == len(df)


def test_scaffold_split_deterministic():
    df = _df()
    train1, test1 = scaffold_split(df, test_size=0.34, seed=42)
    train2, test2 = scaffold_split(df, test_size=0.34, seed=42)
    assert list(train1["canonical_smiles"]) == list(train2["canonical_smiles"])
    assert list(test1["canonical_smiles"]) == list(test2["canonical_smiles"])


def test_dataset_hash_stable_and_order_independent():
    df = _df()
    h1 = dataset_hash(df)
    h2 = dataset_hash(df.iloc[::-1].reset_index(drop=True))
    assert h1 == h2
    assert len(h1) == 64
