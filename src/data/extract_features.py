#!/usr/bin/env python3
"""
Tier 1 featurization: 2048-bit Morgan (ECFP4) fingerprints + 7 physicochemical
descriptors for every compound.

Uses the current RDKit ``rdFingerprintGenerator.GetMorganGenerator`` API
(``AllChem.GetMorganFingerprintAsBitVect`` is deprecated).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

FP_BITS = 2048
FP_RADIUS = 2
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

DESC_ORDER = ["mw", "logp", "hbd", "hba", "tpsa", "rotatable_bonds", "aromatic_rings"]
ID_COLS = ["molecule_chembl_id", "canonical_smiles", "pIC50", "is_active"]
CARRY_COLS = ["variant", "n_measurements", "document_year"]  # kept if present


def calculate_rdkit_features(smiles: str):
    """Return (fp_uint8_array[2048], descriptor_dict) or (None, None) on parse failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    fp = _MORGAN_GEN.GetFingerprint(mol)
    fp_array = np.zeros((FP_BITS,), dtype=np.int8)
    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(fp, fp_array)

    descriptors = {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
    }
    return fp_array, descriptors


def process_dataset(input_parquet_path: str, output_parquet_path: str) -> pd.DataFrame:
    print(f"[+] Reading {input_parquet_path}...")
    df = pd.read_parquet(input_parquet_path)

    fps, descs, keep = [], [], []
    print(f"[+] Featurizing {len(df)} compounds (MorganGenerator, {FP_BITS} bits, r={FP_RADIUS})...")
    for idx, row in df.iterrows():
        fp, desc = calculate_rdkit_features(row["canonical_smiles"])
        if fp is not None:
            fps.append(fp)
            descs.append(desc)
            keep.append(idx)

    df_valid = df.loc[keep].reset_index(drop=True)
    id_cols = [c for c in ID_COLS + CARRY_COLS if c in df_valid.columns]

    df_desc = pd.DataFrame(descs)[DESC_ORDER]
    df_fp = pd.DataFrame(fps, columns=[f"fp_{i}" for i in range(FP_BITS)])
    df_features = pd.concat([df_valid[id_cols], df_desc, df_fp], axis=1)

    print(f"[+] {len(df_features)} valid molecules, {df_features.shape[1]} columns")
    df_features.to_parquet(output_parquet_path, index=False)
    print(f"[OK] {output_parquet_path}")
    return df_features


def main() -> None:
    base = os.path.join(os.path.dirname(__file__), "../../data/processed")
    inp = os.path.join(base, "egfr_compounds_clean.parquet")
    out = os.path.join(base, "egfr_features.parquet")
    if not os.path.exists(inp):
        print(f"[!] {inp} not found. Run fetch_chembl_data.py first.")
        sys.exit(1)
    process_dataset(inp, out)

    # Also featurize the full variant table when it exists (for the selectivity model).
    var_in = os.path.join(base, "egfr_by_variant.parquet")
    if os.path.exists(var_in):
        process_dataset(var_in, os.path.join(base, "egfr_features_by_variant.parquet"))


if __name__ == "__main__":
    main()
