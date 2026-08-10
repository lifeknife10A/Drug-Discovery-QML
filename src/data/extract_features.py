#!/usr/bin/env python3
"""
Phase 1: RDKit Feature Extraction & Molecular Representation Engine
Extracts 2048-bit Morgan Fingerprints (ECFP4) and Physicochemical Descriptors
"""

import os
import sys
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

def calculate_rdkit_features(smiles):
    """
    Computes 2048-bit Morgan Fingerprint and 6 key physicochemical descriptors for a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
        
    # 1. 2048-bit Morgan Fingerprint (Radius 2 = ECFP4)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = np.zeros((2048,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)
    
    # 2. Key Physicochemical Descriptors (Lipinski's Rule of 5 parameters + TPSA)
    descriptors = {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol)
    }
    
    return fp_array, descriptors

def process_dataset(input_parquet_path, output_parquet_path):
    """
    Processes the raw SMILES dataset and builds the final feature matrix.
    """
    print(f"[+] Reading dataset from {input_parquet_path}...")
    df = pd.read_parquet(input_parquet_path)
    
    fps = []
    descriptor_list = []
    valid_indices = []
    
    print(f"[+] Calculating RDKit features for {len(df)} compounds...")
    for idx, row in df.iterrows():
        smiles = row["canonical_smiles"]
        fp, desc = calculate_rdkit_features(smiles)
        
        if fp is not None and desc is not None:
            fps.append(fp)
            descriptor_list.append(desc)
            valid_indices.append(idx)
            
    # Filter valid molecules
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    
    # Create Fingerprint DataFrame
    fp_columns = [f"fp_{i}" for i in range(2048)]
    df_fp = pd.DataFrame(fps, columns=fp_columns)
    
    # Create Descriptors DataFrame
    df_desc = pd.DataFrame(descriptor_list)
    
    # Combine into single feature matrix
    df_features = pd.concat([df_valid[["molecule_chembl_id", "canonical_smiles", "pIC50", "is_active"]], df_desc, df_fp], axis=1)
    
    print(f"[+] Successfully featurized {len(df_features)} valid molecules.")
    print(f"[+] Total feature columns: {df_features.shape[1]}")
    
    df_features.to_parquet(output_parquet_path, index=False)
    print(f"[✓] Feature matrix saved to {output_parquet_path}")
    return df_features

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
    input_path = os.path.join(base_dir, "egfr_compounds_clean.parquet")
    output_path = os.path.join(base_dir, "egfr_features.parquet")
    
    if not os.path.exists(input_path):
        print(f"[!] Input file {input_path} not found. Please run fetch_chembl_data.py first.")
        sys.exit(1)
        
    process_dataset(input_path, output_path)

if __name__ == "__main__":
    main()
