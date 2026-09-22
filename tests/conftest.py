import numpy as np
import pandas as pd
import pytest

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

SMILES_POOL = [
    "c1ccccc1CCN", "c1ccccc1CCO", "c1ccncc1CCN", "C1CCCCC1CCN",
    "c1ccc2ccccc2c1CCN", "CCCCCCCC", "c1ccc(cc1)C(=O)O", "c1ccc(cc1)N",
    "c1ccc2[nH]ccc2c1", "C1CCNCC1", "c1ccsc1", "c1ccoc1",
    "c1ccc(Cl)cc1", "c1ccc(F)cc1", "c1ccc(Br)cc1", "CCOC(=O)c1ccccc1",
    "c1ccc2c(c1)ncnc2N", "c1ccc2c(c1)ccnc2N", "c1ccc2c(c1)cccc2N",
    "c1cnc2[nH]ccc2c1",
]


def _feature_row(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=32)
    fp_arr = np.zeros((32,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, fp_arr)
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        **{f"fp_{i}": int(v) for i, v in enumerate(fp_arr)},
    }


@pytest.fixture
def tiny_feature_parquet(tmp_path):
    rng = np.random.RandomState(42)
    rows = []
    for i, smiles in enumerate(SMILES_POOL):
        row = _feature_row(smiles)
        row["molecule_chembl_id"] = f"CHEMBL{i}"
        row["canonical_smiles"] = smiles
        row["is_active"] = int(rng.rand() > 0.5)
        row["pIC50"] = float(5 + rng.rand() * 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    path = tmp_path / "tiny_features.parquet"
    df.to_parquet(path, index=False)
    return str(path)
