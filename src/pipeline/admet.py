import sys
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import FilterCatalog

try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
except ImportError:
    sascorer = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIABILITY_COLS = ["pains_alert", "brenk_alert", "herg_liability", "cyp_liability", "poor_solubility"]

class ADMETPredictor:
    """
    Tier 4 ADMET Predictor:
    Evaluates Synthetic Accessibility (SAScore), PAINS/BRENK filters,
    and provides heuristic flags for hERG, CYP, and solubility based on
    physicochemical properties and structural alerts.
    """
    def __init__(self):
        # Initialize PAINS catalog
        pains_params = FilterCatalog.FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        self.pains_catalog = FilterCatalog.FilterCatalog(pains_params)

        # Initialize BRENK catalog
        brenk_params = FilterCatalog.FilterCatalogParams()
        brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        self.brenk_catalog = FilterCatalog.FilterCatalog(brenk_params)
        
        # Naive proxy SMARTS for hERG (basic amine) and CYP (imidazole/pyridine)
        self.herg_smarts = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]") 
        self.cyp_smarts = Chem.MolFromSmarts("n1cncc1")

    def compute_admet(self, smiles: str) -> dict:
        """
        Compute ADMET properties for a single SMILES string.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "sascore": None,
                "pains_alert": False,
                "brenk_alert": False,
                "herg_liability": False,
                "cyp_liability": False,
                "poor_solubility": False,
                "clogp": None,
                "tpsa": None
            }

        # SAScore
        sa_score = None
        if sascorer is not None:
            try:
                sa_score = sascorer.calculateScore(mol)
            except Exception:
                pass
                
        # PAINS & Brenk
        pains_alert = self.pains_catalog.HasMatch(mol)
        brenk_alert = self.brenk_catalog.HasMatch(mol)

        # Physicochemical
        clogp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # Solubility (Lipinski logP > 5 as poor aqueous solubility flag)
        poor_solubility = clogp > 5.0
        
        # hERG / CYP naive flags
        herg_liability = mol.HasSubstructMatch(self.herg_smarts) if self.herg_smarts else False
        cyp_liability = mol.HasSubstructMatch(self.cyp_smarts) if self.cyp_smarts else False
        
        return {
            "sascore": sa_score,
            "pains_alert": pains_alert,
            "brenk_alert": brenk_alert,
            "herg_liability": herg_liability,
            "cyp_liability": cyp_liability,
            "poor_solubility": poor_solubility,
            "clogp": clogp,
            "tpsa": tpsa
        }

    def process_dataframe(self, df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
        """
        Process a DataFrame of SMILES and append ADMET columns.
        """
        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not found in DataFrame.")
            
        admet_results = df[smiles_col].apply(self.compute_admet)
        admet_df = pd.DataFrame(admet_results.tolist(), index=df.index)

        return pd.concat([df, admet_df], axis=1)


def score_admet_row(row: pd.Series) -> float:
    """Fold Tier 4 outputs into a single 0-1 'higher is better' admet_score.

    Half from synthetic accessibility (SAScore 1-10, lower/easier is better),
    half from the fraction of liability flags (PAINS/BRENK/hERG/CYP/solubility)
    that did NOT fire. A missing SAScore (parse failure or sascorer unavailable)
    falls back to a neutral 0.5 rather than penalizing the compound.
    """
    sascore = row.get("sascore")
    sa_component = 0.5 if sascore is None or pd.isna(sascore) else float(np.clip((10.0 - sascore) / 9.0, 0.0, 1.0))
    liabilities = [bool(row.get(col, False)) for col in LIABILITY_COLS]
    liability_component = 1.0 - (sum(liabilities) / len(liabilities))
    return round(0.5 * sa_component + 0.5 * liability_component, 4)


def run_admet(feature_path: str = None, artifacts_dir: str = None, smoke: bool = False) -> pd.DataFrame:
    """Tier 4 entrypoint: score every Tier 1 candidate for ADMET risk.

    Writes artifacts/admet_scores.csv with molecule_chembl_id + admet_score
    (plus the raw Tier 4 columns), consumed by src/pipeline/rank.py.
    """
    feature_path = feature_path or os.path.join(REPO_ROOT, "data/processed/egfr_features.parquet")
    artifacts_dir = artifacts_dir or os.path.join(REPO_ROOT, "artifacts")

    if not os.path.exists(feature_path):
        print(f"[admet] skipping: {feature_path} not found.")
        return None

    df = pd.read_parquet(feature_path)[["molecule_chembl_id", "canonical_smiles"]].rename(
        columns={"canonical_smiles": "smiles"}
    )
    if smoke:
        df = df.head(50).reset_index(drop=True)

    print(f"[admet] scoring {len(df)} candidates (SAScore + PAINS/BRENK/hERG/CYP/solubility flags)...")
    predictor = ADMETPredictor()
    scored = predictor.process_dataframe(df, smiles_col="smiles")
    scored["admet_score"] = scored.apply(score_admet_row, axis=1)

    os.makedirs(artifacts_dir, exist_ok=True)
    out_path = os.path.join(artifacts_dir, "admet_scores.csv")
    scored.to_csv(out_path, index=False)
    print(f"[✓] Tier 4 ADMET scores written to {out_path}")
    return scored


def main():
    run_admet()


if __name__ == "__main__":
    main()
