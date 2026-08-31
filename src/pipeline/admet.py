import sys
import os
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
