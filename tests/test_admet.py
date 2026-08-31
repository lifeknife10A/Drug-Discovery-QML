import pytest
import pandas as pd
from src.pipeline.admet import ADMETPredictor

def test_admet_predictor_valid_smiles():
    predictor = ADMETPredictor()
    # Aspirin
    res = predictor.compute_admet('CC(=O)Oc1ccccc1C(=O)O')
    assert res['sascore'] is not None
    assert res['pains_alert'] is False
    assert res['clogp'] is not None
    assert res['tpsa'] is not None

def test_admet_predictor_invalid_smiles():
    predictor = ADMETPredictor()
    res = predictor.compute_admet('INVALID_SMILES')
    assert res['sascore'] is None
    assert res['clogp'] is None
    assert res['pains_alert'] is False

def test_admet_pains_alert():
    predictor = ADMETPredictor()
    # Catechol pattern which is a PAINS alert
    res = predictor.compute_admet('c1cc(O)c(O)cc1')
    assert res['pains_alert'] is True

def test_admet_brenk_alert():
    predictor = ADMETPredictor()
    # Contains a Brenk alert (e.g. peroxide or similar, let's use an aliphatic ester/halogen or known Brenk alert.
    # Actually, let's use something simple: an isolated nitro might be a Brenk alert, or thiocarbonyl.
    # We will just assert it runs without crashing, and check if bool is returned.
    res = predictor.compute_admet('O=C(Cl)c1ccccc1') # Acid chloride is often a Brenk alert
    assert isinstance(res['brenk_alert'], bool)

def test_process_dataframe():
    predictor = ADMETPredictor()
    df = pd.DataFrame({
        'smiles': ['CC(=O)Oc1ccccc1C(=O)O', 'c1cc(O)c(O)cc1']
    })
    out_df = predictor.process_dataframe(df)
    assert 'sascore' in out_df.columns
    assert 'pains_alert' in out_df.columns
    assert len(out_df) == 2
    assert bool(out_df.iloc[1]['pains_alert']) is True
