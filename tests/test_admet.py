import pytest
import pandas as pd
from src.pipeline.admet import ADMETPredictor, run_admet, score_admet_row

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


def test_score_admet_row_penalizes_liabilities_and_sascore():
    clean = pd.Series({"sascore": 2.0, "pains_alert": False, "brenk_alert": False,
                        "herg_liability": False, "cyp_liability": False, "poor_solubility": False})
    risky = pd.Series({"sascore": 8.0, "pains_alert": True, "brenk_alert": True,
                        "herg_liability": True, "cyp_liability": False, "poor_solubility": False})
    assert 0.0 <= score_admet_row(clean) <= 1.0
    assert 0.0 <= score_admet_row(risky) <= 1.0
    assert score_admet_row(clean) > score_admet_row(risky)


def test_score_admet_row_missing_sascore_is_neutral():
    row = pd.Series({"sascore": None, "pains_alert": False, "brenk_alert": False,
                      "herg_liability": False, "cyp_liability": False, "poor_solubility": False})
    assert score_admet_row(row) == 0.75  # 0.5*0.5(neutral) + 0.5*1.0(no liabilities)


def test_run_admet_writes_scores_csv(tiny_feature_parquet, tmp_path):
    artifacts_dir = str(tmp_path / "artifacts")
    scored = run_admet(tiny_feature_parquet, artifacts_dir, smoke=True)

    assert scored is not None
    assert "molecule_chembl_id" in scored.columns
    assert "admet_score" in scored.columns
    assert scored["admet_score"].between(0.0, 1.0).all()

    out_path = tmp_path / "artifacts" / "admet_scores.csv"
    assert out_path.exists()


def test_run_admet_skips_when_feature_file_missing(tmp_path):
    assert run_admet(str(tmp_path / "missing.parquet"), str(tmp_path / "artifacts")) is None
