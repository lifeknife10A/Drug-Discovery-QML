import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.predictor import InvalidSmilesError, PredictionService, molecular_features


def test_fallback_prediction_is_deterministic_and_ranked():
    service = PredictionService(model_dir="missing-model-directory")
    prediction = service.predict("CCO", ["CCN", "c1ccccc1"])

    assert prediction.fallback is True
    assert prediction.model_source == "deterministic_scaffold_fallback"
    assert 1 <= prediction.rank <= prediction.set_size == 3
    assert 0 <= prediction.active_probability <= 1
    assert 0 <= prediction.composite_score <= 1
    assert service.predict("CCO").pic50 == service.predict("CCO").pic50


def test_invalid_smiles_is_rejected():
    with pytest.raises(InvalidSmilesError):
        molecular_features("not-a-smiles")


def test_public_dict_uses_pic50_spelling():
    result = PredictionService(model_dir="missing-model-directory").predict("CCO").to_dict()
    assert "pIC50" in result
    assert "pic50" not in result
