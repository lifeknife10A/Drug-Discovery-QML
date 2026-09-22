"""Prediction primitives shared by the FastAPI service and Streamlit UI.

The service deliberately has a deterministic, chemistry-aware fallback so the
front door remains usable before a trained Tier 1 bundle is available.  The
fallback is labelled in every response and must not be treated as a validated
predictive result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import os
from pathlib import Path
from typing import Any

import numpy as np

try:  # RDKit is optional for import-time compatibility in lightweight deploys.
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
except ImportError:  # pragma: no cover - exercised only in minimal deployments
    Chem = DataStructs = AllChem = Crippen = Descriptors = Lipinski = None


# Names match src/models/train.py's joblib.dump targets. Order = preference
# (random forest has the best scaffold-split ROC-AUC, so it is tried first).
MODEL_FILENAMES = {
    "regressor": ("xgboost_regressor.joblib", "pic50_model.joblib", "regressor.joblib"),
    "classifier": (
        "random_forest_classifier.joblib",
        "xgboost_classifier.joblib",
        "rbf_svm_classifier.joblib",
        "activity_model.joblib",
    ),
}


class InvalidSmilesError(ValueError):
    """Raised when a submitted SMILES string cannot be parsed."""


@dataclass(frozen=True)
class Prediction:
    smiles: str
    pic50: float
    active_probability: float
    composite_score: float
    rank: int
    set_size: int
    model_source: str
    fallback: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Keep the requested scientific spelling in the public JSON contract.
        result["pIC50"] = result.pop("pic50")
        return result


def validate_smiles(smiles: str) -> str:
    canonical = smiles.strip() if isinstance(smiles, str) else ""
    if not canonical:
        raise InvalidSmilesError("SMILES must be a non-empty string.")
    if Chem is not None and Chem.MolFromSmiles(canonical) is None:
        raise InvalidSmilesError("SMILES could not be parsed by RDKit.")
    return canonical


def molecular_features(smiles: str) -> np.ndarray:
    """Return the 7 descriptors plus the 2,048-bit Morgan fingerprint."""
    smiles = validate_smiles(smiles)
    if Chem is None:
        # A deterministic stand-in allows a plain API installation to start.
        digest = sha256(smiles.encode()).digest()
        return np.fromiter((digest[index % len(digest)] / 255 for index in range(2055)), float)

    molecule = Chem.MolFromSmiles(smiles)
    descriptors = np.array(
        [
            Descriptors.MolWt(molecule),
            Crippen.MolLogP(molecule),
            Lipinski.NumHDonors(molecule),
            Lipinski.NumHAcceptors(molecule),
            Descriptors.TPSA(molecule),
            Lipinski.NumRotatableBonds(molecule),
            Lipinski.NumAromaticRings(molecule),
        ],
        dtype=float,
    )
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)
    bits = np.zeros(2048, dtype=float)
    DataStructs.ConvertToNumpyArray(fingerprint, bits)
    return np.concatenate((descriptors, bits))


class PredictionService:
    """Load Toby's persisted estimators when present, otherwise use a fallback."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.model_dir = Path(model_dir or os.getenv("EGFR_MODEL_DIR", root / "models"))
        self._models: dict[str, Any] | None = None

    def _load_models(self) -> dict[str, Any]:
        # Re-check on each service construction; Streamlit's resource cache owns reuse.
        if self._models is not None:
            return self._models
        try:
            import joblib
        except ImportError:  # pragma: no cover
            self._models = {}
            return self._models

        models: dict[str, Any] = {}
        for role, names in MODEL_FILENAMES.items():
            for name in names:
                candidate = self.model_dir / name
                if candidate.exists():
                    models[role] = joblib.load(candidate)
                    break
        self._models = models
        return models

    @staticmethod
    def _fit_width(model: Any, features: np.ndarray) -> np.ndarray:
        width = getattr(model, "n_features_in_", features.size)
        if width == features.size:
            return features.reshape(1, -1)
        if width == 2048:
            return features[7:].reshape(1, -1)
        if width == 7:
            return features[:7].reshape(1, -1)
        raise ValueError(f"Persisted model expects {width} features; API provides {features.size}.")

    @staticmethod
    def _fallback_scores(smiles: str, features: np.ndarray) -> tuple[float, float, float]:
        # Stable bootstrap-only values. They make integration testable, not scientific.
        seed = int.from_bytes(sha256(smiles.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        fp_density = float(features[7:].mean()) if features.size > 7 else float(features.mean())
        pic50 = float(np.clip(5.2 + 1.8 * fp_density + rng.normal(0, 0.15), 3.0, 9.0))
        active = float(1 / (1 + np.exp(-(pic50 - 6.0))))
        composite = float(np.clip(0.35 + 0.45 * fp_density + rng.normal(0, 0.03), 0.0, 1.0))
        return pic50, active, composite

    def _score_one(self, smiles: str) -> tuple[float, float, float, str, bool]:
        features = molecular_features(smiles)
        models = self._load_models()
        regressor, classifier = models.get("regressor"), models.get("classifier")
        if regressor is None or classifier is None:
            p, a, q = self._fallback_scores(smiles, features)
            return p, a, q, "deterministic_scaffold_fallback", True

        pic50 = float(np.asarray(regressor.predict(self._fit_width(regressor, features))).ravel()[0])
        if hasattr(classifier, "predict_proba"):
            active = float(np.asarray(classifier.predict_proba(self._fit_width(classifier, features)))[0, -1])
        else:
            active = float(np.asarray(classifier.predict(self._fit_width(classifier, features))).ravel()[0])
        # Bounded ranking feature derived from Tier 1 outputs only. Not sourced
        # from src/quantum's Tier 3 kernel benchmark and not a quantum-advantage claim.
        composite = float(np.clip((active + np.tanh((pic50 - 5.5) / 2)) / 2, 0.0, 1.0))
        return pic50, active, composite, "persisted_tier1_models", False

    def predict(self, smiles: str, reference_smiles: list[str] | None = None) -> Prediction:
        smiles = validate_smiles(smiles)
        references = [validate_smiles(item) for item in (reference_smiles or [])]
        candidates = list(dict.fromkeys([smiles, *references]))
        scores = {item: self._score_one(item) for item in candidates}
        ranking = sorted(candidates, key=lambda item: (scores[item][1], scores[item][0]), reverse=True)
        pic50, active, composite, source, fallback = scores[smiles]
        return Prediction(
            smiles=smiles,
            pic50=round(pic50, 4),
            active_probability=round(active, 4),
            composite_score=round(composite, 4),
            rank=ranking.index(smiles) + 1,
            set_size=len(candidates),
            model_source=source,
            fallback=fallback,
        )
