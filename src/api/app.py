"""FastAPI front door for EGFR candidate scoring.

Run with: ``uvicorn src.api.app:app --reload``.
"""

from __future__ import annotations

from typing import Annotated

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - gives an actionable local error
    raise ImportError("FastAPI is required for src.api.app. Install requirements.txt.") from exc

from src.api.predictor import InvalidSmilesError, PredictionService


class PredictionRequest(BaseModel):
    smiles: Annotated[str, Field(min_length=1, description="Candidate molecule as a SMILES string")]
    reference_smiles: list[str] = Field(
        default_factory=list,
        description="Optional candidate set used to compute rank-in-set.",
    )


class PredictionResponse(BaseModel):
    smiles: str
    pIC50: float
    active_probability: float
    composite_score: float
    rank: int
    set_size: int
    model_source: str
    fallback: bool


app = FastAPI(title="EGFR Screening API", version="0.1.0")
_service = PredictionService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> dict[str, object]:
    """Score a molecule and rank it against any supplied reference set."""
    try:
        return _service.predict(payload.smiles, payload.reference_smiles).to_dict()
    except InvalidSmilesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"Model bundle is incompatible: {exc}") from exc
