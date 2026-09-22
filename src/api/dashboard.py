"""Streamlit dashboard for the EGFR screening API primitives.

Run with: ``streamlit run src/api/dashboard.py``.
"""

from __future__ import annotations

from io import BytesIO

import streamlit as st

from src.api.predictor import InvalidSmilesError, Prediction, PredictionService, validate_smiles

DEFAULT_SMILES = "COc1ccc(Nc2ncnc3cc(OC)c(OC)cc23)cc1NC(=O)C=C"


@st.cache_resource
def get_service() -> PredictionService:
    return PredictionService()


def molecule_image(smiles: str) -> BytesIO | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    image = Draw.MolToImage(molecule, size=(560, 360))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def prediction_card(prediction: Prediction) -> None:
    st.subheader("Prediction")
    a, b, c = st.columns(3)
    a.metric("Predicted pIC50", f"{prediction.pic50:.2f}")
    b.metric("Active probability", f"{prediction.active_probability:.1%}")
    c.metric("Composite score", f"{prediction.composite_score:.2f}")
    st.metric("Rank in candidate set", f"#{prediction.rank} of {prediction.set_size}")
    if prediction.fallback:
        st.warning(
            "Bootstrap-only deterministic fallback in use. It is for interface testing, "
            "not a validated prediction or a quantum-advantage result."
        )
    else:
        st.caption(f"Model source: {prediction.model_source}")


def main() -> None:
    st.set_page_config(page_title="EGFR Candidate Screen", page_icon="🧬", layout="wide")
    st.title("EGFR reversible-inhibitor screen")
    st.caption("T790M/C797S candidate triage • experimental predictions require follow-up validation")

    left, right = st.columns((1, 1), gap="large")
    with left:
        smiles = st.text_input("Candidate SMILES", value=DEFAULT_SMILES, help="Paste a valid SMILES string.")
        set_text = st.text_area(
            "Rank against a candidate set (optional)",
            help="One SMILES per line. The entered candidate is included automatically.",
            height=150,
        )
        submitted = st.button("Score candidate", type="primary", use_container_width=True)
    with right:
        try:
            canonical = validate_smiles(smiles)
            image = molecule_image(canonical)
            if image is not None:
                st.image(image, caption="2D molecular depiction")
            else:
                st.info("2D depiction becomes available when RDKit is installed.")
        except InvalidSmilesError:
            st.info("Enter a valid SMILES to render its 2D structure.")

    if submitted:
        references = [line.strip() for line in set_text.splitlines() if line.strip()]
        try:
            prediction = get_service().predict(smiles, references)
        except InvalidSmilesError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.error(f"The persisted model bundle cannot score this feature set: {exc}")
        else:
            prediction_card(prediction)


if __name__ == "__main__":
    main()
