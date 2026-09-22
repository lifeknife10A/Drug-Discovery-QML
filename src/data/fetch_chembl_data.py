#!/usr/bin/env python3
"""
Tier 1 data ingestion: EGFR bioactivity from the ChEMBL REST API.

Target CHEMBL203 (human EGFR kinase domain). ChEMBL stores wild-type and
resistance-mutant measurements under the SAME target id, distinguished at the
assay level by ``assay_variant_mutation`` (e.g. "T790M", "L858R,T790M",
"L858R,T790M,C797S"). This fetcher captures that field and buckets every IC50
record into a canonical variant so downstream code can model mutant activity and
mutant-vs-WT selectivity.

Outputs (data/processed/):
  egfr_compounds_clean.parquet/.csv  - combined, one row per (compound, variant)
  egfr_by_variant.parquet            - same, kept explicitly for the variant split
  chembl_raw_activities.parquet      - the raw pull, for provenance
  ingestion_summary.json             - counts per variant + dataset hash

There is NO synthetic fallback. If the API returns nothing, this raises.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

BASE = "https://www.ebi.ac.uk"
TARGET = "CHEMBL203"
OUT_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed")

# Canonical variant buckets, matched case/'/'-insensitively against the raw
# assay_variant_mutation string (which uses commas and varies in ordering).
VARIANT_RULES = [
    ("L858R/T790M/C797S", {"L858R", "T790M", "C797S"}),
    ("T790M/C797S", {"T790M", "C797S"}),
    ("L858R/T790M", {"L858R", "T790M"}),
    ("T790M", {"T790M"}),
    ("L858R", {"L858R"}),
    ("del19", {"DEL19"}),
]


def canonical_variant(mutation_str: str | None) -> str:
    """Map a raw ChEMBL assay_variant_mutation string to a canonical bucket."""
    if not mutation_str or not str(mutation_str).strip():
        return "WT"
    toks = {t.strip().upper() for t in str(mutation_str).replace("/", ",").split(",") if t.strip()}
    for name, needed in VARIANT_RULES:
        if needed.issubset(toks):
            return name
    return "other_mutant"


def fetch_egfr_activities(target_chembl_id: str = TARGET, max_records: int = 60000) -> pd.DataFrame:
    """Page through every IC50 activity for the target, keeping variant annotation."""
    url = (
        f"{BASE}/chembl/api/data/activity.json"
        f"?target_chembl_id={target_chembl_id}&standard_type=IC50&limit=1000"
    )
    rows: list[dict] = []
    page = 0
    while url and len(rows) < max_records:
        page += 1
        print(f"    page {page}  ({len(rows)} records)...")
        res = requests.get(url, timeout=30)
        res.raise_for_status()
        data = res.json()
        for item in data.get("activities", []):
            smiles = item.get("canonical_smiles")
            value = item.get("standard_value")
            units = item.get("standard_units")
            if not (smiles and value is not None and units == "nM"):
                continue
            try:
                val_nm = float(value)
            except (TypeError, ValueError):
                continue
            if val_nm <= 0:
                continue
            pic50 = -np.log10(val_nm * 1e-9)
            rows.append({
                "molecule_chembl_id": item.get("molecule_chembl_id"),
                "canonical_smiles": smiles,
                "assay_chembl_id": item.get("assay_chembl_id"),
                "assay_variant_mutation": item.get("assay_variant_mutation"),
                "variant": canonical_variant(item.get("assay_variant_mutation")),
                "standard_value_nm": val_nm,
                "relation": item.get("standard_relation"),
                "pIC50": pic50,
                "is_active": 1 if pic50 >= 6.0 else 0,  # IC50 <= 1 uM
                "document_year": item.get("document_year"),
            })
        nxt = data.get("page_meta", {}).get("next")
        url = BASE + nxt if nxt else None
        time.sleep(0.1)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"ChEMBL returned zero usable IC50 records for {target_chembl_id}. "
            "Refusing to continue with no data (no synthetic fallback)."
        )
    return df


def collapse_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (compound, variant): median pIC50 across replicate assays."""
    agg = (
        df.groupby(["molecule_chembl_id", "canonical_smiles", "variant"], as_index=False)
        .agg(
            pIC50=("pIC50", "median"),
            standard_value_nm=("standard_value_nm", "median"),
            n_measurements=("pIC50", "size"),
            document_year=("document_year", "min"),
        )
    )
    agg["is_active"] = (agg["pIC50"] >= 6.0).astype(int)
    return agg.sort_values(["variant", "pIC50"], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[+] Fetching EGFR ({TARGET}) IC50 activities from ChEMBL...")
    raw = fetch_egfr_activities()
    print(f"[+] Raw usable records: {len(raw)}")

    raw.to_parquet(os.path.join(OUT_DIR, "chembl_raw_activities.parquet"), index=False)

    clean = collapse_duplicates(raw)
    counts = clean["variant"].value_counts().to_dict()
    print("[+] Compounds per variant bucket:")
    for k, v in counts.items():
        act = int(clean.loc[clean.variant == k, "is_active"].sum())
        print(f"      {k:22s} {v:6d}   (active {act})")

    # Combined table (all variants) + explicit variant-split copy.
    clean.to_parquet(os.path.join(OUT_DIR, "egfr_by_variant.parquet"), index=False)

    # Backwards-compatible primary file = WT rows only (what Tier 1 has always used).
    wt = clean[clean.variant == "WT"].drop(columns=["variant"]).reset_index(drop=True)
    wt.to_parquet(os.path.join(OUT_DIR, "egfr_compounds_clean.parquet"), index=False)
    wt.to_csv(os.path.join(OUT_DIR, "egfr_compounds_clean.csv"), index=False)

    ds_hash = hashlib.sha256(
        "\n".join(sorted(clean["canonical_smiles"].astype(str))).encode()
    ).hexdigest()[:16]
    summary = {
        "target": TARGET,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_raw_records": int(len(raw)),
        "n_compound_variant_rows": int(len(clean)),
        "variant_counts": counts,
        "variant_active_counts": {
            k: int(clean.loc[clean.variant == k, "is_active"].sum()) for k in counts
        },
        "dataset_hash": ds_hash,
        "notes": "assay_variant_mutation bucketed via canonical_variant(); "
                 "duplicates collapsed to median pIC50 per (compound, variant).",
    }
    with open(os.path.join(OUT_DIR, "ingestion_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] WT primary file: {len(wt)} compounds -> egfr_compounds_clean.parquet")
    print(f"[OK] Variant split:   {len(clean)} rows -> egfr_by_variant.parquet")
    print(f"[OK] Summary:          data/processed/ingestion_summary.json")


if __name__ == "__main__":
    main()
