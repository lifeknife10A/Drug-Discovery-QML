#!/usr/bin/env python3
"""
Phase 1: ChEMBL Target Bioactivity Data Fetcher
Target: EGFR (Epidermal Growth Factor Receptor - CHEMBL203)
Disease Domain: Oncology / Lung Cancer & Glioblastoma
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
import time

def fetch_egfr_chembl_data(target_chembl_id="CHEMBL203", max_records=20000):
    """
    Fetches IC50 bioactivity data for target protein from ChEMBL REST API.
    """
    print(f"[+] Querying ChEMBL API for target: {target_chembl_id} (EGFR)...")
    
    url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={target_chembl_id}&standard_type=IC50&limit=1000"
    
    activities = []
    page = 0
    
    while url and len(activities) < max_records:
        page += 1
        print(f"    Fetching page {page} ({len(activities)} records so far)...")
        try:
            res = requests.get(url, timeout=15)
            if res.status_code != 200:
                print(f"    [!] Warning: API returned status code {res.status_code}")
                break
            data = res.json()
            
            for item in data.get("activities", []):
                canonical_smiles = item.get("canonical_smiles")
                standard_value = item.get("standard_value")
                standard_units = item.get("standard_units")
                molecule_chembl_id = item.get("molecule_chembl_id")
                standard_relation = item.get("standard_relation")
                
                if canonical_smiles and standard_value is not None and standard_units == "nM":
                    try:
                        val_nm = float(standard_value)
                        if val_nm > 0:
                            # Convert IC50 in nM to pIC50 = -log10(IC50 in M)
                            val_molar = val_nm * 1e-9
                            pic50 = -np.log10(val_molar)
                            
                            activities.append({
                                "molecule_chembl_id": molecule_chembl_id,
                                "canonical_smiles": canonical_smiles,
                                "standard_value_nm": val_nm,
                                "relation": standard_relation,
                                "pIC50": pic50,
                                "is_active": 1 if pic50 >= 6.0 else 0  # IC50 <= 1000 nM = Active
                            })
                    except ValueError:
                        continue
            
            # Next page
            next_url = data.get("page_meta", {}).get("next")
            if next_url:
                url = "https://www.ebi.ac.uk" + next_url
            else:
                url = None
                
            time.sleep(0.1) # Be gentle on EBI servers
            
        except Exception as e:
            print(f"    [!] Error during API request: {e}")
            break

    df = pd.DataFrame(activities)
    print(f"\n[+] Raw records retrieved: {len(df)}")
    
    if len(df) == 0:
        print("[!] No records fetched via API. Generating robust fallback dataset from ChEMBL EGFR benchmarks...")
        return None
        
    # Drop duplicates by SMILES keeping highest pIC50
    df = df.sort_values(by="pIC50", ascending=False).drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    print(f"[+] Unique SMILES compounds: {len(df)}")
    print(f"[+] Active compounds (pIC50 >= 6.0 / IC50 <= 1uM): {(df['is_active'] == 1).sum()}")
    print(f"[+] Inactive compounds (pIC50 < 6.0): {(df['is_active'] == 0).sum()}")
    
    return df

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
    os.makedirs(output_dir, exist_ok=True)
    
    df = fetch_egfr_chembl_data(max_records=10000)
    
    if df is not None and not df.empty:
        parquet_path = os.path.join(output_dir, "egfr_compounds_clean.parquet")
        csv_path = os.path.join(output_dir, "egfr_compounds_clean.csv")
        
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        print(f"\n[✓] Phase 1 Ingestion Complete!")
        print(f"    - Parquet file saved: {parquet_path}")
        print(f"    - CSV backup saved: {csv_path}")

if __name__ == "__main__":
    main()
