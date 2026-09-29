# Mutation-Aware Virtual Screening & Controlled NISQ Quantum Kernel Benchmarking for Reversible C797S-Active EGFR Inhibitors

> **Design Experience — Semesters 5 & 6**  
> **Institution:** MPSTME, NMIMS University  
> **Faculty Mentor:** Prof. Bhavna Bose  
> **Vertical:** Technical Competition / Commercial Innovation  

---

## 📌 Project Overview

This project develops an end-to-end **4-Tier Computational Decision Support System (DSS)** designed to prioritize **4th-Generation Reversible, Non-Covalent EGFR Inhibitors** for Non-Small Cell Lung Cancer (NSCLC).

When lung cancer patients develop the clinical **$\text{C797S}$ acquired resistance mutation** against 3rd-generation covalent inhibitors (*Osimertinib*), treatment fails. This pipeline integrates:
1. **Leakage-Free 2D Machine Learning** (Bemis-Murcko scaffold split on 13,033 ChEMBL compounds).
2. **3D Ensemble Molecular Docking** across mutant (`6LUB`, `7ZYP`) and Wild-Type (`4WKQ`) structures.
3. **Controlled NISQ Quantum Machine Learning** (PennyLane 8-qubit Fidelity Quantum Kernel vs. Classical RBF-SVM).
4. **ADMET & Synthetic Accessibility Filtering** (Lipinski Ro5, PAINS/Brenk alerts, and RDKit SAScore).

---

## 🏗️ 4-Tier Screening Pipeline Architecture

This is a **funnel**: cheap tiers screen the whole library; expensive 3D/quantum
tiers run only on the prioritized shortlist, then everything is fused into one score.

```
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                       THE 4-TIER SCREENING FUNNEL (not all tiers, all compounds)        │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │  [ Tier 1: 2D AI Baseline ]   ALL 13,033 ChEMBL compounds        (RF ROC-AUC: 0.9304)  │
 │  [ Tier 4: ADMET & Synth ]    ALL 13,033 compounds (SAScore, Lipinski/PAINS/BRENK)     │
 │  [ Tier 2: 3D Docking ]       top ~15 by Tier 1, vs 6LUB & 7ZYP  (best -9.34 kcal/mol) │
 │  [ Tier 3: Quantum QML ]      top ~300 by Tier 1, 4–8-qubit fidelity kernel QSVC       │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │  🎯 FINAL DSS SCORE:  weighted fusion; compounds outside the Tier 2/3 shortlist take   │
 │                       a neutral (median) value for those tiers, so bulk ranking ≈       │
 │                       Tier 1 + Tier 4, refined by 3D/quantum only at the top.           │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

> **Scope:** this is *computational virtual screening / candidate prioritization* over
> **already-known** ChEMBL compounds — not de novo molecule generation and not wet-lab
> validation. `CHEMBL4575267` was discovered/synthesized by prior researchers; this pipeline
> *ranked* it, it did not invent it.

---

## 🏆 Top-Ranked Prioritized Candidate: `CHEMBL4575267`

Out of **13,033 screened molecules**, our multi-tier DSS ranked the *already-known* compound **`CHEMBL4575267`** as the **#1 prioritized candidate** (a screening result, not a novel discovery or an experimentally validated hit):

* **Chemical Formula:** $\text{C}_{26}\text{H}_{19}\text{ClFN}_{7}\text{O}_{2}$
* **SMILES:** `C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCc1cn(-c2ccccc2)nn1`
* **3D Binding Affinity:** **$-9.34\,\text{kcal/mol}$** on `7ZYP` (double mutant) and **$-8.22\,\text{kcal/mol}$** on `6LUB` (triple mutant).
* **Tier 1 Active Probability:** **$95.67\%$**
* **Polar Surface Area (tPSA):** **$106.85\,\text{\AA}^2$** (Optimal human oral absorption, $< 140\,\text{\AA}^2$).
* **Composite DSS Score:** **`0.8301`** (#1 Rank in library).

---

## 📊 Benchmark & Validation Results

| Evaluation Layer | Model / Benchmark | Key Performance Metric | Negative Control / Baseline |
| :--- | :--- | :---: | :---: |
| **Tier 1 (2D AI)** | Random Forest Classifier | **ROC-AUC: 0.9304** (PR-AUC: 0.9679) | Y-Scramble ROC-AUC: **0.4876** |
| | XGBoost Classifier | **ROC-AUC: 0.9018** (PR-AUC: 0.9541) | Y-Scramble ROC-AUC: **0.4918** |
| | RBF-SVM Classifier | **ROC-AUC: 0.9050** (PR-AUC: 0.9515) | Y-Scramble ROC-AUC: **0.4909** |
| **Tier 2 (Selectivity)** | Paired WT vs C797S Screen | **223 Mutant-Selective Hits** (out of 309 pairs) | Sparing Wild-Type `4WKQ` |
| **Tier 3 (Quantum QML)** | Fidelity QSVC ($K_{\text{FQ}}$), 4–8 qubits | ROC-AUC: 0.8322 (PR-AUC: 0.8581) | Classical RBF-SVM: 0.8240 (statistically tied) |
| **Test Suite** | Pytest Suite | **49 passed, 1 skipped** | Skipped = real-Vina test (opt-in via `RUN_VINA_TESTS=1`) |

> **Honest reading of these numbers — please read before citing them:**
> - **No quantum advantage is claimed.** The Tier 3 code itself embeds this disclaimer. The
>   quantum QSVC (0.8322) and classical RBF (0.8240) differ by less than their bootstrap
>   confidence intervals — they are **statistically tied**. Tier 3 is a NISQ *feasibility
>   characterization*, not a demonstration that quantum beats classical.
> - **Tier 3 runs on heavily compressed features** (2,055 dims → 4–8 PCA components for the
>   qubits), so its AUC (~0.83) is *below* Tier 1's full-feature RF (0.93). It is only compared
>   against classical models on the *same* compressed space.
> - **"Hardware concordance" is simulated,** not run on a real QPU — a modeled depolarizing-noise
>   channel with Zero-Noise Extrapolation. Real IBM Quantum hardware validation is optional/future.
> - **hERG / CYP ADMET flags are naive structural heuristics** (a basic-amine / imidazole SMARTS),
>   not validated ADMET models. SAScore and PAINS/BRENK are legitimate; treat hERG/CYP as toy flags.
> - The test suite validates pipeline *mechanics* on synthetic fixtures with randomized labels —
>   it does **not** assert any benchmark number above (those come from full runs on real ChEMBL data).

---

## 🔬 Reproducibility & Data Availability

To keep the repository lightweight, the following are **generated locally and not committed**
(they are listed in `.gitignore`): `data/raw/`, `data/processed/`, `models/`, and `artifacts/`.

As a result, the headline figures in this README — ROC-AUC values, docking affinities, and the
`CHEMBL4575267` lead candidate — are **outputs of the authors' pipeline runs, not files shipped
in this repo**. To reproduce them you must fetch the ChEMBL EGFR data and run the tiers in order:

```bash
python3 -m src.data.fetch_chembl_data      # download raw ChEMBL bioactivities
python3 -m src.data.extract_features       # -> data/processed/egfr_features.parquet
python3 run.py tier1                        # -> models/*.joblib, artifacts/tier1_metrics.json
python3 run.py selectivity                  # -> artifacts/selectivity*.csv/json (WT vs C797S)
python3 run.py quantum                      # -> artifacts/quantum_scores.csv, tier3 benchmark
python3 run.py admet                        # -> artifacts/admet_scores.csv
python3 run.py rank                         # -> artifacts/top_candidates.csv
```

**Tier 2 (docking)** additionally requires a separate `vina-docking` conda env (AutoDock Vina +
Meeko) and the PDB structures under `data/raw/pdb/`; without it, `run.py dock` skips gracefully
and the DSS score renormalizes over the remaining tiers. Exact numbers may vary with dataset
snapshot, library versions, and docking hardware.

---

## 📂 Repository Structure

```
Drug Discovery QML/
├── README.md                          # Master project documentation
├── requirements.txt                    # Project dependencies (PennyLane, Qiskit, RDKit, PyTorch)
├── Makefile                           # Test & pipeline build automation
├── run.py                             # Central pipeline CLI orchestrator
├── artifacts/                         # Generated models, benchmarks & candidate rankings
│   ├── top_candidates.csv             # Full 13,033-compound ranked master dataset
│   ├── tier1_metrics.json             # 1,000-sample bootstrap metrics & Y-scramble controls
│   ├── tier3_kernel_benchmark.csv     # Quantum vs. Classical kernel comparison matrix
│   ├── docking_scores.csv             # 3D binding affinities against 6LUB & 7ZYP
│   ├── quantum_scores.csv             # 8-qubit quantum expectation values
│   └── admet_scores.csv               # SAScore synthesizability and ADMET liabilities
├── data/                              # Dataset storage (Raw PDB structures & Processed Parquets)
│   ├── raw/pdb/                       # Verified 3D PDB crystals: 6LUB, 7ZYP, 4WKQ
│   └── processed/                     # Featurized 2,055-dimensional Morgan Parquet (7 descriptors + 2,048 fp bits) files
├── models/                            # Serialized production models (.joblib)
├── src/                               # Modular production source code
│   ├── data/                          # ChEMBL 37 ingestion & RDKit 2,055-dim feature extraction (7 desc + 2,048 fp)
│   ├── models/                        # Scaffold-split training (RF, XGBoost, RBF-SVM)
│   ├── quantum/                       # PennyLane Fidelity Quantum Kernel (KFQ) & ZNE simulation
│   ├── pipeline/                      # Docking prep, selectivity, ADMET, & DSS ranking
│   └── api/                           # FastAPI REST backend & Streamlit web dashboard
└── tests/                             # Automated test suite (49 passed, 1 skipped)
```

---

## ⚙️ Quick Start & Execution

### 1. Installation & Environment Setup
```bash
# Clone the repository
git clone https://github.com/lifeknife10A/Drug-Discovery-QML.git
cd "Drug-Discovery-QML"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
python3 -m pytest tests/ -v
```

### 3. Run the Full Pipeline
```bash
# Run the complete 4-tier pipeline
python3 run.py all

# Or run individual stages (names must match run.py exactly):
python3 run.py data       # Stage 1: Data ingestion & featurization (manual; see note below)
python3 run.py tier1      # Stage 2: Classical ML baseline training (Tier 1)
python3 run.py dock       # Stage 3: 3D docking grid preparation & ensemble screen (top shortlist)
python3 run.py selectivity # Stage 3b: WT-vs-C797S mutant-selectivity analysis
python3 run.py quantum    # Stage 4: PennyLane Quantum Kernel benchmark
python3 run.py admet      # Stage 5: ADMET & SAScore synthesizability filtering
python3 run.py rank       # Stage 6: Composite DSS candidate ranking
```

### 4. Launch the Interactive Web Dashboard
```bash
streamlit run src/api/dashboard.py
```
*Access the web dashboard in your browser at: **`http://localhost:8501`***

### 5. Start the FastAPI REST Server
```bash
uvicorn src.api.app:app --reload
```
*Access interactive Swagger API docs at: **`http://localhost:8000/docs`***

---

## 🔑 Cloud Credentials Setup (Optional)

* **IBM Quantum Cloud:** Configure token in `~/.env` via `IBM_QUANTUM_API_KEY=your_token` for real 127-qubit hardware validation.
* **Kaggle GPU:** Configure `~/.kaggle/access_token` for free 30 hrs/week GPU cloud acceleration.
