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

```
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                              THE 4-TIER SCREENING CASCADE                              │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │  [ Tier 1: 2D AI Baseline ]   Screen 13,033 ChEMBL compounds (RF ROC-AUC: 0.9304)      │
 │  [ Tier 2: 3D Docking ]       Ensemble docking against 6LUB & 7ZYP (-9.34 kcal/mol)    │
 │  [ Tier 3: Quantum QML ]      PennyLane 8-Qubit Fidelity Kernel (QSVC ROC-AUC: 0.8322) │
 │  [ Tier 4: ADMET & Synth ]    SAScore (< 4.5) & Lipinski / PAINS Liability Filters     │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │  🎯 FINAL DSS SCORE:          Ranked Candidate Hit Selection (13,033 compounds scored) │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏆 Top Discovered Lead Candidate: `CHEMBL4575267`

Out of **13,033 screened molecules**, our multi-tier DSS prioritized **`CHEMBL4575267`** as the **#1 Discovered Hit**:

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
| **Tier 3 (Quantum QML)** | 8-Qubit Fidelity QSVC ($K_{\text{FQ}}$) | **ROC-AUC: 0.8322** (PR-AUC: 0.8581) | Classical RBF-SVM: **0.8240** |
| **Test Suite** | Pytest Suite | **49 / 49 Passed** (100% test pass rate) | 0 Failures |

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
│   └── processed/                     # Featurized 2,059-dimensional Morgan Parquet files
├── models/                            # Serialized production models (.joblib)
├── src/                               # Modular production source code
│   ├── data/                          # ChEMBL 37 ingestion & RDKit 2,059-dim feature extraction
│   ├── models/                        # Scaffold-split training (RF, XGBoost, RBF-SVM)
│   ├── quantum/                       # PennyLane Fidelity Quantum Kernel (KFQ) & ZNE simulation
│   ├── pipeline/                      # Docking prep, selectivity, ADMET, & DSS ranking
│   └── api/                           # FastAPI REST backend & Streamlit web dashboard
└── tests/                             # Automated test suite (50 test cases, 49 passed)
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

# Or run individual stages:
python3 run.py data       # Stage 1: Data ingestion & featurization
python3 run.py train      # Stage 2: Classical ML baseline training
python3 run.py dock       # Stage 3: 3D docking grid preparation & selectivity
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
