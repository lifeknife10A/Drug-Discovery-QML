# Design Experience (Semesters 5 & 6) — Master Project Blueprint

# Hybrid Quantum-Classical AI Platform for Sub-Atomic Molecular Docking & Target Binding Discovery

> **Master Specification & Implementation Document**  
> **Student:** Krishna Solanki	
> **Course:** Design Experience (5-Credit Integrated Course across Semesters 4, 5 & 6)  
> **Institution:** MPSTME, NMIMS University  
> **Vertical:** Technical Competition / Commercial Startup Incubation  
> **Faculty In-Charge:** Prof. Bhavna Bose  

---

## 1. Course Context & Academic History

- **Course:** Design Experience (Semesters 4, 5, and 6 — Total 5 Credits).
- **Selected Vertical:** **Technical Competition** — aiming to represent MPSTME & NMIMS at national/global platforms (Smart India Hackathon [SIH], Kaggle/Zindi, NASA Space Apps, e-Yantra, GSoC, IEEE).
- **Faculty Mentor:** **Prof. Bhavna Bose**.
- **Completed Project 1 (Sem 4 -> Sem 5 W1):** *Battery Materials DSS QML* (Lithium compound stability screening using simulated QML + XGBoost across 210,000+ Materials Project compounds).
- **Official Project 2 (Sem 5 & 6):** **Hybrid Quantum-Classical AI Platform for Sub-Atomic Molecular Docking & Target Binding Discovery**.

---

## 2. Master Problem Statement & Commercial Vision

### The Real-World Business Problem
* **Financial & Time Loss:** Biopharma companies spend over **$2 Billion and 10 to 15 years** to bring a single new drug to market.
* **The 90% Failure Bottleneck:** Over **90% of candidate molecules fail** in wet-lab tests and clinical trials because classical computers cannot accurately calculate sub-atomic electronic binding interactions between drug molecules and human target proteins without exponential approximation errors ($O(2^N)$ complexity).
* **The Opportunity:** A hybrid software platform combining rapid classical screening with quantum-accurate sub-atomic binding simulation can reduce early-stage drug discovery timelines from years to weeks, saving biopharma firms millions in wasted lab experiments.

### The Hybrid Solution Strategy
* **Stage 1 (Classical Macro-Filter):** Use a **PyTorch Geometric Graph Neural Network (GNN)** to rapidly screen 1,000,000+ candidate SMILES molecules from ChEMBL down to the top 1,000 candidates.
* **Stage 2 (Quantum Micro-Scorer):** Use **PennyLane + Qiskit (VQE / Quantum Kernel Estimation)** to simulate exact sub-atomic electronic binding forces on candidate target protein pockets.
* **Commercial Deliverable Output:** A web-based Drug Discovery SaaS Dashboard & API for biopharma R&D teams, ready for startup incubation, IP filing (under NMIMS 50-50 IP policy), and technical competition grand prizes.

---

## 3. System Architecture & Technology Stack

```
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                         DATA INGESTION & FEATURE PIPELINE                       │
 │  ChEMBL v33 API / PubChem ──► RDKit ──► Morgan Fingerprints + 3D Conformer Graphs │
 └────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                      STAGE 1: CLASSICAL GNN MACRO-FILTER                         │
 │  PyTorch Geometric Graph Neural Network (GNN) ──► Screen 1,000,000 -> Top 1,000 │
 └────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                      STAGE 2: PENNYLANE + QISKIT QUANTUM CORE                    │
 │  PennyLane (`qml.device('qiskit.aer')`) ──► 4–8 Qubit VQE & Quantum Kernel (QSVC)│
 └────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                      HYBRID DSS DECISION ENGINE & MLOPS                          │
 │  MLflow Experiment Tracking + DVC Versioning + FastAPI Backend + Streamlit UI    │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

### Complete Software & Tooling Stack
- **Core Logic & ML:** Python 3.10+, PyTorch 2.x, PyTorch Geometric, scikit-learn, XGBoost, RDKit (Cheminformatics).
- **Quantum Stack:** **PennyLane** (Quantum ML / PyTorch Autograd) + **Qiskit** (`pennylane-qiskit` plugin for `qiskit.aer` simulation & `qiskit-ibm-runtime` for real qubit execution).
- **Data Engineering:** DuckDB, Polars, PyArrow, Pandas.
- **MLOps & Quality:** MLflow (experiment tracking), DVC (data version control), Evidently AI (drift monitoring).
- **Deployment & UI:** FastAPI (REST API), Streamlit + py3Dmol (interactive 3D molecular visualization).

---

## 4. Hardware Feasibility & Zero-Cost Cloud Strategy

| Component | Execution Platform | Specs & Cost | Purpose |
| :--- | :--- | :--- | :--- |
| **Local Data Cleaning & Quantum Kernels** | **Local Mac Laptop** | CPU / 4-8 Qubit Simulator (**$0 Cost**) | Feature extraction, 4-8 qubit PennyLane simulation, API testing |
| **Heavy GNN Model Training** | **Kaggle GPU** | NVIDIA P100 / Dual T4 (**30 Free hrs/week**) | Training PyTorch Geometric GNN on 50,000+ compounds |
| **Real Qubit Validation** | **IBM Quantum Cloud** | Real 127-Qubit IBM Quantum Hardware (**$0 Cost**) | Final circuit execution via `qiskit-ibm-runtime` |
| **Web UI & API Deployment** | **DigitalOcean / Azure** | GitHub Student Pack (**$100-$200 Free Credits**) | Hosting FastAPI backend & Streamlit web dashboard |

---

## 5. Phased Implementation Roadmap (Semesters 5 & 6)

---

### **PHASE 1 (Semester 5, Weeks 1–4): Target Selection, Data Pipeline & Baseline Setup**
- **Objective:** Select target protein, fetch bioactivity datasets, and establish classical benchmark models.
- **Detailed Tasks:**
  1. Select a high-value disease target protein (e.g., **EGFR** for oncology / lung cancer or **HER2**).
  2. Query **ChEMBL v33 API** to fetch 30,000+ small-molecule compounds with logged IC50 / Ki bioactivity values against the target protein.
  3. Use **RDKit** to convert SMILES strings into molecular graphs, 2D Morgan Fingerprints (2048-bit), and 3D conformer coordinates.
  4. Train a classical baseline model (**XGBoost / Random Forest**) on 2D Morgan fingerprints to establish ROC-AUC, F1-score, and RMSE baselines.
- **Phase 1 Output:** Verified dataset CSV/Parquet files (`data/processed/target_egfr_compounds.parquet`) and baseline XGBoost model evaluation logs.

---

### **PHASE 2 (Semester 5, Weeks 5–10): Classical Graph Neural Network (GNN) Macro-Filter**
- **Objective:** Build a deep Graph Neural Network to screen large compound libraries.
- **Detailed Tasks:**
  1. Implement a **PyTorch Geometric (PyG)** GNN architecture (GCN / GAT / GraphSAGE) representing atoms as nodes (atomic number, hybridization, formal charge) and bonds as edges (single, double, aromatic).
  2. Train the GNN on Kaggle Free GPU (NVIDIA P100) using multi-task loss (predicting binding affinity $pK_i$ + LogP solubility).
  3. Track all hyperparameters, training loss curves, and model metrics using **MLflow**.
- **Phase 2 Output:** Trained PyTorch Geometric GNN model capable of screening 1,000,000+ compounds and outputting top 1,000 candidates with >92% validation accuracy.

---

### **PHASE 3 (Semester 5, Weeks 11–16): Quantum Core Implementation (PennyLane + Qiskit)**
- **Objective:** Build the sub-atomic quantum circuit simulation engine for precise binding affinity scoring.
- **Detailed Tasks:**
  1. Formulate sub-atomic protein-ligand electronic interactions as a **4-to-8 qubit Variational Quantum Eigensolver (VQE)** / **Quantum Kernel Estimation (QSVC)** problem.
  2. Implement PennyLane quantum circuits using `AngleEmbedding` for feature mapping and `StronglyEntanglingLayers` / `BasicEntanglerLayers` for parameterization.
  3. Connect PennyLane to the `qiskit.aer` local simulator backend via the `pennylane-qiskit` plugin.
  4. Perform exhaustive quantum feature selection and SVM hyperparameter tuning (SVM $C$, angle scales $\pi/2, \pi, 2\pi$).
  5. Execute the top-performing quantum circuits on **real 127-qubit IBM Quantum Hardware** via `qiskit-ibm-runtime` API for experimental validation.
- **Phase 3 Output:** Functional QML kernel module (`src/quantum/qml_kernel.py`) validated on both `qiskit.aer` simulator and IBM Quantum Cloud hardware.

---

### **PHASE 4 (Semester 6, Weeks 1–6): Hybrid Decision Engine & MLOps Integration**
- **Objective:** Integrate classical macro-filter and quantum micro-scorer into a unified Decision Support System (DSS) with production MLOps.
- **Detailed Tasks:**
  1. Build a hybrid scoring formula combining GNN macro probability, QML sub-atomic confidence, toxicity penalty, and model disagreement flags:
     $$\text{Hybrid Score} = w_1 \cdot P_{\text{GNN}} + w_2 \cdot P_{\text{QML}} - \text{Toxicity Penalty} - \text{Disagreement Penalty}$$
  2. Implement **DVC (Data Version Control)** to track dataset iterations.
  3. Implement **Evidently AI** monitoring to detect feature drift and model degradation over time.
- **Phase 4 Output:** Fully functional hybrid decision pipeline script (`src/pipeline/hybrid_dss_engine.py`) and automated MLOps tracking suite.

---

### **PHASE 5 (Semester 6, Weeks 7–12): Production SaaS API, Interactive Web Dashboard & Competition Pitch**
- **Objective:** Package the project into a commercial-grade product UI, REST API, and submission-ready pitch deck.
- **Detailed Tasks:**
  1. Develop a **FastAPI** REST backend serving real-time endpoints for compound scoring (`/api/v1/predict_binding`).
  2. Build a modern, responsive **Streamlit / React** web dashboard featuring:
     - **3D Molecular Viewer (py3Dmol):** Interactive rendering of 3D drug-protein binding pockets.
     - **Hybrid DSS Leaderboard:** Interactive compound ranking table with filterable QML, GNN, and toxicity scores.
     - **Exportable PDF Reports:** Business-ready screening summaries for biopharma teams.
  3. Prepare paper manuscript, GitHub repository documentation, and competition pitch deck for **Smart India Hackathon (SIH)**, IEEE, and NMIMS IP evaluation.
- **Phase 5 Output:** Deployed web dashboard, REST API endpoints, published GitHub repository, and competition pitch deck.

---

## 6. Phase Summary & Milestone Checklist

| Phase | Timeline | Core Deliverable | Key Tech Used | Milestone Verification |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Sem 5, W1–4 | Target Dataset & XGBoost Baseline | ChEMBL API, RDKit, XGBoost | Baseline ROC-AUC logged in MLflow |
| **Phase 2** | Sem 5, W5–10 | GNN Macro-Filter Model | PyTorch Geometric, Kaggle GPU | Top 1,000 candidates extracted |
| **Phase 3** | Sem 5, W11–16 | PennyLane + Qiskit Quantum Engine | PennyLane, Qiskit, IBM Quantum | QML kernel validated on IBM hardware |
| **Phase 4** | Sem 6, W1–6 | Hybrid DSS Engine & MLOps Pipeline | MLflow, DVC, Evidently AI | End-to-end automated hybrid scoring |
| **Phase 5** | Sem 6, W7–12 | SaaS REST API, Web UI & Pitch Deck | FastAPI, Streamlit, py3Dmol | Deployed dashboard & SIH submission |
