# Hybrid Quantum-Classical AI Platform for Sub-Atomic Molecular Docking & Target Binding Discovery

> **Design Experience — Project 2 (Semesters 5 & 6)**  
> **Institution:** MPSTME, NMIMS University  
> **Faculty Mentor:** Prof. Bhavna Bose  
> **Vertical:** Technical Competition / Commercial Startup Incubation  

---

## 📌 Project Overview

This project builds a commercial-grade **Hybrid Quantum-Classical Decision Support System (DSS)** for early-stage drug discovery. By fusing **Classical Graph Neural Networks (PyTorch Geometric)** for fast macro-screening with **Quantum Machine Learning (PennyLane + Qiskit VQE/QSVC)** for sub-atomic electronic binding affinity simulation, the platform reduces candidate drug screening timelines from years to weeks.

---

## 👥 Team Onboarding & Workspace Structure

Whether working individually or in a team, this repository maintains a clean, modular structure so all team members can contribute, track progress, and run experiments.

```
Drug_Discovery_QML/
├── README.md                          # Main repository guide & team onboarding
├── requirements.txt                    # Project dependencies (PennyLane, Qiskit, PyTorch, RDKit)
├── .gitignore                          # Excludes large data & sensitive credentials
├── docs/                               # Architecture blueprints & meeting notes
│   └── project2_blueprint.md          # Full 5-phase project blueprint
├── data/                               # Data storage (Raw & Processed Parquet files)
│   ├── raw/                            # Raw ChEMBL downloads
│   └── processed/                      # Extracted feature sets
├── notebooks/                          # Jupyter Notebooks for research & experiments
├── src/                                # Modular production Python package
│   ├── data/                           # ChEMBL ingestion & RDKit feature extraction
│   ├── models/                         # GNN & XGBoost classical models
│   ├── quantum/                        # PennyLane & Qiskit quantum circuits
│   ├── pipeline/                       # Hybrid DSS scoring formula
│   └── api/                            # FastAPI backend & Streamlit web dashboard
└── tests/                              # Automated tests
```

---

## ⚙️ Quick Start Setup

### 1. Prerequisites & Virtual Environment
Ensure Python 3.10+ is installed on your machine.

```bash
# Clone repository (if fetching from GitHub)
# git clone <your-repo-url>
cd Drug_Discovery_QML

# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

### 2. Free Cloud & API Credentials Setup

This project uses **100% free cloud resources**. Follow the setup guide below to configure your credentials:

#### A. IBM Quantum Experience (Free Real Qubit & Cloud Simulator)
1. Sign up / Log in at [quantum.ibm.com](https://quantum.ibm.com/).
2. Copy your API token from the dashboard.
3. Save it to your local environment by running:
   ```bash
   printf "Enter IBM_QUANTUM_API_KEY (typing hidden): " && read -s val && echo && echo "IBM_QUANTUM_API_KEY=$val" >> ~/.env && echo "Saved."
   ```

#### B. Kaggle API Key (30 Free GPU Hours/Week)
1. Log in to [kaggle.com](https://www.kaggle.com/) -> Settings -> API -> Click **Create New Token**.
2. Save the downloaded `kaggle.json` file to `~/.kaggle/kaggle.json`:
   ```bash
   mkdir -p ~/.kaggle && cp ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```

---

## 🚀 5-Phase Implementation Roadmap

- [ ] **Phase 1 (Sem 5, W1-4):** Target selection (EGFR oncology receptor), ChEMBL data pipeline, RDKit feature extraction & XGBoost baseline.
- [ ] **Phase 2 (Sem 5, W5-10):** PyTorch Geometric Graph Neural Network (GNN) macro-filter trained on Kaggle GPU.
- [ ] **Phase 3 (Sem 5, W11-16):** PennyLane + Qiskit 4-8 Qubit VQE / QSVC simulation & IBM Quantum Cloud execution.
- [ ] **Phase 4 (Sem 6, W1-6):** Hybrid DSS Scoring Formula, MLflow experiment tracking & DVC versioning.
- [ ] **Phase 5 (Sem 6, W7-12):** FastAPI REST backend, Streamlit 3D molecular visualization dashboard & competition submission.
