# Design Experience (Semesters 5 & 6) — Master Project Blueprint (Audited v2.0)

# De Novo Virtual Screening & NISQ Quantum Kernel Benchmarking for 4th-Generation Mutant-Selective EGFR Inhibitors

> **Audited & Peer-Reviewed Capstone Specification**  
> **Student:** Krish  
> **Course:** Design Experience (5-Credit Integrated Course across Semesters 5 & 6)  
> **Institution:** MPSTME, NMIMS University  
> **Vertical:** Technical Competition / Commercial Innovation  
> **Faculty In-Charge:** Prof. Bhavna Bose  

---

## 1. Executive Summary & Clinical Objective

* **Disease Domain:** Non-Small Cell Lung Cancer (NSCLC) harboring acquired resistance mutations.
* **Target Objective:** Discover and prioritize **4th-Generation Reversible, Non-Covalent EGFR Inhibitors** with high selectivity for the triple-mutant `EGFR L858R/T790M/C797S` and `Exon19del/T790M/C797S` variants while **sparing Wild-Type EGFR ($EGFR^{WT}$)** to eliminate dose-limiting cutaneotoxicity.
* **The Computational Deliverable:** A leak-free, 3D-aware virtual screening and QML benchmarking pipeline that outputs ranked, synthesizable candidate leads with validated binding poses, ADMET compliance, and NISQ quantum kernel representation evaluations.

---

## 2. Refined 3-Stage System Architecture

```
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                    STAGE 1: CURATION, 3D DOCKING & GNN SCREENING                │
 │  • ChEMBL 37 / Curated Bioactivity Data (EGFR WT vs T790M/C797S Mutants)         │
 │  • 3D Target Structures: PDB 8A27 (C797S), 7ZYP (T790M/C797S), 7LG8 (WT counter) │
 │  • Molecular Docking (AutoDock Vina / Smina) -> 3D Binding Poses                 │
 │  • PyTorch Geometric 3D-GNN (EGNN / SchNet) + Bemis-Murcko Scaffold Split        │
 └────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                 STAGE 2: CONTROLLED NISQ QUANTUM KERNEL BENCHMARK                │
 │  • Projected Quantum Kernel (PQK) / QSVC in PennyLane on compressed feature space│
 │  • Matched Classical Controls: RBF-SVM, Kernel Ridge, Random Forest, XGBoost    │
 │  • Rigorous Negative Controls: Y-Scrambling, Feature Permutation, Untrained QNN  │
 │  • Hardware Validation: IBM Quantum QPU execution with Zero-Noise Extrapolation │
 └────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                 STAGE 3: CANDIDATE PRIORITIZATION & IN SILICO ADMET              │
 │  • Synthetic Accessibility Filtering: RDKit SAScore (< 4.5 = Makeable in wet lab)│
 │  • Full ADMET Profiling: Lipinski Ro5, hERG cardiac toxicity, PAMPA permeability │
 │  • Final Output: Top 3 Ranked 4th-Gen Mutant-Selective Candidate Leads           │
 └──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Grounded Quantum Computing Formulation

We explicitly avoid unprovable "Quantum Advantage" claims. Instead, we frame our quantum investigation as an **academic NISQ feasibility case study**:

1. **Quantum Machine Learning (QML):**
   * Construct a **Projected Quantum Kernel (PQK)** via parameterized quantum circuits in PennyLane (`AngleEmbedding` + `BasicEntanglerLayers`).
   * Compare classification fidelity ($T790M/C797S$ active vs. inactive) against matched classical non-linear kernels (RBF-SVM, Random Fourier Features).
2. **Hardware Validation & Error Mitigation:**
   * Pre-optimize circuits locally on `qiskit.aer`.
   * Execute final representative circuits on IBM Quantum 127-qubit QPUs (`qiskit-ibm-runtime`) using **Zero-Noise Extrapolation (ZNE)** to quantify hardware noise impact within the 10 min/month free allocation.

---

## 4. Required Scientific Controls & Benchmarks

| Control Layer | Benchmark / Experiment | Purpose |
| :--- | :--- | :--- |
| **Data Splitting** | Bemis-Murcko Scaffold Split & Temporal Split | Prevents model memorization of chemical series |
| **Negative Controls** | Y-Scrambling (Shuffled Labels) & Random Permutation | Proves model learns true structure-activity signals |
| **Decoy Benchmarking** | DUD-E / LIT-PCBA Decoy Screening ($EF_{1\%}$, ROC-AUC) | Evaluates enrichment capacity in realistic low-hit rate scenarios |
| **Synthesizability** | RDKit SAScore + Retrosynthesis Flagging | Ensures in silico candidates are chemically feasible to synthesize |
| **Hardware Concordance**| Noiseless Simulator vs. Noisy Simulator vs. Real QPU | Characterizes NISQ gate error and decoherence effects |

---

## 5. Phased Implementation Roadmap (Semesters 5 & 6)

* **Phase 1 (Sem 5, W1–4):** Curate ChEMBL 37 dataset (WT, T790M, C797S assays), clean SMILES, download verified PDB co-crystals (`8A27`, `7ZYP`), and establish Murcko scaffold splits.
* **Phase 2 (Sem 5, W5–10):** 3D molecular docking (AutoDock Vina), 3D interaction feature extraction, and PyTorch Geometric GNN training with DUD-E decoy evaluation.
* **Phase 3 (Sem 5, W11–16):** PennyLane Projected Quantum Kernel (PQK) construction, classical kernel benchmarking (RBF-SVM/XGBoost), and IBM Quantum hardware validation.
* **Phase 4 (Sem 6, W1–6):** SAScore synthesizability screening, ADMET filtering, and candidate ranking for mutant-selective leads.
* **Phase 5 (Sem 6, W7–12):** Comprehensive benchmark reporting, paper manuscript drafting, and technical competition submission.
