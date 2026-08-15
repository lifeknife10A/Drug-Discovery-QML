# Design Experience (Semesters 5 & 6) — Master Project Blueprint (v3.0)

# Mutation-Aware Virtual Screening & Controlled NISQ Quantum Kernel Benchmarking for Reversible C797S-Active EGFR Inhibitor Candidates

> **Scientifically Audited Capstone Specification (Version 3.0)**  
> **Student:** Krish  
> **Course:** Design Experience (5-Credit Integrated Course across Semesters 4, 5 & 6)  
> **Institution:** MPSTME, NMIMS University  
> **Vertical:** Technical Competition / Commercial Innovation  
> **Faculty In-Charge:** Prof. Bhavna Bose  

---

## 1. Project Scope & Clinical Objective

* **Disease Indication:** Non-Small Cell Lung Cancer (NSCLC) exhibiting acquired resistance to 3rd-generation covalent EGFR inhibitors (*Osimertinib*).
* **Target Objective:** Prioritize **Reversible, Non-Covalent Small-Molecule Inhibitor Candidates** active against the `EGFR L858R/T790M/C797S` triple mutant and `T790M/C797S` double mutant variants, while maintaining selectivity over **Wild-Type EGFR ($EGFR^{WT}$)** to reduce on-target cutaneotoxicity risk.
* **Final Deliverable:** A reproducible, leakage-controlled computational pipeline that outputs **Top 3 Ranked Computational Candidate Hits** with protocol-validated predicted binding poses, estimated mutant-vs-WT selectivity ratios, retrosynthetic route evaluations (SAScore), predicted ADMET risk profiles, and a controlled classical-vs-quantum kernel benchmark.

---

## 2. Structural Biology & PDB Reference Panel

To prevent crystallographic construct artifacts from distorting docking grids and selectivity predictions, all target structures are verified and annotated residue-by-residue:

| Role | PDB ID | Resolution | Target Variant & Construct Details | Bound Ligand |
| :--- | :---: | :---: | :--- | :--- |
| **Primary Target (Triple Mutant)** | **`6LUB`** | $2.60\,\text{\AA}$ | `EGFR L858R/T790M/C797S` (Clinical Resistance) | Reversible allosteric lead |
| **Primary Target (Double Mutant)** | **`7ZYP`** | $2.30\,\text{\AA}$ | `EGFR T790M/C797S` (Reversible inhibitor complex) | Reversible aminopyrimidine |
| **Secondary Target (Structural Lead)**| **`9D3W`** | $2.10\,\text{\AA}$ | `EGFR L858R/T790M/C797S` (High-resolution crystal) | 4th-Gen candidate inhibitor |
| **WT Counter-Screen Baseline** | **`4WKQ`** | $2.02\,\text{\AA}$ | **Genuine Wild-Type EGFR ($EGFR^{WT}$)** | Gefitinib (Active conformation) |
| **WT Alternative Conformation** | **`2ITY`** | $3.00\,\text{\AA}$ | **Wild-Type EGFR ($EGFR^{WT}$)** | Erlotinib (Kinase domain) |
| **Allosteric Reference (Annotated)** | **`8A27`** | $2.40\,\text{\AA}$ | EGFR Kinase Domain (WT allosteric reference) | Compound 57 |
| **Construct Reference (Annotated)** | **`7LG8`** | $2.80\,\text{\AA}$ | `EGFR T790M/V948R` (Contains crystallographic $V948R$ mutation) | Naquotinib + Allosteric binder |

---

## 3. The 4-Tier Screening & Benchmarking Pipeline

```
[ ChEMBL 37 Curated Bioactivity Data (EGFR WT vs Mutant Assays) ]
                            │
                            ▼
    [ Tier 1: 2D Ligand Baseline & Leakage-Controlled Screening ]
  • Feature Encoding: 2048-bit Morgan Fingerprints (ECFP4) + RDKit Descriptors
  • Splitting Strategy: Bemis-Murcko Scaffold Split & Temporal Series Holdout
  • Baseline Models: XGBoost, Random Forest, RBF-SVM, 2D GCN/GAT
  • Output: Narrows candidate pool down to Top 10,000 Scaffolds
                            │
                            ▼
    [ Tier 2: 3D Ensemble Molecular Docking & Pose Rescoring ]
  • Ensemble Docking (AutoDock Vina / Smina) against 6LUB, 7ZYP, and 4WKQ
  • Redocking Protocol Validation (Cognate pose RMSD < 2.0 Å)
  • Interaction Rescoring: ProLIF 3D Protein-Ligand Interaction Fingerprints
  • Output: Narrows down to Top 100–500 Scaffolds with estimated mutant selectivity
                            │
                            ▼
    [ Tier 3: Controlled Classical vs. NISQ Quantum Kernel Benchmark ]
  • Dimensionality Reduction: Matched 4-to-8 latent components (PCA/Autoencoder)
  • Quantum Kernel: Fidelity Quantum Kernel K_FQ(x_i, x_j) = |⟨ψ(x_i)|ψ(x_j)⟩|² in PennyLane
  • Classical Controls: Linear SVM, RBF-SVM, Polynomial Kernel, Kernel Ridge
  • Negative Controls: Y-Scrambling (Shuffled Labels), Feature Permutation, Fixed Random Circuit
  • Hardware Concordance: IBM Quantum QPU execution on 20x20 stratified subset with ZNE
                            │
                            ▼
    [ Tier 4: ADMET Risk Assessment & Synthetic Accessibility Review ]
  • Synthetic Accessibility: RDKit SAScore (< 4.5 prioritization) + AiZynthFinder retrosynthesis
  • Predicted ADMET Profiling: cLogP, TPSA, hERG risk, PAMPA permeability, PAINS filter
  • Final Output: Top 3 Ranked Computational Candidate Hits
```

---

## 4. Grounded Quantum Computing Formulation

We explicitly declare that this study is a **controlled NISQ feasibility & representational capacity benchmark**, with no prior presumption of quantum advantage:

### A. Mathematical Kernel Definition
1. **Fidelity Quantum Kernel ($K_{\text{FQ}}$):**
   $$|\psi(x)\rangle = U(x)|0\rangle, \quad K_{\text{FQ}}(x_i, x_j) = |\langle \psi(x_i) | \psi(x_j) \rangle|^2$$
   Implemented via `qml.AngleEmbedding` + `qml.BasicEntanglerLayers` in PennyLane.
2. **Fair Classical Comparison:** Classical models (RBF-SVM, Kernel Ridge) are evaluated on the exact same 4-to-8 compressed input features, with full-feature classical baselines reported as a documented reference.
3. **Pre-Registered Hypothesis:** If $K_{\text{FQ}}$ performs comparably or inferior to classical RBF-SVM, this will be reported transparently as a characterization of current NISQ feature-mapping limits on heterogeneous bioactivity data.

### B. IBM Quantum Hardware Budgeting
* Full $N \times N$ kernel matrices ($O(N^2)$ pairs) will be computed locally on `qiskit.aer`.
* A **representative $20 \times 20$ stratified subset** (active-active, active-inactive, inactive-inactive pairs) will be executed on IBM Quantum QPUs (`qiskit-ibm-runtime`) to measure hardware gate error, decoherence, and Zero-Noise Extrapolation (ZNE) mitigation without exceeding the free 10 min/month quota.

---

## 5. Required Validation Matrix & Headline Metrics

| Evaluation Layer | Metric / Protocol | Purpose |
| :--- | :--- | :--- |
| **Screening Performance** | PR-AUC, ROC-AUC, $EF_{1\%}$, BEDROC with bootstrap 95% CI | Prevents over-optimistic accuracy reporting on imbalanced data |
| **Negative Controls** | Y-Scrambling ($y$-permuted labels), Feature Randomization | Proves model learns true structure-activity relationships |
| **Decoy Stress Test** | DUD-E EGFR Decoy Benchmark | Secondary check on enrichment factors against property-matched decoys |
| **Docking Validation** | Cognate Ligand Redocking ($\text{RMSD} < 2.0\,\text{\AA}$) | Validates spatial grid and scoring function before virtual screening |
| **Synthesizability** | SAScore + Retrosynthetic Step Count & Starting Material Review | Distinguishes plausible chemical leads from virtual artifacts |
| **ADMET Risk Assessment**| Multi-Model In Silico Flags (hERG, CYP, Solubility, PAINS) | Identifies potential pharmacokinetic and toxicity liabilities |

---

## 6. Phased Implementation Roadmap (Semesters 5 & 6)

### **Phase 1 (Sem 5, W1–4): Data Curation & Baseline Feasibility Gate**
* Curate ChEMBL 37 dataset for EGFR assays (segregating WT vs. T790M vs. C797S measurements).
* Execute Bemis-Murcko scaffold splitting and report baseline data audit (active/inactive ratio, unique scaffolds).
* Download and clean verified PDB structures (`6LUB`, `7ZYP`, `4WKQ`).
* Train classical baselines (Morgan Fingerprints + XGBoost/RBF-SVM).

### **Phase 2 (Sem 5, W5–10): 3D Ensemble Docking & Pose Interaction Rescoring**
* Perform cognate redocking validation on `6LUB` and `4WKQ`.
* Run ensemble docking on the top 10,000 2D-screened compounds.
* Extract 3D interaction fingerprints (ProLIF) to evaluate mutant-vs-WT binding selectivity.

### **Phase 3 (Sem 5, W11–16): Controlled Quantum Kernel Benchmark (PennyLane + IBM QPU)**
* Construct the Fidelity Quantum Kernel ($K_{\text{FQ}}$) on compressed molecular feature spaces.
* Run matched classical kernel comparisons (RBF-SVM, Kernel Ridge, Random Fourier Features).
* Execute the $20 \times 20$ verification subset on IBM Quantum hardware with Zero-Noise Extrapolation (ZNE).

### **Phase 4 (Sem 6, W1–6): ADMET Risk Profiling & Retrosynthesis Screening**
* Filter top candidates through RDKit SAScore and multi-model in silico ADMET predictors.
* Conduct retrosynthesis feasibility checks on final candidate scaffolds.
* Select the **Top 3 Computational Candidate Hits**.

### **Phase 5 (Sem 6, W7–12): Final Documentation, Reporting & Competition Submission**
* Complete reproducible code repository with DVC data tracking and MLflow benchmark logs.
* Prepare publication-ready capstone thesis and presentation deck for **Prof. Bhavna Bose** and technical competition judges.
