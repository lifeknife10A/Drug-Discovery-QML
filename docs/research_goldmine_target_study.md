# Research Study: Target Selection & Disease Vector Analysis

# "Hitting the Goldmine" in Computational & Quantum Drug Discovery

> **Research & Strategic Positioning Document**  
> **Course:** Design Experience (Semesters 5 & 6)  
> **Institution:** MPSTME, NMIMS University  
> **Faculty Mentor:** Prof. Bhavna Bose  
> **Authors:** Krish & Team  

---

## 1. Executive Summary & Research Strategy

To achieve commercial impact, technical competition awards, and paper publication, we must avoid generic molecular screening. Instead, we must target a **specific, high-value disease vector** where:
1. **Gold-standard FDA-approved precedent drugs exist** (Drug A, Drug B).
2. **A clear, unsolved clinical gap / resistance mutation exists** that current drugs fail to address.
3. **Sub-atomic quantum interactions** (hydrogen bonding, covalent-to-noncovalent shifts, allosteric pocket dynamics) dictate binding affinity, giving Quantum Machine Learning (QML/VQE) a genuine mathematical edge over classical ML.
4. **Rich open datasets** (ChEMBL, PubChem, PDB) exist with thousands of tested compounds and bioactivity logs.

---

## 2. Deep Dive: Top 3 "Goldmine" Disease Vectors

---

### 🎯 VECTOR 1: ONCOLOGY — Overcoming 4th-Generation EGFR Resistance in Non-Small Cell Lung Cancer (NSCLC)

#### **A. Precedent FDA-Approved Drugs (Drug A, B, C)**
* **1st Generation:** *Gefitinib* (Iressa) & *Erlotinib* (Tarceva) — targeted wild-type EGFR.
* **2nd Generation:** *Afatinib* (Gilotrif) — irreversible covalent binding.
* **3rd Generation (Current Gold Standard):** *Osimertinib* (Tagrisso) — specifically engineered to target the **T790M resistance mutation** in lung cancer patients.

#### **B. The Unsolved Clinical Bottleneck ("The Goldmine")**
* Patients treated with Osimertinib inevitably develop the **C797S point mutation** in the ATP-binding pocket of EGFR.
* The C797S mutation replaces Cysteine with Serine, destroying the covalent bond Osimertinib relies on and causing total treatment failure.
* **The Clinical Goal:** Discover novel **4th-generation EGFR inhibitors** that can potently inhibit **both T790M and C797S mutations** while sparing wild-type EGFR (to prevent severe skin/gut toxicity).

#### **C. Why Quantum ML Wins Here**
* The shift from covalent binding (Cysteine) to non-covalent sub-atomic interactions (Serine) involves delicate sub-atomic electron cloud rearrangements and hydrogen bonding networks.
* Classical 2D/3D graph models struggle to distinguish subtle binding energy shifts caused by single-atom amino acid mutations (Cysteine vs. Serine). **Quantum Variational Circuits (VQE/QML)** excel at modeling these exact sub-atomic electronic state changes.

#### **D. Data Availability**
* **ChEMBL Targets:** `CHEMBL203` (EGFR wild-type), `CHEMBL3038469` (EGFR T790M mutant), `CHEMBL4523143` (EGFR T790M/C797S double mutant).
* **Data Volume:** 15,000+ compounds with logged $IC_{50}$ values and SMILES structures.

---

### 🎯 VECTOR 2: ONCOLOGY — Targeting "Undruggable" KRAS G12D/G12C Mutations in Pancreatic & Colorectal Cancer

#### **A. Precedent FDA-Approved Drugs**
* **Sotorasib** (Lumakras) & **Adagrasib** (Krazati) — first FDA-approved covalent inhibitors targeting the **KRAS G12C** mutation in lung and colorectal cancers.

#### **B. The Unsolved Clinical Bottleneck ("The Goldmine")**
* KRAS was considered "undruggable" for 40 years due to its smooth surface without deep binding pockets.
* While Sotorasib works for KRAS G12C, it is useless against **KRAS G12D** (the predominant mutation driving **90% of Pancreatic Cancers**, one of the deadliest cancers with <10% 5-year survival).
* Furthermore, patients rapidly acquire secondary resistance mutations (like **KRAS Y96D**).

#### **C. Why Quantum ML Wins Here**
* Inhibiting KRAS G12D requires discovering **allosteric (switch-I / switch-II pocket) inhibitors** that stabilize the inactive GDP-bound state through weak, non-covalent sub-atomic electrostatic forces.
* Quantum state vector simulations can model the subtle conformational energy landscapes of the switch-I/II allosteric loops better than classical molecular dynamics.

#### **D. Data Availability**
* **ChEMBL Target:** `CHEMBL2111394` (KRAS), `CHEMBL4523588` (KRAS G12C), PDB 3D structures (PDB: 6OIM, 7LGI, 7R00).

---

### 🎯 VECTOR 3: NEURODEGENERATIVE — Oral Small-Molecule NLRP3 Inflammasome Inhibitors for Alzheimer's & Parkinson's

#### **A. Precedent Approved Treatments**
* **Lecanemab** (Leqembi) & **Donanemab** — monoclonal antibody IV infusions targeting Beta-Amyloid plaques.

#### **B. The Unsolved Clinical Bottleneck ("The Goldmine")**
* Current antibody treatments require bi-weekly IV hospital infusions, cost $26,000/year, cause brain swelling/bleeding (ARIA side effects), and struggle to cross the Blood-Brain Barrier (BBB).
* **The Clinical Goal:** Discover **oral, small-molecule drugs** that easily cross the Blood-Brain Barrier to selectively inhibit the **NLRP3 inflammasome**, stopping neuro-inflammation, Microglial destruction, and Tau tangle formation.

#### **C. Why Quantum ML Wins Here**
* Small-molecule NLRP3 inhibitors must balance high target binding affinity with strict Blood-Brain Barrier (BBB) permeability parameters (polar surface area, charge distribution, hydrogen bond donors).
* Quantum feature embedding maps non-linear electronic properties (like dipole moments and frontier orbital energies HOMO/LUMO) directly into Hilbert space to optimize BBB penetration alongside binding potency.

#### **D. Data Availability**
* **ChEMBL Target:** `CHEMBL4523910` (NLRP3 Inflammasome), 3,500+ compounds with tested $IC_{50}$ values.

---

## 3. Comparison Matrix for Disease Vector Selection

| Feature | Vector 1: EGFR 4th-Gen (Lung Cancer) | Vector 2: KRAS G12D (Pancreatic Cancer) | Vector 3: NLRP3 (Alzheimer's/Parkinson's) |
| :--- | :--- | :--- | :--- |
| **Precedent Drugs** | Gefitinib, Erlotinib, Osimertinib | Sotorasib, Adagrasib | Lecanemab (Antibody precedent) |
| **Target Protein** | EGFR (Wild-type, T790M, C797S) | KRAS (G12C, G12D, Y96D) | NLRP3 Inflammasome / Microglia |
| **Unsolved Clinical Gap** | C797S double mutation causes complete resistance to Osimertinib | G12D mutation in Pancreatic Cancer has NO approved targeted drug | Need oral small-molecule crossing Blood-Brain Barrier |
| **Quantum Mechanism** | Modeling covalent-to-noncovalent binding energy shift (Serine mutation) | Modeling switch-I/II allosteric pocket electronic conformations | Mapping HOMO/LUMO dipole moments for BBB + binding balance |
| **Data Volume in ChEMBL**| **15,000+ compounds** (Highest data density) | **5,000+ compounds** + high-res 3D PDB structures | **3,500+ compounds** + BBB permeability parameters |
| **Hackathon & Pitch Appeal**| ⭐⭐⭐⭐⭐ (Massive commercial impact & clear drug progression story) | ⭐⭐⭐⭐⭐ (Solves deadliest pancreatic cancer target) | ⭐⭐⭐⭐ (High MedTech interest) |

---

## 4. Recommendation for Team Alignment

**Recommended Selection: Vector 1 (EGFR 4th-Gen Mutant Inhibitors)**

### Why Vector 1 is the Ultimate "Goldmine":
1. **Clear Precedent Story for Faculty & Hackathon Judges:**
   * "Drug A (Gefitinib) worked initially -> Cancer mutated -> Drug B (Osimertinib) solved 3rd-gen T790M mutation -> Cancer mutated again (C797S) -> **Our Hybrid Quantum Platform discovers 4th-Gen Dual-Mutant Inhibitors.**"
2. **Abundant Data:** ChEMBL has over 15,000+ compounds with exact $IC_{50}$ numbers across Wild-Type, T790M, and C797S mutants.
3. **Demonstrable Quantum Edge:** Predicting how a single Cysteine-to-Serine amino acid mutation disrupts drug binding requires precise sub-atomic quantum modeling that classical 2D ML cannot solve.
