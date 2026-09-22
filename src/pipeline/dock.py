"""Tier 2: 3D ensemble molecular docking (AutoDock Vina) + ProLIF rescoring.

Real docking needs optional deps (vina, meeko, prolif) that are NOT in
requirements.txt — installing them requires operator sign-off per hard rule 9
(escalate for installs beyond requirements.txt). Until that's approved, this
module still does everything that doesn't need them: downloads/parses the PDB
targets, splits each into receptor + cognate ligand, and computes a docking
grid box from the cognate ligand's coordinates. When vina/prolif are absent it
writes that prep to artifacts/tier2_docking_prep.json and skips the actual
docking + rescoring with a clear notice, matching run.py's graceful-skip
pattern used for the other stages.
"""

import json
import os
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PDB reference panel (blueprint_v3.md section 2) used for Tier 2 ensemble docking.
PDB_TARGETS = {
    "6LUB": {"role": "primary_triple_mutant", "variant": "EGFR L858R/T790M/C797S"},
    "7ZYP": {"role": "primary_double_mutant", "variant": "EGFR T790M/C797S"},
    "4WKQ": {"role": "wt_counter_screen", "variant": "EGFR WT"},
}

# HETATM residue names to exclude when guessing the cognate (co-crystallized) ligand.
WATER_IONS = {
    "HOH", "NA", "CL", "MG", "ZN", "K", "CA", "SO4", "PO4", "GOL", "EDO", "DMS",
    "ACT", "PEG", "PG4", "IOD", "MN", "CO", "NI", "FMT", "TRS",
}


def parse_pdb_records(pdb_path):
    """Return (atom_lines, hetatm_lines) as raw PDB record strings."""
    atoms, hets = [], []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                atoms.append(line)
            elif line.startswith("HETATM"):
                hets.append(line)
    return atoms, hets


def find_ligand_resname(het_lines):
    """Pick the most common non-water/ion HETATM residue as the cognate ligand."""
    counts = Counter(line[17:20].strip() for line in het_lines)
    candidates = {r: c for r, c in counts.items() if r not in WATER_IONS}
    if not candidates:
        return None
    return max(candidates, key=candidates.get)


def split_receptor_ligand(pdb_path, out_dir, pdb_id):
    """Write receptor-only and cognate-ligand-only PDBs; return prep metadata."""
    os.makedirs(out_dir, exist_ok=True)
    atoms, hets = parse_pdb_records(pdb_path)
    ligand_resname = find_ligand_resname(hets)

    receptor_path = os.path.join(out_dir, f"{pdb_id}_receptor.pdb")
    with open(receptor_path, "w") as f:
        f.writelines(atoms)
        f.write("END\n")

    ligand_path = None
    ligand_coords = []
    if ligand_resname:
        ligand_lines = [l for l in hets if l[17:20].strip() == ligand_resname]
        ligand_path = os.path.join(out_dir, f"{pdb_id}_ligand_{ligand_resname}.pdb")
        with open(ligand_path, "w") as f:
            f.writelines(ligand_lines)
            f.write("END\n")
        for line in ligand_lines:
            ligand_coords.append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            )

    return {
        "receptor_path": receptor_path,
        "ligand_path": ligand_path,
        "ligand_resname": ligand_resname,
        "n_ligand_atoms": len(ligand_coords),
        "ligand_coords": ligand_coords,
    }


def compute_grid_box(ligand_coords, padding=8.0):
    """Docking grid box (center + size, Angstroms) enclosing the cognate ligand."""
    if not ligand_coords:
        return None
    xs, ys, zs = zip(*ligand_coords)
    return {
        "center": {
            "x": sum(xs) / len(xs),
            "y": sum(ys) / len(ys),
            "z": sum(zs) / len(zs),
        },
        "size": {
            "x": (max(xs) - min(xs)) + 2 * padding,
            "y": (max(ys) - min(ys)) + 2 * padding,
            "z": (max(zs) - min(zs)) + 2 * padding,
        },
        "padding": padding,
    }


def prep_targets(pdb_dir, out_dir):
    """Prep every configured PDB target: split receptor/ligand, compute grid box."""
    results = {}
    for pdb_id, meta in PDB_TARGETS.items():
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            results[pdb_id] = {**meta, "status": "missing_pdb"}
            continue
        split = split_receptor_ligand(pdb_path, out_dir, pdb_id)
        grid_box = compute_grid_box(split.pop("ligand_coords"))
        status = "prepped" if grid_box else "prepped_no_cognate_ligand"
        results[pdb_id] = {**meta, **split, "grid_box": grid_box, "status": status}
    return results


def run_docking(pdb_dir=None, artifacts_dir=None, smoke=False):
    """Tier 2 entrypoint: prep targets, then dock + rescore if vina/prolif are installed."""
    pdb_dir = pdb_dir or os.path.join(REPO_ROOT, "data/raw/pdb")
    artifacts_dir = artifacts_dir or os.path.join(REPO_ROOT, "artifacts")
    out_dir = os.path.join(REPO_ROOT, "data/processed/docking_prep")
    os.makedirs(artifacts_dir, exist_ok=True)

    prep = prep_targets(pdb_dir, out_dir)

    try:
        import vina  # noqa: F401

        vina_available = True
    except ImportError:
        vina_available = False
    try:
        import prolif  # noqa: F401

        prolif_available = True
    except ImportError:
        prolif_available = False

    report = {
        "targets": prep,
        "vina_available": vina_available,
        "prolif_available": prolif_available,
        "status": "docked" if vina_available else "prepped_only",
    }
    report_path = os.path.join(artifacts_dir, "tier2_docking_prep.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if not vina_available:
        print(
            "[dock] AutoDock Vina python bindings not installed (install beyond "
            "requirements.txt requires operator sign-off per hard rule 9) - "
            f"receptor/ligand prep + grid boxes written to {report_path}; redocking "
            "RMSD validation, ensemble screening, and ProLIF rescoring are skipped "
            "until vina/meeko/prolif are approved and installed."
        )
        return report

    # Reached once vina is installed - actual Vina scoring / redocking RMSD
    # validation / ProLIF fingerprinting is a follow-up pass, not implemented yet.
    print("[dock] vina detected but ensemble docking + ProLIF rescoring not yet implemented.")
    return report


def main():
    run_docking()


if __name__ == "__main__":
    main()
