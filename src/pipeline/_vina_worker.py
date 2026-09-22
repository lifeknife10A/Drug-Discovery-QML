#!/usr/bin/env python3
"""Tier 2 docking worker — MUST run under the dedicated ``vina-docking`` conda
env (Python 3.10 + AutoDock Vina + meeko + RDKit), not the project's main
environment. vina's Python bindings only ship wheels/conda builds up to
Python 3.10, and installing them (plus meeko/ProLIF) requires operator
sign-off per hard rule 9 (installs beyond requirements.txt) — see
src/pipeline/dock.py for how that env gets created and how this worker is
invoked as a subprocess from the main environment.

Usage: <vina-docking python> _vina_worker.py <job.json>
Reads a job description, writes a JSON list of per-task results to
job["output_path"]. One task failing (bad SMILES, embed failure, etc.)
never aborts the batch — its result just carries status="error".
"""

import json
import sys
import time
import traceback
import urllib.request
from pathlib import Path

from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from vina import Vina

RMSD_PASS_THRESHOLD_A = 2.0


def _prepare_ligand_pdbqt(mol):
    preparator = MoleculePreparation()
    setups = preparator.prepare(mol)
    pdbqt_string, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise RuntimeError(f"meeko ligand prep failed: {err}")
    return pdbqt_string


def _embed(mol, seed):
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        if AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True) != 0:
            raise RuntimeError("RDKit 3D conformer embedding failed")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass  # MMFF failing (e.g. unparameterized atom) is non-fatal; keep the embedded geometry
    return mol


def _dock(receptor_pdbqt, center, box_size, ligand_pdbqt, exhaustiveness, n_poses, seed):
    v = Vina(sf_name="vina", cpu=0, seed=seed, verbosity=0)
    v.set_receptor(receptor_pdbqt)
    v.compute_vina_maps(center=center, box_size=box_size)
    v.set_ligand_from_string(ligand_pdbqt)
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    best_affinity = float(v.energies(n_poses=n_poses)[0][0])
    poses_pdbqt = v.poses(n_poses=1)
    return best_affinity, poses_pdbqt


def _native_ligand_mol(native_pdb_path, ligand_resname, cache_dir):
    """Bond-correct the crystal-pose ligand via the PDB Chemical Component
    Dictionary's ideal SDF (AssignBondOrdersFromTemplate): keeps the observed
    (crystal) 3D coordinates but fixes bond orders/aromaticity, which raw
    PDB HETATM records don't encode."""
    cache_path = Path(cache_dir) / f"{ligand_resname}_ideal.sdf"
    if cache_path.exists():
        sdf_text = cache_path.read_text()
    else:
        url = f"https://files.rcsb.org/ligands/download/{ligand_resname}_ideal.sdf"
        sdf_text = urllib.request.urlopen(url, timeout=30).read().decode()
        cache_path.write_text(sdf_text)

    template = Chem.MolFromMolBlock(sdf_text, sanitize=True)
    if template is None:
        raise RuntimeError(f"could not parse ideal SDF template for {ligand_resname}")

    raw = Chem.MolFromPDBFile(native_pdb_path, sanitize=False, removeHs=False)
    if raw is None:
        raise RuntimeError(f"could not parse native ligand PDB {native_pdb_path}")

    native = AllChem.AssignBondOrdersFromTemplate(template, raw)
    Chem.SanitizeMol(native)
    return native


def run_redock_validate(task, cache_dir):
    """Redocking protocol validation: re-dock the cognate ligand into its own
    crystal pocket from a fresh conformer and measure RMSD back to the
    observed crystal pose. Blueprint threshold: RMSD < 2.0 A."""
    native = _native_ligand_mol(task["native_pdb"], task["ligand_resname"], cache_dir)
    native_noH = Chem.RemoveHs(native)

    redock_mol = _embed(Chem.Mol(native), task.get("seed", 42))
    ligand_pdbqt = _prepare_ligand_pdbqt(redock_mol)

    t0 = time.time()
    best_affinity, poses_pdbqt = _dock(
        task["receptor_pdbqt"], task["center"], task["box_size"], ligand_pdbqt,
        task.get("exhaustiveness", 8), task.get("n_poses", 5), task.get("seed", 42),
    )
    dock_seconds = time.time() - t0

    pdbqt_mol = PDBQTMolecule(poses_pdbqt, skip_typing=True)
    posed_noH = Chem.RemoveHs(RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)[0])
    rmsd = float(rdMolAlign.GetBestRMS(posed_noH, native_noH))

    return {
        "tag": task["tag"],
        "status": "ok",
        "best_affinity_kcal_mol": best_affinity,
        "redocking_rmsd_A": rmsd,
        "rmsd_pass": rmsd < RMSD_PASS_THRESHOLD_A,
        "dock_seconds": round(dock_seconds, 2),
    }


def run_screen(task):
    """Dock one candidate SMILES into one prepped receptor; return best affinity."""
    mol = Chem.MolFromSmiles(task["smiles"])
    if mol is None:
        return {
            "molecule_chembl_id": task["molecule_chembl_id"],
            "tag": task["tag"],
            "status": "invalid_smiles",
        }
    mol = _embed(mol, task.get("seed", 42))
    ligand_pdbqt = _prepare_ligand_pdbqt(mol)
    best_affinity, _ = _dock(
        task["receptor_pdbqt"], task["center"], task["box_size"], ligand_pdbqt,
        task.get("exhaustiveness", 8), task.get("n_poses", 3), task.get("seed", 42),
    )
    return {
        "molecule_chembl_id": task["molecule_chembl_id"],
        "tag": task["tag"],
        "status": "ok",
        "best_affinity_kcal_mol": best_affinity,
    }


def main():
    job_path = sys.argv[1]
    with open(job_path) as f:
        job = json.load(f)

    results = []
    for task in job["jobs"]:
        try:
            if job["mode"] == "redock_validate":
                results.append(run_redock_validate(task, job["cache_dir"]))
            else:
                results.append(run_screen(task))
        except Exception as e:
            fail = {"tag": task.get("tag"), "status": "error", "error": f"{type(e).__name__}: {e}"}
            if "molecule_chembl_id" in task:
                fail["molecule_chembl_id"] = task["molecule_chembl_id"]
            fail["traceback"] = traceback.format_exc(limit=3)
            results.append(fail)

    with open(job["output_path"], "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
