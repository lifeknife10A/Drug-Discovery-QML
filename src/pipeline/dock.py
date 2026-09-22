"""Tier 2: 3D ensemble molecular docking (AutoDock Vina) + redocking validation.

AutoDock Vina's Python bindings only ship wheels/conda builds up to Python
3.10, and pulling vina + meeko + ProLIF into the project's main environment
(which runs 3.13, per requirements.txt) isn't possible without a downgrade.
Per hard rule 9 (escalate for installs beyond requirements.txt), those tools
live in a separate, operator-approved conda env ("vina-docking", Python 3.10)
instead of the main one. This module always does the dependency-free prep
(download/parse the PDB targets, split receptor + cognate ligand, compute a
grid box) directly. For the docking itself it shells out to
src/pipeline/_vina_worker.py running under that dedicated env's interpreter:

  - Receptor prep: meeko's `mk_prepare_receptor.py` CLI (PDB -> PDBQT).
  - Redocking validation: re-dock each cognate ligand into its own pocket
    from a fresh conformer and check RMSD < 2.0 A against the crystal pose
    (blueprint section 3, Tier 2).
  - Ensemble screening: dock a bounded top-N slice of Tier 1's ranked
    candidates against each mutant target (6LUB, 7ZYP), writing
    artifacts/docking_scores.csv for src/pipeline/rank.py's DSS fusion.

If the vina-docking env isn't present (e.g. a grader's machine, or CI without
it set up), everything gracefully degrades to the prep-only behavior this
module always had — no crash, just a clear notice.

ProLIF interaction-fingerprint rescoring is intentionally NOT implemented
yet: it needs MDAnalysis Universe construction from the docked poses with
residue-name alignment against the receptor, which is a distinct follow-up
pass, not a small addition to this one.
"""

import ast
import json
import os
import subprocess
import tempfile
from collections import Counter

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vina_worker.py")

# Dedicated docking-tooling interpreter (see module docstring). Override with
# VINA_DOCKING_PYTHON if the env lives somewhere else on your machine.
DEFAULT_VINA_PYTHON = "/opt/anaconda3/envs/vina-docking/bin/python"

# PDB reference panel (blueprint_v3.md section 2) used for Tier 2 ensemble docking.
PDB_TARGETS = {
    "6LUB": {"role": "primary_triple_mutant", "variant": "EGFR L858R/T790M/C797S"},
    "7ZYP": {"role": "primary_double_mutant", "variant": "EGFR T790M/C797S"},
    "4WKQ": {"role": "wt_counter_screen", "variant": "EGFR WT"},
}
MUTANT_TARGETS = [pdb_id for pdb_id, meta in PDB_TARGETS.items() if meta["role"].startswith("primary_")]

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


def _resolve_vina_python():
    candidate = os.environ.get("VINA_DOCKING_PYTHON", DEFAULT_VINA_PYTHON)
    if not os.path.exists(candidate):
        return None
    check = subprocess.run(
        [candidate, "-c", "import vina, meeko, rdkit"],
        capture_output=True, text=True, timeout=30,
    )
    return candidate if check.returncode == 0 else None


def _prepare_receptor_pdbqt(pdb_id, target, out_dir, vina_python):
    """PDB -> PDBQT via meeko's mk_prepare_receptor.py. Residues that don't
    match a template (usually unresolved side-chain atoms in the crystal
    structure) are dropped rather than blocking the whole target — logged
    here so it's visible, not silently lossy."""
    mk_prepare_receptor = os.path.join(os.path.dirname(vina_python), "mk_prepare_receptor.py")
    out_base = os.path.join(out_dir, f"{pdb_id}_receptor")
    box = target["grid_box"]
    cmd = [
        vina_python, mk_prepare_receptor,
        "--read_pdb", target["receptor_path"],
        "-o", out_base, "-p",
        "--box_center", str(box["center"]["x"]), str(box["center"]["y"]), str(box["center"]["z"]),
        "--box_size", str(box["size"]["x"]), str(box["size"]["y"]), str(box["size"]["z"]),
        "--default_altloc", "A", "-x", "--forgive_extra_bonds",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    pdbqt_path = out_base + ".pdbqt"
    dropped = []
    for line in (result.stdout + result.stderr).splitlines():
        if "Template matching failed for:" not in line:
            continue
        raw_list = line.split("Template matching failed for:", 1)[1].split("Ignored", 1)[0].strip()
        try:
            dropped.extend(ast.literal_eval(raw_list))
        except (ValueError, SyntaxError):
            dropped.append(raw_list)  # unexpected format; keep the raw text rather than lose it
    if result.returncode != 0 or not os.path.exists(pdbqt_path):
        return {"status": "receptor_prep_failed", "stderr": result.stderr[-2000:]}
    return {"status": "ok", "receptor_pdbqt": pdbqt_path, "dropped_residues": dropped}


def _run_worker(mode, jobs, cache_dir, vina_python):
    if not jobs:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        job_path = os.path.join(tmp, "job.json")
        output_path = os.path.join(tmp, "result.json")
        with open(job_path, "w") as f:
            json.dump({"mode": mode, "jobs": jobs, "cache_dir": cache_dir, "output_path": output_path}, f)
        result = subprocess.run(
            [vina_python, WORKER_PATH, job_path],
            capture_output=True, text=True, timeout=3600,
        )
        if not os.path.exists(output_path):
            raise RuntimeError(f"vina worker ({mode}) produced no output.\nstderr: {result.stderr[-2000:]}")
        with open(output_path) as f:
            return json.load(f)


def _select_screen_candidates(artifacts_dir, top_n, models_dir=None, feature_path=None):
    """Top-N candidates by Tier 1 activity probability — docking a bounded,
    already-prioritized slice instead of the full library keeps ensemble
    screening runtime sane (each dock is tens of seconds)."""
    from src.pipeline.rank import _load_tier1_scores  # local import: avoid a hard dependency for prep-only runs

    models_dir = models_dir or os.path.join(REPO_ROOT, "models")
    feature_path = feature_path or os.path.join(REPO_ROOT, "data/processed/egfr_features.parquet")
    try:
        scored = _load_tier1_scores(artifacts_dir, models_dir, feature_path)
    except FileNotFoundError:
        return []
    scored = scored.sort_values("tier1_activity_raw", ascending=False).head(top_n)
    return scored[["molecule_chembl_id", "canonical_smiles"]].to_dict("records")


def run_redocking_validation(prep, out_dir, vina_python, exhaustiveness, n_poses):
    receptor_reports, jobs = {}, []
    for pdb_id, target in prep.items():
        if target["status"] != "prepped":
            continue
        receptor_reports[pdb_id] = _prepare_receptor_pdbqt(pdb_id, target, out_dir, vina_python)
        if receptor_reports[pdb_id]["status"] != "ok":
            continue
        box = target["grid_box"]
        jobs.append({
            "tag": pdb_id,
            "receptor_pdbqt": receptor_reports[pdb_id]["receptor_pdbqt"],
            "center": [box["center"]["x"], box["center"]["y"], box["center"]["z"]],
            "box_size": [box["size"]["x"], box["size"]["y"], box["size"]["z"]],
            "native_pdb": target["ligand_path"],
            "ligand_resname": target["ligand_resname"],
            "exhaustiveness": exhaustiveness,
            "n_poses": n_poses,
            "seed": 42,
        })

    results = _run_worker("redock_validate", jobs, out_dir, vina_python)
    return receptor_reports, {r["tag"]: r for r in results}


def run_ensemble_screen(prep, receptor_reports, candidates, out_dir, artifacts_dir, vina_python, exhaustiveness, n_poses):
    jobs = []
    for candidate in candidates:
        for pdb_id in MUTANT_TARGETS:
            target, receptor = prep.get(pdb_id), receptor_reports.get(pdb_id)
            if not target or target["status"] != "prepped" or not receptor or receptor["status"] != "ok":
                continue
            box = target["grid_box"]
            jobs.append({
                "tag": pdb_id,
                "molecule_chembl_id": candidate["molecule_chembl_id"],
                "smiles": candidate["canonical_smiles"],
                "receptor_pdbqt": receptor["receptor_pdbqt"],
                "center": [box["center"]["x"], box["center"]["y"], box["center"]["z"]],
                "box_size": [box["size"]["x"], box["size"]["y"], box["size"]["z"]],
                "exhaustiveness": exhaustiveness,
                "n_poses": n_poses,
                "seed": 42,
            })

    results = _run_worker("screen", jobs, out_dir, vina_python)
    if not results:
        return None, 0, 0

    rows = pd.DataFrame(results)
    ok = rows[rows["status"] == "ok"].copy()
    if ok.empty:
        return None, 0, len(rows)

    pivot = ok.pivot_table(index="molecule_chembl_id", columns="tag", values="best_affinity_kcal_mol", aggfunc="min")
    pivot.columns = [f"best_affinity_{c}_kcal_mol" for c in pivot.columns]
    mutant_cols = [c for c in pivot.columns if any(t in c for t in MUTANT_TARGETS)]
    # More negative affinity = tighter binding; flip sign so higher docking_score is better,
    # matching the convention rank.py already uses for tier3_quantum / tier4_admet.
    pivot["docking_score"] = -pivot[mutant_cols].median(axis=1, skipna=True)
    pivot = pivot.reset_index()

    out_path = os.path.join(artifacts_dir, "docking_scores.csv")
    pivot.to_csv(out_path, index=False)
    print(f"[dock] Tier 2 ensemble screen: {len(pivot)} candidates scored -> {out_path}")
    return pivot, len(pivot), len(rows)


def run_docking(pdb_dir=None, artifacts_dir=None, models_dir=None, feature_path=None, smoke=False):
    """Tier 2 entrypoint: prep targets, then dock + validate if the vina-docking env is present."""
    pdb_dir = pdb_dir or os.path.join(REPO_ROOT, "data/raw/pdb")
    artifacts_dir = artifacts_dir or os.path.join(REPO_ROOT, "artifacts")
    out_dir = os.path.join(REPO_ROOT, "data/processed/docking_prep")
    os.makedirs(artifacts_dir, exist_ok=True)

    prep = prep_targets(pdb_dir, out_dir)
    vina_python = _resolve_vina_python()

    report = {
        "targets": prep,
        "vina_available": vina_python is not None,
        "prolif_available": False,  # ProLIF rescoring not implemented yet; see module docstring.
        "status": "docked" if vina_python else "prepped_only",
    }
    report_path = os.path.join(artifacts_dir, "tier2_docking_prep.json")

    if vina_python is None:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(
            "[dock] vina-docking conda env not found (looked for "
            f"{os.environ.get('VINA_DOCKING_PYTHON', DEFAULT_VINA_PYTHON)}; install beyond "
            "requirements.txt requires operator sign-off per hard rule 9) - receptor/ligand "
            f"prep + grid boxes written to {report_path}; redocking RMSD validation and "
            "ensemble screening are skipped until that env is set up."
        )
        return report

    exhaustiveness = 4 if smoke else 8
    n_poses = 3 if smoke else 5
    top_n = 2 if smoke else 15

    receptor_reports, redocking = run_redocking_validation(prep, out_dir, vina_python, exhaustiveness, n_poses)
    report["receptor_prep"] = receptor_reports
    report["redocking_validation"] = redocking
    n_targets, n_pass = len(redocking), sum(1 for r in redocking.values() if r.get("rmsd_pass"))
    print(f"[dock] redocking validation: {n_pass}/{n_targets} targets pass RMSD < 2.0 A")

    candidates = _select_screen_candidates(artifacts_dir, top_n, models_dir=models_dir, feature_path=feature_path)
    if candidates:
        _, n_scored, n_jobs = run_ensemble_screen(
            prep, receptor_reports, candidates, out_dir, artifacts_dir, vina_python,
            exhaustiveness, min(n_poses, 3),
        )
        report["screen_n_candidates"] = len(candidates)
        report["screen_n_jobs"] = n_jobs
        report["screen_n_scored"] = n_scored
    else:
        print("[dock] no Tier 1 ranking available yet (run `run.py tier1` first) — skipping ensemble screen.")
        report["screen_n_candidates"] = 0

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    run_docking()


if __name__ == "__main__":
    main()
