import json
import os

import pandas as pd
import pytest

from src.pipeline import dock
from src.pipeline.dock import (
    _select_screen_candidates,
    compute_grid_box,
    find_ligand_resname,
    parse_pdb_records,
    run_docking,
    run_ensemble_screen,
    split_receptor_ligand,
)

SYNTHETIC_PDB = """\
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00  0.00           C
HETATM    3  O   HOH A 101      20.000  20.000  20.000  1.00  0.00           O
HETATM    4  C1  LIG A 200       1.000   2.000   3.000  1.00  0.00           C
HETATM    5  C2  LIG A 200       3.000   4.000   5.000  1.00  0.00           C
END
"""


def _write_synthetic_pdb(tmp_path):
    pdb_path = tmp_path / "TEST.pdb"
    pdb_path.write_text(SYNTHETIC_PDB)
    return str(pdb_path)


def test_parse_pdb_records(tmp_path):
    pdb_path = _write_synthetic_pdb(tmp_path)
    atoms, hets = parse_pdb_records(pdb_path)
    assert len(atoms) == 2
    assert len(hets) == 3


def test_find_ligand_resname_ignores_water(tmp_path):
    pdb_path = _write_synthetic_pdb(tmp_path)
    _, hets = parse_pdb_records(pdb_path)
    assert find_ligand_resname(hets) == "LIG"


def test_split_receptor_ligand(tmp_path):
    pdb_path = _write_synthetic_pdb(tmp_path)
    out_dir = tmp_path / "prep"
    result = split_receptor_ligand(pdb_path, str(out_dir), "TEST")
    assert result["ligand_resname"] == "LIG"
    assert result["n_ligand_atoms"] == 2
    assert os.path.exists(result["receptor_path"])
    assert os.path.exists(result["ligand_path"])
    with open(result["receptor_path"]) as f:
        receptor_text = f.read()
    assert "HETATM" not in receptor_text


def test_compute_grid_box():
    coords = [(0.0, 0.0, 0.0), (2.0, 4.0, 6.0)]
    box = compute_grid_box(coords, padding=8.0)
    assert box["center"] == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert box["size"]["x"] == 2.0 + 16.0
    assert box["padding"] == 8.0


def test_compute_grid_box_empty():
    assert compute_grid_box([]) is None


def test_run_docking_skips_gracefully_without_vina(tmp_path, monkeypatch):
    monkeypatch.setattr(dock, "_resolve_vina_python", lambda: None)
    pdb_dir = tmp_path / "pdb"
    pdb_dir.mkdir()
    artifacts_dir = tmp_path / "artifacts"

    report = run_docking(pdb_dir=str(pdb_dir), artifacts_dir=str(artifacts_dir))

    assert report["status"] == "prepped_only"
    report_path = artifacts_dir / "tier2_docking_prep.json"
    assert report_path.exists()
    with open(report_path) as f:
        saved = json.load(f)
    assert set(saved["targets"].keys()) == {"6LUB", "7ZYP", "4WKQ"}
    for target in saved["targets"].values():
        assert target["status"] == "missing_pdb"


def test_resolve_vina_python_missing_path(monkeypatch):
    monkeypatch.setenv("VINA_DOCKING_PYTHON", "/not/a/real/interpreter")
    assert dock._resolve_vina_python() is None


def test_run_ensemble_screen_writes_docking_scores_and_flips_affinity_sign(tmp_path, monkeypatch):
    # Fake worker: 6LUB/7ZYP affinities for two candidates, no real vina/network involved.
    fake_results = [
        {"molecule_chembl_id": "CHEMBL1", "tag": "6LUB", "status": "ok", "best_affinity_kcal_mol": -9.0},
        {"molecule_chembl_id": "CHEMBL1", "tag": "7ZYP", "status": "ok", "best_affinity_kcal_mol": -7.0},
        {"molecule_chembl_id": "CHEMBL2", "tag": "6LUB", "status": "ok", "best_affinity_kcal_mol": -5.0},
        {"molecule_chembl_id": "CHEMBL2", "tag": "7ZYP", "status": "invalid_smiles"},
    ]
    monkeypatch.setattr(dock, "_run_worker", lambda mode, jobs, cache_dir, vina_python: fake_results)

    prep = {
        "6LUB": {"status": "prepped", "grid_box": {"center": {"x": 0, "y": 0, "z": 0}, "size": {"x": 1, "y": 1, "z": 1}}},
        "7ZYP": {"status": "prepped", "grid_box": {"center": {"x": 0, "y": 0, "z": 0}, "size": {"x": 1, "y": 1, "z": 1}}},
    }
    receptor_reports = {
        "6LUB": {"status": "ok", "receptor_pdbqt": "6lub.pdbqt"},
        "7ZYP": {"status": "ok", "receptor_pdbqt": "7zyp.pdbqt"},
    }
    candidates = [
        {"molecule_chembl_id": "CHEMBL1", "canonical_smiles": "CCO"},
        {"molecule_chembl_id": "CHEMBL2", "canonical_smiles": "CCN"},
    ]
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    pivot, n_scored, n_jobs = run_ensemble_screen(
        prep, receptor_reports, candidates, str(tmp_path), str(artifacts_dir), "unused-python", 8, 3
    )

    assert n_jobs == 4
    assert n_scored == 2
    # CHEMBL1: median(-9.0, -7.0) = -8.0 -> docking_score = 8.0
    # CHEMBL2: only 6LUB scored (-5.0) -> docking_score = 5.0
    chembl1 = pivot.set_index("molecule_chembl_id").loc["CHEMBL1"]
    chembl2 = pivot.set_index("molecule_chembl_id").loc["CHEMBL2"]
    assert chembl1["docking_score"] == pytest.approx(8.0)
    assert chembl2["docking_score"] == pytest.approx(5.0)

    out_path = artifacts_dir / "docking_scores.csv"
    assert out_path.exists()
    on_disk = pd.read_csv(out_path)
    assert set(on_disk["molecule_chembl_id"]) == {"CHEMBL1", "CHEMBL2"}


def test_select_screen_candidates_uses_passed_models_dir_not_repo_root(tiny_feature_parquet, tmp_path):
    # Regression: _select_screen_candidates used to hardcode REPO_ROOT/models and the real
    # egfr_features.parquet instead of honoring caller-supplied paths, so a tmp-dir test setup
    # would load the wrong (2055-feature) production model against a tiny (39-feature) parquet.
    from src.models.train import run_tier1

    models_dir = str(tmp_path / "models")
    artifacts_dir = str(tmp_path / "artifacts")
    run_tier1(tiny_feature_parquet, models_dir, artifacts_dir, n_boot=20)

    candidates = _select_screen_candidates(
        artifacts_dir, top_n=3, models_dir=models_dir, feature_path=tiny_feature_parquet
    )

    assert len(candidates) == 3
    assert "molecule_chembl_id" in candidates[0]
    assert "canonical_smiles" in candidates[0]


def test_run_ensemble_screen_returns_none_with_no_jobs(tmp_path):
    pivot, n_scored, n_jobs = run_ensemble_screen({}, {}, [], str(tmp_path), str(tmp_path), "unused-python", 8, 3)
    assert pivot is None
    assert (n_scored, n_jobs) == (0, 0)


@pytest.mark.skipif(
    os.environ.get("RUN_VINA_TESTS") != "1",
    reason="Real AutoDock Vina docking: needs the vina-docking conda env, local PDB data, "
           "and a network fetch of the ligand's ideal SDF from RCSB. Opt in with RUN_VINA_TESTS=1.",
)
def test_run_docking_real_pipeline_end_to_end():
    report = run_docking(smoke=True)
    assert report["status"] == "docked"
    assert "redocking_validation" in report
    for result in report["redocking_validation"].values():
        assert result["status"] == "ok"
        assert "redocking_rmsd_A" in result
