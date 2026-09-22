import json
import os

from src.pipeline.dock import (
    compute_grid_box,
    find_ligand_resname,
    parse_pdb_records,
    run_docking,
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


def test_run_docking_skips_gracefully_without_vina(tmp_path):
    pdb_dir = tmp_path / "pdb"
    pdb_dir.mkdir()
    artifacts_dir = tmp_path / "artifacts"

    report = run_docking(pdb_dir=str(pdb_dir), artifacts_dir=str(artifacts_dir))

    assert report["status"] in ("prepped_only", "docked")
    report_path = artifacts_dir / "tier2_docking_prep.json"
    assert report_path.exists()
    with open(report_path) as f:
        saved = json.load(f)
    assert set(saved["targets"].keys()) == {"6LUB", "7ZYP", "4WKQ"}
    for target in saved["targets"].values():
        assert target["status"] == "missing_pdb"
