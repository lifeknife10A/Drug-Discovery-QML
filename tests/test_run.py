import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run_stage(stage, extra_args=None):
    cmd = [sys.executable, os.path.join(REPO_ROOT, "run.py"), stage]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=240)


def test_data_stage_skips_gracefully():
    result = _run_stage("data")
    assert result.returncode == 0
    assert "[data]" in result.stdout


def test_quantum_stage_does_not_crash():
    # src/quantum is owned by Angela; run.py should either skip gracefully with
    # a [quantum] notice (module absent) or run her benchmark (module present) —
    # either way it must exit 0.
    result = _run_stage("quantum", extra_args=["--smoke"])
    assert result.returncode == 0


def test_invalid_stage_rejected():
    result = _run_stage("not_a_real_stage")
    assert result.returncode != 0
