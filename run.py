#!/usr/bin/env python3
"""Pipeline entrypoint: python run.py <stage>

Stages: data | tier1 | dock | quantum | rank | all
Each stage skips gracefully with a notice if its module is not implemented yet.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

FEATURE_PATH = os.path.join(REPO_ROOT, "data/processed/egfr_features.parquet")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")


def run_data(smoke: bool = False):
    print("[data] skipping: src/data pipeline is owned by Kelly; run manually if needed.")


def run_tier1(smoke: bool = False):
    try:
        from src.models.train import run_tier1 as _run_tier1
    except ImportError as e:
        print(f"[tier1] skipping: src/models/train.py not importable ({e})")
        return
    if not os.path.exists(FEATURE_PATH):
        print(f"[tier1] skipping: {FEATURE_PATH} not found.")
        return
    _run_tier1(FEATURE_PATH, MODELS_DIR, ARTIFACTS_DIR, smoke=smoke)


def run_dock(smoke: bool = False):
    try:
        from src.pipeline.dock import run_docking as _run_docking
    except ImportError as e:
        print(f"[dock] skipping: src/pipeline/dock.py not importable ({e})")
        return
    _run_docking(artifacts_dir=ARTIFACTS_DIR, smoke=smoke)


def run_quantum(smoke: bool = False):
    try:
        from src.quantum import run_tier3_benchmark
    except ImportError as e:
        print(f"[quantum] skipping: src/quantum benchmark module not importable ({e})")
        return
    if not os.path.exists(FEATURE_PATH):
        print(f"[quantum] skipping: {FEATURE_PATH} not found.")
        return
    print(f"[quantum] running Tier 3 classical vs NISQ quantum kernel benchmark (smoke={smoke})...")
    run_tier3_benchmark(FEATURE_PATH, ARTIFACTS_DIR, smoke=smoke)


def run_rank(smoke: bool = False):
    try:
        from src.pipeline.rank import run_rank as _run_rank
    except ImportError as e:
        print(f"[rank] skipping: src/pipeline/rank.py not importable ({e})")
        return
    _run_rank(ARTIFACTS_DIR)


STAGES = {
    "data": run_data,
    "tier1": run_tier1,
    "dock": run_dock,
    "quantum": run_quantum,
    "rank": run_rank,
}


def main():
    parser = argparse.ArgumentParser(description="Drug Discovery QML pipeline runner")
    parser.add_argument(
        "stage",
        choices=["data", "tier1", "dock", "quantum", "rank", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Run a small smoke subset for CI"
    )
    args = parser.parse_args()

    if args.stage == "all":
        for name, fn in STAGES.items():
            print(f"\n=== stage: {name} ===")
            fn(smoke=args.smoke)
    else:
        STAGES[args.stage](smoke=args.smoke)


if __name__ == "__main__":
    main()
