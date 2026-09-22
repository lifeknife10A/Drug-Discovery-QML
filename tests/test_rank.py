import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.models.train import run_tier1
from src.pipeline.rank import run_rank


def test_run_rank_produces_top_candidates(tiny_feature_parquet, tmp_path):
    models_dir = str(tmp_path / "models")
    artifacts_dir = str(tmp_path / "artifacts")

    run_tier1(tiny_feature_parquet, models_dir, artifacts_dir, n_boot=20)
    ranking = run_rank(artifacts_dir, models_dir=models_dir, feature_path=tiny_feature_parquet, top_n=3)

    assert "dss_score" in ranking.columns
    assert "rank" in ranking.columns
    assert list(ranking["rank"]) == sorted(ranking["rank"])
    assert ranking["dss_score"].is_monotonic_decreasing

    out_path = os.path.join(artifacts_dir, "top_candidates.csv")
    assert os.path.exists(out_path)


def test_run_rank_requires_tier1_metrics(tmp_path):
    artifacts_dir = str(tmp_path / "empty_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        run_rank(artifacts_dir, models_dir=str(tmp_path / "models"), feature_path="unused.parquet")
