import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.models.train import run_tier1


def test_run_tier1_writes_metrics_and_models(tiny_feature_parquet, tmp_path):
    models_dir = str(tmp_path / "models")
    artifacts_dir = str(tmp_path / "artifacts")

    results = run_tier1(tiny_feature_parquet, models_dir, artifacts_dir, n_boot=20)

    assert results["seed"] == 42
    assert results["split_type"] == "bemis_murcko_scaffold"
    assert set(results["classifiers"].keys()) == {"xgboost", "random_forest", "rbf_svm"}
    for name, payload in results["classifiers"].items():
        assert "roc_auc" in payload["metrics"]
        assert "y_scramble_control" in payload
        assert os.path.exists(os.path.join(models_dir, f"{name}_classifier.joblib"))

    assert "xgboost" in results["regression"]
    for key in ["rmse", "mae", "r2"]:
        assert key in results["regression"]["xgboost"]

    metrics_path = os.path.join(artifacts_dir, "tier1_metrics.json")
    assert os.path.exists(metrics_path)
