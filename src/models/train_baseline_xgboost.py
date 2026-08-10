#!/usr/bin/env python3
"""
Phase 1: Classical Baseline XGBoost Model Training & Benchmarking
Trains XGBoost Regressor and Classifier on RDKit Morgan Fingerprints
"""

import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, f1_score, accuracy_score
import xgboost as xgb

def train_xgboost_models(feature_parquet_path):
    """
    Trains XGBoost Regression (predicting pIC50) and Classification (predicting active/inactive).
    """
    print(f"[+] Loading featurized dataset from {feature_parquet_path}...")
    df = pd.read_parquet(feature_parquet_path)
    
    # Feature columns (descriptors + 2048 Morgan fingerprint bits)
    desc_cols = ["mw", "logp", "hbd", "hba", "tpsa", "rotatable_bonds", "aromatic_rings"]
    fp_cols = [f"fp_{i}" for i in range(2048)]
    feature_cols = desc_cols + fp_cols
    
    X = df[feature_cols].values
    y_reg = df["pIC50"].values
    y_cls = df["is_active"].values
    
    # 80/20 Train/Test Split
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    
    print(f"[+] Dataset Split: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples.")
    
    # 1. XGBoost Regressor (pIC50 Prediction)
    print("\n--- Training XGBoost Regressor (pIC50 Binding Affinity) ---")
    reg_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    reg_model.fit(X_train, y_train_reg)
    
    y_pred_reg = reg_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2 = r2_score(y_test_reg, y_pred_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    
    print(f"    - Regression RMSE : {rmse:.4f}")
    print(f"    - Regression R^2  : {r2:.4f}")
    print(f"    - Regression MAE  : {mae:.4f}")
    
    # 2. XGBoost Classifier (Active vs Inactive)
    print("\n--- Training XGBoost Classifier (Active vs Inactive) ---")
    cls_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    cls_model.fit(X_train, y_train_cls)
    
    y_pred_cls = cls_model.predict(X_test)
    y_prob_cls = cls_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test_cls, y_pred_cls)
    f1 = f1_score(y_test_cls, y_pred_cls)
    roc_auc = roc_auc_score(y_test_cls, y_prob_cls)
    
    print(f"    - Classification Accuracy : {acc:.4f}")
    print(f"    - Classification F1-Score : {f1:.4f}")
    print(f"    - Classification ROC-AUC  : {roc_auc:.4f}")
    
    # Save benchmark metrics summary
    summary_path = os.path.join(os.path.dirname(feature_parquet_path), "baseline_metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== CLASSICAL XGBOOST BENCHMARK RESULTS ===\n")
        f.write(f"Dataset Size: {len(df)} compounds (Target: EGFR / CHEMBL203)\n")
        f.write(f"Regression RMSE : {rmse:.4f}\n")
        f.write(f"Regression R^2  : {r2:.4f}\n")
        f.write(f"Regression MAE  : {mae:.4f}\n")
        f.write(f"Classification Accuracy : {acc:.4f}\n")
        f.write(f"Classification F1-Score : {f1:.4f}\n")
        f.write(f"Classification ROC-AUC  : {roc_auc:.4f}\n")
        
    print(f"\n[✓] Benchmark metrics saved to {summary_path}")

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")
    feature_path = os.path.join(base_dir, "egfr_features.parquet")
    
    if not os.path.exists(feature_path):
        print(f"[!] Feature file {feature_path} not found. Run extract_features.py first.")
        sys.exit(1)
        
    train_xgboost_models(feature_path)

if __name__ == "__main__":
    main()
