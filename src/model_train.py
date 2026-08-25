"""
OpticCrop: Multi-Modal Model Training & Feature Engineering Pipeline
Implements mathematical feature formulations (R_NP, R_NK, R_PK, THI, MAI),
trains regularized XGBoost with Hist tree method per TRD v1.0.0,
and precomputes k-Means (k=100) background for sub-150ms TreeSHAP attribution.
Eliminates CSV dependencies - loads directly from SQLite / Parquet.
"""

import sys
import os

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import joblib
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
import shap

from src.db import load_dataset_from_db

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

EPSILON = 1e-5


def synthesize_features(df: pd.DataFrame, mu_rain: float = None, sigma_rain: float = None) -> Tuple[pd.DataFrame, float, float]:
    """
    Transforms raw 7-dimensional soil-meteorological metrics into a dense
    multi-modal feature representation X in R^12 per TRD Section 3.2:
      - R_NP = N / (P + epsilon)
      - R_NK = N / (K + epsilon)
      - R_PK = P / (K + epsilon)
      - THI = 0.8 * T + (RH / 100) * (T - 14.4) + 46.4
      - MAI = (Rainfall - mu_hist) / sigma_hist
    """
    df_feat = df.copy()

    # Raw features
    N = df_feat["nitrogen"].astype(float)
    P = df_feat["phosphorus"].astype(float)
    K = df_feat["potassium"].astype(float)
    T = df_feat["temperature"].astype(float)
    RH = df_feat["humidity"].astype(float)
    pH = df_feat["ph"].astype(float)
    Rain = df_feat["rainfall"].astype(float)

    # 1. Soil Nutrient Ratios
    df_feat["r_np"] = N / (P + EPSILON)
    df_feat["r_nk"] = N / (K + EPSILON)
    df_feat["r_pk"] = P / (K + EPSILON)

    # 2. Temperature-Humidity Index (THI)
    df_feat["thi"] = 0.8 * T + (RH / 100.0) * (T - 14.4) + 46.4

    # 3. Moisture Availability Index (MAI)
    if mu_rain is None:
        mu_rain = float(Rain.mean())
    if sigma_rain is None:
        sigma_rain = float(Rain.std()) if float(Rain.std()) > 0 else 1.0

    df_feat["mai"] = (Rain - mu_rain) / (sigma_rain + EPSILON)

    feature_cols = [
        "nitrogen",
        "phosphorus",
        "potassium",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
        "r_np",
        "r_nk",
        "r_pk",
        "thi",
        "mai",
    ]

    return df_feat[feature_cols], mu_rain, sigma_rain, feature_cols


def benchmark_models(X_train, X_test, y_train, y_test):
    """Benchmarks XGBoost against standard ML algorithms."""
    models = {
        "XGBoost (Hist Tree)": xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            objective="multi:softprob",
            random_state=42,
            eval_metric="mlogloss",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=12, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "SVM (RBF)": SVC(probability=True, random_state=42),
    }

    benchmark_results = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        benchmark_results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
        }
        print(f"[{name:20s}] Accuracy: {acc * 100:6.2f}% | F1-Score: {f1:.4f}")

    return benchmark_results, models["XGBoost (Hist Tree)"]


def train_and_export():
    """Executes full training pipeline and serializes production artifacts."""
    print("=" * 70)
    print("OpticCrop: Multi-Modal XGBoost & TreeSHAP Pipeline Starting...")
    print("=" * 70)

    # 1. Load data from SQLite/Parquet (Zero CSV)
    raw_df = load_dataset_from_db()
    print(f"Loaded {len(raw_df)} agricultural records from SQLite/Parquet database.")

    # 2. Encode Labels
    le = LabelEncoder()
    y = le.fit_transform(raw_df["crop"])
    num_classes = len(le.classes_)
    print(f"Registered {num_classes} crop species: {', '.join(le.classes_)}")

    # 3. Feature Synthesis
    X, mu_rain, sigma_rain, feature_cols = synthesize_features(raw_df)
    print(f"\nEngineered {len(feature_cols)} multimodal features: {feature_cols}")
    print(f"Historical Rainfall Baseline: Mean = {mu_rain:.2f} mm | Std = {sigma_rain:.2f} mm")

    # 4. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 5. Benchmarking & Training
    print("\nEvaluating Model Architectures:")
    benchmarks, best_xgb = benchmark_models(X_train, X_test, y_train, y_test)

    # 6. Evaluation on test set
    y_pred_xgb = best_xgb.predict(X_test)
    final_acc = accuracy_score(y_test, y_pred_xgb)
    final_f1 = f1_score(y_test, y_pred_xgb, average="macro")
    cm = confusion_matrix(y_test, y_pred_xgb).tolist()

    print(f"\nModel KPI Verification:")
    print(f"  • Validation Accuracy: {final_acc * 100:.2f}% (TRD Target: >= 92%)")
    print(f"  • Macro F1-Score:      {final_f1:.4f} (TRD Target: >= 0.89)")

    # 7. Compute k-Means background cluster (k=100) per TRD Section 3.3
    print("\nComputing k-Means background cluster (k=100) for sub-150ms TreeSHAP...")
    from sklearn.cluster import KMeans
    kmeans_model = KMeans(n_clusters=100, random_state=42, n_init=10)
    kmeans_model.fit(X_train)
    background_centroids = pd.DataFrame(kmeans_model.cluster_centers_, columns=feature_cols)

    # C++ accelerated TreeExplainer
    explainer = shap.TreeExplainer(
        best_xgb,
        feature_perturbation="tree_path_dependent"
    )

    # Global feature importances
    sample_shap = explainer.shap_values(X_test.iloc[:80])
    if isinstance(sample_shap, list):
        global_shap_importance = np.mean([np.abs(c).mean(axis=0) for c in sample_shap], axis=0)
    elif len(sample_shap.shape) == 3:
        global_shap_importance = np.abs(sample_shap).mean(axis=(0, 2))
    else:
        global_shap_importance = np.abs(sample_shap).mean(axis=0)

    feature_importances = {
        feat: float(imp) for feat, imp in zip(feature_cols, global_shap_importance)
    }

    # 8. Physiological crop profiles for Agronomic Advisories
    crop_profiles = {}
    for crop in le.classes_:
        cdf = raw_df[raw_df["crop"] == crop]
        crop_profiles[crop] = {
            "nitrogen": {"mean": float(cdf["nitrogen"].mean()), "std": float(cdf["nitrogen"].std())},
            "phosphorus": {"mean": float(cdf["phosphorus"].mean()), "std": float(cdf["phosphorus"].std())},
            "potassium": {"mean": float(cdf["potassium"].mean()), "std": float(cdf["potassium"].std())},
            "temperature": {
                "mean": float(cdf["temperature"].mean()),
                "min": float(cdf["temperature"].min()),
                "max": float(cdf["temperature"].max()),
            },
            "humidity": {
                "mean": float(cdf["humidity"].mean()),
                "min": float(cdf["humidity"].min()),
                "max": float(cdf["humidity"].max()),
            },
            "ph": {
                "mean": float(cdf["ph"].mean()),
                "min": float(cdf["ph"].min()),
                "max": float(cdf["ph"].max()),
            },
            "rainfall": {
                "mean": float(cdf["rainfall"].mean()),
                "min": float(cdf["rainfall"].min()),
                "max": float(cdf["rainfall"].max()),
            },
        }

    # 9. Save production artifacts
    print("\nSerializing production ML artifacts to 'models/'...")
    joblib.dump(best_xgb, os.path.join(MODELS_DIR, "xgboost_crop_model.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    joblib.dump(explainer, os.path.join(MODELS_DIR, "shap_explainer.joblib"))
    joblib.dump(background_centroids, os.path.join(MODELS_DIR, "shap_background.joblib"))

    with open(os.path.join(MODELS_DIR, "crop_profiles.json"), "w") as f:
        json.dump(crop_profiles, f, indent=2)

    metadata = {
        "version": "1.0.0",
        "features": feature_cols,
        "classes": list(le.classes_),
        "num_classes": len(le.classes_),
        "mu_rain": mu_rain,
        "sigma_rain": sigma_rain,
        "benchmarks": benchmarks,
        "accuracy": final_acc,
        "macro_f1": final_f1,
        "confusion_matrix": cm,
        "feature_importances": feature_importances,
    }

    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 70)
    print(f"Training Complete! XGBoost Accuracy: {final_acc * 100:.2f}% | Macro F1: {final_f1:.4f}")
    print("All artifacts successfully saved to 'models/' directory.")
    print("=" * 70)


if __name__ == "__main__":
    train_and_export()
