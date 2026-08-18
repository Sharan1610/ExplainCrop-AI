"""
ExplainCrop-AI: Model Training & Benchmarking Pipeline
Trains an XGBoost Multi-Class Classifier with SHAP explainability support,
benchmarks against alternative ML algorithms, and exports artifacts.
"""

import sys
import os

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import joblib
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

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Crop_Recommendation.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_preprocess_data(csv_path: str):
    """Loads dataset and standardizes column names."""
    df = pd.read_csv(csv_path)

    # Normalize column names
    col_mapping = {
        "Nitrogen": "N",
        "Phosphorus": "P",
        "Potassium": "K",
        "Temperature": "temperature",
        "Humidity": "humidity",
        "pH_Value": "ph",
        "ph": "ph",
        "Rainfall": "rainfall",
        "Crop": "label",
        "label": "label",
    }
    df = df.rename(columns=col_mapping)
    df["label"] = df["label"].str.strip().str.capitalize()

    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols]
    y_raw = df["label"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    return df, X, y, le, feature_cols


def benchmark_models(X_train, X_test, y_train, y_test):
    """Compares XGBoost with standard ML classifiers."""
    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric="mlogloss",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "SVM (RBF)": SVC(probability=True, random_state=42),
    }

    benchmark_results = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        benchmark_results[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
        }
        print(f"[{name}] Accuracy: {acc * 100:.2f}% | F1-Score: {f1:.4f}")

    return benchmark_results, models["XGBoost"]


def train_and_export():
    """Main training routine."""
    print("=" * 60)
    print("ExplainCrop-AI: Training Pipeline Starting...")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df, X, y, le, feature_cols = load_and_preprocess_data(DATA_PATH)
    print(f"Loaded dataset with {len(df)} samples across {len(le.classes_)} crops:")
    print(", ".join(le.classes_))

    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # Benchmark models
    print("\nBenchmarking ML Models:")
    benchmark_results, best_xgb = benchmark_models(X_train, X_test, y_train, y_test)

    # Detailed metrics for XGBoost
    y_pred_xgb = best_xgb.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_xgb).tolist()

    print("\nInitializing SHAP TreeExplainer...")
    # TreeExplainer with tree_path_dependent feature perturbation
    explainer = shap.TreeExplainer(best_xgb, feature_perturbation="tree_path_dependent")

    # Compute sample SHAP values
    sample_shap = explainer.shap_values(X_test.iloc[:100])
    if isinstance(sample_shap, list):
        global_shap_importance = np.mean(
            [np.abs(cls_shap).mean(axis=0) for cls_shap in sample_shap], axis=0
        )
    elif len(sample_shap.shape) == 3:
        global_shap_importance = np.abs(sample_shap).mean(axis=(0, 2))
    else:
        global_shap_importance = np.abs(sample_shap).mean(axis=0)

    feature_importance_dict = {
        feat: float(imp)
        for feat, imp in zip(feature_cols, global_shap_importance)
    }

    # Save artifacts
    print("\nSaving model artifacts to 'models/'...")
    joblib.dump(best_xgb, os.path.join(MODELS_DIR, "xgboost_crop_model.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    joblib.dump(explainer, os.path.join(MODELS_DIR, "shap_explainer.joblib"))

    # Summary Statistics per crop for agronomic advisories
    crop_profiles = {}
    for crop in le.classes_:
        crop_df = df[df["label"] == crop]
        crop_profiles[crop] = {
            "N": {"mean": float(crop_df["N"].mean()), "std": float(crop_df["N"].std())},
            "P": {"mean": float(crop_df["P"].mean()), "std": float(crop_df["P"].std())},
            "K": {"mean": float(crop_df["K"].mean()), "std": float(crop_df["K"].std())},
            "temperature": {
                "mean": float(crop_df["temperature"].mean()),
                "min": float(crop_df["temperature"].min()),
                "max": float(crop_df["temperature"].max()),
            },
            "humidity": {
                "mean": float(crop_df["humidity"].mean()),
                "min": float(crop_df["humidity"].min()),
                "max": float(crop_df["humidity"].max()),
            },
            "ph": {
                "mean": float(crop_df["ph"].mean()),
                "min": float(crop_df["ph"].min()),
                "max": float(crop_df["ph"].max()),
            },
            "rainfall": {
                "mean": float(crop_df["rainfall"].mean()),
                "min": float(crop_df["rainfall"].min()),
                "max": float(crop_df["rainfall"].max()),
            },
        }

    with open(os.path.join(MODELS_DIR, "crop_profiles.json"), "w") as f:
        json.dump(crop_profiles, f, indent=2)

    metadata = {
        "features": feature_cols,
        "classes": list(le.classes_),
        "num_classes": len(le.classes_),
        "benchmarks": benchmark_results,
        "xgboost_accuracy": benchmark_results["XGBoost"]["accuracy"],
        "confusion_matrix": cm,
        "feature_importances": feature_importance_dict,
    }

    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nTraining & Serialization Complete!")
    print(f"Final XGBoost Accuracy: {benchmark_results['XGBoost']['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    train_and_export()
