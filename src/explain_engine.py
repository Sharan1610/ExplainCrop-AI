"""
ExplainCrop-AI: Inference & Explainability Engine
Generates top-K crop recommendations, computes local SHAP feature attributions,
and provides natural-language agronomic advice and fertilizer recommendations.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class ExplainCropEngine:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.explainer = None
        self.metadata = {}
        self.crop_profiles = {}
        self.feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        self.feature_labels = {
            "N": "Nitrogen (N)",
            "P": "Phosphorus (P)",
            "K": "Potassium (K)",
            "temperature": "Temperature (°C)",
            "humidity": "Humidity (%)",
            "ph": "Soil pH",
            "rainfall": "Rainfall (mm)",
        }
        self.load_artifacts()

    def load_artifacts(self):
        """Loads serialized model, encoder, explainer and metadata."""
        model_path = os.path.join(MODELS_DIR, "xgboost_crop_model.joblib")
        encoder_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
        explainer_path = os.path.join(MODELS_DIR, "shap_explainer.joblib")
        meta_path = os.path.join(MODELS_DIR, "metadata.json")
        profiles_path = os.path.join(MODELS_DIR, "crop_profiles.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact missing at {model_path}. Please run 'src/model_train.py' first."
            )

        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        self.explainer = joblib.load(explainer_path)

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

        if os.path.exists(profiles_path):
            with open(profiles_path, "r") as f:
                self.crop_profiles = json.load(f)

    def predict_and_explain(
        self,
        N: float,
        P: float,
        K: float,
        temperature: float,
        humidity: float,
        ph: float,
        rainfall: float,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Executes prediction, calculates SHAP contributions for the top recommendation,
        and generates agronomic guidance.
        """
        input_data = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=self.feature_names,
        )

        # Probabilities
        probabilities = self.model.predict_proba(input_data)[0]
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        recommendations = []
        for rank, idx in enumerate(top_indices, start=1):
            crop_name = self.encoder.inverse_transform([idx])[0]
            conf = float(probabilities[idx])
            recommendations.append(
                {
                    "rank": rank,
                    "crop": crop_name,
                    "confidence": round(conf * 100, 2),
                    "probability": round(conf, 4),
                    "class_index": int(idx),
                }
            )

        top_crop = recommendations[0]["crop"]
        top_idx = recommendations[0]["class_index"]

        # Compute SHAP values
        shap_values = self.explainer.shap_values(input_data)

        # Handle various SHAP multi-class output formats
        if isinstance(shap_values, list):
            # list of length num_classes, each element is (1, n_features)
            crop_shap = shap_values[top_idx][0]
        elif len(shap_values.shape) == 3:
            # shape (1, n_features, num_classes) or (1, num_classes, n_features)
            if shap_values.shape[2] == len(self.encoder.classes_):
                crop_shap = shap_values[0, :, top_idx]
            else:
                crop_shap = shap_values[0, top_idx, :]
        else:
            crop_shap = shap_values[0]

        feature_contributions = []
        for feat, val, s_val in zip(self.feature_names, input_data.iloc[0], crop_shap):
            feature_contributions.append(
                {
                    "feature": feat,
                    "label": self.feature_labels[feat],
                    "input_value": float(val),
                    "shap_value": float(s_val),
                    "impact": "positive" if s_val >= 0 else "negative",
                    "abs_impact": abs(float(s_val)),
                }
            )

        # Sort contributions by absolute SHAP impact
        feature_contributions.sort(key=lambda x: x["abs_impact"], reverse=True)

        # Agronomic explanation & Fertilizer recommendations
        explanation_text, advisory_tips = self._generate_agronomic_advisory(
            top_crop, input_data.iloc[0], feature_contributions
        )

        return {
            "top_crop": top_crop,
            "top_confidence": recommendations[0]["confidence"],
            "recommendations": recommendations,
            "feature_contributions": feature_contributions,
            "explanation": explanation_text,
            "advisory": advisory_tips,
            "raw_inputs": {
                "N": N,
                "P": P,
                "K": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            },
        }

    def _generate_agronomic_advisory(
        self, crop: str, inputs: pd.Series, contributions: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Generates natural language breakdown and fertilizer guidance."""
        profile = self.crop_profiles.get(crop, {})

        pos_factors = [c["label"] for c in contributions if c["shap_value"] > 0][:3]
        neg_factors = [c["label"] for c in contributions if c["shap_value"] < 0][:2]

        explanation_lines = [
            f"**{crop}** is recommended as the most suitable crop with optimal soil-climate alignment."
        ]

        if pos_factors:
            explanation_lines.append(
                f"**Key Supporting Factors**: {', '.join(pos_factors)} strongly aligned with the crop's physiological needs."
            )
        if neg_factors:
            explanation_lines.append(
                f"**Minor Limiting Factors**: {', '.join(neg_factors)} slightly deviate from baseline benchmarks but remain viable."
            )

        explanation_text = " ".join(explanation_lines)

        # Advisory tips
        advisories = []

        # Nitrogen advice
        n_val = inputs["N"]
        if profile and "N" in profile:
            opt_n = profile["N"]["mean"]
            if n_val < opt_n - 25:
                advisories.append(
                    {
                        "category": "Fertilizer (Nitrogen)",
                        "status": "Deficient",
                        "recommendation": f"Soil Nitrogen ({n_val:.0f}) is below the {crop} benchmark ({opt_n:.0f}). Apply Urea or organic compost / FYM during pre-sowing and tillering.",
                    }
                )
            elif n_val > opt_n + 35:
                advisories.append(
                    {
                        "category": "Fertilizer (Nitrogen)",
                        "status": "Excess",
                        "recommendation": f"High soil Nitrogen ({n_val:.0f}). Avoid excessive nitrogenous fertilizers to prevent vegetative overgrowth and lodging.",
                    }
                )
            else:
                advisories.append(
                    {
                        "category": "Fertilizer (Nitrogen)",
                        "status": "Optimal",
                        "recommendation": f"Nitrogen level ({n_val:.0f}) is well balanced for {crop}.",
                    }
                )

        # Phosphorus advice
        p_val = inputs["P"]
        if profile and "P" in profile:
            opt_p = profile["P"]["mean"]
            if p_val < opt_p - 20:
                advisories.append(
                    {
                        "category": "Fertilizer (Phosphorus)",
                        "status": "Deficient",
                        "recommendation": f"Phosphorus ({p_val:.0f}) is low. Apply Diammonium Phosphate (DAP) or Single Super Phosphate (SSP) for robust root development.",
                    }
                )

        # Potassium advice
        k_val = inputs["K"]
        if profile and "K" in profile:
            opt_k = profile["K"]["mean"]
            if k_val < opt_k - 20:
                advisories.append(
                    {
                        "category": "Fertilizer (Potassium)",
                        "status": "Deficient",
                        "recommendation": f"Potassium ({k_val:.0f}) is low. Apply Muriate of Potash (MOP) or Potassium Sulfate to enhance disease resistance and grain quality.",
                    }
                )

        # pH Management
        ph_val = inputs["ph"]
        if ph_val < 5.5:
            advisories.append(
                {
                    "category": "Soil pH Correction",
                    "status": "Acidic",
                    "recommendation": f"Soil pH ({ph_val:.1f}) is acidic. Incorporate agricultural lime (Calcium Carbonate) or dolomite at 200-400 kg/acre to neutralize acidity.",
                }
            )
        elif ph_val > 7.8:
            advisories.append(
                {
                    "category": "Soil pH Correction",
                    "status": "Alkaline",
                    "recommendation": f"Soil pH ({ph_val:.1f}) is alkaline. Apply agricultural gypsum and incorporate green manure / sulfur to lower alkalinity.",
                }
            )
        else:
            advisories.append(
                {
                    "category": "Soil pH",
                    "status": "Optimal",
                    "recommendation": f"Soil pH ({ph_val:.1f}) is within the optimal neutral range (5.5 - 7.8).",
                }
            )

        # Water & Irrigation
        rain_val = inputs["rainfall"]
        if profile and "rainfall" in profile:
            opt_rain = profile["rainfall"]["mean"]
            if rain_val < opt_rain - 50:
                advisories.append(
                    {
                        "category": "Irrigation Management",
                        "status": "Supplementary Needed",
                        "recommendation": f"Rainfall estimate ({rain_val:.0f} mm) is below optimal ({opt_rain:.0f} mm). Schedule drip or furrow irrigation during critical growth stages.",
                    }
                )

        return explanation_text, advisories


# Singleton instance helper
_engine_instance = None


def get_engine() -> ExplainCropEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ExplainCropEngine()
    return _engine_instance


def predict_and_explain(
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
    top_k: int = 3,
) -> Dict[str, Any]:
    engine = get_engine()
    return engine.predict_and_explain(
        N, P, K, temperature, humidity, ph, rainfall, top_k=top_k
    )
