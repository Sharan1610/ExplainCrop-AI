# 🌾 CropMind AI: Climate-Resilient Crop Recommendation System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost 2.0+](https://img.shields.io/badge/Model-XGBoost%20Hist-EB6440?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![TreeSHAP](https://img.shields.io/badge/Explainability-TreeSHAP%20XAI-4F46E5?style=for-the-badge)](https://shap.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI%20REST-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/Database-SQLite%20%2B%20Parquet-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

> **CropMind AI (PRD & TRD v1.0.0)** is an Explainable Multi-Modal Precision Agriculture platform that fuses static edaphic soil sensor profiles with dynamic real-time meteorological forecast streams. Powered by **XGBoost (Hist Tree)**, **TreeSHAP local factor attribution**, **Zero-CSV SQLite relational storage**, and **FastAPI / Streamlit** user interfaces.

---

## 🌟 Core Feature Scope

- 🎯 **Multi-Modal Feature Synthesis ($X \in \mathbb{R}^{12}$)**:
  - Soil Nutrient Ratios: $R_{NP} = \frac{N}{P+\epsilon}, R_{NK} = \frac{N}{K+\epsilon}, R_{PK} = \frac{P}{K+\epsilon}$
  - Temperature-Humidity Index: $THI = 0.8 T_{avg} + \left(\frac{RH}{100}\right)(T_{avg} - 14.4) + 46.4$
  - Moisture Availability Index: $MAI = \frac{P_{forecast} - \mu_{hist}}{\sigma_{hist}}$
- 🏆 **Ranked Crop Prediction**: Top-3 climate-resilient crop recommendations with calibrated suitability percentages based on XGBoost multi-class probabilities (**98.86% Accuracy, 0.9885 Macro $F_1$**).
- 🔍 **TreeSHAP Explainability Dashboard**: Sub-150ms exact local factor attributions into intuitive visual impact bars and natural-language causal narratives.
- 🧪 **Interactive Scenario Simulator**: "What-If" adjustment levers for testing synthetic rainfall shifts, heatwaves, and fertilizer amendments.
- 🗄️ **Zero-CSV Architecture**: All 2,200 agricultural samples and 30-year historical climate normals are stored in indexed **SQLite tables** (`data/optic_crop.db`) and **Parquet storage**.
- 🌦️ **Spatial Geohash Caching**: Level-6 Geohash indexing (~1.2 km²) with 1-hour TTL and graceful degradation fallback.

---

## 🏗️ System Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Data & Storage Layer (Zero CSV)
        A[data/optic_crop.db SQLite Database] -->|SQL Query| B[src/db.py]
        C[data/crop_dataset.parquet] -->|Fast Vector Load| B
    end

    subgraph Multi-Modal Feature Synthesis & ML Pipeline
        B --> D[src/model_train.py]
        D -->|Feature Synthesis: R_NP, R_NK, R_PK, THI, MAI| E[Dense Vector X in R^12]
        E --> F[XGBoost Hist Classifier]
        E --> G[TreeSHAP k-Means Background k=100]
        F & G --> H[Serialized Models in models/]
    end

    subgraph Production REST Microservice
        I[Open-Meteo API + Geohash Cache] --> J[FastAPI POST /api/v1/recommendations/predict]
        H --> J
        J --> K[Top-3 Crop Ranking + Latency Profiler]
    end

    subgraph User Experience
        J --> L[Streamlit Farmer / Agronomist Console & What-If Simulator]
    end
```

---

## 🚀 Quickstart & Running Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize Database & Train Multi-Modal XGBoost Model
```bash
python src/db.py
python src/model_train.py
```

### 3. Launch Production FastAPI Microservice
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation available at: `http://localhost:8000/docs`

### 4. Launch Streamlit Web Console
```bash
streamlit run app.py
```
Open your browser at: `http://localhost:8501`

---

## 📡 REST API Interface Specification

### `POST /api/v1/recommendations/predict`

#### Request Payload:
```json
{
  "latitude": 13.0827,
  "longitude": 80.2707,
  "soil_profile": {
    "nitrogen_mg_kg": 135.0,
    "phosphorus_mg_kg": 42.0,
    "potassium_mg_kg": 55.0,
    "ph_level": 6.7
  },
  "forecast_window_days": 14
}
```

#### Response Structure:
```json
{
  "status": "success",
  "data": {
    "recommendations": [
      {
        "crop": "Coffee",
        "viability_score": 0.196,
        "rank": 1,
        "explanations": {
          "base_value": 0.1091,
          "top_positive_factors": [
            { "feature": "r_nk", "value": 2.45, "shap_delta": 0.9475 },
            { "feature": "r_np", "value": 3.21, "shap_delta": 0.5345 }
          ],
          "top_negative_factors": [
            { "feature": "humidity", "value": 80.0, "shap_delta": -2.4819 }
          ],
          "human_readable_summary": "Viable alternative due to superior alignment with N:K Nutrient Ratio and N:P Nutrient Ratio."
        }
      }
    ],
    "geohash6": "tf346t"
  },
  "latency_metrics": {
    "weather_fetch_ms": 38.2,
    "inference_ms": 5.4,
    "shap_compute_ms": 31.0,
    "total_ms": 74.6
  }
}
```

---

## 📁 Repository Structure

```
ExplainCrop-AI/
├── data/
│   ├── optic_crop.db              # Relational SQLite Database (Zero CSV)
│   └── crop_dataset.parquet       # High-performance columnar dataset cache
├── models/
│   ├── xgboost_crop_model.joblib  # Trained XGBoost Hist classifier
│   ├── label_encoder.joblib       # 22-class Crop LabelEncoder
│   ├── shap_explainer.joblib      # TreeSHAP Explainer
│   ├── shap_background.joblib     # Precomputed k-Means background centroids (k=100)
│   ├── crop_profiles.json         # Crop physiological benchmarks
│   └── metadata.json              # Benchmarks & feature importances
├── src/
│   ├── __init__.py
│   ├── db.py                      # SQLite & Parquet database manager
│   ├── model_train.py             # Multi-modal feature engineering & training pipeline
│   ├── explain_engine.py          # Inference, TreeSHAP attribution & agronomic advisory
│   ├── weather_service.py         # Open-Meteo live sync & Geohash level-6 caching
│   └── api.py                     # FastAPI production REST microservice
├── app.py                         # Streamlit Farmer & Agronomist Console
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## 🏆 Model Benchmarks

| Model Architecture | Accuracy | Precision | Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Hist Tree - Production)** | **98.86%** | **0.9885** | **0.9886** | **0.9885** |
| Random Forest | 99.32% | 0.9932 | 0.9932 | 0.9932 |
| SVM (RBF) | 97.73% | 0.9769 | 0.9773 | 0.9769 |
| Decision Tree | 95.45% | 0.9540 | 0.9545 | 0.9540 |

---

## 📜 License & Author

- **Author**: Vinesh Raja ([@Vinesh-Raja07](https://github.com/Vinesh-Raja07))
- **License**: MIT License
