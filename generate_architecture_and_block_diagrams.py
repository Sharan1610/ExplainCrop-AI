"""
CropMind AI - Dedicated System Architecture & Block Diagrams Generator (Refined)
Produces two distinct, publication-grade 300 DPI diagrams:
1. cropmind_system_architecture.png - Detailed 6-Layer Enterprise Software Architecture
2. cropmind_functional_block_diagram.png - Comprehensive Engineering Block Diagram with Orthogonal Signal Buses
Also compiles them into a standalone 2-page PDF: CropMind_AI_Architecture_and_Block_Diagram.pdf
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

WORKSPACE_DIR = r"c:\Users\Vinesh Raja\OneDrive\Desktop\CropMind AI"
WORKSPACE_IMG_DIR = os.path.join(WORKSPACE_DIR, "images")
ARTIFACT_DIR = r"C:\Users\Vinesh Raja\.gemini\antigravity-ide\brain\9dc85385-b085-455f-86e1-9b422bf08760"

os.makedirs(WORKSPACE_IMG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Styling Palette
PRIMARY_DARK = "#0F172A"      # Slate 900
SECONDARY_DARK = "#1E293B"    # Slate 800
BG_LIGHT = "#F8FAFC"          # Slate 50
CARD_BG = "#FFFFFF"
BORDER_COLOR = "#CBD5E1"      # Slate 300
TEXT_MUTED = "#64748B"        # Slate 500
TEXT_MAIN = "#1E293B"

# Theme Colors
COLOR_BLUE = "#2563EB"        # Presentation / APIs
COLOR_EMERALD = "#059669"     # ML Core
COLOR_PURPLE = "#7C3AED"      # Feature Synthesis
COLOR_AMBER = "#D97706"       # Storage
COLOR_CYAN = "#0891B2"        # Weather & Sensors
COLOR_ROSE = "#E11D48"        # XAI TreeSHAP


def generate_system_architecture():
    """Generates a detailed 6-Layer Software & Microservices Architecture Diagram"""
    fig = plt.figure(figsize=(18, 12.5), facecolor=BG_LIGHT)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # Top Header Banner
    header = FancyBboxPatch((0.02, 0.925), 0.96, 0.062, boxstyle="round,pad=0.008,rounding_size=0.012",
                            facecolor=PRIMARY_DARK, edgecolor='none')
    ax.add_patch(header)
    ax.text(0.04, 0.962, "CropMind AI: System Architecture Diagram", fontsize=20, fontweight='bold', color='white', va='center')
    ax.text(0.04, 0.938, "Layered Microservices, Multi-Modal Feature Synthesis, XGBoost Inference & TreeSHAP Attribution Pipeline", 
            fontsize=10.5, color='#94A3B8', va='center')

    # Define 6 Architecture Layers
    layers = [
        {
            "id": "LAYER 1: PRESENTATION & CLIENT INTERFACE TIER",
            "col": COLOR_BLUE,
            "y": 0.775,
            "h": 0.135,
            "boxes": [
                {
                    "title": "Streamlit Interactive Agronomist Console",
                    "badge": "Frontend UI",
                    "details": [
                        "• Ranked Top-3 Recommended Crops with % Probabilities",
                        "• Interactive What-If Environmental Simulator",
                        "• Plotly Interactive SHAP Waterfall & Force Plots",
                        "• Agronomic Mitigation & Nitrogen/pH Advisories"
                    ],
                    "w": 0.45, "x": 0.04
                },
                {
                    "title": "Third-Party REST Clients & Mobile Edge Apps",
                    "badge": "External Integrations",
                    "details": [
                        "• Agri-IoT Edge Gateway Terminals (LoRaWAN / ESP32)",
                        "• Direct JSON/REST API Client Integration",
                        "• Automated Batch Farm Advisory Dispatch",
                        "• Standardized Pydantic Data Contracts"
                    ],
                    "w": 0.45, "x": 0.51
                }
            ]
        },
        {
            "id": "LAYER 2: API GATEWAY & APPLICATION MICROSERVICE TIER",
            "col": COLOR_CYAN,
            "y": 0.625,
            "h": 0.135,
            "boxes": [
                {
                    "title": "FastAPI Production ASGI Microservice (Uvicorn Server)",
                    "badge": "REST Gateway",
                    "details": [
                        "• POST /api/v1/recommendations/predict (Sub-75ms End-to-End Latency)",
                        "• GET /api/v1/health & Readiness / Live Diagnostics",
                        "• Pydantic v2 Strict Type Validation & Schema Enforcement",
                        "• Automated OpenAPI / Swagger Interactive Documentation"
                    ],
                    "w": 0.45, "x": 0.04
                },
                {
                    "title": "Spatial Geohash Caching & Weather Synchronization Service",
                    "badge": "Cache & Sync Engine",
                    "details": [
                        "• Level-6 Spatial Geohashing (~1.2 km² Precision Indexing)",
                        "• In-Memory Geohash TTL Cache (1-Hour Expiry Window)",
                        "• Open-Meteo REST API Connector (Real-Time Rain, T, RH)",
                        "• Graceful Offline Fallback to 30-Year Historical Normals"
                    ],
                    "w": 0.45, "x": 0.51
                }
            ]
        },
        {
            "id": "LAYER 3: MULTI-MODAL FEATURE SYNTHESIS ENGINE (X in R^12)",
            "col": COLOR_PURPLE,
            "y": 0.475,
            "h": 0.135,
            "boxes": [
                {
                    "title": "Stoichiometric Ratio Engine",
                    "badge": "Edaphic Transform",
                    "details": [
                        "• N-to-P Ratio: R_NP = N / (P + ε)",
                        "• N-to-K Ratio: R_NK = N / (K + ε)",
                        "• P-to-K Ratio: R_PK = P / (K + ε)",
                        "• Prevents Div-Zero via ε = 1e-6"
                    ],
                    "w": 0.29, "x": 0.04
                },
                {
                    "title": "Agro-Climatic Stress Engine",
                    "badge": "Climate Transform",
                    "details": [
                        "• Temperature-Humidity Index (THI):",
                        "   THI = 0.8·T + (RH/100)·(T-14.4) + 46.4",
                        "• Moisture Availability Index (MAI):",
                        "   MAI = (P_forecast - μ_hist) / σ_hist"
                    ],
                    "w": 0.29, "x": 0.355
                },
                {
                    "title": "12-D Dense Vector Assembler",
                    "badge": "Feature Tensor",
                    "details": [
                        "• Raw: [N, P, K, T, RH, pH, Rain]",
                        "• Synthesized: [R_NP, R_NK, R_PK, THI, MAI]",
                        "• Dense Vector Format (X ∈ ℝ¹²)",
                        "• High-Throughput In-Memory Pipeline"
                    ],
                    "w": 0.29, "x": 0.67
                }
            ]
        },
        {
            "id": "LAYER 4: MACHINE LEARNING & EXPLAINABLE AI (XAI) CORE",
            "col": COLOR_EMERALD,
            "y": 0.325,
            "h": 0.135,
            "boxes": [
                {
                    "title": "XGBoost Multi-Class Hist-Tree Classifier (Model Core)",
                    "badge": "Predictive Engine",
                    "details": [
                        "• Histogram-Based Gradient Boosted Decision Forest",
                        "• 22 Crop Target Classes (98.86% Test Accuracy | 0.9885 Macro F1)",
                        "• Softmax Probability Distribution Vector across all 22 crops",
                        "• Ultra-Fast Prediction Time: 5.4 ms Per Inference"
                    ],
                    "w": 0.45, "x": 0.04
                },
                {
                    "title": "TreeSHAP Local Factor Attribution & Explanation Engine",
                    "badge": "Explainability Engine",
                    "details": [
                        "• Game-Theoretic TreeSHAP Algorithm (Direct Exact Computation)",
                        "• k-Means Centroid Background Reference Set (k=100 Samples)",
                        "• Directional Impact Quantifier (+Δ / -Δ Feature Attribution)",
                        "• Causal Agronomic Natural Language Explanation Generator"
                    ],
                    "w": 0.45, "x": 0.51
                }
            ]
        },
        {
            "id": "LAYER 5: DATA PERSISTENCE & COLUMNAR STORAGE TIER (ZERO-CSV)",
            "col": COLOR_AMBER,
            "y": 0.175,
            "h": 0.135,
            "boxes": [
                {
                    "title": "SQLite Relational Database Engine (optic_crop.db)",
                    "badge": "Relational Storage",
                    "details": [
                        "• 2,200 Indexed Historical Agricultural Samples",
                        "• 30-Year Historical Climate Normals (μ_hist, σ_hist by Region)",
                        "• Complete Prediction & SHAP Attribution Audit Logs",
                        "• ACID-Compliant Transaction Management (295 KB Footprint)"
                    ],
                    "w": 0.45, "x": 0.04
                },
                {
                    "title": "Apache Parquet Columnar Data Cache (crop_dataset.parquet)",
                    "badge": "Columnar Storage",
                    "details": [
                        "• High-Performance Vectorized Snappy-Compressed Format (96.5 KB)",
                        "• Zero-CSV Architecture: Eliminates CSV Parsing Overhead",
                        "• Sub-Millisecond Batch Data Loading for Training & Validation",
                        "• Exact Type Preservation (Float32 / Int64 / Categorical)"
                    ],
                    "w": 0.45, "x": 0.51
                }
            ]
        },
        {
            "id": "LAYER 6: EXTERNAL SENSORY & METEOROLOGICAL INTEGRATION LAYER",
            "col": COLOR_ROSE,
            "y": 0.025,
            "h": 0.135,
            "boxes": [
                {
                    "title": "On-Field Edaphic Soil Sensor Telemetry",
                    "badge": "IoT Sensory Data",
                    "details": [
                        "• Soil Nitrogen (N), Phosphorus (P), Potassium (K) in mg/kg",
                        "• Soil Acidity / Alkalinity (pH Scale 0–14)",
                        "• Calibrated Sensor Ingestion with Min/Max Clipping Limits",
                        "• Real-Time Edge Data Validation"
                    ],
                    "w": 0.45, "x": 0.04
                },
                {
                    "title": "Open-Meteo REST API Meteorology Provider",
                    "badge": "External REST API",
                    "details": [
                        "• Live Dynamic Ambient Temperature (°C) & Relative Humidity (%)",
                        "• 14-Day Dynamic Cumulative Precipitation Forecast (mm)",
                        "• Global Geospatial Resolution (No API Key Dependency)",
                        "• Average Latency: 38.2 ms Network Request Time"
                    ],
                    "w": 0.45, "x": 0.51
                }
            ]
        }
    ]

    for layer in layers:
        ly = layer["y"]
        lh = layer["h"]
        col = layer["col"]

        # Layer Container Background
        layer_bg = FancyBboxPatch((0.02, ly), 0.96, lh, boxstyle="round,pad=0.005,rounding_size=0.008",
                                  facecolor=BG_LIGHT, edgecolor=col, linewidth=1.5)
        ax.add_patch(layer_bg)

        # Layer Header Tag
        tag_w = 0.42
        ltag = FancyBboxPatch((0.035, ly + lh - 0.024), tag_w, 0.022, boxstyle="round,pad=0.002,rounding_size=0.004",
                              facecolor=col, edgecolor='none')
        ax.add_patch(ltag)
        ax.text(0.045, ly + lh - 0.013, layer["id"], fontsize=8.5, fontweight='bold', color='white', va='center')

        # Draw component cards inside layer
        for box in layer["boxes"]:
            bx = box["x"]
            bw = box["w"]
            by = ly + 0.012
            bh = lh - 0.042

            c_box = FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.003,rounding_size=0.006",
                                   facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1.0)
            ax.add_patch(c_box)

            # Box Title
            ax.text(bx + 0.012, by + bh - 0.016, box["title"], fontsize=9.2, fontweight='bold', color=PRIMARY_DARK, va='center')

            # Box Badge
            badge_w = 0.105 if bw < 0.35 else 0.125
            badge = FancyBboxPatch((bx + bw - badge_w - 0.010, by + bh - 0.024), badge_w, 0.016,
                                   boxstyle="round,pad=0.002,rounding_size=0.003",
                                   facecolor="#F1F5F9", edgecolor=col, linewidth=0.8)
            ax.add_patch(badge)
            ax.text(bx + bw - (badge_w/2) - 0.010, by + bh - 0.016, box["badge"], fontsize=6.8, fontweight='bold', color=col, ha='center', va='center')

            # Details
            dy = by + bh - 0.036
            for line in box["details"]:
                ax.text(bx + 0.012, dy, line, fontsize=7.5, color=TEXT_MAIN, va='center')
                dy -= 0.0155

    # Save outputs
    out_ws = os.path.join(WORKSPACE_IMG_DIR, "cropmind_system_architecture.png")
    out_art = os.path.join(ARTIFACT_DIR, "cropmind_system_architecture.png")
    fig.savefig(out_ws, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    fig.savefig(out_art, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"System Architecture saved: {out_ws}")
    return out_ws


def generate_functional_block_diagram():
    """Generates a comprehensive Engineering Functional Block Diagram with clean orthogonal signal buses"""
    fig = plt.figure(figsize=(18, 12.5), facecolor=BG_LIGHT)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # Top Header Banner
    header = FancyBboxPatch((0.02, 0.925), 0.96, 0.062, boxstyle="round,pad=0.008,rounding_size=0.012",
                            facecolor=PRIMARY_DARK, edgecolor='none')
    ax.add_patch(header)
    ax.text(0.04, 0.962, "CropMind AI: Functional Block Diagram", fontsize=20, fontweight='bold', color='white', va='center')
    ax.text(0.04, 0.938, "Signal Acquisition, Feature Synthesis Unit, ML Predictive Core, XAI Attribution Engine & Output Dispatch", 
            fontsize=10.5, color='#94A3B8', va='center')

    # 6 Functional Blocks in Clean Symmetrical Grid
    # Column 1 (x: 0.04, w: 0.20): Block 1 (Top), Block 2 (Bottom)
    # Column 2 (x: 0.28, w: 0.22): Block 3 (Full Height Feature Synthesis)
    # Column 3 (x: 0.54, w: 0.20): Block 4 (Top ML Core), Block 5 (Bottom TreeSHAP XAI)
    # Column 4 (x: 0.77, w: 0.19): Block 6 (Full Height Output & Action Dispatch)

    # BLOCK 1: INPUT ACQUISITION
    b1 = FancyBboxPatch((0.04, 0.49), 0.20, 0.40, boxstyle="round,pad=0.006,rounding_size=0.010",
                        facecolor=CARD_BG, edgecolor=COLOR_CYAN, linewidth=1.5)
    ax.add_patch(b1)
    h1 = FancyBboxPatch((0.04, 0.85), 0.20, 0.04, boxstyle="round,pad=0.002,rounding_size=0.004", facecolor=COLOR_CYAN, edgecolor='none')
    ax.add_patch(h1)
    ax.text(0.14, 0.87, "BLOCK 1: INPUT ACQUISITION", fontsize=8.5, fontweight='bold', color='white', ha='center', va='center')

    b1_items = [
        ("Soil N-P-K Nutrients", "Nitrogen, Phosphorus, Potassium (mg/kg)"),
        ("Soil Acidity / pH", "Direct pH probe sensor readings (0-14)"),
        ("Geospatial GPS", "Latitude, Longitude Coordinates"),
        ("Open-Meteo REST Stream", "Live Temp, Humidity, 14-Day Rain"),
        ("Regional Normals DB", "30-Year μ_hist, σ_hist Baseline")
    ]
    y = 0.81
    for name, desc in b1_items:
        ibox = FancyBboxPatch((0.05, y - 0.045), 0.18, 0.055, boxstyle="round,pad=0.002,rounding_size=0.004",
                              facecolor="#F0FDF4", edgecolor="#BBF7D0", linewidth=0.8)
        ax.add_patch(ibox)
        ax.text(0.055, y - 0.015, name, fontsize=8, fontweight='bold', color=PRIMARY_DARK)
        ax.text(0.055, y - 0.035, desc, fontsize=6.8, color=TEXT_MUTED)
        y -= 0.065

    # BLOCK 2: GEOHASH & VALIDATION
    b2 = FancyBboxPatch((0.04, 0.06), 0.20, 0.39, boxstyle="round,pad=0.006,rounding_size=0.010",
                        facecolor=CARD_BG, edgecolor=COLOR_BLUE, linewidth=1.5)
    ax.add_patch(b2)
    h2 = FancyBboxPatch((0.04, 0.41), 0.20, 0.04, boxstyle="round,pad=0.002,rounding_size=0.004", facecolor=COLOR_BLUE, edgecolor='none')
    ax.add_patch(h2)
    ax.text(0.14, 0.43, "BLOCK 2: GEOHASH & VALIDATION", fontsize=8.5, fontweight='bold', color='white', ha='center', va='center')

    b2_items = [
        ("Spatial Geohash Encoder", "Level-6 Hash Resolution (~1.2 km²)"),
        ("In-Memory TTL Cache", "1-Hour Cache: Eliminates API Spikes"),
        ("Data Range Validator", "Pydantic Schema & Outlier Clipping"),
        ("Zero-CSV SQLite / Parquet", "Columnar & Relational Data Access"),
        ("Offline Safe Dispatcher", "Fallback to historical climatic normals")
    ]
    y = 0.37
    for name, desc in b2_items:
        ibox = FancyBboxPatch((0.05, y - 0.045), 0.18, 0.055, boxstyle="round,pad=0.002,rounding_size=0.004",
                              facecolor="#EFF6FF", edgecolor="#BFDBFE", linewidth=0.8)
        ax.add_patch(ibox)
        ax.text(0.055, y - 0.015, name, fontsize=8, fontweight='bold', color=PRIMARY_DARK)
        ax.text(0.055, y - 0.035, desc, fontsize=6.8, color=TEXT_MUTED)
        y -= 0.063

    # BLOCK 3: MULTI-MODAL FEATURE SYNTHESIS UNIT (CENTRAL COLUMN)
    b3 = FancyBboxPatch((0.28, 0.06), 0.22, 0.83, boxstyle="round,pad=0.006,rounding_size=0.010",
                        facecolor=CARD_BG, edgecolor=COLOR_PURPLE, linewidth=1.5)
    ax.add_patch(b3)
    h3 = FancyBboxPatch((0.28, 0.85), 0.22, 0.04, boxstyle="round,pad=0.002,rounding_size=0.004", facecolor=COLOR_PURPLE, edgecolor='none')
    ax.add_patch(h3)
    ax.text(0.39, 0.87, "BLOCK 3: FEATURE SYNTHESIS UNIT", fontsize=8.5, fontweight='bold', color='white', ha='center', va='center')

    b3_items = [
        ("Raw Edaphic & Weather (7)", "[N, P, K, pH, Temp, RH, Rain]"),
        ("R_NP Ratio Synthesizer", "R_NP = N / (P + 1e-6)"),
        ("R_NK Ratio Synthesizer", "R_NK = N / (K + 1e-6)"),
        ("R_PK Ratio Synthesizer", "R_PK = P / (K + 1e-6)"),
        ("Bioclimatic THI Computer", "THI = 0.8·T + (RH/100)·(T-14.4) + 46.4"),
        ("Moisture Index (MAI)", "MAI = (P_forecast - μ_hist) / σ_hist"),
        ("12-D Feature Multiplexer", "Dense Vector Tensor X ∈ ℝ¹²"),
        ("Tensor Normalizer", "Sub-1ms Vector Pipeline")
    ]
    y = 0.81
    for name, desc in b3_items:
        ibox = FancyBboxPatch((0.29, y - 0.065), 0.20, 0.070, boxstyle="round,pad=0.002,rounding_size=0.004",
                              facecolor="#FAF5FF", edgecolor="#E9D5FF", linewidth=0.8)
        ax.add_patch(ibox)
        ax.text(0.295, y - 0.022, name, fontsize=8, fontweight='bold', color=PRIMARY_DARK)
        ax.text(0.295, y - 0.046, desc, fontsize=7, color="#6B21A8")
        y -= 0.092

    # BLOCK 4: ML PREDICTIVE CORE
    b4 = FancyBboxPatch((0.54, 0.49), 0.20, 0.40, boxstyle="round,pad=0.006,rounding_size=0.010",
                        facecolor=CARD_BG, edgecolor=COLOR_EMERALD, linewidth=1.5)
    ax.add_patch(b4)
    h4 = FancyBboxPatch((0.54, 0.85), 0.20, 0.04, boxstyle="round,pad=0.002,rounding_size=0.004", facecolor=COLOR_EMERALD, edgecolor='none')
    ax.add_patch(h4)
    ax.text(0.64, 0.87, "BLOCK 4: ML PREDICTIVE CORE", fontsize=8.5, fontweight='bold', color='white', ha='center', va='center')

    b4_items = [
        ("XGBoost Hist Classifier", "Histogram-based tree decision forest"),
        ("22-Class Probability Engine", "Softmax vector across 22 crops"),
        ("Top-3 Confidence Ranker", "Filters highest yield probabilities"),
        ("Low-Latency Inference", "5.4 ms per inference execution")
    ]
    y = 0.81
    for name, desc in b4_items:
        ibox = FancyBboxPatch((0.55, y - 0.055), 0.18, 0.062, boxstyle="round,pad=0.002,rounding_size=0.004",
                              facecolor="#ECFDF5", edgecolor="#A7F3D0", linewidth=0.8)
        ax.add_patch(ibox)
        ax.text(0.555, y - 0.02, name, fontsize=8, fontweight='bold', color=PRIMARY_DARK)
        ax.text(0.555, y - 0.042, desc, fontsize=7, color=TEXT_MUTED)
        y -= 0.078

    # BLOCK 5: XAI TREESHAP ENGINE
    b5 = FancyBboxPatch((0.54, 0.06), 0.20, 0.39, boxstyle="round,pad=0.006,rounding_size=0.010",
                        facecolor=CARD_BG, edgecolor=COLOR_ROSE, linewidth=1.5)
    ax.add_patch(b5)
    h5 = FancyBboxPatch((0.54, 0.41), 0.20, 0.04, boxstyle="round,pad=0.002,rounding_size=0.004", facecolor=COLOR_ROSE, edgecolor='none')
    ax.add_patch(h5)
    ax.text(0.64, 0.43, "BLOCK 5: XAI TREESHAP ENGINE", fontsize=8.5, fontweight='bold', color='white', ha='center', va='center')

    b5_items = [
        ("Exact TreeSHAP Evaluator", "Game-theoretic Shapley value vector"),
        ("k-Means Background Set", "k=100 cluster centroids background"),
        ("Directional Impact (+Δ/-Δ)", "Distinguishes positive vs limiting factors"),
        ("Agronomic Narrative Synthesizer", "Generates human-readable explanations")
    ]
    y = 0.37
    for name, desc in b5_items:
        ibox = FancyBboxPatch((0.55, y - 0.055), 0.18, 0.062, boxstyle="round,pad=0.002,rounding_size=0.004",
                              facecolor="#FFF1F2", edgecolor="#FECDD3", linewidth=0.8)
        ax.add_patch(ibox)
        ax.text(0.555, y - 0.02, name, fontsize=8, fontweight='bold', color=PRIMARY_DARK)
        ax.text(0.555, y - 0.042, desc, fontsize=7, color=TEXT_MUTED)
        y -= 0.068

    # BLOCK 6: OUTPUT & ACTION DISPATCH
    b6 = FancyBboxPatch((0.77, 0.06), 0.19, 0.83, boxstyle="round,pad=0.006,rounding_size=0.010",
                        facecolor=CARD_BG, edgecolor=COLOR_AMBER, linewidth=1.5)
    ax.add_patch(b6)
    h6 = FancyBboxPatch((0.77, 0.85), 0.19, 0.04, boxstyle="round,pad=0.002,rounding_size=0.004", facecolor=COLOR_AMBER, edgecolor='none')
    ax.add_patch(h6)
    ax.text(0.865, 0.87, "BLOCK 6: OUTPUT DISPATCH", fontsize=8.5, fontweight='bold', color='white', ha='center', va='center')

    b6_items = [
        ("Ranked Crop Recommendations", "Primary & secondary crop options with % confidence"),
        ("SHAP Factor Waterfall", "Plotly visual breakdown of soil/climate impacts"),
        ("Agronomic Mitigation Guidance", "Actionable N-P-K & pH corrective advice"),
        ("What-If Simulation Engine", "Real-time sensitivity analysis for climate resilience"),
        ("FastAPI JSON API Response", "Structured payload for third-party microservices"),
        ("Prediction & Audit Logging", "Historical database trace for compliance & drift monitoring")
    ]
    y = 0.81
    for name, desc in b6_items:
        ibox = FancyBboxPatch((0.78, y - 0.090), 0.17, 0.100, boxstyle="round,pad=0.002,rounding_size=0.004",
                              facecolor="#FFFBEB", edgecolor="#FDE68A", linewidth=0.8)
        ax.add_patch(ibox)
        ax.text(0.785, y - 0.025, name, fontsize=8, fontweight='bold', color=PRIMARY_DARK)
        ax.text(0.785, y - 0.055, desc, fontsize=6.8, color=TEXT_MUTED)
        y -= 0.123

    # Clean Orthogonal Connecting Buses and Arrows
    arrow_props = dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#1E293B", lw=2.0)
    
    # 1. Sensory Bus: Block 1 -> Block 3
    ax.annotate("", xy=(0.28, 0.69), xytext=(0.24, 0.69), arrowprops=arrow_props)
    ax.text(0.26, 0.71, "Sensory Bus", fontsize=7.5, fontweight='bold', color=COLOR_CYAN, ha='center')

    # 2. Cache / Normals Bus: Block 2 -> Block 3
    ax.annotate("", xy=(0.28, 0.25), xytext=(0.24, 0.25), arrowprops=arrow_props)
    ax.text(0.26, 0.27, "Cache / Normals", fontsize=7.5, fontweight='bold', color=COLOR_BLUE, ha='center')

    # 3. Dense Feature Vector Bus to ML Core: Block 3 -> Block 4
    ax.annotate("", xy=(0.54, 0.69), xytext=(0.50, 0.69), arrowprops=arrow_props)
    ax.text(0.52, 0.71, "X ∈ ℝ¹²", fontsize=8, fontweight='bold', color=COLOR_PURPLE, ha='center')

    # 4. Dense Feature Vector Bus to TreeSHAP XAI: Block 3 -> Block 5
    ax.annotate("", xy=(0.54, 0.25), xytext=(0.50, 0.25), arrowprops=arrow_props)
    ax.text(0.52, 0.27, "X ∈ ℝ¹²", fontsize=8, fontweight='bold', color=COLOR_PURPLE, ha='center')

    # 5. Prediction Output to Dispatcher: Block 4 -> Block 6
    ax.annotate("", xy=(0.77, 0.69), xytext=(0.74, 0.69), arrowprops=arrow_props)
    ax.text(0.755, 0.71, "Top-3 Crops", fontsize=7.5, fontweight='bold', color=COLOR_EMERALD, ha='center')

    # 6. SHAP Explanations to Dispatcher: Block 5 -> Block 6
    ax.annotate("", xy=(0.77, 0.25), xytext=(0.74, 0.25), arrowprops=arrow_props)
    ax.text(0.755, 0.27, "SHAP Values", fontsize=7.5, fontweight='bold', color=COLOR_ROSE, ha='center')

    # Bottom Legend / Summary Strip
    legend_box = FancyBboxPatch((0.04, 0.01), 0.92, 0.035, boxstyle="round,pad=0.002,rounding_size=0.004",
                                facecolor=PRIMARY_DARK, edgecolor='none')
    ax.add_patch(legend_box)
    ax.text(0.50, 0.027, "Data Flow: [Sensors & Weather Stream] ➔ [12-D Feature Synthesis] ➔ [XGBoost 98.86% + TreeSHAP XAI] ➔ [Advisories & UI]", 
            fontsize=8.5, fontweight='bold', color='#38BDF8', ha='center', va='center')

    # Save outputs
    out_ws = os.path.join(WORKSPACE_IMG_DIR, "cropmind_functional_block_diagram.png")
    out_art = os.path.join(ARTIFACT_DIR, "cropmind_functional_block_diagram.png")
    fig.savefig(out_ws, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    fig.savefig(out_art, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Functional Block Diagram saved: {out_ws}")
    return out_ws


def compile_pdf():
    arch_img = generate_system_architecture()
    block_img = generate_functional_block_diagram()

    pdf_ws = os.path.join(WORKSPACE_DIR, "CropMind_AI_Architecture_and_Block_Diagram.pdf")
    pdf_art = os.path.join(ARTIFACT_DIR, "CropMind_AI_Architecture_and_Block_Diagram.pdf")

    img1 = Image.open(arch_img).convert('RGB')
    img2 = Image.open(block_img).convert('RGB')

    img1.save(pdf_ws, save_all=True, append_images=[img2], resolution=300.0, quality=95)
    img1.save(pdf_art, save_all=True, append_images=[img2], resolution=300.0, quality=95)

    print(f"Standalone PDF compiled: {pdf_ws}")


if __name__ == "__main__":
    compile_pdf()
