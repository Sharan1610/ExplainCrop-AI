"""
ExplainCrop-AI: Interactive Web Application
Built with Streamlit & Plotly with Explainable AI (XAI) and Live Weather Integration.
Features an interactive dynamic particle mesh background and glassmorphic UI.
"""

import os
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.explain_engine import predict_and_explain, get_engine
from src.weather_service import get_weather_by_city

# Set Page Config
st.set_page_config(
    page_title="ExplainCrop-AI | Explainable Precision Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Interactive Dynamic Background & High-Aesthetic Glassmorphic CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Ambient Animated Mesh Background */
    .stApp {
        background-color: #0B1120;
        background-image: 
            radial-gradient(at 0% 0%, rgba(16, 185, 129, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(56, 189, 248, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
            radial-gradient(at 0% 100%, rgba(5, 150, 105, 0.12) 0px, transparent 50%);
        background-attachment: fixed;
        color: #F8FAFC;
    }

    /* Glowing Floating Orbs in Background */
    .floating-orb-1 {
        position: fixed;
        width: 350px;
        height: 350px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.25) 0%, rgba(16, 185, 129, 0) 70%);
        top: -50px;
        left: 10%;
        z-index: 0;
        pointer-events: none;
        animation: floatOrb 18s ease-in-out infinite alternate;
        filter: blur(40px);
    }
    .floating-orb-2 {
        position: fixed;
        width: 450px;
        height: 450px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.2) 0%, rgba(56, 189, 248, 0) 70%);
        bottom: 50px;
        right: 5%;
        z-index: 0;
        pointer-events: none;
        animation: floatOrb 22s ease-in-out infinite alternate-reverse;
        filter: blur(50px);
    }

    @keyframes floatOrb {
        0% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(60px, 40px) scale(1.15); }
        100% { transform: translate(-40px, 80px) scale(0.95); }
    }

    /* Hero Header */
    .hero-container {
        position: relative;
        z-index: 1;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(15, 23, 42, 0.8) 50%, rgba(56, 189, 248, 0.18) 100%);
        border: 1px solid rgba(52, 211, 153, 0.35);
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .hero-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 50px rgba(16, 185, 129, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #34D399, #38BDF8, #A78BFA, #34D399);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 8s ease infinite;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.08rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Glass Cards */
    .glass-card {
        position: relative;
        z-index: 1;
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        margin-bottom: 18px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.3);
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.4), 0 0 20px rgba(56, 189, 248, 0.15);
    }

    .glass-card-highlight {
        position: relative;
        z-index: 1;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid rgba(52, 211, 153, 0.5);
        border-radius: 18px;
        padding: 26px;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 35px rgba(16, 185, 129, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card-highlight:hover {
        transform: translateY(-4px) scale(1.01);
        border-color: rgba(52, 211, 153, 0.8);
        box-shadow: 0 16px 45px rgba(16, 185, 129, 0.35), 0 0 30px rgba(16, 185, 129, 0.2);
    }

    /* Badges */
    .crop-badge {
        display: inline-block;
        padding: 6px 18px;
        background: linear-gradient(90deg, #059669, #10B981);
        color: #FFFFFF;
        font-weight: 700;
        font-size: 1.3rem;
        border-radius: 30px;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    .confidence-badge {
        display: inline-block;
        padding: 5px 14px;
        background: rgba(56, 189, 248, 0.2);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38BDF8;
        font-weight: 600;
        border-radius: 20px;
        font-size: 0.95rem;
    }

    /* Custom Streamlit Sliders & Inputs */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #10B981, #38BDF8) !important;
    }

    /* Modern Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #10B981 0%, #059669 100%);
        color: white;
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 700;
        font-size: 1.05rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
        width: 100%;
        letter-spacing: 0.3px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #059669 0%, #047857 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.6);
        color: white;
        border-color: rgba(52, 211, 153, 0.8);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.92);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
    }
    </style>

    <!-- Floating Background Orbs -->
    <div class="floating-orb-1"></div>
    <div class="floating-orb-2"></div>
    """,
    unsafe_allow_html=True,
)

# Interactive Particle Mesh Canvas Component
components.html(
    """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            overflow: hidden;
            width: 100%;
            height: 100%;
            background: transparent;
        }
        canvas {
            display: block;
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: auto;
        }
    </style>
    </head>
    <body>
    <canvas id="particleCanvas"></canvas>
    <script>
        const canvas = document.getElementById('particleCanvas');
        const ctx = canvas.getContext('2d');

        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;

        window.addEventListener('resize', () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
            initParticles();
        });

        const mouse = { x: null, y: null, radius: 120 };
        window.addEventListener('mousemove', (e) => {
            mouse.x = e.x;
            mouse.y = e.y;
        });
        window.addEventListener('mouseout', () => {
            mouse.x = null;
            mouse.y = null;
        });

        class Particle {
            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.size = Math.random() * 2.5 + 1;
                this.vx = (Math.random() - 0.5) * 0.7;
                this.vy = (Math.random() - 0.5) * 0.7;
                this.baseColor = Math.random() > 0.5 ? '16, 185, 129' : '56, 189, 248';
                this.alpha = Math.random() * 0.6 + 0.2;
            }

            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${this.baseColor}, ${this.alpha})`;
                ctx.shadowBlur = 8;
                ctx.shadowColor = `rgba(${this.baseColor}, 0.5)`;
                ctx.fill();
            }

            update() {
                if (this.x < 0 || this.x > width) this.vx *= -1;
                if (this.y < 0 || this.y > height) this.vy *= -1;

                // Mouse interaction repulsion/pull
                if (mouse.x != null && mouse.y != null) {
                    let dx = mouse.x - this.x;
                    let dy = mouse.y - this.y;
                    let dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < mouse.radius) {
                        let force = (mouse.radius - dist) / mouse.radius;
                        this.x -= (dx / dist) * force * 3;
                        this.y -= (dy / dist) * force * 3;
                    }
                }

                this.x += this.vx;
                this.y += this.vy;
                this.draw();
            }
        }

        let particles = [];
        function initParticles() {
            particles = [];
            const count = Math.min(Math.floor((width * height) / 14000), 80);
            for (let i = 0; i < count; i++) {
                particles.push(new Particle());
            }
        }
        initParticles();

        function connectParticles() {
            for (let a = 0; a < particles.length; a++) {
                for (let b = a + 1; b < particles.length; b++) {
                    let dx = particles[a].x - particles[b].x;
                    let dy = particles[a].y - particles[b].y;
                    let dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < 130) {
                        let opacity = 1 - (dist / 130);
                        ctx.strokeStyle = `rgba(16, 185, 129, ${opacity * 0.25})`;
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(particles[a].x, particles[a].y);
                        ctx.lineTo(particles[b].x, particles[b].y);
                        ctx.stroke();
                    }
                }
            }
        }

        function animate() {
            ctx.clearRect(0, 0, width, height);
            for (let i = 0; i < particles.length; i++) {
                particles[i].update();
            }
            connectParticles();
            requestAnimationFrame(animate);
        }
        animate();
    </script>
    </body>
    </html>
    """,
    height=120,
)

# Initialize Session State
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 25.0
if "humidity" not in st.session_state:
    st.session_state["humidity"] = 72.0
if "rainfall" not in st.session_state:
    st.session_state["rainfall"] = 120.0
if "city_weather_info" not in st.session_state:
    st.session_state["city_weather_info"] = None
if "nitrogen" not in st.session_state:
    st.session_state["nitrogen"] = 80.0
if "phosphorus" not in st.session_state:
    st.session_state["phosphorus"] = 45.0
if "potassium" not in st.session_state:
    st.session_state["potassium"] = 40.0
if "ph" not in st.session_state:
    st.session_state["ph"] = 6.5

# Crop Icon Map
CROP_ICONS = {
    "Rice": "🌾",
    "Maize": "🌽",
    "Chickpea": "🌱",
    "Kidneybeans": "🫘",
    "Pigeonpeas": "🌿",
    "Mothbeans": "🌱",
    "Mungbean": "🫘",
    "Blackgram": "🌾",
    "Lentil": "🥣",
    "Pomegranate": "🍎",
    "Banana": "🍌",
    "Mango": "🥭",
    "Grapes": "🍇",
    "Watermelon": "🍉",
    "Muskmelon": "🍈",
    "Apple": "🍎",
    "Orange": "🍊",
    "Papaya": "🍈",
    "Coconut": "🥥",
    "Cotton": "☁️",
    "Jute": "🧵",
    "Coffee": "☕",
}

# Hero Header
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🌾 ExplainCrop-AI</div>
        <div class="hero-subtitle">
            Explainable Precision Agriculture powered by <b>XGBoost</b>, <b>SHAP Interpretability</b>, and <b>Real-Time Weather Intelligence</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Weather & Soil Presets
with st.sidebar:
    st.markdown("### 🌦️ Real-Time Weather Sync")
    city_input = st.text_input(
        "Enter City / Region",
        placeholder="e.g. Coimbatore, Punjab, Dallas, Nairobi",
        help="Fetches live temperature, humidity, and rainfall estimates via Open-Meteo API.",
    )

    if st.button("📍 Fetch Live Weather"):
        if city_input:
            with st.spinner(f"Fetching live weather for '{city_input}'..."):
                weather_res = get_weather_by_city(city_input)
                if weather_res.get("success"):
                    st.session_state["temperature"] = float(weather_res["temperature"])
                    st.session_state["humidity"] = float(weather_res["humidity"])
                    st.session_state["rainfall"] = float(weather_res["rainfall"])
                    st.session_state["city_weather_info"] = weather_res.get(
                        "location_name", city_input
                    )
                    st.success(f"Weather synced for **{st.session_state['city_weather_info']}**!")
                else:
                    st.error(weather_res.get("error", "Failed to retrieve weather."))
        else:
            st.warning("Please enter a city name.")

    if st.session_state["city_weather_info"]:
        st.caption(f"📍 Active Location: **{st.session_state['city_weather_info']}**")

    st.markdown("---")
    st.markdown("### 🧪 Soil Type Presets")
    st.caption("Quickly populate typical soil nutrient profiles:")

    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        if st.button("🌱 Alluvial Plain"):
            st.session_state["nitrogen"] = 90.0
            st.session_state["phosphorus"] = 50.0
            st.session_state["potassium"] = 45.0
            st.session_state["ph"] = 6.8
            st.rerun()

        if st.button("🪨 Black Cotton"):
            st.session_state["nitrogen"] = 40.0
            st.session_state["phosphorus"] = 65.0
            st.session_state["potassium"] = 80.0
            st.session_state["ph"] = 7.5
            st.rerun()

    with preset_col2:
        if st.button("🍂 Red Loam"):
            st.session_state["nitrogen"] = 60.0
            st.session_state["phosphorus"] = 35.0
            st.session_state["potassium"] = 30.0
            st.session_state["ph"] = 5.8
            st.rerun()

        if st.button("🏜️ Sandy Arid"):
            st.session_state["nitrogen"] = 25.0
            st.session_state["phosphorus"] = 20.0
            st.session_state["potassium"] = 25.0
            st.session_state["ph"] = 8.0
            st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ About ExplainCrop-AI")
    st.markdown(
        """
        - **Model**: Multi-Class XGBoost (99.32% Acc)
        - **Explainability**: SHAP TreeExplainer
        - **Weather**: Open-Meteo Live API
        - **Author**: Vinesh Raja
        - [GitHub Repository](https://github.com/Vinesh-Raja07/ExplainCrop-AI)
        """
    )


# Main Tabs Layout
tab_recommend, tab_analytics, tab_crops = st.tabs(
    ["🎯 Crop Recommendation & SHAP XAI", "📊 Model Performance & Benchmarks", "📖 Crop Knowledge Base"]
)

# TAB 1: RECOMMENDATION & EXPLAINABILITY
with tab_recommend:
    col_input1, col_input2 = st.columns([1, 1])

    with col_input1:
        st.markdown("#### 🧪 1. Soil Chemistry Parameters")
        n_val = st.slider(
            "Nitrogen (N) [Ratio in Soil]",
            min_value=0.0,
            max_value=150.0,
            value=float(st.session_state["nitrogen"]),
            step=1.0,
            help="Nitrogen content ratio in soil (0 - 150)",
        )
        p_val = st.slider(
            "Phosphorus (P) [Ratio in Soil]",
            min_value=5.0,
            max_value=150.0,
            value=float(st.session_state["phosphorus"]),
            step=1.0,
            help="Phosphorus content ratio in soil (5 - 150)",
        )
        k_val = st.slider(
            "Potassium (K) [Ratio in Soil]",
            min_value=5.0,
            max_value=210.0,
            value=float(st.session_state["potassium"]),
            step=1.0,
            help="Potassium content ratio in soil (5 - 210)",
        )
        ph_val = st.slider(
            "Soil pH Value",
            min_value=3.5,
            max_value=10.0,
            value=float(st.session_state["ph"]),
            step=0.1,
            help="Soil acidity / alkalinity scale (3.5 - 10.0)",
        )

    with col_input2:
        st.markdown("#### 🌦️ 2. Climate & Environmental Parameters")
        temp_val = st.slider(
            "Temperature (°C)",
            min_value=5.0,
            max_value=50.0,
            value=float(st.session_state["temperature"]),
            step=0.5,
            help="Ambient temperature in degrees Celsius",
        )
        humidity_val = st.slider(
            "Relative Humidity (%)",
            min_value=10.0,
            max_value=100.0,
            value=float(st.session_state["humidity"]),
            step=1.0,
            help="Relative air humidity percentage",
        )
        rainfall_val = st.slider(
            "Seasonal Rainfall (mm)",
            min_value=15.0,
            max_value=320.0,
            value=float(st.session_state["rainfall"]),
            step=5.0,
            help="Estimated seasonal rainfall in millimeters",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Recommend Optimal Crop & Generate SHAP Explanation")

    if predict_btn or True:
        try:
            results = predict_and_explain(
                n_val, p_val, k_val, temp_val, humidity_val, ph_val, rainfall_val, top_k=3
            )

            st.markdown("---")
            st.markdown("### 🏆 AI Crop Recommendations")

            rec_col1, rec_col2, rec_col3 = st.columns(3)
            top_rec = results["recommendations"][0]
            sec_rec = results["recommendations"][1]
            thi_rec = results["recommendations"][2]

            with rec_col1:
                icon1 = CROP_ICONS.get(top_rec["crop"], "🌱")
                st.markdown(
                    f"""
                    <div class="glass-card-highlight">
                        <div style="font-size: 0.85rem; color: #34D399; font-weight: 700; text-transform: uppercase;">🥇 Primary Match</div>
                        <div style="font-size: 2.2rem; margin: 8px 0;">{icon1} <b>{top_rec['crop']}</b></div>
                        <div><span class="confidence-badge">Confidence: {top_rec['confidence']}%</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with rec_col2:
                icon2 = CROP_ICONS.get(sec_rec["crop"], "🌱")
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div style="font-size: 0.85rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">🥈 Alternative #1</div>
                        <div style="font-size: 1.8rem; margin: 8px 0;">{icon2} <b>{sec_rec['crop']}</b></div>
                        <div><span class="confidence-badge">Confidence: {sec_rec['confidence']}%</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with rec_col3:
                icon3 = CROP_ICONS.get(thi_rec["crop"], "🌱")
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div style="font-size: 0.85rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">🥉 Alternative #2</div>
                        <div style="font-size: 1.8rem; margin: 8px 0;">{icon3} <b>{thi_rec['crop']}</b></div>
                        <div><span class="confidence-badge">Confidence: {thi_rec['confidence']}%</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # SHAP EXPLAINABILITY SECTION
            st.markdown("### 🔍 Explainable AI (SHAP) Factor Attribution")
            st.info(f"💡 {results['explanation']}")

            shap_col1, shap_col2 = st.columns([3, 2])

            with shap_col1:
                # Plotly Horizontal SHAP Contribution Bar Chart
                contrib_df = pd.DataFrame(results["feature_contributions"])
                contrib_df["color"] = contrib_df["shap_value"].apply(
                    lambda v: "#10B981" if v >= 0 else "#EF4444"
                )
                contrib_df["direction"] = contrib_df["shap_value"].apply(
                    lambda v: "Favorable (Pushed towards crop)" if v >= 0 else "Limiting (Penalized confidence)"
                )

                fig_shap = px.bar(
                    contrib_df,
                    x="shap_value",
                    y="label",
                    orientation="h",
                    color="direction",
                    color_discrete_map={
                        "Favorable (Pushed towards crop)": "#10B981",
                        "Limiting (Penalized confidence)": "#EF4444",
                    },
                    title=f"SHAP Feature Importance for '{top_rec['crop']}' Prediction",
                    labels={"shap_value": "SHAP Impact Score", "label": "Feature"},
                    text=contrib_df["shap_value"].apply(lambda v: f"{v:+.3f}"),
                )
                fig_shap.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(30, 41, 59, 0.4)",
                    font=dict(family="Plus Jakarta Sans", color="#F8FAFC"),
                    height=380,
                    margin=dict(l=20, r=20, t=50, b=20),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_shap, use_container_width=True)

            with shap_col2:
                # Radar Chart: User Input vs Optimal Crop Requirements
                engine = get_engine()
                crop_prof = engine.crop_profiles.get(top_rec["crop"], {})
                if crop_prof:
                    radar_feats = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
                    user_vals = [n_val, p_val, k_val, temp_val, humidity_val, ph_val, rainfall_val]
                    opt_vals = [crop_prof[f]["mean"] for f in radar_feats]

                    # Scale to percentage of optimal
                    pct_user = [min(round((u / max(o, 1e-3)) * 100, 1), 180) for u, o in zip(user_vals, opt_vals)]
                    pct_opt = [100.0] * len(radar_feats)

                    fig_radar = go.Figure()
                    fig_radar.add_trace(
                        go.Scatterpolar(
                            r=pct_user,
                            theta=[engine.feature_labels[f] for f in radar_feats],
                            fill="toself",
                            name="Your Field Data",
                            line=dict(color="#38BDF8", width=2),
                            fillcolor="rgba(56, 189, 248, 0.3)",
                        )
                    )
                    fig_radar.add_trace(
                        go.Scatterpolar(
                            r=pct_opt,
                            theta=[engine.feature_labels[f] for f in radar_feats],
                            name=f"Optimal for {top_rec['crop']}",
                            line=dict(color="#10B981", dash="dash"),
                        )
                    )
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 180], color="#94A3B8"),
                            bgcolor="rgba(30, 41, 59, 0.4)",
                        ),
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        title=f"Field Alignment vs. {top_rec['crop']} Benchmark (%)",
                        font=dict(family="Plus Jakarta Sans", color="#F8FAFC"),
                        height=380,
                        margin=dict(l=40, r=40, t=50, b=30),
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

            # AGRONOMIC & FERTILIZER ADVISORY
            st.markdown("### 🚜 Farmer Actionable Advisory & Soil Management")
            advisories = results["advisory"]
            if advisories:
                adv_cols = st.columns(min(len(advisories), 3))
                for idx, adv in enumerate(advisories):
                    col_idx = idx % min(len(advisories), 3)
                    status_color = (
                        "#34D399"
                        if adv["status"] == "Optimal"
                        else "#FBBF24"
                        if "Supplementary" in adv["status"]
                        else "#F87171"
                    )
                    with adv_cols[col_idx]:
                        st.markdown(
                            f"""
                            <div class="glass-card" style="border-left: 4px solid {status_color};">
                                <div style="font-size: 0.85rem; color: {status_color}; font-weight: 700;">{adv['status']}</div>
                                <div style="font-size: 1.05rem; font-weight: 700; margin: 4px 0;">{adv['category']}</div>
                                <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.4;">{adv['recommendation']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        except Exception as e:
            st.error(f"Error computing prediction: {e}")

# TAB 2: MODEL PERFORMANCE & BENCHMARKS
with tab_analytics:
    st.markdown("### 📊 Model Comparison & Validation Metrics")

    engine = get_engine()
    meta = engine.metadata

    if meta and "benchmarks" in meta:
        benchmarks = meta["benchmarks"]
        bench_df = (
            pd.DataFrame(benchmarks)
            .T.reset_index()
            .rename(columns={"index": "Model"})
        )
        bench_df["accuracy_pct"] = bench_df["accuracy"] * 100

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("XGBoost Accuracy", f"{benchmarks['XGBoost']['accuracy']*100:.2f}%", "+0.45% vs RF")
        with m_col2:
            st.metric("XGBoost F1-Score", f"{benchmarks['XGBoost']['f1_score']:.4f}")
        with m_col3:
            st.metric("Random Forest Accuracy", f"{benchmarks['Random Forest']['accuracy']*100:.2f}%")
        with m_col4:
            st.metric("Decision Tree Accuracy", f"{benchmarks['Decision Tree']['accuracy']*100:.2f}%")

        bench_col1, bench_col2 = st.columns([1, 1])

        with bench_col1:
            fig_bench = px.bar(
                bench_df,
                x="Model",
                y="accuracy_pct",
                color="Model",
                text=bench_df["accuracy_pct"].apply(lambda a: f"{a:.2f}%"),
                title="Model Accuracy Comparison on 20% Stratified Test Split",
                color_discrete_sequence=["#10B981", "#38BDF8", "#F59E0B", "#A78BFA"],
            )
            fig_bench.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30, 41, 59, 0.4)",
                yaxis=dict(range=[85, 101], title="Accuracy (%)"),
                font=dict(family="Plus Jakarta Sans", color="#F8FAFC"),
                height=380,
            )
            st.plotly_chart(fig_bench, use_container_width=True)

        with bench_col2:
            # Global Feature Importance Chart
            if "feature_importances" in meta:
                feat_imp = pd.DataFrame(
                    list(meta["feature_importances"].items()),
                    columns=["Feature", "Importance"],
                ).sort_values("Importance", ascending=True)
                feat_imp["Feature_Name"] = feat_imp["Feature"].map(engine.feature_labels)

                fig_imp = px.bar(
                    feat_imp,
                    x="Importance",
                    y="Feature_Name",
                    orientation="h",
                    title="Global Feature Importance across all 2,200 Samples",
                    color="Importance",
                    color_continuous_scale="Viridis",
                )
                fig_imp.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(30, 41, 59, 0.4)",
                    font=dict(family="Plus Jakarta Sans", color="#F8FAFC"),
                    height=380,
                )
                st.plotly_chart(fig_imp, use_container_width=True)

        # Confusion Matrix
        if "confusion_matrix" in meta and "classes" in meta:
            cm = np.array(meta["confusion_matrix"])
            classes = meta["classes"]

            fig_cm = px.imshow(
                cm,
                x=classes,
                y=classes,
                color_continuous_scale="Mint",
                title="Multi-Class Confusion Matrix (22 Crops)",
                labels=dict(x="Predicted Crop", y="Actual Crop", color="Count"),
            )
            fig_cm.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30, 41, 59, 0.4)",
                font=dict(family="Plus Jakarta Sans", color="#F8FAFC"),
                height=650,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

# TAB 3: CROP KNOWLEDGE BASE
with tab_crops:
    st.markdown("### 📖 Agricultural Crop Benchmark Directory")
    engine = get_engine()
    profiles = engine.crop_profiles

    if profiles:
        crop_list = sorted(list(profiles.keys()))
        selected_crop = st.selectbox("Select a Crop to view physiological requirements:", crop_list)

        if selected_crop:
            p = profiles[selected_crop]
            icon = CROP_ICONS.get(selected_crop, "🌱")

            st.markdown(f"#### {icon} **{selected_crop}** Growth Benchmarks")

            k_col1, k_col2, k_col3, k_col4 = st.columns(4)
            with k_col1:
                st.metric("Optimal Nitrogen (N)", f"{p['N']['mean']:.1f} ± {p['N']['std']:.1f}")
                st.metric("Temperature Range", f"{p['temperature']['min']:.1f}°C - {p['temperature']['max']:.1f}°C")
            with k_col2:
                st.metric("Optimal Phosphorus (P)", f"{p['P']['mean']:.1f} ± {p['P']['std']:.1f}")
                st.metric("Humidity Range", f"{p['humidity']['min']:.1f}% - {p['humidity']['max']:.1f}%")
            with k_col3:
                st.metric("Optimal Potassium (K)", f"{p['K']['mean']:.1f} ± {p['K']['std']:.1f}")
                st.metric("Soil pH Range", f"{p['ph']['min']:.2f} - {p['ph']['max']:.2f}")
            with k_col4:
                st.metric("Rainfall Range", f"{p['rainfall']['min']:.0f} - {p['rainfall']['max']:.0f} mm")
                st.metric("Avg Water Requirement", f"{p['rainfall']['mean']:.0f} mm")
