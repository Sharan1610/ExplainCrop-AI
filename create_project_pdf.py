"""
CropMind AI - Multi-Page PDF Document Generator
Compiles all project visuals, architecture diagrams, requirements, and acceptance testing dashboards
into a single, high-quality, shareable PDF document.
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

PDF_NAME = "CropMind_AI_Project_Visuals_Report.pdf"
WORKSPACE_PDF_PATH = os.path.join(WORKSPACE_DIR, PDF_NAME)
ARTIFACT_PDF_PATH = os.path.join(ARTIFACT_DIR, PDF_NAME)

# Palette
PRIMARY_DARK = "#0F172A"
SECONDARY_DARK = "#1E293B"
ACCENT_EMERALD = "#059669"
ACCENT_BLUE = "#2563EB"
ACCENT_PURPLE = "#7C3AED"
ACCENT_CYAN = "#0891B2"
BG_LIGHT = "#F8FAFC"
CARD_BG = "#FFFFFF"
BORDER_COLOR = "#CBD5E1"
TEXT_MUTED = "#64748B"

def generate_cover_page():
    cover_path = os.path.join(WORKSPACE_IMG_DIR, "00_cover_page.png")
    
    fig = plt.figure(figsize=(16, 10), facecolor=PRIMARY_DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # Background accent glow / container
    glow_box = FancyBboxPatch((0.04, 0.05), 0.92, 0.90, boxstyle="round,pad=0.01,rounding_size=0.015",
                              facecolor=SECONDARY_DARK, edgecolor=ACCENT_EMERALD, linewidth=2.0)
    ax.add_patch(glow_box)

    # Top Tag
    tag = FancyBboxPatch((0.08, 0.85), 0.32, 0.045, boxstyle="round,pad=0.005,rounding_size=0.008",
                         facecolor=ACCENT_EMERALD, edgecolor='none')
    ax.add_patch(tag)
    ax.text(0.24, 0.872, "PRECISION AGRICULTURE & EXPLAINABLE AI (XAI)", fontsize=9, fontweight='bold', color='white', ha='center', va='center')

    # Main Title
    ax.text(0.08, 0.78, "CropMind AI", fontsize=38, fontweight='bold', color='white', va='center')
    ax.text(0.08, 0.71, "Climate-Resilient Multi-Modal Crop Recommendation System", 
            fontsize=20, fontweight='bold', color='#38BDF8', va='center')
    ax.text(0.08, 0.655, "System Architecture, Research Literature Survey, Requirements & Module 1 Acceptance Test Report", 
            fontsize=12, color='#94A3B8', va='center')

    # Divider line
    ax.plot([0.08, 0.92], [0.62, 0.62], color='#334155', linewidth=1.5)

    # Table of Contents Grid Cards
    toc_items = [
        {"num": "01", "title": "Literature Survey & Comparative Analysis", "sub": "Evaluation of prior works, ML baselines, and research gaps", "col": ACCENT_BLUE},
        {"num": "02", "title": "Proposed Work System Block Diagram", "sub": "5-tier end-to-end architecture from IoT/Weather to Streamlit console", "col": ACCENT_EMERALD},
        {"num": "03", "title": "Proposed System Modules Breakdown", "sub": "Detailed functional specs for Data, ML, XAI, Weather & Full-stack", "col": ACCENT_CYAN},
        {"num": "04", "title": "Software Requirements Specification", "sub": "Runtime, Python 3.11+, XGBoost, TreeSHAP, FastAPI & SQLite", "col": ACCENT_PURPLE},
        {"num": "05", "title": "Hardware Requirements Specification", "sub": "Minimum operational specs vs. Recommended production hardware", "col": "#D97706"},
        {"num": "06", "title": "Module 1 Outcome (Acceptance Test)", "sub": "98.86% Accuracy, 22-class confusion matrix & 12 feature gains", "col": ACCENT_EMERALD}
    ]

    card_w = 0.40
    card_h = 0.12
    coords = [
        (0.08, 0.47), (0.52, 0.47),
        (0.08, 0.32), (0.52, 0.32),
        (0.08, 0.17), (0.52, 0.17)
    ]

    for i, item in enumerate(toc_items):
        cx, cy = coords[i]
        c_box = FancyBboxPatch((cx, cy), card_w, card_h, boxstyle="round,pad=0.005,rounding_size=0.008",
                               facecolor="#0F172A", edgecolor=item["col"], linewidth=1.2)
        ax.add_patch(c_box)

        # Number pill
        num_pill = FancyBboxPatch((cx + 0.015, cy + card_h - 0.045), 0.05, 0.035,
                                  boxstyle="round,pad=0.002,rounding_size=0.004",
                                  facecolor=item["col"], edgecolor='none')
        ax.add_patch(num_pill)
        ax.text(cx + 0.04, cy + card_h - 0.028, item["num"], fontsize=9.5, fontweight='bold', color='white', ha='center', va='center')

        ax.text(cx + 0.075, cy + card_h - 0.028, item["title"], fontsize=10, fontweight='bold', color='white', va='center')
        ax.text(cx + 0.015, cy + 0.035, item["sub"], fontsize=8, color='#94A3B8', va='center')

    # Footer Metadata
    ax.text(0.08, 0.09, "Author / Project Lead: Vinesh Raja (@Vinesh-Raja07)  |  Model: XGBoost Hist + TreeSHAP  |  Version: 1.0.0", 
            fontsize=9.5, color='#64748B', va='center')
    ax.text(0.92, 0.09, "Status: Production Ready (PASSED)", fontsize=9.5, fontweight='bold', color=ACCENT_EMERALD, ha='right', va='center')

    fig.savefig(cover_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Cover page generated: {cover_path}")
    return cover_path

def compile_pdf():
    cover_path = generate_cover_page()
    
    image_files = [
        cover_path,
        os.path.join(WORKSPACE_IMG_DIR, "01_literature_survey.png"),
        os.path.join(WORKSPACE_IMG_DIR, "02_proposed_work_block_diagram.png"),
        os.path.join(WORKSPACE_IMG_DIR, "03_proposed_modules.png"),
        os.path.join(WORKSPACE_IMG_DIR, "04_software_requirements.png"),
        os.path.join(WORKSPACE_IMG_DIR, "05_hardware_requirements.png"),
        os.path.join(WORKSPACE_IMG_DIR, "06_module1_acceptance_test.png")
    ]

    pil_images = []
    for f in image_files:
        if os.path.exists(f):
            img = Image.open(f)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            pil_images.append(img)
            print(f"Loaded page: {os.path.basename(f)}")
        else:
            print(f"Warning: file not found {f}")

    if not pil_images:
        print("Error: No images loaded to compile PDF!")
        return

    # First image is used as the base, others appended
    first_image = pil_images[0]
    rest_images = pil_images[1:]

    # Save to Workspace
    first_image.save(WORKSPACE_PDF_PATH, save_all=True, append_images=rest_images, resolution=300.0, quality=95)
    print(f"PDF saved to workspace: {WORKSPACE_PDF_PATH}")

    # Save to Artifacts directory
    first_image.save(ARTIFACT_PDF_PATH, save_all=True, append_images=rest_images, resolution=300.0, quality=95)
    print(f"PDF saved to artifacts: {ARTIFACT_PDF_PATH}")

if __name__ == "__main__":
    compile_pdf()
