"""
MARS ROVER TERRAIN SEGMENTATION
AI-Powered Martian Surface Analysis Dashboard
Built with Streamlit & PyTorch (NASA JPL AI4Mars Dataset)
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import torch

# Ensure src modules can be imported
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from src.model import build_unet_resnet34
from src.utils import (
    TERRAIN_CLASSES,
    COLOR_MAP,
    COLOR_MAP_NORMALIZED,
    IGNORE_INDEX,
    decode_segmentation_mask,
    create_overlay,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="MARS ROVER TERRAIN SEGMENTATION",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Space-Themed CSS Styles
SPACE_THEME_CSS = """
<style>
    /* Dark Space Background */
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Header Gradient & Mars Orange Accent */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #FF6B4A, #FFA366, #FF4500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        text-transform: uppercase;
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 500;
        margin-bottom: 1.5rem;
        letter-spacing: 1px;
    }

    /* Cards & Containers */
    .stMetric, div[data-testid="stMetricValue"] {
        color: #FFA366 !important;
        font-weight: 700;
    }
    
    .hud-box {
        background: rgba(18, 24, 38, 0.85);
        border: 1px solid rgba(255, 107, 74, 0.25);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .class-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        margin-right: 6px;
        color: #000;
    }
    
    .disclaimer-box {
        background: rgba(220, 38, 38, 0.08);
        border-left: 4px solid #EF4444;
        border-radius: 4px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #FCA5A5;
        margin-top: 25px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #FF5722 0%, #D84315 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        padding: 0.55rem 1.2rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #FF7043 0%, #E64A19 100%);
        box-shadow: 0 0 15px rgba(255, 87, 34, 0.5);
    }
</style>
"""
st.markdown(SPACE_THEME_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_segmentation_model(checkpoint_path: str = "checkpoints/mars_terrain_model.pth"):
    """Loads and caches the PyTorch U-Net (ResNet-34) model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet_resnet34(num_classes=4, pretrained=False).to(device)

    chk_file = Path(checkpoint_path)
    if chk_file.exists():
        try:
            checkpoint = torch.load(chk_file, map_location=device, weights_only=False)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            st.sidebar.success(f"Loaded checkpoint: `{chk_file.name}`")
        except Exception as e:
            st.sidebar.warning(f"Could not load state_dict ({e}). Running in inference mode.")
    else:
        st.sidebar.info("Model running with initialized ResNet-34 backbone weights.")

    model.eval()
    return model, device


def preprocess_image(image: Image.Image, target_size=(256, 256)):
    """Prepares PIL image for ResNet-34 U-Net input."""
    resized = image.convert("RGB").resize(target_size, resample=Image.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0
    normalized = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, resized, np.array(resized)


def run_inference(model, device, tensor):
    """Executes model inference and returns predicted class map and confidence scores."""
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
        confidence = probs.squeeze(0).cpu().numpy()
    return preds, confidence


def compute_terrain_distribution(mask: np.ndarray):
    """Calculates pixel percentages for each of the four terrain categories."""
    total_pixels = mask.size
    distribution = {}
    for cls_id, name in TERRAIN_CLASSES.items():
        count = np.sum(mask == cls_id)
        pct = (count / total_pixels) * 100.0
        distribution[name] = {
            "id": cls_id,
            "count": int(count),
            "percentage": float(pct),
            "color_rgb": COLOR_MAP[cls_id],
            "hex": f"#{COLOR_MAP[cls_id][0]:02x}{COLOR_MAP[cls_id][1]:02x}{COLOR_MAP[cls_id][2]:02x}",
        }
    return distribution


def render_distribution_chart(distribution: dict):
    """Renders a sleek horizontal bar chart of terrain distribution."""
    labels = list(distribution.keys())
    percentages = [distribution[l]["percentage"] for l in labels]
    colors = [f"#{distribution[l]['color_rgb'][0]:02x}{distribution[l]['color_rgb'][1]:02x}{distribution[l]['color_rgb'][2]:02x}" for l in labels]

    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="#111625")
    ax.set_facecolor("#111625")

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, percentages, color=colors, height=0.6, edgecolor="#2D3748")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="#E2E8F0", fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Surface Coverage Percentage (%)", color="#94A3B8", fontsize=9)
    ax.set_xlim(0, 100)
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#2D3748")

    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            ha="left",
            color="#E2E8F0",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def main():
    # Title & Subtitle Banner
    st.markdown('<div class="main-title">MARS ROVER TERRAIN SEGMENTATION</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">AI-POWERED MARTIAN SURFACE ANALYSIS &bull; NASA JPL AI4MARS BENCHMARK &bull; CURIOSITY MASTCAM</div>',
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    st.sidebar.markdown("### 🛰️ Mission Control & Settings")
    checkpoint_input = st.sidebar.text_input(
        "Model Checkpoint Path",
        value="checkpoints/mars_terrain_model.pth",
        help="Path to trained PyTorch weights (.pth)",
    )
    overlay_alpha = st.sidebar.slider(
        "Hazard Overlay Transparency (Alpha)",
        min_value=0.1,
        max_value=0.9,
        value=0.55,
        step=0.05,
    )

    # Model loading
    model, device = load_segmentation_model(checkpoint_input)

    # Class Legend in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏷️ Martian Terrain Legend")
    st.sidebar.markdown(
        """
        <div style='line-height: 2.0;'>
            <span class='class-badge' style='background:#2ECC71;'>SOIL</span> Safe, fine Martian regolith<br>
            <span class='class-badge' style='background:#3498DB;'>BEDROCK</span> Solid rock, firm traction<br>
            <span class='class-badge' style='background:#F1C40F;'>SAND</span> Dunes/Ripples, slippage risk<br>
            <span class='class-badge' style='background:#E74C3C;'>BIG ROCK</span> Boulder, collision obstacle<br>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sample Presets from Dataset if available
    preset_dir = Path(r"D:\mars\ai4mars-dataset-merged-0.6\msl\mcam\images")
    preset_options = ["None (Upload Own Image)"]
    preset_files = []
    if preset_dir.exists():
        preset_files = sorted([f for f in os.listdir(preset_dir) if f.lower().endswith(".jpg")])[:5]
        preset_options.extend([f"Curiosity MastCam #{i+1}: {f}" for i, f in enumerate(preset_files)])

    st.markdown("### 📥 Select or Upload Mars Rover Image")
    col_input1, col_input2 = st.columns([1, 1])

    with col_input1:
        selected_preset = st.selectbox("Load Sample from AI4Mars Dataset:", preset_options)

    with col_input2:
        uploaded_file = st.file_uploader(
            "Or Upload Martian Image (JPG/PNG):",
            type=["jpg", "jpeg", "png"],
        )

    # Resolve target image
    target_image: Image.Image = None
    source_name = ""

    if uploaded_file is not None:
        target_image = Image.open(uploaded_file)
        source_name = uploaded_file.name
    elif selected_preset != "None (Upload Own Image)" and preset_files:
        chosen_fname = selected_preset.split(": ")[-1]
        target_image = Image.open(preset_dir / chosen_fname)
        source_name = chosen_fname
    elif preset_files:
        # Default load first preset
        target_image = Image.open(preset_dir / preset_files[0])
        source_name = preset_files[0]

    if target_image is None:
        st.warning("Please upload a Mars rover image or select a sample preset to proceed.")
        return

    # Processing & Inference
    with st.spinner("Processing Martian surface telemetry and running U-Net ResNet-34 segmentation..."):
        tensor, resized_pil, resized_np = preprocess_image(target_image, target_size=(256, 256))
        preds, probs = run_inference(model, device, tensor)
        distribution = compute_terrain_distribution(preds)
        rgb_mask = decode_segmentation_mask(preds)
        overlay = create_overlay(resized_np, preds, alpha=overlay_alpha)

    # Display Visual Comparison Dashboard
    st.markdown(f"#### 🔭 Telemetry Analysis: `{source_name}` (Resolution: {target_image.size[0]} × {target_image.size[1]} px)")

    tab1, tab2 = st.tabs(["📊 Multi-View Comparison", "🔍 Individual Telemetry Layers"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**1. Raw Mars Rover Image**")
            st.image(resized_pil, use_container_width=True)
        with col2:
            st.markdown("**2. Predicted Segmentation Mask**")
            st.image(rgb_mask, use_container_width=True)
        with col3:
            st.markdown("**3. Terrain Hazard Overlay**")
            st.image(overlay, use_container_width=True)

    with tab2:
        view_option = st.radio("Select View:", ["Original Image", "Segmentation Mask", "Hazard Overlay"], horizontal=True)
        if view_option == "Original Image":
            st.image(resized_pil, width=700)
        elif view_option == "Segmentation Mask":
            st.image(rgb_mask, width=700)
        else:
            st.image(overlay, width=700)

    # Terrain Distribution Section
    st.markdown("---")
    st.markdown("### 📈 Martian Surface Composition & Distribution")

    stat_cols = st.columns(4)
    stat_cols[0].metric("🌿 Soil (Low Hazard)", f"{distribution['Soil']['percentage']:.1f}%", f"{distribution['Soil']['count']:,} px")
    stat_cols[1].metric("🪨 Bedrock (High Traction)", f"{distribution['Bedrock']['percentage']:.1f}%", f"{distribution['Bedrock']['count']:,} px")
    stat_cols[2].metric("⚠️ Sand (Slip Hazard)", f"{distribution['Sand']['percentage']:.1f}%", f"{distribution['Sand']['count']:,} px")
    stat_cols[3].metric("🚨 Big Rock (Obstacle)", f"{distribution['Big Rock']['percentage']:.1f}%", f"{distribution['Big Rock']['count']:,} px")

    # Render Chart & Traversability Advisory
    chart_col, advice_col = st.columns([1.2, 1])

    with chart_col:
        chart_fig = render_distribution_chart(distribution)
        st.pyplot(chart_fig)

    with advice_col:
        st.markdown('<div class="hud-box">', unsafe_allow_html=True)
        st.markdown("**🤖 Autonomy & Traversability Assessment:**")
        sand_pct = distribution["Sand"]["percentage"]
        rock_pct = distribution["Big Rock"]["percentage"]
        bedrock_pct = distribution["Bedrock"]["percentage"]

        if rock_pct > 15.0:
            st.error("⚠️ **HIGH COLLISION HAZARD**: Dense boulder field detected in field of view. AutoNav path replanning advised.")
        elif sand_pct > 40.0:
            st.warning("⚠️ **SLIPPAGE / ENTRAPMENT WARNING**: Significant loose sand dunes detected. Monitor wheel sinkage kinematics closely.")
        elif bedrock_pct > 50.0:
            st.success("✅ **OPTIMAL TRACTION TERRAIN**: Extensive solid bedrock detected. Ideal for efficient drive traverses.")
        else:
            st.info("ℹ️ **STANDARD TERRAIN MIX**: Nominal Martian regolith and mixed surface. Proceed with standard AutoNav drive parameters.")

        st.caption(
            f"Execution Device: `{device.type.upper()}` | Encoder: `ResNet-34` | Input Resolution: `256 × 256`"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Safety Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ OPERATIONAL & SAFETY DISCLAIMER:</strong><br>
            This deep learning semantic segmentation model serves as an informative 2D visual classification tool. 
            <strong>Semantic segmentation alone does NOT guarantee rover safety or successful traversability.</strong> 
            Flight-proven rover navigation systems (such as NASA's AutoNav on Curiosity and Perseverance) require 
            multi-modal sensor fusion incorporating dense 3D stereo point clouds, Digital Elevation Models (DEMs), 
            ground clearance geometry, localized surface slope angles, wheel torque telemetry, and real-time visual odometry slip detection.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
