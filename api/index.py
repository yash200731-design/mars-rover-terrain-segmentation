"""
Vercel Serverless Function: Mars Rover Terrain Segmentation API
FastAPI backend optimized for Vercel Serverless Runtime
"""

import os
import io
import base64
import json
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests

app = FastAPI(
    title="Mars Rover Terrain Segmentation API",
    description="Serverless Martian surface analysis powered by NASA JPL AI4Mars schema",
    version="1.0.0",
)

# Enable CORS for web client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4 Core AI4Mars terrain classes
TERRAIN_CLASSES = {
    0: "Soil",
    1: "Bedrock",
    2: "Sand",
    3: "Big Rock",
}

# Standard high-contrast visualization colors
COLOR_MAP = {
    0: (46, 204, 113),   # Green (Soil)
    1: (52, 152, 219),   # Blue (Bedrock)
    2: (241, 196, 15),   # Yellow (Sand)
    3: (231, 76, 60),    # Red (Big Rock)
    255: (35, 35, 35),   # Dark Gray (Ignored)
}


def encode_pil_to_base64(img: Image.Image, fmt="PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=fmt)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{img_b64}"


def run_local_edge_segmentation(image: Image.Image, target_size=(256, 256)) -> np.ndarray:
    """
    Fast edge-based Martian terrain segmentation algorithm.
    Used when running on Vercel's serverless environment without a remote GPU backend.
    Analyzes chrominance, luminance gradient, and spatial texture characteristics.
    """
    resized = image.convert("RGB").resize(target_size, resample=Image.BILINEAR)
    img_np = np.array(resized, dtype=np.float32)

    # Extract RGB channels
    r = img_np[:, :, 0]
    g = img_np[:, :, 1]
    b = img_np[:, :, 2]
    brightness = 0.299 * r + 0.587 * g + 0.114 * b

    # Local texture gradient (spatial Laplacian approximation)
    grad_y = np.abs(np.diff(brightness, axis=0, prepend=brightness[0:1, :]))
    grad_x = np.abs(np.diff(brightness, axis=1, prepend=brightness[:, 0:1]))
    texture = grad_y + grad_x

    # Martian Redness Index
    redness = (r - g) / (r + g + 1e-5)

    # Initialize mask
    h, w = target_size
    mask = np.zeros((h, w), dtype=np.uint8)

    # Classification heuristics based on AI4Mars spectral properties:
    # 1. Sand (Class 2): High brightness, smooth texture (low gradient), warm orange/yellow hue
    sand_mask = (brightness > np.percentile(brightness, 55)) & (texture < np.percentile(texture, 40)) & (redness > 0.15)
    
    # 2. Big Rocks (Class 3): High local texture, sharp shadow edges, elevated intensity contrast
    rock_mask = (texture > np.percentile(texture, 82)) & (brightness < np.percentile(brightness, 85))

    # 3. Bedrock (Class 1): Moderate to high brightness, distinct fractured texture, lower redness (bluer/grayer rock tone)
    bedrock_mask = (brightness > np.percentile(brightness, 40)) & (texture >= np.percentile(texture, 40)) & (redness <= 0.18) & (~rock_mask)

    # 4. Soil (Class 0): Nominal background regolith
    mask[:] = 0  # Soil default
    mask[bedrock_mask] = 1
    mask[sand_mask] = 2
    mask[rock_mask] = 3

    return mask, resized


def decode_mask_to_rgb(mask_2d: np.ndarray) -> np.ndarray:
    h, w = mask_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, col in COLOR_MAP.items():
        rgb[mask_2d == cid] = col
    return rgb


def create_overlay(image_rgb: np.ndarray, mask_2d: np.ndarray, alpha=0.55) -> np.ndarray:
    rgb_mask = decode_mask_to_rgb(mask_2d)
    valid = (mask_2d != 255)[:, :, np.newaxis]
    blended = np.where(valid, (1 - alpha) * image_rgb + alpha * rgb_mask, image_rgb)
    return np.clip(blended, 0, 255).astype(np.uint8)


@app.get("/api")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Mars Rover Terrain Segmentation API",
        "platform": "Vercel Serverless Function",
        "remote_backend_configured": bool(os.getenv("MODEL_BACKEND_URL")),
    }


@app.post("/api/segment")
async def segment_image(
    file: UploadFile = File(...),
    alpha: float = Form(0.55),
):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Check if user has connected a remote PyTorch backend (e.g. Hugging Face Space / Render)
    remote_backend = os.getenv("MODEL_BACKEND_URL")
    if remote_backend:
        try:
            resp = requests.post(
                f"{remote_backend.rstrip('/')}/predict",
                files={"file": (file.filename, contents, file.content_type)},
                data={"alpha": alpha},
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"Remote backend failed ({e}). Falling back to local edge segmentation engine.")

    # Local edge segmentation
    mask, resized_img = run_local_edge_segmentation(pil_img, target_size=(256, 256))
    img_np = np.array(resized_img)
    overlay_np = create_overlay(img_np, mask, alpha=alpha)

    # Class percentages
    total_pixels = mask.size
    distribution = {}
    for cid, cname in TERRAIN_CLASSES.items():
        cnt = int(np.sum(mask == cid))
        pct = float((cnt / total_pixels) * 100.0)
        distribution[cname] = {
            "id": cid,
            "count": cnt,
            "percentage": round(pct, 2),
            "color_rgb": list(COLOR_MAP[cid]),
            "color_hex": f"#{COLOR_MAP[cid][0]:02x}{COLOR_MAP[cid][1]:02x}{COLOR_MAP[cid][2]:02x}",
        }

    # Traversability Advisory
    sand_pct = distribution["Sand"]["percentage"]
    rock_pct = distribution["Big Rock"]["percentage"]
    bedrock_pct = distribution["Bedrock"]["percentage"]

    if rock_pct > 15.0:
        advisory_level = "CRITICAL"
        advisory_title = "HIGH COLLISION HAZARD"
        advisory_msg = "Dense boulder field detected in field of view. AutoNav path replanning advised to prevent undercarriage impact."
    elif sand_pct > 40.0:
        advisory_level = "WARNING"
        advisory_title = "SLIPPAGE & ENTRAPMENT HAZARD"
        advisory_msg = "Significant loose sand dune / ripple field detected. Monitor wheel sinkage kinematics closely."
    elif bedrock_pct > 50.0:
        advisory_level = "OPTIMAL"
        advisory_title = "FIRM TRACTION TERRAIN"
        advisory_msg = "Extensive solid bedrock detected. Nominal grip and zero sinkage for efficient drive traverses."
    else:
        advisory_level = "NOMINAL"
        advisory_title = "STANDARD MARTIAN SURFACE"
        advisory_msg = "Nominal mix of regolith soil and bedrock. Standard drive parameters approved."

    # Convert results to Base64
    mask_rgb_pil = Image.fromarray(decode_mask_to_rgb(mask))
    overlay_pil = Image.fromarray(overlay_np)

    return {
        "status": "success",
        "engine": "AI4Mars Edge Analysis",
        "image_size": [pil_img.size[0], pil_img.size[1]],
        "processed_size": [256, 256],
        "distribution": distribution,
        "advisory": {
            "level": advisory_level,
            "title": advisory_title,
            "message": advisory_msg,
        },
        "images": {
            "original": encode_pil_to_base64(resized_img, fmt="JPEG"),
            "mask": encode_pil_to_base64(mask_rgb_pil, fmt="PNG"),
            "overlay": encode_pil_to_base64(overlay_pil, fmt="PNG"),
        },
    }
