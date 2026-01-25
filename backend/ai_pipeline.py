"""
ai_pipeline.py — LungCare Triage
==================================
Handles PyTorch model loading and lung nodule risk classification.

Responsibilities:
    1. Load the trained 3D CNN weights from best_lung_nodule_model.pth.
    2. Expose `calculate_risk_score()` — rule-based WHO threshold classifier.
    3. Expose `run_inference()` — full model inference stub for .mhd uploads.
    4. Expose `calculate_growth_rate()` — compares historical diameters.
"""

import os
import logging
from typing import Literal, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ─── Model Path ───────────────────────────────────────────────────────────────
# Resolving absolute path directly since relative paths are failing during execution
MODEL_PATH = r"/Users/anurag/College/lungCanerProject/lungcare_triage/machine_learning/models/best_lung_nodule_model.pth"

RiskLevel = Literal["Low", "Medium", "High"]


# ─── 3D CNN Architecture (RESTORED TO MATCH KAGGLE) ─────────────────────────

class NoduleNet3D(nn.Module):
    """
    Original Kaggle Architecture: Two 3D Conv layers with MaxPool, 
    flattening into two Linear layers ending in a Sigmoid.
    """
    def __init__(self):
        super(NoduleNet3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(32 * 8 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x)) 
        return x


# ─── Model Loader (singleton pattern) ─────────────────────────────────────────

_model: Optional[NoduleNet3D] = None
_device = torch.device("cpu")  # Forced to CPU for inference


def load_model() -> NoduleNet3D:
    """
    Load the PyTorch model weights once and cache globally.
    """
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        logger.error(f"[AI Pipeline] Model weights not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model weights not found at: {MODEL_PATH}")

    logger.info(f"[AI Pipeline] Loading model from {MODEL_PATH} on {_device}...")
    model = NoduleNet3D().to(_device)

    # Load state dict
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=_device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        _model = model
        logger.info("[AI Pipeline] Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"[AI Pipeline] Failed to load model weights: {e}")
        raise


# ─── Risk Score Calculator ────────────────────────────────────────────────────

def calculate_risk_score(tumor_diameter_mm: float) -> RiskLevel:
    """
    Thresholds: < 6 mm (Low), 6-8 mm (Medium), > 8 mm (High)
    """
    if tumor_diameter_mm < 6.0:
        return "Low"
    elif tumor_diameter_mm <= 8.0:
        return "Medium"
    else:
        return "High"


# ─── Full Inference Function ──────────────────────────────────────────────────

def predict_nodule(image_bytes: bytes) -> dict:
    """
    Takes an uploaded 2D image (from Streamlit), resizes to 32x32, and expands 
    into a pseudo-3D tensor (1, 1, 32, 32, 32) so the 3D CNN processes it.
    """
    from PIL import Image
    import io
    import numpy as np

    try:
        model = load_model()
    except Exception as e:
        raise ValueError("AI Model not available for inference.") from e

    # Preprocess 2D image -> pseudo 3D tensor
    try:
        # 1. Load image and convert to grayscale
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        
        # 2. Resize to 32x32
        img = img.resize((32, 32))
        
        # 3. Convert to float array [0, 1]
        arr_2d = np.array(img, dtype=np.float32) / 255.0
        
        # 4. Convert to tensor
        tensor_2d = torch.tensor(arr_2d)
        
        # 5. Expand to (32, 32, 32) by repeating the 2D slice 32 times along depth
        tensor_3d = tensor_2d.unsqueeze(0).repeat(32, 1, 1)
        
        # 6. Add batch and channel dimensions -> (1, 1, 32, 32, 32)
        tensor_input = tensor_3d.unsqueeze(0).unsqueeze(0).to(_device)
        
    except Exception as e:
        logger.error(f"[AI Pipeline] Image preprocessing failed: {e}")
        raise ValueError("Failed to preprocess image.") from e

    # Real inference
    with torch.no_grad():
        prob = model(tensor_input).item()

    predicted_class = "Malignant" if prob >= 0.5 else "Benign"
    logger.info(f"[AI Pipeline] Inference complete — prob={prob:.3f} ({predicted_class})")
    
    return {
        "malignancy_probability": round(prob, 4),
        "predicted_class": predicted_class,
        "confidence": round(prob if prob >= 0.5 else 1 - prob, 4),
    }


# ─── Growth Rate Calculator ───────────────────────────────────────────────────

def calculate_growth_rate(diameters: list[float]) -> dict:
    """
    Calculate nodule growth between the most recent two scans.
    """
    if len(diameters) < 2:
        return {
            "growth_mm"  : 0.0,
            "growth_pct" : 0.0,
            "velocity"   : "Insufficient data",
            "num_scans"  : len(diameters),
        }

    oldest, latest = diameters[0], diameters[-1]
    growth_mm  = round(latest - oldest, 2)
    growth_pct = round((growth_mm / oldest) * 100, 2) if oldest > 0 else 0.0

    if growth_pct < 5:
        velocity = "Stable"
    elif growth_pct < 25:
        velocity = "Slow Growth"
    else:
        velocity = "Rapid Growth"

    return {
        "growth_mm"  : growth_mm,
        "growth_pct" : growth_pct,
        "velocity"   : velocity,
        "num_scans"  : len(diameters),
    }