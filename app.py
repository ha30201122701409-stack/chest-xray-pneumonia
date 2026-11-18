"""
app.py — FastAPI backend
Loads medical_model.pth, runs a QA check, DenseNet121 inference and Grad-CAM.
Run:  uvicorn app:app --reload --port 8000
"""

import io
import base64
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Config ───────────────────────────────────────────────────
MODEL_PATH = "medical_model.pth"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
LAP_THRESH = 20.0


# ── Model ────────────────────────────────────────────────────
def load_model():
    """Same DenseNet121 architecture as training, with saved weights."""
    m = models.densenet121(weights=None)
    m.classifier = nn.Sequential(
        nn.Linear(m.classifier.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2),
    )
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    return m


model = load_model()

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── QA filter ────────────────────────────────────────────────
def qa_ok(gray: np.ndarray):
    """Blur / exposure check. Returns (passed, reason)."""
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap < LAP_THRESH:
        return False, f"BLURRY (lap={lap:.1f})"
    mean_px = float(np.mean(gray))
    if mean_px < 10:
        return False, "TOO_DARK"
    if mean_px > 245:
        return False, "TOO_BRIGHT"
    return True, "OK"


# ── Grad-CAM ─────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.act = None
        self.grad = None
        layer.register_forward_hook(
            lambda m, i, o: setattr(self, "act", o.detach()))
        layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grad", go[0].detach()))

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(tensor)
        logits[0, class_idx].backward()
        weights = self.grad[0].mean(dim=(1, 2))
        cam = torch.zeros(self.act.shape[2:], device=DEVICE)
        for i, w in enumerate(weights):
            cam += w * self.act[0, i]
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cv2.resize(cam.cpu().numpy(), (IMG_SIZE, IMG_SIZE))


gc = GradCAM(model, model.features.denseblock4)


def arr_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Schema ───────────────────────────────────────────────────
class Result(BaseModel):
    predicted_class: str
    confidence: float
    normal_prob: float
    pneumonia_prob: float
    qa_passed: bool
    qa_reason: str
    overlay_b64: Optional[str] = None


# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="Chest X-Ray Classifier",
    description="DenseNet121 + QA filter + Grad-CAM",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict", response_model=Result)
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Cannot read image file.")

    # QA filter
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    passed, reason = qa_ok(gray)
    if not passed:
        return Result(predicted_class="REJECTED", confidence=0,
                      normal_prob=0, pneumonia_prob=0,
                      qa_passed=False, qa_reason=reason)

    # Inference
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = tfm(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = probs.argmax().item()

    # Grad-CAM overlay
    heatmap = gc.generate(tfm(pil).unsqueeze(0).to(DEVICE), idx)
    orig = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    hmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    hmap = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig, 0.55, hmap, 0.45, 0)

    return Result(
        predicted_class=CLASS_NAMES[idx],
        confidence=round(probs[idx].item(), 4),
        normal_prob=round(probs[0].item(), 4),
        pneumonia_prob=round(probs[1].item(), 4),
        qa_passed=True,
        qa_reason="OK",
        overlay_b64=arr_to_b64(overlay),
    )
