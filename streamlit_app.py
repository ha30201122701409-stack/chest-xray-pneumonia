"""
streamlit_app.py — Clinician portal
Upload an X-Ray -> prediction + Grad-CAM overlay.
Run:  streamlit run streamlit_app.py   (API must be running on :8000)
"""

import io
import base64
import json

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="X-Ray Classifier", page_icon="🫁", layout="wide")

API_URL = "http://localhost:8000/predict"


def b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def prob_bar(label, value, color):
    pct = int(value * 100)
    st.markdown(
        f"""
        <div style="margin-bottom:8px">
          <span style="color:#AAAAAA">{label}</span>
          <div style="background:#1A1D23;border-radius:6px;height:24px;margin-top:4px">
            <div style="width:{pct}%;background:{color};height:24px;border-radius:6px;
                        text-align:right;padding-right:8px;line-height:24px;
                        color:white;font-weight:bold;font-size:13px">{pct}%</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.markdown(
        "**Model**: DenseNet121  \n"
        "**Task**: NORMAL vs PNEUMONIA  \n"
        "**XAI**: Grad-CAM  \n"
        "**QA**: Laplacian blur filter"
    )
    st.divider()
    st.markdown("🔴 **Red / Yellow** = model focused here  \n"
                "🔵 **Blue** = model ignored here")
    st.divider()
    try:
        with open("eval_metrics.json") as f:
            m = json.load(f)
        st.header("📊 Test Set Performance")
        st.metric("Accuracy", f"{m['accuracy'] * 100:.1f}%")
        st.metric("Sensitivity", f"{m['sensitivity'] * 100:.1f}%")
        st.metric("Specificity", f"{m['specificity'] * 100:.1f}%")
        st.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    except FileNotFoundError:
        st.info("Run the training notebook first to generate eval_metrics.json.")
    st.caption("⚠️ Research only. Not a medical device.")

# ── Main ─────────────────────────────────────────────────────
st.title("🫁 Chest X-Ray Pneumonia Classifier")
st.markdown(
    "Upload a chest X-Ray to get an AI prediction with a **Grad-CAM heatmap** "
    "showing which regions influenced the decision."
)
st.divider()

uploaded = st.file_uploader("📂 Upload Chest X-Ray (.jpg / .png)",
                            type=["jpg", "jpeg", "png"])

if uploaded:
    raw_bytes = uploaded.read()
    orig_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    with st.spinner("🔍 Analysing image ..."):
        try:
            res = requests.post(
                API_URL,
                files={"file": ("xray.jpg", raw_bytes, "image/jpeg")},
                timeout=60,
            ).json()
        except requests.exceptions.ConnectionError:
            st.error("❌ API not running. Start it with:\n"
                     "```\nuvicorn app:app --reload --port 8000\n```")
            st.stop()

    if not res["qa_passed"]:
        st.error(f"🚫 Image rejected by QA filter: `{res['qa_reason']}`")
        st.info("Please upload a clear, well-exposed chest X-Ray.")
        st.stop()

    st.success("✅ Image passed QA validation")

    pred = res["predicted_class"]
    bg = "#1A4731" if pred == "NORMAL" else "#4A1A1A"
    tc = "#2ECC71" if pred == "NORMAL" else "#E74C3C"

    st.subheader("🩺 Result")
    st.markdown(
        f"""
        <div style="background:{bg};border-radius:12px;padding:24px;
                    text-align:center;margin-bottom:20px">
          <div style="font-size:34px;font-weight:bold;color:{tc}">{pred}</div>
          <div style="color:#AAAAAA;margin-top:6px">
            Confidence: {res['confidence'] * 100:.1f}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    prob_bar("NORMAL", res["normal_prob"], "#2ECC71")
    prob_bar("PNEUMONIA", res["pneumonia_prob"], "#E74C3C")

    st.divider()
    st.subheader("🔬 Grad-CAM Overlay")
    st.caption("Red/yellow = high attention | Blue = low attention")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original X-Ray**")
        st.image(orig_img, use_container_width=True)
    with col2:
        st.markdown("**Grad-CAM Overlay**")
        st.image(b64_to_image(res["overlay_b64"]), use_container_width=True)

    with st.expander("🛠️ Raw API response"):
        st.json({k: v for k, v in res.items() if k != "overlay_b64"})
