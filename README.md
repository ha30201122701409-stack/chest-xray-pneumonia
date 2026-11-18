# 🫁 Chest X-Ray Pneumonia Classifier (DenseNet121 + Grad-CAM)

End-to-end medical computer vision pipeline that classifies chest X-rays as **NORMAL** or **PNEUMONIA**, explains each prediction with **Grad-CAM** heatmaps, and serves it through a **FastAPI** backend and a **Streamlit** clinician portal.

> ⚠️ Research / educational project. Not a medical device.

## Features

- **Pre-inference QA filter** (OpenCV): rejects blurry, too-dark or over-exposed images using Laplacian variance and brightness checks.
- **CLAHE** contrast enhancement for visual EDA.
- **Imbalance handling**: inverse-frequency class weights in the loss + `WeightedRandomSampler`.
- **Transfer learning**: ImageNet-pretrained DenseNet121, only `denseblock4` + a custom head are fine-tuned.
- **Training**: Adam, `ReduceLROnPlateau`, gradient clipping, label smoothing, early stopping.
- **Clinical metrics**: Sensitivity, Specificity, F1, ROC-AUC, confusion matrix.
- **Explainability**: custom Grad-CAM implemented with PyTorch forward/backward hooks.
- **Deployment**: FastAPI `/predict` endpoint + Streamlit UI showing the X-ray next to its Grad-CAM overlay.

## Project structure

```
chest-xray-pneumonia/
├── notebooks/
│   ├── medical_training_pipeline.ipynb   # data prep, training, evaluation, Grad-CAM
│   └── medical_app_deployment.ipynb      # writes app.py / streamlit_app.py / requirements.txt
├── app.py              # FastAPI backend
├── streamlit_app.py    # Streamlit frontend
├── requirements.txt
└── README.md
```

## Dataset

[Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Unzip so the folder looks like:

```
chest_xray/
├── train/{NORMAL,PNEUMONIA}
├── val/{NORMAL,PNEUMONIA}
└── test/{NORMAL,PNEUMONIA}
```

## Quick start

```bash
git clone https://github.com/<your-username>/chest-xray-pneumonia.git
cd chest-xray-pneumonia
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Train

Open `notebooks/medical_training_pipeline.ipynb`, set the 3 dataset paths in **Cell 2**, and run all cells. This produces `medical_model.pth` and `eval_metrics.json`. Copy both next to `app.py`.

### 2. Serve

```bash
# Terminal 1 — API
uvicorn app:app --reload --port 8000

# Terminal 2 — UI
streamlit run streamlit_app.py
```

- API docs: http://localhost:8000/docs
- UI: http://localhost:8501

## How it works

```
X-ray → QA filter → resize/normalize → DenseNet121 → softmax
                                            └→ Grad-CAM (denseblock4) → overlay
```

## Results

Fill in after training (see `eval_metrics.json`):

| Metric | Value |
|---|---|
| Accuracy | – |
| Sensitivity | – |
| Specificity | – |
| F1 | – |
| ROC-AUC | – |

## Tech stack

PyTorch · torchvision · OpenCV · scikit-learn · FastAPI · Streamlit

## Disclaimer

For research and educational purposes only. Predictions must not be used for clinical decisions.
