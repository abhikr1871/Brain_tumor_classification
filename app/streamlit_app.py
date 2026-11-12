"""
Streamlit Web Application: Brain Tumor Classification & Explainability System.
Allows interactive MRI upload, classical IPCV preprocessing (skull-stripping,
homomorphic filtering, bilateral denoising, gamma, CLAHE, edge extraction),
morphological tumor segmentation, multi-model inference with TTA, Grad-CAM XAI,
clinical confidence guardrails, downloadable HTML reports, and benchmark inspection.
"""

import base64
import html
import io
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    CHECKPOINTS_DIR,
    CLASSES,
    DATA_RAW,
    DEVICE,
    FIGURES_DIR,
    IMG_SIZE,
    REPORTS_DIR,
)
from src.explainability.gradcam import GradCAM, overlay_cam_on_image
from src.image_processing.enhancement import (
    adjust_gamma,
    apply_bilateral_filter,
    apply_clahe,
    apply_homomorphic_filter,
    extract_edge_map,
)
from src.image_processing.feature_extraction import (
    extract_all_classical_features,
    extract_glcm_features,
    extract_intensity_statistics,
)
from src.image_processing.segmentation import (
    draw_segmentation_overlay,
    extract_morphological_features,
    segment_tumor_region,
)
from src.image_processing.skull_stripping import crop_brain_contour
from src.models.custom_cnn import BrainTumorCNN
from src.models.transfer_models import build_transfer_model

st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="wide", page_icon="🧠")

MODEL_SPECS = {
    "ResNet18 (Transfer)": ("resnet18", lambda: build_transfer_model("resnet18", num_classes=len(CLASSES), pretrained=False)),
    "ResNet50 (Transfer)": ("resnet50", lambda: build_transfer_model("resnet50", num_classes=len(CLASSES), pretrained=False)),
    "DenseNet121 (Transfer)": ("densenet121", lambda: build_transfer_model("densenet121", num_classes=len(CLASSES), pretrained=False)),
    "EfficientNet-B0 (Transfer)": ("efficientnet_b0", lambda: build_transfer_model("efficientnet_b0", num_classes=len(CLASSES), pretrained=False)),
    "Custom CNN": ("custom_cnn", lambda: BrainTumorCNN(num_classes=len(CLASSES))),
}

CLASSICAL_PKLS = {
    "SVM (RBF)": CHECKPOINTS_DIR / "svm_rbf.pkl",
    "Random Forest": CHECKPOINTS_DIR / "random_forest.pkl",
}

CONFIDENCE_THRESHOLD = 70.0


def _checkpoint_exists(model_key: str) -> bool:
    return (CHECKPOINTS_DIR / f"{model_key}_best.pth").exists()


def _default_model_index() -> int:
    labels = list(MODEL_SPECS.keys())
    for idx, label in enumerate(labels):
        if _checkpoint_exists(MODEL_SPECS[label][0]):
            return idx
    return 0


@st.cache_resource
def load_dl_model(model_key: str, mtime: float) -> torch.nn.Module:
    """Loads and caches a trained PyTorch checkpoint by model key and file mtime."""
    _ = mtime
    factory = next(spec[1] for spec in MODEL_SPECS.values() if spec[0] == model_key)
    model = factory()
    checkpoint = torch.load(CHECKPOINTS_DIR / f"{model_key}_best.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_classical_models() -> Dict[str, object]:
    """Loads available scikit-learn classical ML pipelines."""
    loaded = {}
    for name, path in CLASSICAL_PKLS.items():
        if path.exists():
            with open(path, "rb") as f:
                loaded[name] = pickle.load(f)
    return loaded


def _list_sample_test_images() -> Dict[str, Path]:
    """Collects a few sample MRI paths per class from data/raw/Testing/ for quick demos."""
    samples: Dict[str, Path] = {}
    testing_dir = DATA_RAW / "Testing"
    if not testing_dir.exists():
        return samples
    for cls_name in CLASSES:
        cls_dir = testing_dir / cls_name
        if cls_dir.is_dir():
            for img_path in sorted(cls_dir.glob("*"))[:5]:
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    samples[f"{cls_name} / {img_path.name}"] = img_path
    return samples


def _preprocess_image(
    rgb_image: np.ndarray,
    enable_crop: bool,
    enable_homomorphic: bool,
    enable_bilateral: bool,
    gamma_val: float,
    enable_clahe: bool,
    clahe_clip: float,
) -> np.ndarray:
    processed = rgb_image.copy()
    if enable_crop:
        processed = crop_brain_contour(processed)
    if enable_homomorphic:
        processed = apply_homomorphic_filter(processed, color_space="RGB")
    if enable_bilateral:
        processed = apply_bilateral_filter(processed)
    if abs(gamma_val - 1.0) > 1e-3:
        processed = adjust_gamma(processed, gamma=gamma_val)
    if enable_clahe:
        processed = apply_clahe(processed, clip_limit=clahe_clip, color_space="RGB")
    return processed


def _encode_png_b64(rgb_or_gray: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb_or_gray).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_html_report(
    sample_name: str,
    model_choice: str,
    use_tta: bool,
    pred_class: str,
    confidence: float,
    probs: np.ndarray,
    images_rgb: Dict[str, np.ndarray],
    descriptors: Dict[str, float],
) -> str:
    """Builds a self-contained HTML diagnostic report for download."""
    status_badge = (
        f"LOW CONFIDENCE (&lt;{CONFIDENCE_THRESHOLD:.0f}%) - REVIEW RECOMMENDED"
        if confidence < CONFIDENCE_THRESHOLD
        else "HIGH CONFIDENCE"
    )
    img_cards = "".join(
        f'<div class="card"><h4>{html.escape(title)}</h4>'
        f'<img src="data:image/png;base64,{_encode_png_b64(arr)}" /></div>'
        for title, arr in images_rgb.items()
    )
    prob_rows = "".join(
        f"<tr><td>{html.escape(c.capitalize())}</td><td>{p*100:.2f}%</td></tr>"
        for c, p in zip(CLASSES, probs)
    )
    desc_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v:.4f}</td></tr>"
        for k, v in descriptors.items()
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MRI Diagnostic Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 16px 0; }}
.card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 10px; width: 220px; text-align: center; }}
.card img {{ max-width: 200px; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 420px; margin-bottom: 16px; }}
th, td {{ border: 1px solid #d1d5db; padding: 6px 10px; text-align: left; }}
th {{ background: #f3f4f6; }}
.badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; background: #eef2ff; font-weight: bold; }}
</style></head><body>
<h2>Brain Tumor MRI Diagnostic Report</h2>
<p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
<strong>Sample:</strong> {html.escape(sample_name)} |
<strong>Model:</strong> {html.escape(model_choice)} (TTA={use_tta})</p>
<p class="badge">Predicted Class: {html.escape(pred_class.upper())} ({confidence:.2f}%) — {status_badge}</p>
<div class="grid">{img_cards}</div>
<h3>Class Probabilities</h3>
<table><tr><th>Class</th><th>Probability</th></tr>{prob_rows}</table>
<h3>IPCV Texture &amp; Morphological Descriptors</h3>
<table><tr><th>Descriptor</th><th>Value</th></tr>{desc_rows}</table>
<p><em>Academic IPCV diagnostic demonstration only — not a substitute for clinical radiology review.</em></p>
</body></html>"""


def _load_benchmark_rows() -> pd.DataFrame:
    """Aggregates saved JSON evaluation reports into a comparison table."""
    rows = []

    classical_path = REPORTS_DIR / "classical_ml_report.json"
    if classical_path.exists():
        with open(classical_path, encoding="utf-8") as f:
            classical_data = json.load(f)
        for name, metrics in classical_data.items():
            macro = metrics["classification_report"]["macro avg"]
            rows.append({
                "Model": name,
                "Family": "Classical ML (GLCM+LBP)",
                "Test Accuracy (%)": round(metrics["accuracy"] * 100, 2),
                "Macro Precision": round(macro["precision"], 4),
                "Macro Recall": round(macro["recall"], 4),
                "Macro F1": round(macro["f1-score"], 4),
                "Macro ROC-AUC": None,
            })

    for report_path in sorted(REPORTS_DIR.glob("*_report.json")):
        if report_path.name == "classical_ml_report.json":
            continue
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        rep = data["classification_report"]
        macro = rep["macro avg"]
        auc_val = data.get("macro_roc_auc")
        rows.append({
            "Model": data.get("model", report_path.stem.replace("_report", "")),
            "Family": "Deep Learning (TTA)" if data.get("use_tta") else "Deep Learning",
            "Test Accuracy (%)": round(rep["accuracy"] * 100, 2),
            "Macro Precision": round(macro["precision"], 4),
            "Macro Recall": round(macro["recall"], 4),
            "Macro F1": round(macro["f1-score"], 4),
            "Macro ROC-AUC": round(auc_val, 4) if auc_val is not None else None,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by="Test Accuracy (%)", ascending=False).reset_index(drop=True)


st.title("🧠 Brain Tumor Classification & Explainability System")
st.markdown(
    "*Image Processing & Computer Vision Project — Skull-Stripping, Homomorphic FFT Filtering, "
    "CLAHE, Morphological Segmentation, Texture/Shape Analysis, Transfer Learning & Grad-CAM.*"
)

# Sidebar settings
st.sidebar.header("Model & Inference")
model_labels = list(MODEL_SPECS.keys())
model_choice = st.sidebar.selectbox(
    "Select Model Architecture",
    model_labels,
    index=_default_model_index(),
    format_func=lambda label: f"{label} ({'✅ Ready' if _checkpoint_exists(MODEL_SPECS[label][0]) else 'Not trained'})",
)
use_tta = st.sidebar.checkbox("Enable Test-Time Augmentation (TTA)", value=True)

st.sidebar.subheader("Classical IPCV Pipeline")
enable_crop = st.sidebar.checkbox("Skull-Stripping / Contour Crop", value=True)
enable_homomorphic = st.sidebar.checkbox("Homomorphic FFT Filter", value=False)
enable_clahe = st.sidebar.checkbox("CLAHE Contrast Enhancement", value=True)
clahe_clip = st.sidebar.slider("CLAHE Clip Limit", 1.0, 4.0, 2.0, 0.5, disabled=not enable_clahe)
enable_bilateral = st.sidebar.checkbox("Bilateral Denoising Filter", value=False)
gamma_val = st.sidebar.slider("Gamma Adjustment (1.0 = off)", 0.5, 2.0, 1.0, 0.1)
show_segmentation = st.sidebar.checkbox("Morphological Tumor Segmentation", value=True)
show_edges = st.sidebar.checkbox("Show Structural Edge Map", value=False)
edge_method = st.sidebar.selectbox("Edge Operator", ["Canny", "Sobel"], disabled=not show_edges)

st.sidebar.subheader("Input MRI Source")
sample_map = _list_sample_test_images()
source_options = ["Upload MRI Image"] + (["Use Testing Set Sample"] if sample_map else [])
input_mode = st.sidebar.radio("Source", source_options)

pil_image: Optional[Image.Image] = None
image_caption = ""

if input_mode == "Upload MRI Image":
    uploaded_file = st.sidebar.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_caption = uploaded_file.name
else:
    chosen_sample = st.sidebar.selectbox("Select Testing Sample", list(sample_map.keys()))
    pil_image = Image.open(sample_map[chosen_sample]).convert("RGB")
    image_caption = chosen_sample

tab_diag, tab_bench = st.tabs(["🔬 Live MRI Diagnosis & XAI", "📊 Model Benchmarks & Figures"])

with tab_diag:
    if pil_image is None:
        st.info("👈 Upload a brain MRI scan (or pick a Testing set sample) from the sidebar to run the pipeline.")
    else:
        np_image = np.array(pil_image)
        processed_np = _preprocess_image(
            np_image,
            enable_crop=enable_crop,
            enable_homomorphic=enable_homomorphic,
            enable_bilateral=enable_bilateral,
            gamma_val=gamma_val,
            enable_clahe=enable_clahe,
            clahe_clip=clahe_clip,
        )
        processed_resized = cv2.resize(processed_np, IMG_SIZE)
        resized_bgr = cv2.cvtColor(processed_resized, cv2.COLOR_RGB2BGR)

        seg_mask = segment_tumor_region(processed_resized, is_rgb=True)
        seg_overlay = draw_segmentation_overlay(processed_resized, seg_mask, is_rgb=True)
        morph_stats = extract_morphological_features(seg_mask)
        glcm_stats = extract_glcm_features(resized_bgr)
        intensity_stats = extract_intensity_statistics(resized_bgr)
        all_descriptors = {**glcm_stats, **intensity_stats, **morph_stats}

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("1. Original MRI Scan")
            st.image(pil_image, caption=image_caption or "Input MRI", use_container_width=True)
            if show_edges:
                edge_map = extract_edge_map(processed_resized, method=edge_method.lower(), color_space="RGB")
                st.image(edge_map, caption=f"{edge_method} Edge Map", use_container_width=True)

        with col2:
            st.subheader("2. Preprocessed & Segmented")
            st.image(processed_np, caption="Enhanced Brain ROI", use_container_width=True)
            if show_segmentation:
                st.image(
                    seg_overlay,
                    caption=f"Morphological Lesion Mask ({morph_stats['tumor_area_px']:.0f} px)",
                    use_container_width=True,
                )
            with st.expander("Texture, Intensity & Shape Descriptors"):
                st.dataframe(
                    pd.DataFrame(
                        [{"Descriptor": k, "Value": round(v, 4)} for k, v in all_descriptors.items()]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

        with col3:
            st.subheader("3. Clinical Inference & XAI")
            transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = transform(Image.fromarray(processed_np)).unsqueeze(0).to(DEVICE)

            model_key, _ = MODEL_SPECS[model_choice]
            checkpoint_path = CHECKPOINTS_DIR / f"{model_key}_best.pth"

            if checkpoint_path.exists():
                model = load_dl_model(model_key, checkpoint_path.stat().st_mtime)

                with torch.no_grad():
                    probs_t = torch.softmax(model(input_tensor), dim=1)
                    if use_tta:
                        probs_t = 0.5 * (probs_t + torch.softmax(model(torch.flip(input_tensor, dims=[3])), dim=1))
                    probs = probs_t.squeeze(0).cpu().numpy()
                    pred_idx = int(np.argmax(probs))
                    pred_class = CLASSES[pred_idx]
                    confidence = float(probs[pred_idx] * 100)

                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning(
                        f"⚠️ **Uncertain Prediction ({model_choice}):** `{pred_class.upper()}` "
                        f"({confidence:.2f}% — below {CONFIDENCE_THRESHOLD:.0f}% threshold; clinical review recommended)"
                    )
                else:
                    st.success(f"**Predicted Diagnosis ({model_choice}):** `{pred_class.upper()}` ({confidence:.2f}%)")

                st.write("Class Probabilities:")
                for c_name, prob in zip(CLASSES, probs):
                    st.progress(float(prob), text=f"{c_name.capitalize()}: {prob*100:.1f}%")

                overlay = processed_resized
                target_layer = getattr(model, "target_gradcam_layer", None)
                if target_layer is not None:
                    cam_generator = GradCAM(model, target_layer)
                    try:
                        heatmap = cam_generator.generate_heatmap(input_tensor, target_class=pred_idx)
                    finally:
                        cam_generator.remove_hooks()
                    overlay = overlay_cam_on_image(processed_resized, heatmap, is_rgb=True)
                    st.image(overlay, caption="Grad-CAM Saliency Map", use_container_width=True)

                report_html = _build_html_report(
                    sample_name=image_caption or "uploaded_mri",
                    model_choice=model_choice,
                    use_tta=use_tta,
                    pred_class=pred_class,
                    confidence=confidence,
                    probs=probs,
                    images_rgb={
                        "1. Original MRI": cv2.resize(np_image, IMG_SIZE),
                        "2. Preprocessed ROI": processed_resized,
                        "3. Lesion Segmentation": seg_overlay,
                        "4. Grad-CAM Saliency": overlay,
                    },
                    descriptors=all_descriptors,
                )
                st.download_button(
                    "📄 Download Diagnostic Report (HTML)",
                    data=report_html,
                    file_name=f"mri_report_{pred_class}.html",
                    mime="text/html",
                    use_container_width=True,
                )
            else:
                st.info(
                    f"Checkpoint `{checkpoint_path.name}` not found. Train it with:\n\n"
                    f"`python run_pipeline.py --mode train-dl --model {model_key}`"
                )

            classical_models = load_classical_models()
            if classical_models:
                with st.expander("Classical ML Baseline Comparison (SVM & RF)"):
                    feat_vec = extract_all_classical_features(resized_bgr).reshape(1, -1)
                    for ml_name, clf in classical_models.items():
                        ml_probs = clf.predict_proba(feat_vec)[0]
                        ml_idx = int(np.argmax(ml_probs))
                        st.write(
                            f"**{ml_name}:** `{CLASSES[ml_idx].upper()}` "
                            f"({ml_probs[ml_idx]*100:.1f}% confidence)"
                        )

with tab_bench:
    st.subheader("Quantitative Evaluation on Held-Out Test Split (1,600 MRI Scans)")
    bench_df = _load_benchmark_rows()
    if bench_df.empty:
        st.info("No evaluation reports found in `outputs/reports/` yet.")
    else:
        st.dataframe(bench_df, hide_index=True, use_container_width=True)

    fig_paths = sorted(p for p in FIGURES_DIR.glob("*.png"))
    if fig_paths:
        st.subheader("Saved Training Curves, Confusion Matrices, ROC & Error Analysis")
        cols = st.columns(2)
        for idx, fig_path in enumerate(fig_paths):
            with cols[idx % 2]:
                st.image(str(fig_path), caption=fig_path.stem.replace("_", " ").title(), use_container_width=True)
