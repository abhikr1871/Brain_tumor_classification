"""
Streamlit Web Application: Brain Tumor Classification & Explainability System.
Allows interactive MRI upload, automated image preprocessing, multi-model inference,
and visual Grad-CAM interpretability.
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import CLASSES, IMG_SIZE, CHECKPOINTS_DIR, DEVICE
from src.image_processing.skull_stripping import crop_brain_contour
from src.image_processing.enhancement import apply_clahe
from src.models.custom_cnn import BrainTumorCNN
from src.models.transfer_models import build_transfer_model
from src.explainability.gradcam import GradCAM, overlay_cam_on_image
from torchvision import transforms

st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="wide", page_icon="🧠")

st.title("🧠 Brain Tumor Classification & Explainability System")
st.markdown("""
*An Academic & Clinical Diagnostic Demonstration for Image Processing and Computer Vision.*
Upload a T1/T2-weighted MRI scan to perform **Skull-Stripping**, **CLAHE enhancement**, **Deep Learning Classification**, and **Grad-CAM Saliency Analysis**.
""")

# Sidebar settings
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Select Model Architecture", ["ResNet50 (Transfer)", "Custom CNN", "DenseNet121"])
enable_clahe = st.sidebar.checkbox("Enable CLAHE Enhancement", value=True)
enable_crop = st.sidebar.checkbox("Enable Skull-Stripping / Crop", value=True)

uploaded_file = st.sidebar.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

col1, col2, col3 = st.columns(3)

if uploaded_file is not None:
    # 1. Display raw image
    pil_image = Image.open(uploaded_file).convert("RGB")
    np_image = np.array(pil_image)
    
    with col1:
        st.subheader("1. Original MRI Scan")
        st.image(pil_image, use_container_width=True)

    # 2. Image Processing Pipeline Preview
    processed_np = np_image.copy()
    if enable_crop:
        processed_np = crop_brain_contour(processed_np)
    if enable_clahe:
        processed_np = apply_clahe(processed_np, clip_limit=2.0)
        
    with col2:
        st.subheader("2. Preprocessed & Enhanced")
        st.image(processed_np, use_container_width=True, caption="Contour Cropped + CLAHE")

    # 3. Model Inference & Grad-CAM
    with col3:
        st.subheader("3. Clinical Inference & XAI")
        
        # Prepare input tensor
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(Image.fromarray(processed_np)).unsqueeze(0).to(DEVICE)
        
        # Load model weights if available, else informative notice
        checkpoint_path = CHECKPOINTS_DIR / f"{model_choice.split()[0].lower()}_best.pth"
        
        if checkpoint_path.exists():
            if "ResNet50" in model_choice:
                model = build_transfer_model("resnet50", num_classes=len(CLASSES), pretrained=False)
            else:
                model = BrainTumorCNN(num_classes=len(CLASSES))
                
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(DEVICE)
            model.eval()

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                pred_idx = np.argmax(probs)
                pred_class = CLASSES[pred_idx]
                confidence = probs[pred_idx] * 100

            st.success(f"**Predicted Diagnosis:** `{pred_class.upper()}` ({confidence:.2f}% confidence)")
            
            # Probability breakdown
            st.write("Class Probabilities:")
            for c_name, prob in zip(CLASSES, probs):
                st.progress(float(prob), text=f"{c_name.capitalize()}: {prob*100:.1f}%")
                
            # Grad-CAM heatmap
            target_layer = getattr(model, "target_gradcam_layer", None)
            if target_layer is not None:
                cam_generator = GradCAM(model, target_layer)
                heatmap = cam_generator.generate_heatmap(input_tensor, target_class=pred_idx)
                overlay = overlay_cam_on_image(cv2.resize(processed_np, IMG_SIZE), heatmap)
                st.image(overlay, caption="Grad-CAM Saliency Map", use_container_width=True)
        else:
            st.info(f"Checkpoint `{checkpoint_path.name}` not found. Train the model using `src/training/train.py` to view live diagnostic predictions and Grad-CAM.")
            st.write("Ready for inference once training checkpoint is generated.")
else:
    st.info("👈 Please upload a brain MRI scan from the sidebar to test the pipeline.")
