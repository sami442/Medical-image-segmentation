import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
from huggingface_hub import hf_hub_download
import os

# Page config
st.set_page_config(
    page_title="Brain MRI Tumor Segmentation",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("🧠 Brain MRI Tumor Segmentation")
st.markdown("### Using U-Net Deep Learning Architecture")
st.markdown("---")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This app detects and segments brain tumors 
in MRI scans using U-Net architecture.

**Model Performance:**
- Accuracy: 99.37%
- Dice Coefficient: 0.3147

**Developer:** Samina Mazhar
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Links")
st.sidebar.markdown("[GitHub](https://github.com/sami442)")
st.sidebar.markdown("[Hugging Face](https://huggingface.co/mazharsamina26)")

# Load Model
@st.cache_resource
def load_model():
    with st.spinner("Loading AI model... ⏳"):
        model_path = hf_hub_download(
            repo_id="mazharsamina26/brain-mri-segmentation",
            filename="model.h5"
        )
        
        def dice_coefficient(y_true, y_pred):
            smooth = 1e-6
            y_true_f = tf.keras.backend.flatten(y_true)
            y_pred_f = tf.keras.backend.flatten(y_pred)
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (
                tf.keras.backend.sum(y_true_f) +
                tf.keras.backend.sum(y_pred_f) + smooth)
        
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'dice_coefficient': dice_coefficient,
                          'bce_dice_loss': lambda y_true, y_pred: y_pred}
        )
        return model

model = load_model()
st.sidebar.success("✅ Model loaded!")

# Sample Images
GITHUB_RAW = "https://raw.githubusercontent.com/sami442/medical-image-segmentation/main/samples"

sample_images = {
    "Sample MRI 1": f"{GITHUB_RAW}/sample_1.png",
    "Sample MRI 2": f"{GITHUB_RAW}/sample_2.png",
    "Sample MRI 3": f"{GITHUB_RAW}/sample_3.png",
}

# Input Method
st.markdown("### 🖼️ Choose Input Method")
input_method = st.radio(
    "Select how to provide MRI scan:",
    ["📂 Upload Your Own", "🔬 Use Sample Images"]
)

col1, col2 = st.columns(2)
image = None
IMG_SIZE = 128

if input_method == "📂 Upload Your Own":
    with col1:
        st.markdown("### 📤 Upload MRI Scan")
        uploaded_file = st.file_uploader(
            "Choose an MRI image...",
            type=['png', 'jpg', 'jpeg', 'tif']
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
else:
    with col1:
        st.markdown("### 🔬 Select Sample Image")
        selected_sample = st.selectbox(
            "Choose a sample MRI:",
            list(sample_images.keys())
        )
        try:
            response = requests.get(sample_images[selected_sample])
            image = Image.open(BytesIO(response.content))
            st.success(f"✅ {selected_sample} loaded!")
        except:
            st.error("❌ Could not load sample image")

# Predict
if image is not None:
    with col1:
        st.markdown("### 🧠 Input MRI Scan")
        st.image(image, caption="Input MRI Scan",
                use_container_width=True)

    with col2:
        st.markdown("### 🎯 Tumor Segmentation")
        with st.spinner("Analyzing MRI scan... ⏳"):
            # Preprocess
            img_array = np.array(image.convert('RGB'))
            img_resized = np.array(
                Image.fromarray(img_array).resize((IMG_SIZE, IMG_SIZE))
            )
            img_normalized = img_resized / 255.0
            img_input = np.expand_dims(
                img_normalized, axis=0
            ).astype(np.float32)

            # Predict
            prediction = model.predict(img_input)
            mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255

            # Display
            st.image(mask, caption="Predicted Tumor Mask",
                    use_container_width=True, clamp=True)

            # Stats
            tumor_percent = (mask > 0).mean() * 100
            st.markdown("### 📊 Analysis Results")

            if tumor_percent > 1:
                st.error(f"⚠️ Tumor Detected: {tumor_percent:.2f}% of scan")
            else:
                st.success(f"✅ No Tumor Detected: {tumor_percent:.2f}%")

            st.metric("Tumor Region", f"{tumor_percent:.2f}%")

# Performance
st.markdown("---")
st.markdown("### 📊 Model Performance")

col3, col4, col5 = st.columns(3)
with col3:
    st.metric("Test Accuracy", "99.37%", "+28%")
with col4:
    st.metric("Dice Coefficient", "0.3147", "+2211%")
with col5:
    st.metric("Test Loss", "0.7156", "-0.27")

# Results
st.markdown("---")
st.markdown("### 📈 Training Results")

col6, col7 = st.columns(2)
with col6:
    st.image(
        "https://raw.githubusercontent.com/sami442/medical-image-segmentation/main/Results/training_history_v2.png",
        caption="Training History",
        use_container_width=True
    )
with col7:
    st.image(
        "https://raw.githubusercontent.com/sami442/medical-image-segmentation/main/Results/predictions_v2.png",
        caption="Segmentation Predictions",
        use_container_width=True
    )

st.markdown("---")
st.markdown("**Developed by Samina Mazhar** | BS Artificial Intelligence")
