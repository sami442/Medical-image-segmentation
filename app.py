import streamlit as st
import numpy as np
import cv2
from PIL import Image

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
st.sidebar.markdown("[LinkedIn](https://linkedin.com/in/YOUR-LINKEDIN)")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📤 Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose an MRI image...",
        type=['png', 'jpg', 'jpeg', 'tif']
    )

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    with col1:
        st.image(image, caption="Uploaded MRI Scan", 
                use_column_width=True)

    # Preprocess
    IMG_SIZE = 128
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Handle grayscale images
    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    elif img_resized.shape[2] == 4:
        img_resized = img_resized[:, :, :3]
    
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0).astype(np.float32)

    with col2:
        st.markdown("### 🎯 Segmentation Result")
        with st.spinner("Analyzing MRI scan... ⏳"):
            st.info("""
            ⚠️ **Demo Mode**
            
            To see real predictions, the trained 
            model needs to be added to this repo.
            
            **Current Results on Test Set:**
            - ✅ Accuracy: 99.37%
            - ✅ Dice Score: 0.3147
            """)

# Results section
st.markdown("---")
st.markdown("### 📊 Model Performance")

col3, col4, col5 = st.columns(3)
with col3:
    st.metric("Test Accuracy", "99.37%", "+28% improvement")
with col4:
    st.metric("Dice Coefficient", "0.3147", "+2211% improvement")
with col5:
    st.metric("Test Loss", "0.7156", "-0.27 improvement")

# Show result images
st.markdown("---")
st.markdown("### 📈 Training Results")

col6, col7 = st.columns(2)
with col6:
    st.image(
        "https://raw.githubusercontent.com/sami442/medical-image-segmentation/main/Results/training_history_v2.png",
        caption="Training History",
        use_column_width=True
    )
with col7:
    st.image(
        "https://raw.githubusercontent.com/sami442/medical-image-segmentation/main/Results/predictions_v2.png",
        caption="Segmentation Predictions",
        use_column_width=True
    )

st.markdown("---")
st.markdown("**Developed by Samina Mazhar** | BS Artificial Intelligence")
