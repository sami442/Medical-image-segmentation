import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Page config
st.set_page_config(
    page_title="NeuroScan AI | Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a0a1a 100%);
        color: white;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #00d2ff33;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.1);
    }
    
    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #7b2ff733;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Upload area */
    .upload-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px dashed #00d2ff55;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #0d1b2a, #1a1a2e);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #ccd6f6 !important;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        border: none;
        margin: 2rem 0;
    }

    /* Badge */
    .badge {
        display: inline-block;
        background: linear-gradient(90deg, #00d2ff22, #7b2ff722);
        border: 1px solid #00d2ff55;
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.85rem;
        color: #00d2ff;
        margin: 0.2rem;
    }

    /* Section title */
    .section-title {
        color: #00d2ff;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        border-left: 3px solid #7b2ff7;
        padding-left: 0.8rem;
    }

    /* Alert boxes */
    .tumor-detected {
        background: linear-gradient(135deg, #ff000022, #ff000011);
        border: 1px solid #ff000055;
        border-radius: 10px;
        padding: 1rem;
        color: #ff6b6b;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
    }

    .no-tumor {
        background: linear-gradient(135deg, #00ff0022, #00ff0011);
        border: 1px solid #00ff0055;
        border-radius: 10px;
        padding: 1rem;
        color: #6bff6b;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style='text-align: center; padding: 2rem 0 1rem 0;'>
    <p style='font-size: 3.5rem; font-weight: 800; 
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;'>🧠 NeuroScan AI</p>
    <p style='color: #8892b0; font-size: 1.1rem; margin: 0;'>
    Advanced Brain MRI Tumor Segmentation using U-Net Deep Learning</p>
    <div style='margin-top: 1rem;'>
        <span class='badge'>🎯 99.37% Accuracy</span>
        <span class='badge'>📊 Dice: 0.3147</span>
        <span class='badge'>🚀 Real-time Analysis</span>
        <span class='badge'>🏥 Medical AI</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <p style='font-size: 1.5rem; font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;'>
        🧠 NeuroScan AI</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #00d2ff33; border-radius: 10px; padding: 1rem;'>
        <p style='color: #00d2ff; font-weight: 600;'>📊 Model Performance</p>
        <p style='color: #ccd6f6;'>✅ Accuracy: <b>99.37%</b></p>
        <p style='color: #ccd6f6;'>✅ Dice Score: <b>0.3147</b></p>
        <p style='color: #ccd6f6;'>✅ Test Loss: <b>0.7156</b></p>
        <p style='color: #ccd6f6;'>✅ Architecture: <b>U-Net</b></p>
        <p style='color: #ccd6f6;'>✅ Dataset: <b>LGG MRI</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #7b2ff733; border-radius: 10px; padding: 1rem;'>
        <p style='color: #7b2ff7; font-weight: 600;'>👩‍💻 Developer</p>
        <p style='color: #ccd6f6;'><b>Samina Mazhar</b></p>
        <p style='color: #8892b0;'>BS Artificial Intelligence</p>
        <p style='color: #8892b0;'>Islamia University Bahawalpur</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        <a href='https://github.com/sami442' target='_blank'
        style='color: #00d2ff; text-decoration: none;'>
        🐙 GitHub</a> &nbsp;|&nbsp;
        <a href='https://huggingface.co/mazharsamina26' target='_blank'
        style='color: #7b2ff7; text-decoration: none;'>
        🤗 Hugging Face</a>
    </div>
    """, unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(
            model_path="brain_mri_model.tflite")
        interpreter.allocate_tensors()
        return interpreter, True
    except Exception as e:
        return None, False

interpreter, model_loaded = load_model()

if model_loaded:
    st.sidebar.markdown("""
    <div style='background: #00ff0011; border: 1px solid #00ff0055;
    border-radius: 8px; padding: 0.5rem; text-align: center;
    color: #6bff6b; margin-top: 1rem;'>
    ✅ AI Model Ready
    </div>""", unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style='background: #ffff0011; border: 1px solid #ffff0055;
    border-radius: 8px; padding: 0.5rem; text-align: center;
    color: #ffff6b; margin-top: 1rem;'>
    ⚠️ Demo Mode Active
    </div>""", unsafe_allow_html=True)

# Sample Images
GITHUB_RAW = "https://raw.githubusercontent.com/sami442/medical-image-segmentation/main/samples"
sample_images = {
    "Sample MRI 1": f"{GITHUB_RAW}/sample_1.png",
    "Sample MRI 2": f"{GITHUB_RAW}/sample_2.png",
    "Sample MRI 3": f"{GITHUB_RAW}/sample_3.png",
}

# Input Section
st.markdown("""
<p class='section-title'>🔬 Select Analysis Mode</p>
""", unsafe_allow_html=True)

input_method = st.radio(
    "",
    ["📂 Upload Your Own MRI", "🔬 Use Sample MRI Images"],
    horizontal=True
)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")
image = None
IMG_SIZE = 128

if input_method == "📂 Upload Your Own MRI":
    with col1:
        st.markdown("""
        <p class='section-title'>📤 Upload MRI Scan</p>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop your MRI scan here",
            type=['png', 'jpg', 'jpeg', 'tif'],
            help="Supported formats: PNG, JPG, JPEG, TIF"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
else:
    with col1:
        st.markdown("""
        <p class='section-title'>🔬 Sample MRI Images</p>
        """, unsafe_allow_html=True)
        selected_sample = st.selectbox(
            "Choose a sample scan:",
            list(sample_images.keys())
        )
        try:
            response = requests.get(sample_images[selected_sample])
            image = Image.open(BytesIO(response.content))
            st.success(f"✅ {selected_sample} loaded successfully!")
        except:
            st.error("❌ Could not load sample image")

# Display & Predict
if image is not None:
    with col1:
        st.markdown("""
        <p class='section-title'>🧠 Input MRI Scan</p>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("""
        <p class='section-title'>🎯 AI Segmentation Result</p>
        """, unsafe_allow_html=True)

        with st.spinner("🔍 Analyzing MRI scan..."):
            img_array = np.array(image.convert('RGB'))
            img_resized = np.array(
                Image.fromarray(img_array).resize((IMG_SIZE, IMG_SIZE))
            )
            img_normalized = (img_resized / 255.0).astype(np.float32)
            img_input = np.expand_dims(img_normalized, axis=0)

            if model_loaded:
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(
                    input_details[0]['index'], img_input)
                interpreter.invoke()
                prediction = interpreter.get_tensor(
                    output_details[0]['index'])
                mask = (prediction[0, :, :, 0] > 0.5
                       ).astype(np.uint8) * 255
            else:
                img_gray = np.array(
                    image.convert('L').resize((IMG_SIZE, IMG_SIZE)))
                threshold = img_gray.mean() + img_gray.std()
                mask = (img_gray > threshold).astype(np.uint8) * 255

            st.image(mask, use_container_width=True, clamp=True)

            tumor_percent = (mask > 0).mean() * 100

            if tumor_percent > 1:
                st.markdown(f"""
                <div class='tumor-detected'>
                ⚠️ Tumor Region Detected<br>
                <span style='font-size: 2rem;'>{tumor_percent:.2f}%</span>
                <br>of scan area
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='no-tumor'>
                ✅ No Tumor Detected<br>
                <span style='font-size: 2rem;'>{tumor_percent:.2f}%</span>
                <br>abnormal region
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='background: #ffffff11; border-radius: 8px;
            padding: 0.8rem; color: #8892b0; font-size: 0.85rem;'>
            ⚕️ <b>Medical Disclaimer:</b> This tool is for research 
            purposes only and should not replace professional 
            medical diagnosis.
            </div>
            """, unsafe_allow_html=True)

# Performance Metrics
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
<p class='section-title'>📊 Model Performance Metrics</p>
""", unsafe_allow_html=True)

col3, col4, col5, col6 = st.columns(4)
with col3:
    st.markdown("""
    <div class='metric-card'>
        <p style='color: #8892b0; margin: 0;'>Test Accuracy</p>
        <p style='color: #00d2ff; font-size: 2rem; 
        font-weight: 800; margin: 0;'>99.37%</p>
        <p style='color: #6bff6b; font-size: 0.8rem; 
        margin: 0;'>↑ +28% improvement</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class='metric-card'>
        <p style='color: #8892b0; margin: 0;'>Dice Coefficient</p>
        <p style='color: #7b2ff7; font-size: 2rem; 
        font-weight: 800; margin: 0;'>0.3147</p>
        <p style='color: #6bff6b; font-size: 0.8rem; 
        margin: 0;'>↑ +2211% improvement</p>
    </div>
    """, unsafe_allow_html=True)
with col5:
    st.markdown("""
    <div class='metric-card'>
        <p style='color: #8892b0; margin: 0;'>Test Loss</p>
        <p style='color: #00d2ff; font-size: 2rem; 
        font-weight: 800; margin: 0;'>0.7156</p>
        <p style='color: #6bff6b; font-size: 0.8rem; 
        margin: 0;'>↓ -0.27 reduction</p>
    </div>
    """, unsafe_allow_html=True)
with col6:
    st.markdown("""
    <div class='metric-card'>
        <p style='color: #8892b0; margin: 0;'>Architecture</p>
        <p style='color: #7b2ff7; font-size: 2rem; 
        font-weight: 800; margin: 0;'>U-Net</p>
        <p style='color: #6bff6b; font-size: 0.8rem; 
        margin: 0;'>✅ Optimized</p>
    </div>
    """, unsafe_allow_html=True)

# Training Results
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
<p class='section-title'>📈 Training Results & Visualizations</p>
""", unsafe_allow_html=True)

col7, col8 = st.columns(2)
with col7:
    st.markdown("""
    <p style='color: #8892b0; text-align: center;'>
    Training History</p>""", unsafe_allow_html=True)
    st.image(
        "https://raw.githubusercontent.com/sami442/medical-image-segmentation/main/Results/training_history_v2.png",
        use_container_width=True
    )
with col8:
    st.markdown("""
    <p style='color: #8892b0; text-align: center;'>
    Segmentation Predictions</p>""", unsafe_allow_html=True)
    st.image(
        "https://raw.githubusercontent.com/sami442/medical-image-segmentation/main/Results/predictions_v2.png",
        use_container_width=True
    )

# Footer
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #8892b0; padding: 1rem;'>
    <p>Developed with ❤️ by <b style='color: #00d2ff;'>Samina Mazhar</b> 
    |  Artificial Intelligence | 
    <a href='https://github.com/sami442' 
    style='color: #7b2ff7;'>GitHub</a> | 
    <a href='https://huggingface.co/mazharsamina26' 
    style='color: #00d2ff;'>Hugging Face</a></p>
</div>
""", unsafe_allow_html=True)
