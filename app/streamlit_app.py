"""
Blood Cell Classification & Leukemia Alert System
Streamlit Web Application
M.R. Sakeer BSc (Hons) Medical Laboratory Science - Gold Badge Capstone
"""

import streamlit as st
import numpy as np
import json
import os
import time
from PIL import Image

# ═══════════════════════════════════════════════
# Paths (relative - works on Streamlit Cloud)
# ═══════════════════════════════════════════════
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "blood_cell_classifier.h5")
INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

# ═══════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="Blood Cell AI Analyzer by M.R.Sakeer BSc Hons in MLS",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════
# CSS Styling
# ═══════════════════════════════════════════════
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .main-header {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 50%, #922b21 100%);
        padding: 25px 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(192,57,43,0.3);
    }
    .main-header h1 { font-size: 2.2em; margin: 0; }
    .main-header p  { font-size: 1.0em; margin: 5px 0 0 0; opacity: 0.9; }
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 20px 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 15px;
        border-left: 5px solid #e74c3c;
    }
    .alert-box {
        background: #fdf2f2;
        border: 2px solid #e74c3c;
        border-radius: 10px;
        padding: 15px 20px;
        margin: 10px 0;
    }
    .alert-box.green {
        background: #f0fdf4;
        border-color: #27ae60;
    }
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# Clinical Knowledge Base
# ═══════════════════════════════════════════════
CLINICAL_DATA = {
    "EOSINOPHIL": {
        "full_name":    "Eosinophil (Acidophilic Granulocyte)",
        "normal_range": "1-4%",
        "function":     "Destroys parasites; involved in allergic reactions and asthma.",
        "elevated":     "Allergies, asthma, parasitic infections, drug reactions",
        "decreased":    "Cushing's syndrome, acute bacterial infections",
        "color":        "#f39c12",
        "icon":         "🟡"
    },
    "LYMPHOCYTE": {
        "full_name":    "Lymphocyte (T-cell / B-cell / NK-cell)",
        "normal_range": "20-40%",
        "function":     "Adaptive immune response. B-cells produce antibodies; T-cells kill infected cells.",
        "elevated":     "Viral infections, lymphocytic leukemia, pertussis",
        "decreased":    "HIV/AIDS, immunodeficiency disorders, corticosteroid therapy",
        "color":        "#3498db",
        "icon":         "🔵"
    },
    "MONOCYTE": {
        "full_name":    "Monocyte (precursor to macrophage)",
        "normal_range": "2-8%",
        "function":     "Differentiates into macrophages. Phagocytoses pathogens and debris.",
        "elevated":     "Chronic infections (TB, malaria), monocytic leukemia",
        "decreased":    "Aplastic anemia, acute stress response",
        "color":        "#2ecc71",
        "icon":         "🟢"
    },
    "NEUTROPHIL": {
        "full_name":    "Neutrophil (Polymorphonuclear Leukocyte)",
        "normal_range": "50-70%",
        "function":     "First line of defense against bacterial and fungal infections.",
        "elevated":     "Bacterial infection, inflammation, stress, steroid use",
        "decreased":    "Viral infections, aplastic anemia, chemotherapy",
        "color":        "#e74c3c",
        "icon":         "🔴"
    }
}

# ═══════════════════════════════════════════════
# Load Model (with compatibility patches)
# ═══════════════════════════════════════════════
@st.cache_resource
def load_model_and_assets():
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense as _Dense
        from tensorflow.keras.layers import Conv2D as _Conv2D
        from tensorflow.keras.layers import InputLayer as _InputLayer
        from tensorflow.keras.layers import BatchNormalization as _BatchNormalization
        from tensorflow.keras.layers import DepthwiseConv2D as _DepthwiseConv2D

        class CompatInputLayer(_InputLayer):
            def __init__(self, **kwargs):
                kwargs.pop('optional', None)
                kwargs.pop('quantization_config', None)
                if 'batch_shape' in kwargs:
                    kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
                super().__init__(**kwargs)

        class CompatDense(_Dense):
            def __init__(self, *args, **kwargs):
                kwargs.pop('quantization_config', None)
                super().__init__(*args, **kwargs)

        class CompatConv2D(_Conv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop('quantization_config', None)
                super().__init__(*args, **kwargs)

        class CompatBatchNormalization(_BatchNormalization):
            def __init__(self, *args, **kwargs):
                kwargs.pop('quantization_config', None)
                super().__init__(*args, **kwargs)

        class CompatDepthwiseConv2D(_DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop('quantization_config', None)
                super().__init__(*args, **kwargs)

        custom_objects = {
            'InputLayer':         CompatInputLayer,
            'Dense':              CompatDense,
            'Conv2D':             CompatConv2D,
            'BatchNormalization': CompatBatchNormalization,
            'DepthwiseConv2D':    CompatDepthwiseConv2D,
        }

        if not os.path.exists(MODEL_PATH):
            st.sidebar.error(f"Model not found at: {MODEL_PATH}")
            return None, None, {}, False

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects=custom_objects
        )

        with open(INDICES_PATH, "r") as f:
            class_indices = json.load(f)

        idx_to_class = {v: k for k, v in class_indices.items()}

        metrics = {}
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)

        return model, idx_to_class, metrics, True

    except Exception as e:
        st.sidebar.error(f"Load error: {str(e)}")
        return None, None, {}, False


# ═══════════════════════════════════════════════
# Prediction Functions
# ═══════════════════════════════════════════════
def reinhard_normalise(img_array):
    """
    Reinhard colour normalisation to reduce domain shift.
    Standardises stain colours from different microscopes/labs
    to match the Kaggle Giemsa-stained training dataset profile.
    This fixes misclassification of monocytes from internet images.
    Target stats computed from paultimothymooney/blood-cells dataset.
    """
    target_mean = np.array([148.60, 105.97, 145.90])
    target_std  = np.array([30.22,  40.15,  25.67])

    img      = img_array.astype(np.float32)
    src_mean = img.mean(axis=(0, 1))
    src_std  = img.std(axis=(0, 1))

    for i in range(3):
        img[:, :, i] = (img[:, :, i] - src_mean[i]) / (src_std[i] + 1e-6)
        img[:, :, i] = img[:, :, i] * target_std[i] + target_mean[i]

    return np.clip(img, 0, 255).astype(np.uint8)


def preprocess_image(image):
    img       = image.convert("RGB").resize((224, 224))
    img_array = np.array(img)
    img_array = reinhard_normalise(img_array)   # normalise stain colours
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


def predict(model, image, idx_to_class):
    processed    = preprocess_image(image)
    probs        = model.predict(processed, verbose=0)[0]
    top_idx      = int(np.argmax(probs))
    top_class    = idx_to_class[top_idx]
    top_conf     = float(probs[top_idx])
    all_probs    = {idx_to_class[i]: float(p) for i, p in enumerate(probs)}
    needs_review = top_conf < 0.75
    return {
        "predicted_class": top_class,
        "confidence":      top_conf,
        "all_probs":       all_probs,
        "needs_review":    needs_review
    }


# ═══════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🩸 Blood Cell AI Analyzer By M.R. Sakeer</h1>
    <p>Automated WBC Classification & Abnormal Cell Detection System</p>
    <p><small>Powered by Custom CNN Deep Learning | Reinhard Stain Normalisation | For Screening Purposes Only</small></p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# Load Model
# ═══════════════════════════════════════════════
model, idx_to_class, metrics, model_loaded = load_model_and_assets()

# ═══════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### System Info")
    if model_loaded:
        acc = metrics.get("test_accuracy", 0)
        st.success("Model Loaded Successfully")
        st.metric("Test Accuracy",      f"{acc*100:.1f}%")
        st.metric("Model Architecture", "Custom CNN")
        st.metric("Cell Classes",       "4 WBC Types")
    else:
        st.error("Model not found!")

    st.divider()
    st.markdown("### WBC Reference Ranges")
    for cls, data in CLINICAL_DATA.items():
        st.markdown(f"""
        <div style='padding:8px; margin:5px 0; background:#f8f9fa; border-radius:8px;'>
            {data['icon']} <b>{cls}</b><br>
            <small style='color:#888'>Normal: {data['normal_range']}</small>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.warning("For educational purposes only. Confirm with qualified pathologist.")


# ═══════════════════════════════════════════════
# Main Content
# ═══════════════════════════════════════════════
col_upload, col_result = st.columns([1, 1.4], gap="large")

with col_upload:
    st.markdown("### 🔬 Upload Blood Smear Image")
    uploaded_file = st.file_uploader(
        "Upload a microscope blood smear image",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.markdown("### Patient Info (Optional)")
        col_a, col_b = st.columns(2)
        with col_a:
            patient_id  = st.text_input("Patient ID", placeholder="PT-2026-001")
            patient_age = st.number_input("Age", min_value=0, max_value=120, value=0)
        with col_b:
            patient_sex = st.selectbox("Sex", ["Select", "Male", "Female", "Other"])
            sample_type = st.selectbox("Sample", ["Peripheral Blood Smear", "Bone Marrow", "Other"])

        analyze_btn = st.button("🔬 Analyze Blood Cell", use_container_width=True, type="primary")

with col_result:
    st.markdown("### 📊 Analysis Results")

    if not uploaded_file:
        st.markdown("""
        <div style="background:#f0f4ff; border-radius:12px; padding:30px; text-align:center; color:#666;">
            <h3 style="color:#aaa;">Upload an image to begin</h3>
            <p>AI will identify the WBC type and flag abnormalities</p>
            <br>
            🔴 Neutrophil | 🔵 Lymphocyte | 🟢 Monocyte | 🟡 Eosinophil
        </div>
        """, unsafe_allow_html=True)

    elif not model_loaded:
        st.error("Model not loaded. Check that all model files are in the repository.")

    elif uploaded_file and analyze_btn:
        with st.spinner("Analyzing blood cell..."):
            time.sleep(0.5)
            result = predict(model, image, idx_to_class)

        cls        = result["predicted_class"]
        conf       = result["confidence"]
        clin_info  = CLINICAL_DATA.get(cls, {})
        conf_pct   = conf * 100
        conf_color = "#27ae60" if conf_pct >= 80 else "#f39c12" if conf_pct >= 60 else "#e74c3c"

        st.markdown(f"""
        <div class="result-card">
            <h2 style="margin:0; color:{clin_info.get('color','#e74c3c')};">
                {clin_info.get('icon','🔬')} {cls}
            </h2>
            <p style="font-size:1.1em; color:#444; margin:5px 0;">
                {clin_info.get('full_name', cls)}
            </p>
            <h3 style="color:{conf_color}; margin:10px 0 0 0;">
                Confidence: {conf_pct:.1f}%
            </h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Probability Distribution:**")
        for cell_cls, prob in result["all_probs"].items():
            color = CLINICAL_DATA.get(cell_cls, {}).get("color", "#999")
            icon  = CLINICAL_DATA.get(cell_cls, {}).get("icon", "")
            st.markdown(f"""
            <div style="margin:6px 0;">
                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                    <span>{icon} <b>{cell_cls}</b></span>
                    <span style="color:{color}; font-weight:bold;">{prob*100:.1f}%</span>
                </div>
                <div style="background:#eee; border-radius:10px; height:10px;">
                    <div style="background:{color}; width:{int(prob*100)}%; height:10px; border-radius:10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        if result["needs_review"]:
            st.markdown("""
            <div class="alert-box">
                <h4 style="color:#c0392b; margin:0;">⚠️ PATHOLOGIST REVIEW RECOMMENDED</h4>
                <p style="margin:8px 0 0 0; color:#666;">
                    Low confidence or atypical morphology. Manual review advised.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-box green">
                <h4 style="color:#27ae60; margin:0;">✅ Normal Cell Morphology</h4>
                <p style="margin:8px 0 0 0; color:#666;">
                    Cell identified with high confidence. Normal {cls.lower()} morphology.
                </p>
            </div>
            """, unsafe_allow_html=True)

        if clin_info:
            with st.expander("📋 Clinical Information", expanded=True):
                st.markdown(f"**Normal Range:** `{clin_info['normal_range']}`")
                st.markdown(f"**Function:** {clin_info['function']}")
                col_e, col_d = st.columns(2)
                with col_e:
                    st.markdown("**🔺 Elevated in:**")
                    for item in clin_info["elevated"].split(", "):
                        st.markdown(f"- {item}")
                with col_d:
                    st.markdown("**🔻 Decreased in:**")
                    for item in clin_info["decreased"].split(", "):
                        st.markdown(f"- {item}")

        with st.expander("📄 Generate Lab Report"):
            report = f"""
AUTOMATED BLOOD CELL ANALYSIS REPORT
=====================================
Date/Time:    {time.strftime('%Y-%m-%d %H:%M:%S')}
Patient ID:   {patient_id if patient_id else 'N/A'}
Age / Sex:    {patient_age if patient_age else 'N/A'} / {patient_sex}
Sample:       {sample_type}
Analyzer:     Blood Cell AI (Custom CNN) v1.0
Test Acc:     {metrics.get('test_accuracy', 0)*100:.1f}%

RESULT
------
Cell Type:    {cls}
Full Name:    {clin_info.get('full_name', '')}
Confidence:   {conf_pct:.2f}%
Normal Range: {clin_info.get('normal_range', 'N/A')}

PROBABILITIES
-------------
""" + "\n".join([f"  {k:<15}: {v*100:.2f}%" for k, v in result["all_probs"].items()]) + f"""

CLINICAL FLAG
-------------
{'REVIEW REQUIRED - Low confidence. Manual microscopy recommended.'
 if result['needs_review'] else
 'No abnormality flagged. Normal cell morphology.'}

DISCLAIMER: AI screening tool for educational purposes only.
Not a clinical diagnosis. Confirm with qualified professional.
=====================================
"""
            st.code(report, language="text")
            st.download_button(
                "⬇️ Download Report (.txt)",
                data=report,
                file_name=f"BloodCell_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

    elif uploaded_file:
        st.info("Click **Analyze Blood Cell** to run AI analysis.")

# ═══════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════
st.divider()
st.markdown("""
<p style="text-align:center; color:#aaa; font-size:0.85em;">
    🩸 Blood Cell AI Analyzer | BSc (Hons) Medical Laboratory Science Capstone | Gold Badge
</p>
""", unsafe_allow_html=True)