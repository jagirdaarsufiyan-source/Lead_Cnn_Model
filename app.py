import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Leaf Deficiency Detection",
    page_icon="🌿",
    layout="centered"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>

    /* BACKGROUND IMAGE (leaf field) */
    .stApp {
        background: 
        linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
        url("https://images.unsplash.com/photo-1464226184884-fa280b87c399");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* TITLE */
    h1 {
        color: #00ffae;
        text-align: center;
        font-size: 42px;
        font-weight: bold;
    }

    /* TEXT */
    .stMarkdown, .stText {
        color: white;
        font-size: 18px;
    }

    /* FILE UPLOADER */
    .stFileUploader {
        border: 2px dashed #00ffae;
        padding: 12px;
        border-radius: 12px;
        background-color: rgba(255,255,255,0.05);
    }

    /* BUTTON */
    .stButton>button {
        background-color: #00ffae;
        color: black;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }

    /* RESULT BOX */
    .result-box {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin-top: 10px;
    }

    .healthy {
        background-color: rgba(0, 255, 150, 0.2);
        color: #00ffae;
        border: 1px solid #00ffae;
    }

    .disease {
        background-color: rgba(255, 0, 0, 0.2);
        color: #ff4d4d;
        border: 1px solid #ff4d4d;
    }

    .unsure {
        background-color: rgba(255, 200, 0, 0.2);
        color: #ffd633;
        border: 1px solid #ffd633;
    }

    </style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model("cnn_model.h5")
    resnet = tf.keras.models.load_model("ResNet50.h5")
    mobilenet = tf.keras.models.load_model("mobilenetv2.h5")
    return cnn, resnet, mobilenet

cnn_model, resnet_model, mobilenet_model = load_models()

# -------------------- CLASS NAMES --------------------
class_names = ["boron", "Healthy", "Kalium", "Magnesium", "nitrogen"]

# -------------------- TITLE --------------------
st.markdown("<h1>🌿 Leaf Disease Detection App</h1>", unsafe_allow_html=True)

# -------------------- MODEL SELECTION --------------------
model_option = st.selectbox(
    "🤖 Select Model",
    ["Simple CNN", "ResNet50", "MobileNetV2"]
)

# Assign model + image size
if model_option == "Simple CNN":
    selected_model = cnn_model
    img_size = 128
elif model_option == "ResNet50":
    selected_model = resnet_model
    img_size = 224
else:
    selected_model = mobilenet_model
    img_size = 224

# -------------------- UPLOAD --------------------
uploaded_file = st.file_uploader("📤 Upload a Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # -------------------- PREPROCESS --------------------
    img = image.resize((img_size, img_size))
    img_array = np.array(img) / 255.0

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)

    # -------------------- PREDICT --------------------
    if st.button("🔍 Predict"):
        prediction = selected_model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        # -------------------- OUTPUT --------------------
        st.write(f"### 🧠 Prediction: {predicted_class}")
        st.write(f"### 🔍 Confidence: {confidence:.2f}")

        if confidence < 0.6:
            st.markdown(
                '<div class="result-box unsure">❓ Not sure, try another image</div>',
                unsafe_allow_html=True
            )
        elif predicted_class.lower() == "healthy":
            st.markdown(
                '<div class="result-box healthy">✅ Leaf is Healthy</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-box disease">⚠ Disease Detected</div>',
                unsafe_allow_html=True
            )
