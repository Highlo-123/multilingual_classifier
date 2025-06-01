import streamlit as st
import numpy as np
import cv2
import joblib
import os
from skimage.feature import hog
from skimage import exposure
from PIL import Image

# ---- Set page config ----
st.set_page_config(page_title="Multilingual HOG-SVM Classifier", layout="wide")



# ---- Load SVM models from disk ----
@st.cache_resource
def load_models():
    model1_path = "model/hog_svm_model1.pkl"
    model2_path = "model/hog_svm_model2.pkl"
    model1 = joblib.load(model1_path)
    model2 = joblib.load(model2_path)
    return model1, model2

# ---- HOG feature extractor ----
def extract_hog_features(image: np.ndarray):
    resized_img = cv2.resize(image, (64, 128))  # Consistent size
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    features, _ = hog(
        gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
        visualize=True, multichannel=False
    )
    return features

# ---- Streamlit App ----
st.title("📝 Multilingual Document Classifier (HOG + SVM)")

st.markdown("This tool uses pretrained HOG+SVM models to classify a **handwritten image** as English, Hindi, or Kannada.")

# ---- Sidebar model selection ----
model_choice = st.sidebar.radio(
    "Choose classification model:",
    ("English vs Hindi (Model 1)", "English vs Kannada (Model 2)")
)

uploaded_file = st.file_uploader("Upload a handwritten document image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL to NumPy
    image_np = np.array(image)

    # Extract HOG features
    with st.spinner("Extracting HOG features..."):
        features = extract_hog_features(image_np)

    # Load appropriate model
    model1, model2 = load_models()
    if "Model 1" in model_choice:
        model = model1
        label_map = {0: "English", 1: "Hindi"}
    else:
        model = model2
        label_map = {0: "English", 1: "Kannada"}

    # Predict
    with st.spinner("Classifying..."):
        prediction = model.predict([features])[0]
        predicted_label = label_map[prediction]

    st.success(f"✅ Predicted Language: **{predicted_label}**")

# ---- Optional footer ----

