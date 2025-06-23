import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import joblib
from PIL import Image
import os

# --- Page configuration (MUST be before any other Streamlit code) ---
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    layout="centered"  # Changed from 'heart' to a valid emoji
)

# --- Debugging: Confirm which app.py is running ---
print(f"--- DEBUG: Running app.py from: {os.path.abspath(__file__)} ---")

# --- Model Paths ---
RANDOM_FOREST_MODEL_PATH = 'random_forest_model.pkl'
CNN_MODEL_PATH = 'cnn_model.h5'

# --- Load Models ---
@st.cache_resource
def load_random_forest_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Random Forest model not found at '{path}'.")
        return None
    except Exception as e:
        st.error(f"Error loading Random Forest model: {e}")
        return None

@st.cache_resource
def load_cnn_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: CNN model not found at '{path}'.")
        return None
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        return None

random_forest_model = load_random_forest_model(RANDOM_FOREST_MODEL_PATH)
cnn_model = load_cnn_model(CNN_MODEL_PATH)

# --- Helper Functions ---
def predict_with_rf(model, input_features):
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    return prediction

def process_ecg_img(image_data):
    img_array = np.array(image_data.convert('L'))
    img = cv2.resize(img_array, (224, 224))
    img = img / 255.0
    img = img.reshape(1, 224, 224, 1)
    return img

def predict_with_cnn(model, processed_img):
    prediction = model.predict(processed_img)
    return prediction[0][0]

# --- UI ---
st.title("❤️ Heart Attack Risk Predictor")
st.markdown("---")

option = st.sidebar.radio(
    "Choose a prediction method:",
    ("Predict from Symptoms", "Predict from ECG Image")
)

if option == "Predict from Symptoms":
    st.header("📝 Predict from Symptoms")
    if random_forest_model is None:
        st.warning("Random Forest model not loaded.")
    else:
        st.write("Please enter your details and symptoms below:")

        # ECG Parameters
        st.subheader("ECG Parameters")
        age = st.number_input("Age:", min_value=1, max_value=120, value=30)
        hr = st.number_input("Heart Rate (bpm):", min_value=30, max_value=200, value=70)
        qrs = st.number_input("QRS duration (ms):", min_value=50, max_value=200, value=100)
        qtc = st.number_input("QTc interval (ms):", min_value=200, max_value=600, value=400)
        pr = st.number_input("PR interval (ms):", min_value=80, max_value=250, value=150)

        # Symptoms
        st.subheader("Symptoms")
        chest_pain = 1 if st.radio("Do you have chest pain?", ("No", "Yes"), key="cp") == "Yes" else 0
        shortness_of_breath = 1 if st.radio("Shortness of breath?", ("No", "Yes"), key="sob") == "Yes" else 0
        sweating = 1 if st.radio("Are you sweating?", ("No", "Yes"), key="sweat") == "Yes" else 0
        nausea = 1 if st.radio("Do you feel nauseous?", ("No", "Yes"), key="nausea") == "Yes" else 0
        fatigue = 1 if st.radio("Feeling fatigue?", ("No", "Yes"), key="fatigue") == "Yes" else 0
        dizziness = 1 if st.radio("Do you feel dizzy?", ("No", "Yes"), key="dizzy") == "Yes" else 0
        jaw_pain = 1 if st.radio("Jaw pain?", ("No", "Yes"), key="jaw") == "Yes" else 0
        shoulder_pain = 1 if st.radio("Shoulder pain?", ("No", "Yes"), key="shoulder") == "Yes" else 0

        input_features = [
            age, hr, qrs, qtc, pr,
            chest_pain, shortness_of_breath, sweating, nausea,
            fatigue, dizziness, jaw_pain, shoulder_pain
        ]

        if st.button("Predict Risk (Symptoms)"):
            prediction = predict_with_rf(random_forest_model, input_features)
            if prediction == 1:
                st.error("⚠️ High risk of heart attack. Please consult a doctor immediately.")
            else:
                st.success("✅ No signs of heart attack risk based on the symptoms.")

elif option == "Predict from ECG Image":
    st.header("🖼️ Predict from ECG Image")
    if cnn_model is None:
        st.warning("CNN model not loaded.")
    else:
        st.write("Upload an ECG image (JPG, PNG) for prediction.")
        uploaded_file = st.file_uploader("Choose an ECG image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded ECG Image', use_column_width=True)

            if st.button("Predict Risk (ECG Image)"):
                with st.spinner("Analyzing ECG image..."):
                    processed_img = process_ecg_img(image)
                    prediction_prob = predict_with_cnn(cnn_model, processed_img)

                    st.write(f"Prediction Confidence: {prediction_prob:.4f}")

                    if prediction_prob >= 0.8:
                        st.error("🚨 High chance of heart issues. Please consult a doctor.")
                    else:
                        st.success("👍 ECG shows no significant heart attack risk.")

st.markdown("---")
st.info("Disclaimer: This tool is for informational purposes only and should not be used as a substitute for professional medical advice.")




