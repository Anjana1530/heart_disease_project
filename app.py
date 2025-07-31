import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("🫀 Heart Disease Prediction using Deep Learning + AI Explainability")

# ✅ Load tabular (CSV) model and scaler from ../model
@st.cache_resource
def load_tabular_model():
    try:
        model = load_model("../model/heart_model.keras")
        scaler = joblib.load("../model/scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ CSV model or scaler not found.\n\n{e}")
        return None, None

# ✅ Load image model from ../model
@st.cache_resource
def load_image_model():
    try:
        return load_model("../model/heart_image_model.h5")
    except Exception as e:
        st.warning(f"⚠️ Image model not found.\n\n{e}")
        return None

# ✅ Preprocess image for prediction
def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# ✅ Predict from uploaded CSV
def predict_from_csv(file, model, scaler):
    try:
        df = pd.read_csv(file)
        st.subheader("📄 Uploaded CSV Preview:")
        st.dataframe(df)

        # Preprocess and predict
        X = df.drop(columns=['target']) if 'target' in df.columns else df
        X_scaled = scaler.transform(X)
        predictions = (model.predict(X_scaled) > 0.5).astype("int32")

        df["Prediction"] = predictions
        df["Heart Disease"] = df["Prediction"].apply(lambda x: "Yes" if x == 1 else "No")

        st.subheader("🔍 Predictions:")
        st.dataframe(df)
    except Exception as e:
        st.error(f"❌ Failed to process CSV: {e}")

# ✅ Load models
tabular_model, scaler = load_tabular_model()
image_model = load_image_model()

# ✅ File uploader
st.markdown("### 📂 Upload a patient CSV or X-ray image (PNG, JPG, JPEG)")
uploaded_file = st.file_uploader("Upload CSV or image", type=["csv", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    if uploaded_file.type == "text/csv":
        if tabular_model is not None and scaler is not None:
            predict_from_csv(uploaded_file, tabular_model, scaler)
        else:
            st.error("❌ CSV model or scaler not loaded.")
    elif uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded X-ray", use_container_width=True)

        if image_model is not None:
            processed_img = preprocess_image(image)
            prediction_score = image_model.predict(processed_img)[0][0]

            st.subheader("🧪 Prediction Result:")
            st.write(f"🔢 Raw Prediction Score: `{prediction_score:.4f}`")

            if prediction_score >= 0.5:
                st.success("✅ Heart disease **detected**.")
            else:
                st.info("✅ Heart disease **not detected**.")
        else:
            st.warning("⚠️ Image model not loaded.")
else:
    st.info("📤 Upload a CSV or image to get started.")

st.markdown("---")
st.markdown("🧠 _Developed using Deep Learning + Streamlit_")
