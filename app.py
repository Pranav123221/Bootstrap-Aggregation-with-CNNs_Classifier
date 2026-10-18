
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps




APP_DIR = Path(__file__).resolve().parent

if (APP_DIR / "cnn_1.keras").exists():
    ARTIFACT_DIR = APP_DIR
else:
    ARTIFACT_DIR = APP_DIR / "deepbag_artifacts"

print("App folder:", APP_DIR)
print("Model folder:", ARTIFACT_DIR)
print("Model files:", list(ARTIFACT_DIR.glob("*.keras")))


st.set_page_config(
    page_title="DeepBag | CNN Ensemble",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_models():
    models = [
        tf.keras.models.load_model(
            ARTIFACT_DIR / f"cnn_{i}.keras"
    )
        for i in range(1, 4)
    ]
    baseline = tf.keras.models.load_model(
        ARTIFACT_DIR / "baseline_cnn.keras"
    )
    return models, baseline

with open(r"C:\Users\Smart\deepbag_artifacts\class_names.json",
          "r", encoding="utf-8") as f:
    class_names = json.load(f)
    

models, baseline_model = load_models()

st.title("🧠 DeepBag")
st.subheader("CNN Bagging Ensemble for Fashion-MNIST")
st.write(
    "Upload a Fashion-MNIST-style grayscale image "
    "to compare individual CNNs with the ensemble."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.fit(image, (28, 28))
    image_array = np.asarray(image).astype("float32") / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    st.image(image, caption="Resized 28 × 28 input", width=180)

    baseline_probs = baseline_model.predict(
        image_array, verbose=0
    )[0]

    individual_probs = [
        model.predict(image_array, verbose=0)[0]
        for model in models
    ]

    ensemble_probs = np.mean(individual_probs, axis=0)

    baseline_idx = int(np.argmax(baseline_probs))
    ensemble_idx = int(np.argmax(ensemble_probs))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Baseline CNN")
        st.metric(
            "Prediction",
            class_names[baseline_idx]
        )
        st.metric(
            "Confidence",
            f"{baseline_probs[baseline_idx]:.2%}"
        )

    with col2:
        st.markdown("### Bagging Ensemble")
        st.metric(
            "Prediction",
            class_names[ensemble_idx]
        )
        st.metric(
            "Confidence",
            f"{ensemble_probs[ensemble_idx]:.2%}"
        )

    st.markdown("### Individual CNN predictions")
    for i, probs in enumerate(individual_probs, start=1):
        idx = int(np.argmax(probs))
        st.write(
            f"CNN {i}: **{class_names[idx]}** "
            f"(confidence: {probs[idx]:.2%})"
        )

    st.markdown("### Ensemble class probabilities")
    st.bar_chart({
        class_names[i]: float(ensemble_probs[i])
        for i in range(len(class_names))
    })

    st.caption(
        "Confidence is the model's predicted probability, "
        "not a guarantee that the prediction is correct. "
        "This model is trained on Fashion-MNIST images."
    )
