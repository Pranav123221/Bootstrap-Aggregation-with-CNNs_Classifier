
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DeepBag | CNN Ensemble Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# 2. APP STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .hero {
        padding: 1.5rem;
        border: 1px solid #30363d;
        border-radius: 16px;
        background: linear-gradient(135deg, #161b22, #101d30);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: 0.4rem;
    }

    .muted {
        color: #aab4c3;
    }

    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        padding: 16px;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. CLASS LABELS
# --------------------------------------------------

DEFAULT_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_class_names():
    """Load class labels from JSON, with a Fashion-MNIST fallback."""
    class_file = BASE_DIR / "class_names.json"

    if not class_file.exists():
        st.warning(
            "class_names.json was not found. "
            "Using the default Fashion-MNIST class labels."
        )
        return DEFAULT_CLASS_NAMES

    try:
        with open(class_file, "r", encoding="utf-8") as file:
            names = json.load(file)

        # Supports either a list or a dictionary such as {"0": "T-shirt/top"}.
        if isinstance(names, dict):
            names = [
                names.get(str(i), DEFAULT_CLASS_NAMES[i])
                for i in range(10)
            ]

        if not isinstance(names, list) or len(names) != 10:
            raise ValueError("Expected exactly 10 class names.")

        return names

    except (json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
        st.warning(
            f"Could not read class_names.json: {exc}. "
            "Using default class labels."
        )
        return DEFAULT_CLASS_NAMES


CLASS_NAMES = load_class_names()


# --------------------------------------------------
# 4. MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_trained_model(model_path_string):
    """Load and cache a trained Keras model."""
    model_path = Path(model_path_string)

    if not model_path.is_file():
        return None

    return load_model(model_path, compile=False)


MODEL_FILES = {
    "Baseline CNN": "baseline_cnn.keras",
    "CNN 1": "cnn_1.keras",
    "CNN 2": "cnn_2.keras",
    "CNN 3": "cnn_3.keras",
}


def get_model(model_name, filename):
    model_path = BASE_DIR / filename

    try:
        return load_trained_model(str(model_path))
    except Exception as exc:
        st.error(f"Could not load {model_name}: {exc}")
        return None


# --------------------------------------------------
# 5. IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image):
    """
    Convert an uploaded image into the expected Fashion-MNIST input:
    grayscale, 28x28 pixels, normalized to [0, 1], shape (1, 28, 28, 1).

    Note: preprocessing cannot guarantee that a real-world photo
    matches the distribution of the original Fashion-MNIST dataset.
    """
    image = ImageOps.exif_transpose(image)
    image = ImageOps.grayscale(image)
    image = ImageOps.fit(
        image,
        (28, 28),
        method=Image.Resampling.LANCZOS,
    )

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)

    return image, image_array


# --------------------------------------------------
# 6. PREDICTION HELPERS
# --------------------------------------------------

def predict_probabilities(model, image_array):
    """Return the predicted class probabilities for one image."""
    predictions = model.predict(image_array, verbose=0)
    probabilities = np.asarray(predictions)[0]

    if probabilities.shape != (10,):
        raise ValueError(
            f"Expected 10 class probabilities, got {probabilities.shape}."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model produced invalid probability values.")

    return probabilities


def show_prediction_card(model_name, probabilities):
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index]) * 100.0

    st.markdown(f"**{model_name}**")
    st.markdown(f"### {CLASS_NAMES[predicted_index]}")
    st.metric("Top-class probability", f"{confidence:.2f}%")
    return predicted_index


# --------------------------------------------------
# 7. HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🧠 DeepBag</h1>
        <h3>CNN Bootstrap Aggregation for Fashion-MNIST</h3>
        <p class="muted">
            Compare a baseline CNN, three independently trained CNNs,
            and their soft-voting ensemble.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 8. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("About DeepBag")

    st.write(
        "DeepBag combines the predictions of three CNN models "
        "trained on bootstrap samples of the Fashion-MNIST dataset."
    )

    st.divider()

    st.subheader("Experiment results")

    st.metric("Baseline / best individual CNN", "90.96%")
    st.metric("Bagging ensemble accuracy", "91.57%")
    st.metric("Observed improvement", "+0.61 pp")

    st.caption(
        "Reported test-set results from this experiment. "
        "They are not a guarantee of performance on uploaded photos."
    )

    st.divider()

    st.subheader("Supported classes")
    for class_name in CLASS_NAMES:
        st.write(f"• {class_name}")


# --------------------------------------------------
# 9. LOAD MODELS
# --------------------------------------------------

models = {}

with st.spinner("Loading trained CNN models..."):
    for model_name, filename in MODEL_FILES.items():
        model = get_model(model_name, filename)
        if model is not None:
            models[model_name] = model

if len(models) == 0:
    st.error(
        "No trained models could be loaded. Make sure the .keras files "
        "are uploaded to GitHub in the same directory as app.py."
    )

    st.code(
        "app.py\n"
        "class_names.json\n"
        "baseline_cnn.keras\n"
        "cnn_1.keras\n"
        "cnn_2.keras\n"
        "cnn_3.keras\n"
        "requirements.txt"
    )

    st.stop()


# --------------------------------------------------
# 10. IMAGE UPLOAD
# --------------------------------------------------

st.subheader("Upload a clothing image")

st.write(
    "Upload a Fashion-MNIST image or a similar clothing image. "
    "For the most representative results, use a small grayscale image "
    "with one centered item and a plain background."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded_file is None:
    st.info("Upload an image to compare model predictions.")

    st.subheader("How the ensemble works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1. Bootstrap")
        st.write(
            "Train three CNNs on independently sampled training datasets."
        )

    with col2:
        st.markdown("### 2. Predict")
        st.write(
            "Each CNN estimates probabilities for the 10 clothing classes."
        )

    with col3:
        st.markdown("### 3. Aggregate")
        st.write(
            "Average the three probability vectors and select the "
            "class with the highest mean probability."
        )

    st.stop()


# --------------------------------------------------
# 11. PREPROCESS UPLOADED IMAGE
# --------------------------------------------------

try:
    original_image = Image.open(uploaded_file).convert("RGB")
    processed_image, image_array = preprocess_image(original_image)
except Exception as exc:
    st.error(f"Could not process this image: {exc}")
    st.stop()


left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Input image")
    st.image(original_image, caption="Uploaded image", width="stretch")

    with st.expander("View preprocessed 28 × 28 image"):
        st.image(
            processed_image,
            caption="Grayscale model input",
            width=224,
        )

with right:
    st.subheader("Model predictions")

    prediction_results = {}
    prediction_errors = {}

    for model_name, model in models.items():
        try:
            prediction_results[model_name] = predict_probabilities(
                model, image_array
            )
        except Exception as exc:
            prediction_errors[model_name] = str(exc)

    if prediction_errors:
        for name, error in prediction_errors.items():
            st.warning(f"{name} prediction failed: {error}")

    # Use the three individual CNNs for the ensemble.
    individual_names = ["CNN 1", "CNN 2", "CNN 3"]
    available_individuals = [
        name for name in individual_names
        if name in prediction_results
    ]

    if len(available_individuals) == 3:
        ensemble_probabilities = np.mean(
            [prediction_results[name] for name in individual_names],
            axis=0,
        )

        st.markdown("### 🏆 Bagging ensemble")

        ensemble_index = show_prediction_card(
            "Ensemble prediction",
            ensemble_probabilities,
        )

        st.progress(
            float(np.clip(ensemble_probabilities[ensemble_index], 0, 1)),
            text="Ensemble top-class probability",
        )

    else:
        ensemble_probabilities = None
        st.warning(
            "The ensemble requires all three individual CNN models. "
            "Upload the missing model files and redeploy to enable it."
        )


# --------------------------------------------------
# 12. INDIVIDUAL MODEL COMPARISON
# --------------------------------------------------

st.divider()
st.subheader("Individual CNN comparison")

individual_columns = st.columns(3)

for index, model_name in enumerate(["CNN 1", "CNN 2", "CNN 3"]):
    with individual_columns[index]:
        if model_name in prediction_results:
            show_prediction_card(
                model_name,
                prediction_results[model_name],
            )
        else:
            st.warning(f"{model_name} is unavailable.")


# --------------------------------------------------
# 13. BASELINE COMPARISON
# --------------------------------------------------

if "Baseline CNN" in prediction_results:
    st.divider()
    st.subheader("Baseline CNN")

    baseline_index = show_prediction_card(
        "Baseline CNN",
        prediction_results["Baseline CNN"],
    )

    if ensemble_probabilities is not None:
        ensemble_index = int(np.argmax(ensemble_probabilities))

        if baseline_index == ensemble_index:
            st.success(
                "The baseline and ensemble predict the same class."
            )
        else:
            st.info(
                "The baseline and ensemble predict different classes."
            )


# --------------------------------------------------
# 14. CLASS PROBABILITY COMPARISON
# --------------------------------------------------

if ensemble_probabilities is not None:
    st.divider()
    st.subheader("Ensemble class probabilities")

    probability_table = {
        "Class": CLASS_NAMES,
        "Probability (%)": np.round(
            ensemble_probabilities * 100.0, 2
        ),
    }

    st.bar_chart(
        data={
            name: np.round(
                probabilities * 100.0, 2
            )
            for name, probabilities in [
                ("Ensemble", ensemble_probabilities)
            ]
        }
    )

    st.dataframe(
        probability_table,
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# 15. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "DeepBag — CNN Bootstrap Aggregation Classifier | "
    "Built with Python, TensorFlow/Keras and Streamlit."
)

st.caption(
    "Model probabilities represent the models' outputs, not guaranteed "
    "real-world correctness. Uploaded photographs may differ from "
    "the Fashion-MNIST training distribution."
)
