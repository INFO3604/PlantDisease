from pathlib import Path
import json
import sys

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

# Add src/ to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from plantdisease.data.preprocess.pipeline import PreprocessingPipeline
from plantdisease.features import extract_features_from_pipeline_result


MODEL_PATH = Path("models/exports/xgboost.pkl")
CLASS_NAMES_PATH = Path("models/exports/class_names.json")
FEATURE_COLUMNS_PATH = Path("models/exports/feature_columns.json")


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH.resolve()}")

    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Feature columns file not found: {FEATURE_COLUMNS_PATH.resolve()}")

    model = joblib.load(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    with open(FEATURE_COLUMNS_PATH, "r") as f:
        feature_columns = json.load(f)

    pipeline = PreprocessingPipeline()

    return model, class_names, feature_columns, pipeline


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def format_label(label: str) -> str:
    return label.replace("___", " — ").replace("_", " ")


def predict_image(image_bgr: np.ndarray) -> dict:
    model, class_names, feature_columns, pipeline = load_artifacts()

    result = pipeline.run(image_bgr)
    features = extract_features_from_pipeline_result(result)

    missing_features = [col for col in feature_columns if col not in features]
    if missing_features:
        raise ValueError(
            f"Missing extracted features: {missing_features[:10]}"
            + (" ..." if len(missing_features) > 10 else "")
        )

    feature_vector = [features[col] for col in feature_columns]
    X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

    pred_idx = int(model.predict(X)[0])

    confidence = None
    top_predictions = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        confidence = float(np.max(probs))

        top_indices = np.argsort(probs)[::-1][:3]
        top_predictions = [
            {
                "label": class_names[int(i)],
                "confidence": float(probs[int(i)]),
            }
            for i in top_indices
        ]

    return {
        "label": class_names[pred_idx],
        "confidence": confidence,
        "top_predictions": top_predictions,
    }


def main():
    st.set_page_config(page_title="Plant Disease Detection", layout="centered")

    st.markdown(
        """
        <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background-color: #E8F5E9;
            color: #1B4332;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, p, label, div, span {
            color: #1B4332 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🌿 Plant Leaf Disease Detection")
    st.write(
        "Upload a leaf image or take a photo to predict the disease class using the trained XGBoost model."
    )

    st.subheader("System Check")

    try:
        model, class_names, feature_columns, pipeline = load_artifacts()
        st.success("Artifacts loaded successfully.")
        st.write(f"Classes loaded: {len(class_names)}")
        st.write(f"Feature columns loaded: {len(feature_columns)}")
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        return

    st.subheader("Input Image")
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    camera_file = st.camera_input("Or take a photo")

    selected_file = camera_file if camera_file is not None else uploaded_file
    image = None

    if selected_file is not None:
        try:
            image = Image.open(selected_file)
            st.image(image, caption="Selected image", use_container_width=True)
        except Exception as e:
            st.error(f"Could not open image: {e}")
            return

    if st.button("Run Prediction"):
        if image is None:
            st.warning("Please upload an image or take a photo first.")
            return

        try:
            with st.spinner("Analyzing image..."):
                image_bgr = pil_to_bgr(image)
                prediction = predict_image(image_bgr)

            st.subheader("Prediction Result")
            st.success(f"Predicted disease: {format_label(prediction['label'])}")

            if prediction["confidence"] is not None:
                st.write(f"Confidence: {prediction['confidence']:.2%}")

            if prediction["top_predictions"]:
                st.subheader("Top 3 Predictions")
                for item in prediction["top_predictions"]:
                    st.write(f"- {format_label(item['label'])}: {item['confidence']:.2%}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()