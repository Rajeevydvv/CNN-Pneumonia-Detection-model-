import io
import hashlib

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.resnet50     import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet     import preprocess_input as densenet_preprocess

# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Constants & model registry
# ─────────────────────────────────────────────────────────
IMG_SIZE = 224


MODEL_REGISTRY: dict[str, dict] = {

    # ──────────────────────────────────────────────
    # Custom CNN
    # ──────────────────────────────────────────────
    "Custom CNN": {
        "file": "cnn_best.keras",
        "preprocess_fn": lambda x: x / 255.0,
        "description": "Custom CNN trained from scratch on Chest X-Ray images.",
    },

    # ──────────────────────────────────────────────
    # Phase 1 Transfer Learning Models
    # ──────────────────────────────────────────────
    "DenseNet121 (Phase 1)": {
        "file": "densenet_phase1.keras",
        "preprocess_fn": densenet_preprocess,
        "description": "DenseNet121 feature extraction model trained with frozen ImageNet backbone.",
    },

    "ResNet50 (Phase 1)": {
        "file": "resnet_phase1.keras",
        "preprocess_fn": resnet_preprocess,
        "description": "ResNet50 feature extraction model trained with frozen ImageNet backbone.",
    },

    "EfficientNetB0 (Phase 1)": {
        "file": "effnet_phase1.keras",
        "preprocess_fn": effnet_preprocess,
        "description": "EfficientNetB0 feature extraction model trained with frozen ImageNet backbone.",
    },

    # ──────────────────────────────────────────────
    # Fine-Tuned Models
    # ──────────────────────────────────────────────
    "DenseNet121 (Fine-Tuned)": {
        "file": "densenet_ft_best.keras",
        "preprocess_fn": densenet_preprocess,
        "description": "Best fine-tuned DenseNet121 model.",
    },

    "ResNet50 (Fine-Tuned)": {
        "file": "resnet_ft_best.keras",
        "preprocess_fn": resnet_preprocess,
        "description": "Best fine-tuned ResNet50 model.",
    },

    "EfficientNetB0 (Fine-Tuned)": {
        "file": "effnet_ft_best.keras",
        "preprocess_fn": effnet_preprocess,
        "description": "Best fine-tuned EfficientNetB0 model.",
    },
}

LABEL_NAMES = {0: "Normal", 1: "Pneumonia"}

# ─────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models() -> dict:
    """
    Load every .keras file once per server session and cache the result.
    Missing files emit a warning rather than raising an unhandled exception.
    """
    loaded: dict = {}
    for name, cfg in MODEL_REGISTRY.items():
        try:
            loaded[name] = tf.keras.models.load_model(cfg["file"])
        except (OSError, ValueError) as exc:
            st.warning(
                f"**{name}** could not be loaded — `{cfg['file']}` "
                f"may be missing or corrupt.\n\n> {exc}"
            )
    return loaded


models = load_models()

if not models:
    st.error(
        "No models could be loaded. "
        "Ensure the `.keras` files are in the same directory as `app.py`."
    )
    st.stop()

# ─────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image, preprocess_fn) -> np.ndarray:
    """
    Resize to IMG_SIZE×IMG_SIZE, cast to float32, apply the
    backbone-specific normalisation, and add the batch axis.
    LANCZOS resampling preserves diagnostic detail better than the
    default nearest-neighbour on downscaling paths.
    """
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)   # (224, 224, 3)
    arr = preprocess_fn(arr)                # backbone-specific range
    return np.expand_dims(arr, axis=0)      # (1, 224, 224, 3)

# ─────────────────────────────────────────────────────────
# Cached inference
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_inference(model_name: str, img_hash: str, _img_bytes: bytes) -> float:
    """
    Return the raw Pneumonia probability for one (model, image) pair.

    Cache key  : (model_name, img_hash) – uniquely identifies the pair.
    _img_bytes : leading underscore tells Streamlit to SKIP hashing this
                 argument (bytes objects can be large); img_hash already
                 captures image identity for the key.

    FIX: the original called model.predict() on every Streamlit re-render.
    Now the result is reused from cache until the model or image changes.
    """
    cfg = MODEL_REGISTRY[model_name]
    pil_img = Image.open(io.BytesIO(_img_bytes))
    arr = preprocess_image(pil_img, cfg["preprocess_fn"])
    prob = float(models[model_name].predict(arr, verbose=0)[0][0])
    return prob


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

# ─────────────────────────────────────────────────────────
# Helper: render a single prediction card
# ─────────────────────────────────────────────────────────
def render_result(model_name: str, pneumonia_prob: float, threshold: float) -> None:
    normal_prob  = 1.0 - pneumonia_prob
    is_pneumonia = pneumonia_prob >= threshold

    if is_pneumonia:
        st.error("### 🦠 Pneumonia Detected")
        confidence = pneumonia_prob
    else:
        st.success("### ✅ Normal")
        confidence = normal_prob

    st.metric("Confidence", f"{confidence * 100:.1f}%")

    st.markdown("**Probability breakdown**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption(f"Normal · `{normal_prob:.4f}`")
        st.progress(float(normal_prob))
    with col_b:
        st.caption(f"Pneumonia · `{pneumonia_prob:.4f}`")
        st.progress(float(pneumonia_prob))

    st.caption(
        f"Model: `{model_name}` &nbsp;·&nbsp; "
        f"Threshold: `{threshold:.2f}` &nbsp;·&nbsp; "
        f"{'Above' if is_pneumonia else 'Below'} threshold"
    )

# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    available_models = list(models.keys())

    # FIX: original showed a selectbox AND then a redundant bullet list
    # of the same model names directly below it. Removed the bullet list.
    selected_model_name = st.selectbox("Model", available_models)
    st.caption(MODEL_REGISTRY[selected_model_name]["description"])

    st.markdown("---")

    compare_all = st.checkbox(
        "Compare all models",
        value=False,
        help="Run every loaded model on the same image and show a results table.",
    )


    threshold = st.slider(
        "Decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
        help=(
            "Probability above which the prediction is **Pneumonia**. "
            "Lowering it increases sensitivity (catches more true cases) "
            "but also increases false positives."
        ),
    )

    st.markdown("---")
    st.warning(
        "⚠️ **Research use only.**  \n"
        "This tool is not a substitute for professional medical diagnosis. "
        "Always consult a qualified clinician."
    )

# ─────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────
st.title("🫁 Pneumonia Detection")
st.write(
    "Upload a chest X-ray to classify it as **Normal** or **Pneumonia** "
    "using one of four deep-learning models."
)

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"],
    help="JPEG or PNG chest X-ray image.",
)

if uploaded_file is not None:
    raw_bytes  = uploaded_file.read()
    img_hash   = md5(raw_bytes)
    pil_image  = Image.open(io.BytesIO(raw_bytes))

    col_image, col_result = st.columns([1, 1], gap="large")

    # ── X-ray image ─────────────────────────────────────────────
    with col_image:
        st.subheader("Uploaded X-Ray")
        # Display in greyscale to match clinical convention
        st.image(pil_image.convert("L"), use_container_width=True, clamp=True)
        st.caption(
            f"Original size: {pil_image.width} × {pil_image.height} px  |  "
            f"Colour mode: {pil_image.mode}"
        )

    # ── Prediction(s) ────────────────────────────────────────────
    with col_result:
        if not compare_all:
            # ── Single model ─────────────────────────────────────
            st.subheader(f"Result — {selected_model_name}")
            with st.spinner("Running inference…"):
                prob = run_inference(selected_model_name, img_hash, raw_bytes)
            render_result(selected_model_name, prob, threshold)

        else:
            # ── All-model comparison ──────────────────────────────
            st.subheader("All-Model Comparison")
            progress_bar = st.progress(0.0, text="Running models…")
            rows = []

            for idx, name in enumerate(MODEL_REGISTRY):
                if name not in models:
                    rows.append({
                        "Model": name,
                        "Pneumonia": "—",
                        "Normal":    "—",
                        "Verdict":   "⚠️ Not loaded",
                    })
                    continue

                prob = run_inference(name, img_hash, raw_bytes)
                verdict = "🦠 Pneumonia" if prob >= threshold else "✅ Normal"
                rows.append({
                    "Model":      name,
                    "Pneumonia":  f"{prob:.4f}",
                    "Normal":     f"{1 - prob:.4f}",
                    "Verdict":    verdict,
                })
                progress_bar.progress(
                    (idx + 1) / len(MODEL_REGISTRY),
                    text=f"Finished {name}",
                )

            progress_bar.empty()

            import pandas as pd
            df = pd.DataFrame(rows).set_index("Model")
            st.dataframe(df, use_container_width=True)

            # Show full card for the selected model even in compare mode
            st.markdown("---")
            st.subheader(f"Detail — {selected_model_name}")
            detail_prob = run_inference(selected_model_name, img_hash, raw_bytes)
            render_result(selected_model_name, detail_prob, threshold)

# ─────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Pneumonia Detection System · Deep Learning with CNNs & Transfer Learning"
)