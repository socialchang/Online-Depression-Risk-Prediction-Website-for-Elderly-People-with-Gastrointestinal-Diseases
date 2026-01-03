import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ========== Page config ==========
st.set_page_config(
    page_title="Online Depression Risk Prediction",
    page_icon="🧠",
    layout="wide"
)

# ========== Simple CSS (beautify) ==========
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
      .card {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(0,0,0,0.08);
        background: rgba(255,255,255,0.6);
        margin-bottom: 1rem;
      }
      .big-title {font-size: 1.65rem; font-weight: 750; margin-bottom: 0.3rem;}
      .subtle {color: rgba(0,0,0,0.6); margin-top: -0.1rem;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">Online Depression Risk Prediction Website for Elderly People with Gastrointestinal Diseases</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Enter patient information in the sidebar to obtain predicted depression risk. Optionally generate a SHAP force plot for explanation.</div>', unsafe_allow_html=True)

# ========== Paths ==========
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "ann_model.pkl"
TRAIN_CSV_CANDIDATES = [BASE_DIR / "traindata.csv", BASE_DIR / "train_data.csv", BASE_DIR / "train.csv"]

FEATURES = [
    "Gender", "Pain", "Retire", "Falldown", "Disability",
    "Self_perceived_health", "Life_satisfaction", "Eyesight",
    "ADL_score", "Sleep_time"
]

# ========== Load model artifact ==========
@st.cache_resource
def load_artifact():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

artifact = load_artifact()

model = artifact
scaler = None
if isinstance(artifact, dict) and "model" in artifact:
    model = artifact["model"]
    scaler = artifact.get("scaler", None)

# ========== Sidebar inputs ==========
st.sidebar.header("Patient Info")

gender_value = st.sidebar.selectbox(
    "Gender",
    options=[0, 1],
    format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
)

yesno_value = lambda label: st.sidebar.selectbox(
    label,
    options=[0, 1],
    format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
)

pain_value = yesno_value("Pain")
retire_value = yesno_value("Retire")
falldown_value = yesno_value("Falldown")
disability_value = yesno_value("Disability")

three_map = {"Poor (1)": 1, "Fair (2)": 2, "Good (3)": 3}

def three_class(label):
    k = st.sidebar.selectbox(label, options=list(three_map.keys()))
    return three_map[k]

self_health_value = three_class("Self_perceived_health")
life_satisfaction_value = three_class("Life_satisfaction")
eyesight_value = three_class("Eyesight")

adl_value = st.sidebar.number_input("ADL_score (0-6)", min_value=0, max_value=6, value=0, step=1)
sleep_value = st.sidebar.number_input("Sleep_time (0-24 hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

inputs = {
    "Gender": gender_value,
    "Pain": pain_value,
    "Retire": retire_value,
    "Falldown": falldown_value,
    "Disability": disability_value,
    "Self_perceived_health": self_health_value,
    "Life_satisfaction": life_satisfaction_value,
    "Eyesight": eyesight_value,
    "ADL_score": adl_value,
    "Sleep_time": sleep_value,
}
X_raw = pd.DataFrame([inputs], columns=FEATURES)

# ========== Predict helper ==========
def get_proba_and_pred(X_df: pd.DataFrame):
    X_in = X_df
    if scaler is not None:
        X_in = scaler.transform(X_df)

    if hasattr(model, "predict_proba"):
        proba = np.array(model.predict_proba(X_in))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            p1 = float(proba[0, 1])
        else:
            p1 = float(proba.reshape(-1)[0])
        pred = np.array(model.predict(X_in)).reshape(-1)[0]
        return p1, pred

    pred = np.array(model.predict(X_in)).reshape(-1)[0]
    try:
        pred_float = float(pred)
        if 0.0 <= pred_float <= 1.0:
            return pred_float, int(pred_float >= 0.5)
    except Exception:
        pass
    return None, pred

# ========== Load background for SHAP ==========
@st.cache_data
def load_background_df():
    # Prefer background.csv (public, small) for SHAP background
    bg_path = BASE_DIR / "background.csv"
    if bg_path.exists():
        try:
            df = pd.read_csv(bg_path)
            if all(c in df.columns for c in FEATURES):
                return df[FEATURES].dropna()
        except Exception:
            pass

    # Fallback to training CSV candidates if background.csv is not found
    for p in TRAIN_CSV_CANDIDATES:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if all(c in df.columns for c in FEATURES):
                    return df[FEATURES].dropna()
            except Exception:
                pass
    return None

# ========== Tabs ==========
tab_pred, tab_shap, tab_batch = st.tabs(["🧾 Single Prediction", "🧩 Explain (SHAP Force Plot)", "📦 Batch Predict"])

# -------------------- Tab 1: Prediction --------------------
with tab_pred:
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Input Summary")
        st.dataframe(X_raw.T.rename(columns={0: "Value"}), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")

        if "last_prob" not in st.session_state:
            st.session_state["last_prob"] = None
            st.session_state["last_pred"] = None

        if st.button("Predict", use_container_width=True):
            try:
                prob, pred = get_proba_and_pred(X_raw)
                st.session_state["last_prob"] = prob
                st.session_state["last_pred"] = pred
            except Exception as e:
                st.error("Prediction failed. Please copy the error below to me.")
                st.exception(e)

        prob = st.session_state.get("last_prob", None)
        pred = st.session_state.get("last_pred", None)

        if prob is not None:
            st.metric("Depression Probability", f"{prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
            if prob < 0.5:
                st.success("Low Risk (prob < 0.5)")
            else:
                st.error("High Risk (prob ≥ 0.5)")
            st.write("Predicted label (threshold=0.5):", int(prob >= 0.5))
        elif pred is not None:
            st.success(f"Prediction output: {pred}")

        with st.expander("Debug Info"):
            st.write("Model type:", type(model))
            st.write("Scaler:", "Yes" if scaler is not None else "No")
            st.write("Has predict_proba:", hasattr(model, "predict_proba"))

        st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Tab 2: SHAP Force Plot --------------------
with tab_shap:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("SHAP Force Plot (Local Explanation)")

    if not hasattr(model, "predict_proba"):
        st.warning("Your model does not support predict_proba(), so SHAP probability explanation cannot be computed.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        bg_df = load_background_df()
        if bg_df is None:
            st.warning(
                "Background data not found. Please put background.csv (with the 10 feature columns) in the same folder as app.py "
                "to enable SHAP force plot."
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # SHAP settings
            c1, c2, c3 = st.columns(3)
            with c1:
                bg_n = st.number_input("Background sample size", min_value=20, max_value=500, value=120, step=10)
            with c2:
                nsamples = st.number_input("Kernel SHAP nsamples (speed/quality)", min_value=50, max_value=1000, value=200, step=50)
            with c3:
                run_explain = st.button("Generate SHAP Force Plot", use_container_width=True)

            # prepare background
            if len(bg_df) > bg_n:
                bg_df_use = bg_df.sample(int(bg_n), random_state=0)
            else:
                bg_df_use = bg_df.copy()

            # define f(x) for KernelExplainer: return P(class=1)
            def f_np(X_np):
                X_df = pd.DataFrame(X_np, columns=FEATURES)
                X_in = X_df
                if scaler is not None:
                    X_in = scaler.transform(X_df)
                proba = np.array(model.predict_proba(X_in))
                return proba[:, 1] if (proba.ndim == 2 and proba.shape[1] >= 2) else proba.reshape(-1)

            if run_explain:
                try:
                    import shap
                    with st.spinner("Computing SHAP values... (KernelExplainer may take some time)"):
                        explainer = shap.KernelExplainer(f_np, bg_df_use.values)
                        shap_values = explainer.shap_values(X_raw.values, nsamples=int(nsamples))

                        # shap_values could be array or list; we explain class-1 probability, so it should be (1, features)
                        if isinstance(shap_values, list):
                            sv = np.array(shap_values[0]).reshape(-1)
                        else:
                            sv = np.array(shap_values).reshape(-1)

                        base = explainer.expected_value
                        if isinstance(base, (list, np.ndarray)):
                            base = float(np.array(base).reshape(-1)[0])
                        else:
                            base = float(base)

                        shap.initjs()
                        force = shap.force_plot(
                            base_value=base,
                            shap_values=sv,
                            features=X_raw.iloc[0],
                            feature_names=FEATURES,
                            link="logit"  # show in log-odds space; change to "identity" if you prefer
                        )

                        components.html(
                            f"<head>{shap.getjs()}</head><body>{force.html()}</body>",
                            height=220,
                            scrolling=True
                        )

                        st.caption(
                            "Tip: KernelExplainer is slower. Reduce background size or nsamples if it feels too slow."
                        )
                except Exception as e:
                    st.error("SHAP explanation failed. Please copy the error below to me.")
                    st.exception(e)

            st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Tab 3: Batch Predict --------------------
with tab_batch:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Batch Predict (CSV Upload)")

    uploaded = st.file_uploader("Upload a CSV (must contain the 10 feature columns)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("CSV preview:", df.head())

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"CSV missing columns: {missing}")
        else:
            try:
                Xb = df[FEATURES].copy()
                X_in = Xb
                if scaler is not None:
                    X_in = scaler.transform(Xb)

                if hasattr(model, "predict_proba"):
                    proba = np.array(model.predict_proba(X_in))
                    p1 = proba[:, 1] if (proba.ndim == 2 and proba.shape[1] >= 2) else proba.reshape(-1)
                    out = df.copy()
                    out["depression_probability"] = p1
                    out["predicted_label_0.5"] = (out["depression_probability"] >= 0.5).astype(int)
                else:
                    preds = np.array(model.predict(X_in)).reshape(-1)
                    out = df.copy()
                    out["prediction"] = preds

                st.success("Batch prediction done!")
                st.dataframe(out.head(30), use_container_width=True)

                csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Download result CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error("Batch prediction failed. Please copy the error below to me.")
                st.exception(e)

    st.markdown("</div>", unsafe_allow_html=True)
