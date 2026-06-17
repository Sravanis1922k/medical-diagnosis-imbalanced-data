"""
MEDISCAN — Streamlit Frontend
Upload dataset → Train → Evaluate → Predict
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import json

API = "http://localhost:8000"

st.set_page_config(page_title="MEDISCAN", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.main-title  { font-size:2rem; font-weight:700; color:#D85A30; }
.sub-title   { color:#666; font-size:0.9rem; margin-top:-8px; margin-bottom:16px; }
.pub-badge   { background:#e1f5ee; color:#0F6E56; border-radius:20px;
               padding:4px 14px; font-size:12px; font-weight:600; display:inline-block; margin-bottom:16px; }
.metric-card { background:#fff; border:1px solid #e0e0e0; border-radius:10px;
               padding:1.2rem; text-align:center; }
.metric-num  { font-size:1.8rem; font-weight:700; color:#D85A30; }
.metric-lbl  { font-size:11px; color:#888; margin-top:4px; }
.risk-high   { background:#fdecea; color:#c0392b; border-radius:8px; padding:12px 16px; font-weight:600; }
.risk-low    { background:#e8f8f0; color:#1e8449; border-radius:8px; padding:12px 16px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏥 MEDISCAN</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Medical Diagnosis for Incomplete & Imbalanced Data</div>', unsafe_allow_html=True)
st.markdown('<div class="pub-badge">📄 Published · Springer — Intelligent Data Engineering and Analytics, Feb 2022</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📂 Train Model", "📊 Evaluation", "🔬 Predict Risk"])

# ── Tab 1: Train ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload & Train")
    col1, col2 = st.columns([2,1])
    with col1:
        uploaded = st.file_uploader("Upload healthcare CSV dataset", type=["csv"])
        if uploaded:
            df_preview = pd.read_csv(uploaded)
            st.dataframe(df_preview.head(5), use_container_width=True)
            missing_pct = round(df_preview.isnull().sum().sum() / df_preview.size * 100, 2)
            m1,m2,m3 = st.columns(3)
            m1.metric("Rows", df_preview.shape[0])
            m2.metric("Columns", df_preview.shape[1])
            m3.metric("Missing %", f"{missing_pct}%")

    with col2:
        target_col = st.selectbox("Target column", options=df_preview.columns.tolist() if uploaded else [])
        test_size  = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        epochs     = st.slider("Max epochs", 10, 100, 50, 10)

        if st.button("🚀 Train Model", type="primary") and uploaded and target_col:
            uploaded.seek(0)
            with st.spinner("Preprocessing (KNN Imputation + SMOTE) + Training..."):
                res = requests.post(
                    f"{API}/train",
                    data={"target_column": target_col, "test_size": test_size, "epochs": epochs},
                    files={"file": (uploaded.name, uploaded, "text/csv")}
                )
            if res.status_code == 200:
                d = res.json()
                st.success(f"✅ Trained in {d['epochs_run']} epochs!")
                st.session_state["metrics"]  = d["metrics"]
                st.session_state["columns"]  = [c for c in df_preview.columns if c != target_col]
                st.session_state["trained"]  = True
                m = d["metrics"]
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("ROC-AUC",   m["roc_auc"])
                c2.metric("F1 Score",  m["f1_score"])
                c3.metric("Precision", m["precision"])
                c4.metric("Recall",    m["recall"])
            else:
                st.error(f"Training failed: {res.text}")

# ── Tab 2: Evaluation ────────────────────────────────────────────────────────
with tab2:
    st.subheader("Model Evaluation")
    if st.session_state.get("metrics"):
        m = st.session_state["metrics"]
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-num">{m["roc_auc"]}</div><div class="metric-lbl">ROC-AUC</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-num">{m["f1_score"]}</div><div class="metric-lbl">F1 Score</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-num">{m["precision"]}</div><div class="metric-lbl">Precision</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-num">{m["recall"]}</div><div class="metric-lbl">Recall</div></div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("Confusion Matrix")
        cm = m["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual Healthy","Actual At Risk"], columns=["Pred Healthy","Pred At Risk"])
        st.dataframe(cm_df, use_container_width=False)

        st.divider()
        st.subheader("Classification Report")
        report = m["classification_report"]
        rows = []
        for label in ["Healthy", "At Risk"]:
            if label in report:
                r = report[label]
                rows.append({"Class": label, "Precision": round(r["precision"],3),
                             "Recall": round(r["recall"],3), "F1": round(r["f1-score"],3),
                             "Support": int(r["support"])})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Train a model first in the **Train Model** tab.")

# ── Tab 3: Predict ───────────────────────────────────────────────────────────
with tab3:
    st.subheader("Predict Disease Risk")
    if not st.session_state.get("trained"):
        st.info("Train a model first in the **Train Model** tab.")
    else:
        cols = st.session_state.get("columns", [])
        st.write("Enter patient values:")
        feature_dict = {}
        grid = st.columns(3)
        for i, col in enumerate(cols):
            feature_dict[col] = grid[i % 3].number_input(col, value=0.0, format="%.4f")

        if st.button("🔬 Predict Risk", type="primary"):
            res = requests.post(f"{API}/predict", json={"features": feature_dict, "return_probability": True})
            if res.status_code == 200:
                d = res.json()
                if d["prediction"] == "At Risk":
                    st.markdown(f'<div class="risk-high">⚠️ PREDICTION: AT RISK &nbsp;|&nbsp; Risk Score: {d["risk_score"]} &nbsp;|&nbsp; Confidence: {d["confidence"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low">✅ PREDICTION: HEALTHY &nbsp;|&nbsp; Risk Score: {d["risk_score"]} &nbsp;|&nbsp; Confidence: {d["confidence"]}</div>', unsafe_allow_html=True)

                st.divider()
                c1, c2 = st.columns(2)
                c1.metric("Probability Healthy", f"{d['probability_healthy']*100:.1f}%")
                c2.metric("Probability At Risk",  f"{d['probability_at_risk']*100:.1f}%")
            else:
                st.error(f"Prediction failed: {res.text}")
