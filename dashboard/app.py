from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "fraud_model.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Fraud Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- DARK NAVY THEME ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #07111f 0%, #0b1730 100%);
        color: #edf3ff;
    }
    [data-testid="stSidebar"] {
        background: #091426;
        border-right: 1px solid #24334d;
    }
    h1, h2, h3, p, label {
        color: #edf3ff !important;
    }
    div[data-testid="stMetric"] {
        background: #111f35;
        border: 1px solid #293b57;
        padding: 18px;
        border-radius: 14px;
    }
    div[data-testid="stMetricLabel"] p {
        color: #a8b9d4 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #f5f8ff !important;
    }
    .panel {
        background: #101e33;
        border: 1px solid #263954;
        padding: 18px;
        border-radius: 14px;
        margin-bottom: 12px;
    }
    .small-note {
        color: #9fb0ca;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- LOAD FILES ----------
@st.cache_data
def load_data(path, modified_time):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


@st.cache_data
def load_metrics(path, modified_time):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def get_predictions(data, model_path, model_modified_time):
    loaded_model = joblib.load(model_path)
    features = data.drop(columns=["Class"])
    return loaded_model.predict_proba(features)[:, 1]


# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.title("🛡️ Fraud Intelligence")
st.sidebar.caption("Transaction Monitoring Platform")
st.sidebar.divider()
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Go to",
    ["📊 Overview", "🔎 Transaction Analysis", "🧠 Model Performance"],
    label_visibility="collapsed",
)

st.sidebar.divider()

# ---------- VALIDATE AND LOAD PROJECT FILES ----------
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Place creditcard.csv inside the data/raw/ folder.")
    st.stop()

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("Trained model or metrics file is missing.")
    st.info("Run this command from the project root: python src/train.py")
    st.stop()

try:
    df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    metrics = load_metrics(str(METRICS_PATH), METRICS_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load project files: {exc}")
    st.stop()

if "Class" not in df.columns:
    st.error("Dataset must contain the Class column.")
    st.stop()

feature_columns = [column for column in df.columns if column != "Class"]
model_features = list(getattr(model, "feature_names_in_", feature_columns))
if model_features != feature_columns:
    st.error("Dataset feature columns do not match the trained model.")
    st.stop()

# ---------- SHARED THRESHOLD AND PREDICTIONS ----------
threshold = st.sidebar.slider(
    "Fraud alert threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
)
st.sidebar.caption("Adjust the threshold to change predicted fraud alerts.")

try:
    probabilities = get_predictions(
        df, str(MODEL_PATH), MODEL_PATH.stat().st_mtime
    )
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

df = df.copy()
df["Fraud Probability"] = probabilities
df["Predicted Class"] = (df["Fraud Probability"] >= threshold).astype(int)
df["Actual Status"] = df["Class"].map({0: "Legitimate", 1: "Fraud"})
df["Predicted Status"] = df["Predicted Class"].map(
    {0: "Legitimate", 1: "Fraud Alert"}
)

total_transactions = len(df)
actual_frauds = int((df["Class"] == 1).sum())
fraud_rate = actual_frauds / total_transactions * 100 if total_transactions else 0
predicted_frauds = int((df["Predicted Class"] == 1).sum())

# ---------- COMMON HEADER ----------
st.title("🛡️ Fraud Risk Intelligence Platform")
st.markdown(
    '<p class="small-note">Credit card transaction monitoring • '
    'Machine-learning risk analysis</p>',
    unsafe_allow_html=True,
)
st.divider()

# ==================== OVERVIEW PAGE ====================
if page == "📊 Overview":
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Transactions", f"{total_transactions:,}")
    k2.metric("Actual Fraud Cases", f"{actual_frauds:,}")
    k3.metric("Actual Fraud Rate", f"{fraud_rate:.3f}%")
    k4.metric("Predicted Fraud Alerts", f"{predicted_frauds:,}")

    st.caption(
        "Actual fraud figures use the dataset's Class labels. "
        "Predicted alerts depend on the selected threshold."
    )

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Fraud vs Legitimate Transactions")
        counts = (
            df["Actual Status"]
            .value_counts()
            .reindex(["Legitimate", "Fraud"], fill_value=0)
            .rename_axis("Status")
            .reset_index(name="Transactions")
        )
        fig = px.pie(
            counts,
            names="Status",
            values="Transactions",
            hole=0.68,
            color="Status",
            color_discrete_map={
                "Legitimate": "#25c7a0",
                "Fraud": "#ff5573",
            },
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#edf3ff",
            legend_title_text="",
            margin=dict(t=15, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Fraud Cases Over Transaction Time")
        time_df = df[["Time", "Class"]].copy()
        time_df["Time Window"] = (time_df["Time"] // 3600).astype(int)
        hourly = (
            time_df.groupby("Time Window")["Class"]
            .agg(["sum", "count"])
            .reset_index()
        )
        hourly["Fraud Rate (%)"] = hourly["sum"] / hourly["count"] * 100
        fig = px.line(
            hourly,
            x="Time Window",
            y="Fraud Rate (%)",
            markers=True,
            labels={
                "Time Window": "Time window (hours from dataset start)",
                "Fraud Rate (%)": "Actual fraud rate (%)",
            },
        )
        fig.update_traces(line_color="#45a5ff", marker_color="#45a5ff")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#edf3ff",
            margin=dict(t=15, b=10, l=10, r=10),
        )
        fig.update_xaxes(gridcolor="#263954")
        fig.update_yaxes(gridcolor="#263954")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Feature Importance")
    classifier = (
        model.named_steps.get("classifier")
        if hasattr(model, "named_steps")
        else None
    )

    if classifier is not None and hasattr(classifier, "coef_"):
        importance = pd.DataFrame(
            {
                "Feature": feature_columns,
                "Importance": abs(classifier.coef_[0]),
            }
        ).sort_values("Importance", ascending=False).head(12)

        fig = px.bar(
            importance.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            labels={"Importance": "Absolute coefficient", "Feature": ""},
        )
        fig.update_traces(marker_color="#8b7cff")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#edf3ff",
            margin=dict(t=10, b=10, l=10, r=10),
        )
        fig.update_xaxes(gridcolor="#263954")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Absolute logistic-regression coefficients show model influence "
            "on scaled inputs; they are not causal explanations."
        )
    else:
        st.info("Feature importance is unavailable for this model type.")
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== TRANSACTION ANALYSIS PAGE ====================
elif page == "🔎 Transaction Analysis":
    st.subheader("Transaction Analysis")
    st.caption(
        "Explore transactions from the downloaded credit-card dataset. "
        "These are dataset records, not a live bank feed."
    )

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
    with filter_col1:
        status_filter = st.selectbox(
            "Prediction filter",
            ["All transactions", "Fraud alerts only", "Predicted legitimate only"],
        )
    with filter_col2:
        actual_filter = st.selectbox(
            "Actual label filter",
            ["All labels", "Actual fraud only", "Actual legitimate only"],
        )
    with filter_col3:
        rows_to_show = st.selectbox("Rows to display", [15, 30, 50, 100], index=0)

    filtered = df.copy()
    if status_filter == "Fraud alerts only":
        filtered = filtered[filtered["Predicted Class"] == 1]
    elif status_filter == "Predicted legitimate only":
        filtered = filtered[filtered["Predicted Class"] == 0]

    if actual_filter == "Actual fraud only":
        filtered = filtered[filtered["Class"] == 1]
    elif actual_filter == "Actual legitimate only":
        filtered = filtered[filtered["Class"] == 0]

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.metric("Transactions matching filters", f"{len(filtered):,}")
    display_columns = [
        column
        for column in [
            "Time",
            "Amount",
            "Actual Status",
            "Fraud Probability",
            "Predicted Status",
        ]
        if column in filtered.columns
    ]
    st.dataframe(
        filtered[display_columns].sort_values("Time", ascending=False).head(rows_to_show),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fraud Probability": st.column_config.ProgressColumn(
                "Fraud Probability",
                min_value=0,
                max_value=1,
                format="%.3f",
            ),
            "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        "⬇️ Download filtered transactions as CSV",
        data=filtered[display_columns].to_csv(index=False).encode("utf-8"),
        file_name="filtered_fraud_transactions.csv",
        mime="text/csv",
    )

# ==================== MODEL PERFORMANCE PAGE ====================
elif page == "🧠 Model Performance":
    st.subheader("Model Performance")
    st.caption("Evaluation results saved during training on the held-out test set.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{metrics['precision']:.3f}")
    m2.metric("Recall", f"{metrics['recall']:.3f}")
    m3.metric("F1-score", f"{metrics['f1_score']:.3f}")
    m4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    cm = metrics.get("confusion_matrix")
    if cm and len(cm) == 2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Legitimate", "Actual Fraud"],
            columns=["Predicted Legitimate", "Predicted Fraud"],
        )
        st.dataframe(cm_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # A visual heatmap using the same saved confusion-matrix values.
        import plotly.graph_objects as go

        heatmap = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted Legitimate", "Predicted Fraud"],
                y=["Actual Legitimate", "Actual Fraud"],
                colorscale=[[0, "#101e33"], [1, "#45a5ff"]],
                text=cm,
                texttemplate="%{text}",
                showscale=False,
            )
        )
        heatmap.update_layout(
            title="Confusion Matrix Heatmap",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#edf3ff",
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.warning("A 2×2 confusion matrix was not found in metrics.json.")

    st.info(
        "Precision, recall, F1-score, and ROC-AUC are calculated on the held-out "
        "test set during training. Changing the sidebar alert threshold changes "
        "dashboard alerts, but does not recalculate these saved test metrics."
    )

st.divider()
st.caption(
    "Educational project: model alerts require validation before real-world use. "
    "Dataset predictions are not a live connection to a bank."
)
