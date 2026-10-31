# app.py
#
# Streamlit web app for the assignment.
#   1. Upload a test CSV file (or use the included test_data.csv)
#   2. Pick one of the 5 saved models from a dropdown
#   3. See the 6 metrics, the confusion matrix and the classification report
#   4. Compare all 5 models, with a plain-language reason for the winner
#
# The models are loaded from the "model" folder. Nothing is retrained here.
#
# How to run:   streamlit run app.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

# Reuse the helper functions from train_models.py so the app
# prepares data in exactly the same way as during training.
from train_models import (ALL_INPUT_COLUMNS, TARGET_COLUMN, MODEL_FILE_NAMES,
                          split_inputs_and_answer, calculate_metrics)

# Links shown in the right-hand "Data sources" panel
REPO = "https://github.com/gaurav2026ak05507/ml-assignment2-classificationmodels"
SOURCE_CSV_LINK = REPO + "/blob/main/data/online_shoppers_intention.csv"
TEST_CSV_LINK = REPO + "/blob/main/test_data.csv"
UCI_LINK = "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset"

BLUE = "#2563eb"

st.set_page_config(page_title="Online Shopper Prediction", page_icon="🛒", layout="wide")

# ---------- Page styling (light theme, card look) ----------
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
h1, h2, h3 {color: #0f172a;}
.hero {background: #f8fafc; border: 1px solid #e2e8f0; border-left: 5px solid #2563eb;
       border-radius: 10px; padding: 1rem 1.4rem; margin-bottom: 1.2rem;}
.hero ul {margin: 0.3rem 0 0 0; padding-left: 1.2rem;}
.hero li {margin-bottom: 0.25rem; color: #334155;}
.side-card {background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
            padding: 1rem 1.1rem; margin-bottom: 1rem;}
.side-card h4 {font-size: 1.05rem; margin: 0 0 0.6rem 0;}
.side-card a {color: #2563eb; text-decoration: none; font-weight: 600; word-break: break-all;}
.side-card p {margin: 0.2rem 0 0.7rem 0; color: #475569; font-size: 0.9rem;}
div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e2e8f0;
       border-radius: 10px; padding: 0.6rem 0.9rem;}
.reason {background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px;
         padding: 1rem 1.3rem; color: #1e293b;}
.reason li {margin-bottom: 0.35rem;}
</style>
""", unsafe_allow_html=True)


# ---------- Step 1: Load the saved models ----------
# @st.cache_resource loads the models only once, not on every click.
@st.cache_resource
def load_saved_models():
    models = {}
    for name, file_name in MODEL_FILE_NAMES.items():
        models[name] = joblib.load("model/" + file_name)
    return models


models = load_saved_models()


# ---------- Step 2: Sidebar inputs ----------
st.sidebar.header("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
chosen_model = st.sidebar.selectbox("Choose a model", list(MODEL_FILE_NAMES.keys()))

# Use the uploaded file if there is one, otherwise use the included test_data.csv
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Using uploaded file: " + uploaded_file.name)
else:
    data = pd.read_csv("test_data.csv")
    st.sidebar.info("No file uploaded. Using the included test_data.csv.")

# Check that the file has all the columns the models need
missing_columns = [c for c in ALL_INPUT_COLUMNS + [TARGET_COLUMN] if c not in data.columns]
if missing_columns:
    st.error("These columns are missing from the CSV: " + ", ".join(missing_columns))
    st.stop()

inputs, answer = split_inputs_and_answer(data)


# ---------- Page layout: main area (left) + data sources panel (right) ----------
main, side = st.columns([3.2, 1], gap="large")

with side:
    st.markdown(f"""
    <div class="side-card">
      <h4>📂 Data sources</h4>
      <a href="{SOURCE_CSV_LINK}" target="_blank">online_shoppers_intention.csv ↗</a>
      <p>Full source dataset: 12,330 sessions, 18 columns. Opens on GitHub, where you can scroll and search it.</p>
      <a href="{TEST_CSV_LINK}" target="_blank">test_data.csv ↗</a>
      <p>The 20% hold-out test split (2,466 rows) scored on this page.</p>
      <a href="{UCI_LINK}" target="_blank">UCI dataset page ↗</a>
      <p>Original source, column descriptions and citation.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Peek at the data being scored"):
        st.dataframe(data.head(20), height=260)
    st.download_button("⬇️ Download test_data.csv", open("test_data.csv", "rb").read(),
                       file_name="test_data.csv", mime="text/csv", width="stretch")

with main:
    # ---------- Title and problem statement ----------
    st.title("🛒 Online Shopper Purchase Prediction")
    st.markdown("""
    <div class="hero">
      <b>Problem statement</b>
      <ul>
        <li><b>What we are solving:</b> predict whether a visit to an online shop will end in a purchase,
            using only how the visitor behaved during that session.</li>
        <li><b>Example used:</b> the UCI <i>Online Shoppers Purchasing Intention</i> dataset: 12,330 real
            sessions from one e-commerce site over a year, with 17 features such as pages visited, time spent,
            bounce/exit rates, page value, month and visitor type.</li>
        <li><b>Target:</b> <code>Revenue</code>: True if the session ended in a purchase (only about 15.5% did).</li>
        <li><b>Why this example:</b> it is a real business problem (spotting likely buyers lets a shop target offers),
            it has enough rows and features for all five models, and its imbalance shows why
            accuracy alone can mislead, which makes the model comparison meaningful.</li>
        <li><b>How:</b> 5 classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) were trained on 80%
            of the data. This page scores the remaining 20% with the saved models, with no retraining.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Step 3: Metrics for the chosen model ----------
    model = models[chosen_model]
    predicted = model.predict(inputs)
    metrics = calculate_metrics(model, inputs, answer)

    st.subheader(f"Results for {chosen_model}")
    st.caption(f"Rows tested: {len(answer):,}  |  Actual buyers: {answer.sum()} ({answer.mean():.1%})")

    row1 = st.columns(3)
    row2 = st.columns(3)
    for box, (metric_name, value) in zip(row1 + row2, metrics.items()):
        box.metric(metric_name, f"{value:.4f}")

    # ---------- Step 4: Confusion matrix and classification report ----------
    left, right = st.columns([1, 1.3], gap="medium")

    with left:
        st.markdown("##### Confusion Matrix")
        matrix = confusion_matrix(answer, predicted, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(3.2, 2.6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No buy", "Buy"], yticklabels=["No buy", "Buy"],
                    annot_kws={"size": 11})
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, width="content")

    with right:
        st.markdown("##### Classification Report")
        report = classification_report(answer, predicted, labels=[0, 1],
                                       target_names=["No buy", "Buy"],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().round(3), width="stretch")

        tn, fp, fn, tp = matrix.ravel()
        st.markdown(f"The model caught **{tp}** of {tp + fn} buyers and missed **{fn}**. "
                    f"It wrongly flagged **{fp}** non-buyers as buyers.")

    st.divider()

    # ---------- Step 5: Compare all 5 models ----------
    st.subheader("Comparison of All Models")
    rows = []
    for name, m in models.items():
        rows.append({"Model": name, **calculate_metrics(m, inputs, answer)})
    comparison = pd.DataFrame(rows).set_index("Model")

    st.dataframe(comparison.style.format("{:.4f}").highlight_max(axis=0, color="#dbeafe"),
                 width="stretch")
    st.caption("Shaded cell = best value in that column.")

    # ---------- Step 6: Explain the winner in plain language ----------
    ranked = comparison.sort_values("MCC", ascending=False)
    best, second = ranked.index[0], ranked.index[1]
    most_accurate = comparison["Accuracy"].idxmax()
    baseline = 1 - answer.mean()
    b = comparison.loc[best]

    accuracy_note = ""
    if most_accurate != best:
        accuracy_note = (
            f"<li><b>Why not the most accurate model?</b> {most_accurate} has the highest accuracy "
            f"({comparison.loc[most_accurate, 'Accuracy']:.1%}), but it catches only "
            f"{comparison.loc[most_accurate, 'Recall']:.0%} of buyers. High accuracy here mostly comes "
            f"from correctly saying \"no\" to the many non-buyers.</li>")

    st.markdown(f"""
    <div class="reason">
      <b>🏆 Why {best} is the best model here</b>
      <ul>
        <li><b>Accuracy alone is misleading:</b> only {answer.mean():.1%} of sessions are purchases, so a model that
            always says "no purchase" would already be {baseline:.1%} accurate while finding zero buyers.</li>
        <li><b>So we rank by MCC</b> (Matthews Correlation Coefficient). It uses all four boxes of the confusion
            matrix (caught buyers, missed buyers, false alarms, correct "no"s) and only scores high when the model
            does well on <i>both</i> classes. It ranges from -1 to 1, and 0 means no better than guessing.
            F1 is a second check: it balances precision (how many flagged buyers were real) and recall
            (how many real buyers were found).</li>
        <li><b>{best}</b> has the highest MCC (<b>{b['MCC']:.3f}</b>) and an F1 of {b['F1']:.3f}. It finds
            {b['Recall']:.0%} of real buyers, and {b['Precision']:.0%} of the sessions it flags really are buyers.</li>
        <li><b>Runner-up:</b> {second}, with MCC {ranked.loc[second, 'MCC']:.3f}: very close, so either would be a
            reasonable choice in practice.</li>
        {accuracy_note}
      </ul>
    </div>
    """, unsafe_allow_html=True)
