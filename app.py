# app.py
#
# Streamlit web app for the assignment.
#   1. Upload a test CSV file (or use the included test_data.csv)
#   2. Pick one of the 5 saved models from a dropdown
#   3. See the 6 metrics, the confusion matrix and the classification report
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


st.set_page_config(page_title="Online Shopper Prediction", layout="wide")
st.title("Online Shopper Purchase Prediction")
st.write("Predicts whether a website visit ends in a purchase, using 5 ML models.")


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
st.sidebar.header("Settings")
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


# ---------- Step 3: Metrics for the chosen model ----------
model = models[chosen_model]
predicted = model.predict(inputs)
metrics = calculate_metrics(model, inputs, answer)

st.subheader(f"Results for {chosen_model}")
st.write(f"Rows tested: {len(answer)}  |  Actual buyers: {answer.sum()} "
         f"({answer.mean():.1%})")

# Show the 6 metrics side by side
metric_columns = st.columns(6)
for box, (metric_name, value) in zip(metric_columns, metrics.items()):
    box.metric(metric_name, f"{value:.4f}")


# ---------- Step 4: Confusion matrix and classification report ----------
left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    matrix = confusion_matrix(answer, predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No buy", "Buy"], yticklabels=["No buy", "Buy"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification Report")
    report = classification_report(answer, predicted, labels=[0, 1],
                                   target_names=["No buy", "Buy"],
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3))

    tn, fp, fn, tp = matrix.ravel()
    st.write(f"The model caught **{tp}** of {tp + fn} buyers and missed **{fn}**. "
             f"It wrongly flagged **{fp}** non-buyers as buyers.")


# ---------- Step 5: Compare all 5 models ----------
st.subheader("Comparison of All Models")
rows = []
for name, m in models.items():
    rows.append({"Model": name, **calculate_metrics(m, inputs, answer)})
comparison = pd.DataFrame(rows).set_index("Model").round(4)
st.dataframe(comparison)

best_model = comparison["MCC"].idxmax()
st.write(f"Best model by MCC: **{best_model}**")
