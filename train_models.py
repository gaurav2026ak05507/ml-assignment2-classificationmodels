# train_models.py
#
# What this file does:
#   1. Reads the Online Shoppers dataset
#   2. Splits it into 80% training data and 20% test data
#   3. Trains 5 classification models on the training data
#   4. Tests each model on the test data and prints 6 metrics
#   5. Saves every trained model in the "model" folder, so app.py can use it
#
# How to run:   python train_models.py

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix)


# ---------------------------------------------------------------
# Step 0: Column lists
# ---------------------------------------------------------------

# The column we want to predict: did the visitor buy something? (True/False)
TARGET_COLUMN = "Revenue"

# Number columns: page counts, time spent, bounce/exit rates, etc.
NUMBER_COLUMNS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]

# Category columns. Note: OperatingSystems, Browser, Region and TrafficType
# are stored as numbers, but they are really just ID codes
# (Region 9 is not "bigger" than Region 3), so we treat them as categories.
CATEGORY_COLUMNS = ["Month", "VisitorType", "OperatingSystems",
                    "Browser", "Region", "TrafficType"]

# Yes/No column
YES_NO_COLUMNS = ["Weekend"]

ALL_INPUT_COLUMNS = NUMBER_COLUMNS + CATEGORY_COLUMNS + YES_NO_COLUMNS


# ---------------------------------------------------------------
# Step 1: Small helper functions
# ---------------------------------------------------------------

def true_false_to_number(column):
    """Changes True/False (or "TRUE"/"FALSE") values into 1/0."""
    text = column.astype(str).str.strip().str.lower()
    return text.map({"true": 1, "1": 1, "false": 0, "0": 0}).astype(int)


def split_inputs_and_answer(data):
    """
    Takes the full table and returns two things:
      inputs -> the columns the model looks at (X)
      answer -> the Revenue column the model must predict (y)
    """
    inputs = data[ALL_INPUT_COLUMNS].copy()
    inputs["Weekend"] = true_false_to_number(inputs["Weekend"])

    # Make every category column text, so the encoder treats them as labels
    for col in CATEGORY_COLUMNS:
        inputs[col] = inputs[col].astype(str)

    answer = true_false_to_number(data[TARGET_COLUMN])
    return inputs, answer


def make_data_cleaner():
    """
    Builds the preprocessing step that runs before every model.

    - Number columns: the values are very skewed (most visits are short,
      a few are extremely long). log1p squeezes the big values, then
      StandardScaler puts every column on the same scale.
      KNN, SVM and Logistic Regression need this to work well.
    - Category columns: one-hot encoding turns each category into its
      own 0/1 column (e.g. Month_Nov = 1 or 0).
    - Weekend: already 0/1, so it is passed through unchanged.
    """
    number_steps = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("numbers", number_steps, NUMBER_COLUMNS),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORY_COLUMNS),
        ("yes_no", "passthrough", YES_NO_COLUMNS),
    ])


def get_buy_scores(model, inputs):
    """
    AUC needs a "how likely is a purchase" score for every row, not just a 0/1 guess.
    Most models give a probability through predict_proba.
    Our SVM doesn't, so we use its decision_function instead
    (a bigger value means "more likely to buy"; that's all AUC needs).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(inputs)[:, 1]
    return model.decision_function(inputs)


def calculate_metrics(model, inputs, answer):
    """Runs the model on the given data and returns the 6 required metrics."""
    predicted = model.predict(inputs)
    scores = get_buy_scores(model, inputs)
    return {
        "Accuracy": accuracy_score(answer, predicted),
        "AUC": roc_auc_score(answer, scores),
        "Precision": precision_score(answer, predicted, zero_division=0),
        "Recall": recall_score(answer, predicted, zero_division=0),
        "F1": f1_score(answer, predicted, zero_division=0),
        "MCC": matthews_corrcoef(answer, predicted),
    }


# ---------------------------------------------------------------
# Step 2: The 5 models
# ---------------------------------------------------------------
# Only ~15% of visitors buy, so the data is imbalanced.
# class_weight="balanced" tells a model to pay more attention to the
# rare "buy" class. (KNN and Naive Bayes don't have this option.)
#
# The settings below (C, max_depth, n_neighbors...) were picked by trying
# a few values with 5-fold cross-validation on the training data only.

MODELS = {
    "Logistic Regression": LogisticRegression(C=0.1, class_weight="balanced",
                                              max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=5,
                                            class_weight="balanced",
                                            random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=21, weights="distance"),
    "Naive Bayes": GaussianNB(var_smoothing=1e-3),
    "Support Vector Machine": SVC(kernel="rbf", C=0.5, class_weight="balanced",
                                  random_state=42),
}

# File name used when saving each model
MODEL_FILE_NAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


# ---------------------------------------------------------------
# Step 3: Main program
# ---------------------------------------------------------------

def main():
    # 3a. Load the data
    data = pd.read_csv("data/online_shoppers_intention.csv")
    print("Dataset size:", data.shape)

    # 3b. Split 80% train / 20% test.
    # stratify keeps the same ~15% buyers in both parts.
    # random_state=42 gives the same split every time we run.
    train_data, test_data = train_test_split(
        data, test_size=0.2, stratify=data[TARGET_COLUMN], random_state=42)

    # Save the raw test rows. This is the file we upload in the Streamlit app.
    test_data.to_csv("test_data.csv", index=False)

    train_inputs, train_answer = split_inputs_and_answer(train_data)
    test_inputs, test_answer = split_inputs_and_answer(test_data)
    print("Training rows:", len(train_inputs), "| Test rows:", len(test_inputs))

    # 3c. Train, test and save each model
    os.makedirs("model", exist_ok=True)   # create the folder if it doesn't exist
    all_results = []
    for name, classifier in MODELS.items():
        # A Pipeline = data cleaning + model, joined together.
        # Saving the whole pipeline means the app can feed in the raw CSV directly.
        model = Pipeline([
            ("cleaner", make_data_cleaner()),
            ("classifier", classifier),
        ])
        model.fit(train_inputs, train_answer)
        joblib.dump(model, "model/" + MODEL_FILE_NAMES[name])

        metrics = calculate_metrics(model, test_inputs, test_answer)
        all_results.append({"Model": name, **metrics})

        tn, fp, fn, tp = confusion_matrix(test_answer, model.predict(test_inputs)).ravel()
        print(f"\n{name}")
        print(f"  correct non-buyers={tn}  false alarms={fp}  "
              f"missed buyers={fn}  caught buyers={tp}")

    # 3d. Show and save the comparison table
    results_table = pd.DataFrame(all_results).round(4)
    print("\n", results_table.to_string(index=False))
    results_table.to_csv("model/test_metrics.csv", index=False)

    # 3e. Which features did the decision tree rely on most?
    tree = joblib.load("model/decision_tree.joblib")
    feature_names = tree.named_steps["cleaner"].get_feature_names_out()
    importance = pd.Series(tree.named_steps["classifier"].feature_importances_,
                           index=feature_names)
    print("\nTop 5 features used by the Decision Tree:")
    print(importance.sort_values(ascending=False).head(5).round(3))


if __name__ == "__main__":
    main()
