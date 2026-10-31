# Online Shopper Purchase Prediction

M.Tech (AIML) — Machine Learning, Assignment 2

**Live app:** https://ml-assignment2-classificationmodels-dxp6haqmwv67gtyglkp2u7.streamlit.app/
**Repository:** <https://github.com/gaurav2026ak05507/ml-assignment2-classificationmodels>

---

## a. Problem statement

- **What we are solving:** an online shop wants to know whether a visitor will buy something, based only on how they behave on the website during that visit.
- **Type of problem:** binary classification. The model answers Yes (purchase) or No (no purchase) for each visit, using the `Revenue` column as the answer.
- **Example used:** the UCI *Online Shoppers Purchasing Intention* dataset: 12,330 real visits to one shopping website over a year. For each visit we know things like how many pages were viewed, how long the visitor stayed, bounce and exit rates, page value, the month and the visitor type.
- **Why this example:** it is a real business problem (a shop can show offers to likely buyers), it is big enough for all five models, and only about 15.5% of visits end in a purchase. That imbalance makes it a good test of how to judge models fairly.
- **How we judge the models:** a model that always says "no purchase" would already be 84.5% accurate while finding zero buyers, so accuracy alone is not enough. We mainly compare models on **MCC** and **F1**, which only score high when the model actually finds buyers, and report accuracy, AUC, precision and recall alongside.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository — *Online Shoppers Purchasing Intention Dataset* (ID 468); Sakar et al., *Neural Computing and Applications*, 2019 |
| Instances | 12,330 sessions (each from a different user, over one year) |
| Features | 17 (10 numeric, 7 categorical/binary) |
| Target | `Revenue` — 10,422 False (84.5%), 1,908 True (15.5%) |
| Missing values | None |
| Duplicates | 125 identical rows; kept, since repeated short sessions are plausible real behaviour |

**Numeric features**
- `Administrative`, `Informational`, `ProductRelated`: number of pages of each type visited
- `*_Duration`: total seconds spent on each page type
- `BounceRates`, `ExitRates`, `PageValues`: Google Analytics metrics averaged over the pages visited
- `SpecialDay`: closeness of the session date to a special day such as Valentine's Day (0 to 1)

**Categorical features**
- `Month` (10 values; January and April are absent), `VisitorType` (Returning, New, Other), `Weekend` (bool)
- `OperatingSystems`, `Browser`, `Region`, `TrafficType`: integer codes with no natural order

**Preprocessing (inside each saved sklearn Pipeline)**
- Numeric: `log1p` (all are non-negative and heavily right-skewed), then `StandardScaler`
- Categorical, including the integer-coded ones: `OneHotEncoder(handle_unknown="ignore")`
- `Weekend`: mapped to 0/1
- Split: stratified 80/20, `random_state=42` → 9,864 train / 2,466 test. The raw test split is saved as `test_data.csv`.
- Hyper-parameters: a few values per model were compared with 5-fold stratified CV on the training set only (scored by F1); the chosen values are fixed in `train_models.py`
- Imbalance: `class_weight="balanced"` for Logistic Regression, Decision Tree and SVM (KNN and Gaussian NB have no such option)

## c. GitHub repository

<https://github.com/gaurav2026ak05507/ml-assignment2-classificationmodels>

```
online-shoppers-ml/
├── .streamlit/config.toml  # Light theme for the app
├── app.py                  # Streamlit app (loads saved models; never retrains)
├── train_models.py         # Preprocessing, tuning, training, evaluation
├── requirements.txt
├── README.md
├── test_data.csv           # Hold-out test split (raw features + Revenue)
├── data/
│   └── online_shoppers_intention.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── test_metrics.csv
```

Run locally (from inside the project folder):

```bash
pip install -r requirements.txt
python train_models.py      # trains the 5 models (~20 s), saves model/ and test_data.csv
streamlit run app.py        # opens the app at http://localhost:8501
```

## d. Models used

All five models were trained on the same training split and evaluated on the same 2,466-row hold-out test set. For SVM, AUC is computed from the `decision_function` margin.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8642 | **0.9161** | 0.5423 | 0.7880 | 0.6425 | 0.5771 |
| Decision Tree | 0.8386 | 0.9059 | 0.4879 | **0.8429** | 0.6180 | 0.5563 |
| KNN | **0.8905** | 0.8992 | **0.6944** | 0.5236 | 0.5970 | 0.5423 |
| Naive Bayes (Gaussian) | 0.6164 | 0.7871 | 0.2658 | 0.8377 | 0.4035 | 0.2993 |
| Support Vector Machine (RBF) | 0.8633 | 0.9143 | 0.5386 | 0.8220 | **0.6508** | **0.5901** |

Tuned settings: LR `C=0.1`; Decision Tree `max_depth=6, min_samples_leaf=5`; KNN `k=21, weights="distance"`; Gaussian NB `var_smoothing=1e-3`; SVM `C=0.5`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.916), so it ranks buyers above non-buyers best of all five. With balanced class weights it recovers 79% of buyers at 54% precision. The log transform makes the dominant `PageValues` signal close to linear, which is why a linear model does so well here. Strong regularisation (`C=0.1`) was preferred, suggesting the one-hot features add little beyond the numeric ones. |
| Decision Tree | Highest recall (84%) but the lowest precision of the four competitive models (49%): it flags 338 false positives. `PageValues` carries 81% of the tree's feature importance and `Month=Nov` another 6%, so the tree is effectively a few thresholds on page value plus seasonality. Easy to interpret, but its step-wise splits are less precise than smooth decision boundaries. |
| KNN | Highest accuracy (0.891) and precision (0.694), yet the lowest recall (52%): it misses 182 of 382 buyers. KNN cannot apply class weights, so the majority class dominates most neighbourhoods. This is the clearest example in the table of accuracy being misleading on imbalanced data: best accuracy, but the lowest MCC of the four competitive models. |
| Naive Bayes | Clearly the weakest model. Accuracy (0.616) is below the 84.5% "always no" baseline, and precision is only 27%. Its independence assumption is badly violated: each page count is strongly correlated with its duration, and `BounceRates` with `ExitRates`. Gaussian likelihoods also fit poorly to the many one-hot 0/1 columns. It still catches 84% of buyers, but only by over-predicting purchases. |
| Support Vector Machine | Best overall balance: highest F1 (0.651) and MCC (0.590), with AUC (0.914) nearly matching Logistic Regression. The RBF kernel captures non-linear interactions (for example, page value combined with exit rate) that the linear model misses. It is the slowest model to train and its predictions are the hardest to explain. |
| **Overall winner** | **Support Vector Machine**, on MCC and F1, the metrics that matter for this imbalanced problem. Logistic Regression is a close second (best AUC, MCC only 0.013 lower) and would be a sensible production choice where interpretability and speed matter. |

## Streamlit app features

**Required features**
- **Upload test data (CSV):** upload your own test file from the sidebar. If nothing is uploaded, the app uses the included `test_data.csv`.
- **Model dropdown:** choose any of the five saved models.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model, shown as cards.
- **Confusion matrix and classification report:** shows how many buyers the model caught or missed, and how many non-buyers it wrongly flagged.

**Extra features**
- **Problem statement** at the top of the page, in simple bullet points.
- **Comparison of all five models** in one table, with the best value in each column highlighted.
- **"Why this model is best" explanation** in plain language: why accuracy can mislead on this data, what MCC and F1 measure, and why the winning model beats the others. It is calculated from the actual results, so it stays correct if a different test file is uploaded.
- **Data sources panel** on the right, with clickable links to view the full dataset and the test file on GitHub, plus the original UCI dataset page.
- **Data preview and download:** a quick look at the rows being scored, and a button to download `test_data.csv`.
- **Clean light theme** (white background, blue accents), set in `.streamlit/config.toml`.

**How the app works:** the uploaded CSV must contain all 17 feature columns and the `Revenue` column. The app only loads the saved models from `model/` and uses them to predict. It never retrains a model on the uploaded data.
