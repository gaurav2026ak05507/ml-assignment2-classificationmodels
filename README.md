# Online Shopper Purchase Prediction

M.Tech (AIML) — Machine Learning, Assignment 2

**Live app:** <your-streamlit-app-link>
**Repository:** <https://github.com/gaurav2026ak05507/ml-assignment2-classificationmodels>

---

## a. Problem statement

An e-commerce site wants to know, from the behaviour recorded during a browsing session, whether that session will end in a purchase. This is a **binary classification** problem: predict `Revenue` (True = the session produced a transaction) from page-visit counts, time spent, Google Analytics engagement metrics, and session context such as month, visitor type and traffic source.

The classes are imbalanced (about 15.5% of sessions convert), so a model that always predicts "no purchase" already scores 84.5% accuracy. Models are therefore judged mainly on **F1 and MCC**, which reward correctly finding the minority class, with accuracy and AUC reported alongside.

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

- CSV upload for test data (the included `test_data.csv` is used if nothing is uploaded)
- Model selection dropdown covering all five saved models
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix and classification report for the selected model
- Comparison table of all five models, with the best model by MCC

The uploaded CSV must contain all 17 feature columns and the `Revenue` column. The app loads the saved pipelines and only calls `predict`; it never retrains.
