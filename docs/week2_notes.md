# Week 2 — Modelling & Evaluation

Status: feature engineering, feature selection and the six-model comparison are done.
Still to do: hyperparameter tuning, operating-point tooling, the confusion matrix,
MLflow tracking and saving the best model.

## Scope changes

| Date | Change | Reason |
|---|---|---|
| 2026-11-05 | CatBoost uses the same one-hot encoding as the other models, not its native categorical handling | Keeps one common pipeline for all six models. Comparing the two is a PRD stretch goal. |

## 1. Feature engineering (`src/features/engineering.py`)

Feature engineering means building new columns from existing ones to make a pattern
easier for the model to see. We added total pages, total duration, time per page, time
per product page, and the share of pages and time spent on products.

The function runs as the first step of the pipeline. If it only lived in a notebook, the
API would have to re-implement it, and any small difference would give wrong predictions
without any error (train/serve skew).

Ratios need care: a session with no pages would divide by zero, so `_safe_ratio` returns 0.

**Result:** the engineered columns gave no measurable PR-AUC gain. Every difference is
smaller than the variation between folds. Reporting that is more useful than claiming
an improvement that isn't there.

## 2. Feature selection (`src/features/selection.py`)

We measure importance two ways, because each has blind spots:

- **Gain:** how much a model's splits on a feature improve its fit. It's fast, but it
  favours columns with many distinct values.
- **Permutation:** shuffle one column and see how much PR-AUC drops. It measures real
  impact on new data, but correlated columns can cover for each other, which understates both.

The rule is that a column is weak when its gain share is under 1% **and** its permutation
importance is no bigger than the noise between folds.

**Why we ranked twice.** With PageValues present, Month looked useless. Without it, Month
was no longer weak. A very strong feature can hide others, and PageValues may not be
available in production, so we only dropped columns weak in both rankings:
`Administrative`, `Informational`, `VisitorType` and `Weekend`. PR-AUC did not drop.

Dropped columns are still needed in the input. `Administrative` and `Informational` feed
`total_pages`, and `predict_sessions` currently asks for every raw column.

Selection only uses the training split. Using the test set here would be another form of leakage.

## 3. The common trainer (`src/models/trainer.py`)

`get_models()` lists the six algorithms, and `ModelTrainer` gives each the same three
methods: `build`, `cross_validate` and `fit`. Adding a model takes one line.

| Model | Idea |
|---|---|
| Logistic Regression | Weighted sum of features, turned into a probability |
| Decision Tree | A flowchart of yes/no questions |
| Random Forest | Many trees on random samples of the data, averaged |
| XGBoost, LightGBM, CatBoost | Gradient boosting: each small tree corrects the mistakes of the previous ones |

Each library turns on class weighting differently, which `_balance_params` handles:
`class_weight="balanced"`, `auto_class_weights="Balanced"` or `scale_pos_weight`.

## 4. Results (`reports/model_comparison.csv`)

PR-AUC from 5-fold stratified CV on the training split. These are starting settings,
not tuned ones. Each cell is the mean for that model's best imbalance strategy.

| Model | With PageValues | Without PageValues |
|---|---|---|
| CatBoost | 0.851 ± 0.011 | 0.532 ± 0.021 |
| XGBoost | 0.842 ± 0.007 | 0.517 ± 0.025 |
| LightGBM | 0.835 ± 0.004 | 0.505 ± 0.028 |
| Random Forest | 0.832 ± 0.012 | 0.530 ± 0.022 |
| Logistic Regression | 0.831 ± 0.016 | 0.524 ± 0.015 |
| Decision Tree | 0.807 ± 0.012 | 0.478 ± 0.011 |

A model that guesses scores about 0.157.

**How to read this:**

1. **PageValues matters far more than the choice of model.** Removing it costs about 0.3.
   The gap between the five models above the decision tree is about 0.02.
2. **No winner yet.** CatBoost leads, but its margin over XGBoost is about the size of
   the ± values. Without PageValues the top models are effectively tied.
3. **SMOTE never helped.** It had the lowest mean PR-AUC for every model in both feature
   sets, although some of those gaps are within the noise. It also creates fractional
   values in one-hot columns, which aren't real categories.
4. **Class weighting moves the trade-off, not the ranking.** CatBoost with PageValues
   has about the same PR-AUC either way (0.851 and 0.850), but recall is 0.93 with weights
   and 0.75 without, while precision is 0.60 and 0.78. Week 2's threshold step decides
   where to sit on that trade-off.

Precision and recall above use the placeholder threshold of 0.5.

## 5. Decision: PageValues is left out (data leakage)

We score sessions mid-visit, while a nudge can still change the outcome. PageValues
mostly fills in as a visitor heads to checkout: no buyer has PageValues = 0, and no
session with PageValues = 0 converts. Mid-session it is either unknown or already a
sign that the purchase is happening, so using it would be data leakage.

It is listed under `features.leakage_excluded` in `config.yaml`, separate from the
columns dropped for being weak, so the reason stays visible. The comparison scripts
put it back in for comparison only. The dashboard will show both versions.

The cost on paper is large: cross-validated PR-AUC falls from about 0.85 to about 0.53.
The lower number is the honest one for our use.

**Remaining caveat:** the other columns, such as page counts and durations, are also
whole-session totals. Mid-session they will be smaller than in the training data, so
live performance may be somewhat lower again. Training on partial sessions would need
page-level event data, which this dataset does not have.

## Running it

```bash
source .venv/bin/activate
python -m src.features.selection      # feature rankings -> reports/
python -m src.models.compare_models   # six-model comparison -> reports/
```
