# Week 2 — Modelling & Evaluation

Status: complete. Feature engineering and selection, six models compared and tuned,
operating-point tooling, the final model on the test set, and MLflow tracking.

## Scope changes

| Date | Change | Reason |
|---|---|---|
| 2026-11-13 | PageValues left out of the model | Data leakage: we score mid-session (section 5). |
| 2026-11-13 | No value-based threshold chosen by us | The value of a conversion and the cost of a nudge are unknown. The CRO team owns that trade-off, so we deliver operating-point tooling and a neutral default (section 7). |
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

## 6. Tuning (`src/models/tune_models.py`)

**Hyperparameters** are settings chosen before training, such as tree depth or learning
rate. **Randomised search** tries 20 random combinations per model and keeps the one with
the best mean PR-AUC over the same 5 folds.

Imbalance strategy is now `none` for every model: SMOTE never helped, and class weights
only moved the precision/recall trade-off (which the threshold handles) while inflating
the probabilities the CRO team will read.

| Model | Before tuning | After tuning |
|---|---|---|
| XGBoost | 0.513 | 0.538 ± 0.019 |
| CatBoost | 0.532 | 0.538 ± 0.020 |
| Random Forest | 0.529 | 0.537 ± 0.019 |
| Logistic Regression | 0.525 | 0.535 ± 0.018 |
| LightGBM | 0.500 | 0.524 ± 0.019 |
| Decision Tree | 0.478 | 0.487 ± 0.015 |

XGBoost and CatBoost are tied. XGBoost was picked by the rule decided in advance (highest
mean), but CatBoost would be just as fair a choice. Several best settings sit at the edge
of their range (XGBoost `max_depth` = 2), so the models want to be simple; a wider search
might add a little more. Tuning and choosing on the same folds makes the CV scores
slightly optimistic, which is why the test set is used at the end.

## 7. Operating points instead of one "right" threshold (`src/models/threshold.py`)

Every session gets a probability. A **threshold** turns it into a decision: nudge or not.
Choosing it is a business decision: the value of a reached buyer against the cost of a
nudge. We don't know either number, and the growth/CRO team owns that trade-off, so the
project delivers the tooling to make it:

- `operating_points()`: precision, recall, share of sessions flagged and the confusion counts at every threshold from 0.01 to 0.99.
- `add_expected_value()`: net value = buyers reached × value − sessions nudged × cost.
- `value_threshold()`: the threshold with the best net value, or `None` if nudging never pays.
- `default_threshold()`: a neutral default until the team picks one, the threshold with the best F1.

The default threshold is set from **out-of-fold** predictions on the training split: each
training row is scored by a model trained without it. Choosing it on the test set would
leak test information into a decision.

A useful rule of thumb: nudging a group of sessions pays only if its precision is above
cost ÷ value. "Value" should be the *extra* value a nudge brings, because many flagged
buyers would buy anyway.

## 8. Final model (`src/models/train_final.py`)

Tuned XGBoost, without PageValues. Default threshold 0.27. Test set, 2,400 sessions:

| Metric | Value |
|---|---|
| PR-AUC | 0.559 (CV: 0.538) |
| Recall | 0.610 |
| Precision | 0.529 |
| F1 | 0.567 |
| ROC-AUC | 0.849 |
| Accuracy | 0.853 |

| | Predicted no | Predicted buy |
|---|---|---|
| Actual no | 1,818 | 205 |
| Actual buy | 147 | 230 |

So 18% of sessions get flagged, 53% of those buy (16% of all sessions do), and 61% of
buyers are reached. The test PR-AUC is close to the CV score, so there is no sign of overfitting.

A second model, `purchase_intent_with_page_values`, is trained the same way for the
dashboard comparison only (test PR-AUC 0.867). It reuses the parameters tuned without
PageValues.

Models are saved with version numbers (`models/purchase_intent_v1.joblib` plus a `.json`
with the threshold, metrics and parameters). `predict.py` uses the newest version.

## 9. MLflow (`src/models/tracking.py`)

MLflow keeps a record of every run: parameters, metrics and files. Tuning logs one run
per model; final training logs both variants with the model, its **signature** (expected
input columns and output) and the operating points. The real model is added to the
**model registry** as `purchase-intent`, which gives it a version number to deploy or roll back to.

Models are logged with `cloudpickle`. MLflow's newer default (skops) refuses to load
classes it hasn't been told to trust, which includes our feature step and XGBoost.
Pickle-based files can run code when loaded, so only load models you created yourself.

To look at the runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000. The database and run files stay on your machine
(`mlflow.db`, `mlruns/`) and are not committed.

## Running it

```bash
source .venv/bin/activate
python -m src.features.selection      # feature rankings -> reports/
python -m src.models.compare_models   # six models, with/without PageValues -> reports/
python -m src.models.tune_models      # tuning -> reports/best_params.json + MLflow
python -m src.models.train_final      # final models -> models/ + reports/ + MLflow
```
