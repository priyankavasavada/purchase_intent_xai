# Week 1 — Foundations & Data: what we built and why

## Scope changes (PRD "Document control")

| Date | Change | Reason |
|---|---|---|
| 2026-11-01 | No download script | PRD §2.1: the dataset already ships in `dataset/`. The loader checks that the file exists and is valid instead. |
| 2026-11-01 | Python 3.12 | Chosen over 3.14 for library support. The later libraries install on 3.12; 3.14 was not tested. |

---

## 1. Repo scaffold, config and logging

**Config (`src/config/config.yaml` + `settings.py`).** Every path, seed and column
list lives in one YAML file. `get_settings()` reads it once and returns a typed
`Settings` object. If you later change the test size or drop a column, you change it
in one place and the notebook, training script and (later) API all follow.

**Random seed (42).** Splitting, cross-validation and SMOTE all involve randomness.
A fixed seed makes every run produce the same split and the same numbers.

**Logging (`src/utils/logger.py`).** `logging` instead of `print` gives timestamps,
levels (INFO / WARNING / ERROR), the module name, and a copy in `logs/app.log`.
Each module creates its logger with `get_logger(__name__)`.

## 2. CI skeleton (`.github/workflows/ci.yml`)

CI (continuous integration) means GitHub runs checks automatically on every push:

- **ruff** (linter) catches unused imports, wrong import order and likely bugs.
- **black** (formatter) gives the code one consistent style.
- **pytest** runs the tests in `tests/`.

If any of these fails, the push shows a red ✗ on GitHub. The Docker build step is
added in Week 4.

## 3. Data loader and validation (`src/data/loader.py`)

`load_data()` is the only way the project reads the CSV. It "fails loudly":

| Problem | What happens |
|---|---|
| File missing | `FileNotFoundError` |
| Empty file, missing column, text in a numeric column, negative values, missing values, target not 0/1 | `DataValidationError` with the exact reason |
| Duplicate rows | Warning only (two real sessions can look identical) |

`split_data()` makes a **stratified** 80/20 split: both parts keep the same ~15.7%
buyer rate. Without stratifying, the test set could by chance get noticeably more or
fewer buyers.

## 4. EDA (`notebooks/01_eda.ipynb`)

The key findings and the decisions they led to are in the summary table at the end of
the notebook. The most important ones:

- **Imbalance:** 15.7% buyers. A model that always says "no" is 84.3% accurate and useless.
- **PageValues leakage risk:** *no* buyer has PageValues = 0. It is the strongest
  feature, but it may only be known late in a session. We test this in Week 2.
- **ID columns are categories:** conversion by TrafficType jumps around with no order.

## 5. Preprocessing (`src/preprocessing/pipeline.py`)

**ColumnTransformer** applies a different transformation to each group of columns:

| Columns | Transformation | Why |
|---|---|---|
| Page counts, durations, PageValues | `log1p` then `StandardScaler` | Long right tail and many zeros. `log1p(x) = log(1 + x)` compresses big values and keeps 0 at 0. |
| BounceRates, ExitRates, SpecialDay | `StandardScaler` | Already small and bounded; just centre and scale. |
| Month, VisitorType, Weekend, OperatingSystems, Browser, Region, TrafficType | `OneHotEncoder` | Categories, one 0/1 column per value. Region 3 is not "more" than Region 1. |

`handle_unknown="ignore"` means a category never seen in training (e.g. Month = "Jan")
becomes all zeros instead of crashing the API.

**Why a Pipeline (avoiding data leakage).** A scaler "learns" the mean and standard
deviation; an encoder "learns" the list of categories. If they learn from the whole
dataset before splitting, they have seen the test rows, which is a small leak that
inflates scores. Putting them in the same `Pipeline` as the model means `fit()` only
ever learns from the training data, including inside every cross-validation fold.
The fitted pipeline is saved as one file, so the API later applies exactly the same
transforms (no train/serve skew).

## 6. Imbalance strategy

Three options, all available through `build_pipeline(model, strategy=...)`:

- **none:** train normally, then move the decision threshold (Week 2).
- **class_weight="balanced":** mistakes on buyers cost the model about 5× more during training.
- **smote:** create synthetic buyer rows by interpolating between real ones. It is an
  `imblearn` pipeline step, so it runs **only on training data** and never on test data.

Result of `python -m src.models.compare_imbalance` (Logistic Regression, 5-fold CV on the training set):

| strategy | PR-AUC | recall @0.5 | precision @0.5 |
|---|---|---|---|
| none | 0.830 ± 0.016 | 0.734 | 0.777 |
| class_weight | 0.820 ± 0.017 | 0.924 | 0.595 |
| smote | 0.820 ± 0.016 | 0.913 | 0.619 |

**What this means:** PR-AUC (how well the model *ranks* buyers above non-buyers) is
practically the same for all three. Rebalancing mainly shifts the 0.5 cut-off,
trading precision for recall. **Default chosen: `class_weight`**, because it is simple
and creates no fake rows. The final choice is made in Week 2 together with threshold tuning.

## 7. Metrics cheat-sheet

- **Precision:** of the sessions we flag as buyers, how many really buy.
- **Recall:** of the real buyers, how many we catch.
- **F1:** a single balance of precision and recall.
- **ROC-AUC:** ranking quality; can look good even on imbalanced data.
- **PR-AUC (average precision):** ranking quality focused on the buyer class. A random
  model scores ≈ 0.157 here (the buyer rate). This is our headline metric.
- **Threshold:** the probability above which we say "will buy". 0.5 is a placeholder
  until Week 2.

## 8. Thin slice

```
dataset/ecommerce_sessions.csv
   → load_data() + validate_data()
   → split_data()                     (stratified, seed 42)
   → build_pipeline(LogisticRegression)  (preprocess + class_weight + model)
   → fit → evaluate
   → models/baseline_logreg.joblib + baseline_logreg.json
   → python -m src.models.predict examples/sample_session.json
```

Baseline test results at the end of Week 1, before the Week 2 feature changes: dummy model accuracy 0.843 / recall 0.0; logistic regression
PR-AUC 0.82, recall 0.94, precision 0.60.
