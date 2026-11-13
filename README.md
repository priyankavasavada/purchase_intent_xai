# E-Commerce Purchase-Intent Prediction with Explainable AI

Predicts whether an online shopping session will end in a purchase, explains each
prediction, and (in later weeks) serves it through an API and a dashboard.

> Status: Weeks 1–3 complete. See the notes in `docs/` and the [model card](docs/model_card.md).

## Changes from the brief

The brief asks for scope changes to be recorded. Each one is explained in the notes in `docs/`.

| Date | Change | Reason |
|---|---|---|
| 2026-11-01 | No download script | The dataset ships in `dataset/` (brief §2.1). The loader validates it instead. |
| 2026-11-01 | Python 3.12 | Chosen over 3.14 so every library in the stack installs. |
| 2026-11-05 | CatBoost uses one-hot encoding, not its native categoricals | Keeps one common pipeline for all six models. |
| 2026-11-13 | PageValues left out of the model | Data leakage: sessions are scored mid-visit, and PageValues mostly fills in on the way to checkout. A comparison model with it is kept for the dashboard. |
| 2026-11-13 | No single value-based threshold chosen by us | The value of a conversion and the cost of a nudge are unknown and owned by the CRO team. We deliver operating-point tooling and a neutral best-F1 default (0.27). |
| 2026-11-13 | Imbalance strategy `none` | All three strategies were benchmarked; SMOTE never helped and class weights only moved the precision/recall trade-off. |
| 2026-11-13 | MLflow models saved with cloudpickle | MLflow's skops default refuses the pipeline's custom classes. |
| 2026-11-13 | Final model files committed | So a fresh clone can start the API without retraining. |

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost and LightGBM also need OpenMP: `brew install libomp`.

## Usage

All commands are run from the project root.

```bash
# Train the baseline model -> models/baseline_logreg.joblib (+ .json metadata)
python -m src.models.train_baseline

# Tune the six models, then train and save the final model (logged to MLflow)
python -m src.models.tune_models
python -m src.models.train_final

# Score one or more sessions with the newest saved model
python -m src.models.predict examples/sample_session.json

# Browse experiment runs and the model registry at http://127.0.0.1:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Save global SHAP importance for the dashboard
python -m src.explain.report

# Start the API; interactive docs at http://127.0.0.1:8000/docs
uvicorn src.api.main:app --reload

# Compare imbalance strategies with 5-fold stratified cross-validation
python -m src.models.compare_imbalance

# Rank features and compare feature sets -> reports/
python -m src.features.selection

# Compare the six models, with and without PageValues -> reports/
python -m src.models.compare_models
```

How to call the API and read its answers: [docs/api.md](docs/api.md).

Notebooks: `01_eda.ipynb` (EDA), `02_features.ipynb` (feature selection), `03_modeling.ipynb` (model comparison and operating points) and `04_xai.ipynb` (SHAP and LIME).

## Code quality

```bash
ruff check src tests      # lint
black --check src tests   # formatting
pytest                    # tests
```

The same three checks run in GitHub Actions on every push (`.github/workflows/ci.yml`).

## Project structure

```
dataset/        raw data + data dictionary
notebooks/      exploration and storytelling (imports code from src/)
src/
  api/          FastAPI app, request/response schemas, scoring
  config/       config.yaml + typed settings loader
  data/         loading, validation, train/test split
  explain/      SHAP and LIME explanations
  features/     behavioural features + feature selection
  preprocessing/ ColumnTransformer + imbalance strategy
  models/       training, tuning, thresholds, MLflow, saving/loading, prediction
  utils/        logging
models/         saved model artifacts (not committed)
reports/        results from the comparison scripts
tests/          pytest suite
docs/           documentation
```
