# E-Commerce Purchase-Intent Prediction with Explainable AI

Predicts whether an online shopping session will end in a purchase, explains each
prediction, and (in later weeks) serves it through an API and a dashboard.

> Status: Weeks 1 and 2 complete. See `docs/week1_notes.md` and `docs/week2_notes.md`.

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

# Compare imbalance strategies with 5-fold stratified cross-validation
python -m src.models.compare_imbalance

# Rank features and compare feature sets -> reports/
python -m src.features.selection

# Compare the six models, with and without PageValues -> reports/
python -m src.models.compare_models
```

Notebooks: `01_eda.ipynb` (EDA), `02_features.ipynb` (feature selection) and `03_modeling.ipynb` (model comparison and operating points).

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
  config/       config.yaml + typed settings loader
  data/         loading, validation, train/test split
  features/     behavioural features + feature selection
  preprocessing/ ColumnTransformer + imbalance strategy
  models/       training, tuning, thresholds, MLflow, saving/loading, prediction
  utils/        logging
models/         saved model artifacts (not committed)
reports/        results from the comparison scripts
tests/          pytest suite
docs/           documentation
```
