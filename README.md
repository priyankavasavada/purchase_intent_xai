# E-Commerce Purchase-Intent Prediction with Explainable AI

Predicts whether an online shopping session will end in a purchase, explains each
prediction, and (in later weeks) serves it through an API and a dashboard.

> Status: Week 1 (foundations & data) complete. See `docs/week1_notes.md`.

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

All commands are run from the project root.

```bash
# Train the baseline model -> models/baseline_logreg.joblib (+ .json metadata)
python -m src.models.train_baseline

# Score one or more sessions from a JSON file
python -m src.models.predict examples/sample_session.json

# Compare imbalance strategies with 5-fold stratified cross-validation
python -m src.models.compare_imbalance
```

The EDA notebook is `notebooks/01_eda.ipynb`.

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
  preprocessing/ ColumnTransformer + imbalance strategy
  models/       training, evaluation, saving/loading, prediction
  utils/        logging
models/         saved model artifacts (not committed)
tests/          pytest suite
docs/           documentation
```
