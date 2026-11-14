# E-Commerce Purchase-Intent Prediction with Explainable AI

Predicts, mid-visit, whether an online shopping session will end in a purchase, explains
which behaviours drove each prediction, and serves it through a FastAPI service and a
Streamlit dashboard, so a growth / CRO team can decide who to nudge with an offer.

## Quick start

```bash
git clone https://github.com/priyankavasavada/purchase_intent_xai.git
cd purchase_intent_xai
docker compose up --build
```

- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

The trained model is in the repository, so nothing needs training first. Running without
Docker, training and the notebooks are covered in the [install guide](docs/install.md).

## Results

Tuned XGBoost, scored on 2,400 test sessions the model never saw during training:

| Metric | Value |
|---|---|
| PR-AUC | 0.559 (random guessing: 0.157) |
| Buyers reached (recall) at the default threshold of 0.27 | 61% |
| Nudged sessions that buy (precision) | 53%, against 16% of all sessions |
| ROC-AUC | 0.849 |

PageValues is **left out** to prevent data leakage: it mostly fills in on the way to
checkout, so it isn't known mid-session. A comparison model with it reaches a PR-AUC of
0.867, which is shown in the dashboard for reference only. The CRO team picks the operating
threshold from its own conversion value and nudge cost; see the
[model card](docs/model_card.md) for performance, intended use and limitations.

## Documentation

| Document | For |
|---|---|
| [Install guide](docs/install.md) | Running with Docker or Python, retraining, troubleshooting |
| [User guide](docs/user_guide.md) | Using the dashboard and choosing a threshold |
| [API guide](docs/api.md) | Calling `/predict` and reading its answer |
| [Architecture](docs/architecture.md) | Architecture and workflow diagrams; one session from CSV row to response |
| [Model card](docs/model_card.md) | What the model is for, how well it works, its limits |
| [AWS deployment](docs/deployment_aws.md) | Deploying to ECR and ECS Fargate |
| Notes: [week 1](docs/week1_notes.md), [week 2](docs/week2_notes.md), [week 3](docs/week3_notes.md), [week 4](docs/week4_notes.md) | What was built each week and why |

Notebooks (CRISP-DM): `01_eda` (data understanding), `02_features` (data preparation),
`03_modeling` (modelling and evaluation) and `04_xai` (explainability).

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
| 2026-11-13 | AWS deployment documented, not run | The brief requires no live cloud spend. |

## Commands

All from the project root, with the environment activated (see the install guide).

```bash
# Pipeline, in order
python -m src.features.selection      # feature rankings -> reports/
python -m src.models.compare_models   # six models, with and without PageValues -> reports/
python -m src.models.tune_models      # tuning -> reports/best_params.json + MLflow
python -m src.models.train_final      # final models -> models/ + reports/ + MLflow
python -m src.explain.report          # SHAP importance for the dashboard
python -m src.models.tracking         # MLflow run log -> reports/mlflow_runs.csv

# Serving
uvicorn src.api.main:app --reload     # API at http://127.0.0.1:8000/docs
streamlit run dashboard/app.py        # dashboard at http://localhost:8501
python -m src.models.predict examples/sample_session.json

# Experiment tracking UI at http://127.0.0.1:5000 (after training on this machine)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Week 1 baseline and imbalance comparison
python -m src.models.train_baseline
python -m src.models.compare_imbalance
```

## Code quality

```bash
ruff check src tests dashboard      # lint
black --check src tests dashboard   # formatting
pytest                              # tests
```

GitHub Actions runs these on every push, then builds both Docker images, starts the stack
and checks the API and dashboard answer (`.github/workflows/ci.yml`).

## Project structure

```
dataset/          raw data + data dictionary
notebooks/        exploration and storytelling (imports code from src/)
src/
  api/            FastAPI app, request/response schemas, scoring
  config/         config.yaml (every tunable value) + typed settings loader
  data/           loading, validation, train/test split
  explain/        SHAP and LIME explanations
  features/       behavioural features + feature selection
  preprocessing/  ColumnTransformer + imbalance strategy
  models/         training, tuning, thresholds, MLflow, saving/loading, prediction
  utils/          logging
dashboard/        Streamlit app, one file per section
models/           saved final models (+ metadata)
reports/          results: comparisons, tuning, operating points, SHAP, MLflow run log
tests/            pytest suite
docker/           Dockerfiles for the API and dashboard
deploy/aws/       ECR push script and ECS task definition
docs/             documentation
docker-compose.yml, requirements*.txt, constraints.txt, .env.example
```
