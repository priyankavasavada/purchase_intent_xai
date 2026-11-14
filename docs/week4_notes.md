# Week 4 — Dashboard, Deployment & Docs

Status: dashboard, Docker images, docker-compose, CI image build and run, AWS-ready
config and the documentation set are done. The demo recording is still to do.

## 1. Checking the project against the brief

Before Week 4, every "Done when" in the brief was checked against the code. Fixes made:

- Every tunable value now lives in `config.yaml`, including the API's confidence bands and the
  feature selection settings.
- The EDA notebook states the outcome of each decision (for example, VisitorType was dropped).
- Type hints added where missing; request logs use `key=value` fields.
- The final model files and an MLflow run log (`reports/mlflow_runs.csv`) are committed, so a
  fresh clone can start the API.
- All scope changes are listed in one table in the README.

## 2. The dashboard (`dashboard/`)

Streamlit turns a Python script into a web page. Every time you click something, Streamlit
runs the whole script again from the top. Two things make that fast:

- `st.cache_data` remembers tables and results (the data, the reports, test-set predictions).
- `st.cache_resource` keeps one shared copy of heavy objects (the models, the SHAP and LIME
  explainers).

So the model is loaded once, and the dashboard **never trains**: it only reads `models/` and
`reports/`.

| File | Section |
|---|---|
| `app.py` | Sidebar and page setup |
| `data.py` | Cached loaders, colours, scoring through the API or locally |
| `sections/overview.py` | Headline numbers and how to use the dashboard |
| `sections/scoring.py` | Form → probability, decision, reasons; PageValues what-if |
| `sections/analytics.py` | Funnel and cohorts with filters |
| `sections/comparison.py` | Six models; with vs without PageValues |
| `sections/explainability.py` | Global SHAP; SHAP and LIME for any test session |
| `sections/performance.py` | Metrics at any threshold; value/cost → best threshold; system info |

When `API_URL` is set (docker-compose sets it), live scoring goes through the API, so the
dashboard and API always give the same answer. Without it, the dashboard loads the model itself.

`tests/test_dashboard.py` runs the real app without a browser (Streamlit's `AppTest`), opens
every section, submits the scoring form and checks nothing crashes.

## 3. Docker (`docker/`, `docker-compose.yml`)

A **Docker image** is a packaged copy of the app with its own Python and libraries, so it runs
the same on any machine. A **container** is a running image. **docker-compose** starts several
containers together.

How the images follow the brief's hints:

| Hint | How |
|---|---|
| Slim base | `python:3.12-slim` |
| `.dockerignore` | Keeps `.venv`, notebooks, tests, docs and git history out |
| Pinned versions | Every package version is fixed by `constraints.txt` |
| Non-root user | Containers run as `appuser`, not root |
| `libgomp1` | Installed; XGBoost needs it on Linux |
| Install only what you serve | `requirements-api.txt` (32 packages) and `requirements-dashboard.txt` leave out CatBoost, LightGBM, MLflow and Jupyter |

Each image has a **health check**: Docker regularly calls `/health` (API) or
`/_stcore/health` (dashboard). docker-compose only starts the dashboard once the API is
healthy, and inside compose the dashboard reaches the API at `http://api:8000`, using the
service name as the address.

**Why `constraints.txt`?** `requirements.txt` fixes our direct libraries, but they pull in
about 170 others. A new release of one of those can break a build that worked last week.
`constraints.txt` (from `pip freeze`) fixes all of them.

## 4. CI (`.github/workflows/ci.yml`)

Two jobs run on every push:

1. **lint-and-test**: ruff, black and pytest, installing with `constraints.txt`.
2. **docker**, only if the first passes: builds both images, starts them with
   `docker compose up --wait` (which waits for the health checks), then calls `/health`,
   `/predict` and the dashboard. If anything fails, it prints the container logs.

Docker isn't installed on the development Mac, so this CI job is where the images are actually built
and tested. Before relying on it, the API and dashboard package lists were each installed
into an empty Python environment and run, to make sure nothing was missing.

## 5. AWS (`deploy/aws/`, `docs/deployment_aws.md`)

- **ECR** (Elastic Container Registry) stores Docker images in AWS.
- **ECS Fargate** runs containers without you managing any servers.

`push_images.sh` builds the images for x86_64 (Fargate's architecture, even on an Apple
Silicon Mac), pushes them to ECR and fills in the task definition from `.env`. The task
definition runs both containers in one task, so the dashboard reaches the API at localhost.
The guide covers deploying, updating to a new model, rolling back and deleting everything.
Nothing was deployed: the brief requires no cloud spend.

## 6. Documentation

| Document | Contents |
|---|---|
| `README.md` | Quick start, results, links, scope changes, commands, structure |
| `docs/install.md` | Docker and local setup, retraining, troubleshooting |
| `docs/user_guide.md` | The dashboard, section by section, for the CRO team |
| `docs/api.md` | Endpoints, fields, examples, errors |
| `docs/architecture.md` | Architecture and workflow diagrams; one session from CSV row to response |
| `docs/model_card.md` | Intended use, performance, threshold, explanations, limitations |
| `docs/deployment_aws.md` | AWS deployment steps |

The diagrams are written in Mermaid, a text format that GitHub draws as pictures.

## Still to do

- **Demo recording** (3–5 minutes): show a prediction with its explanation in the API docs,
  then the dashboard sections.
- **Clone-from-scratch test** (brief §9): clone into a new folder and follow the README
  literally.
