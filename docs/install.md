# Install guide

Two ways to run the project: with Docker (quickest, runs the API and dashboard) or
locally with Python (needed for training, notebooks and tests).

## Option 1: Docker

Needs Docker Desktop (or Docker Engine with the Compose plugin).

```bash
git clone https://github.com/priyankavasavada/purchase_intent_xai.git
cd purchase_intent_xai
docker compose up --build
```

The first build takes a few minutes. Then open:

- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

The trained model is in the repository, so nothing needs training first. Stop everything
with `Ctrl+C`, then `docker compose down`.

To use different ports, copy `.env.example` to `.env` and change `API_PORT` and
`DASHBOARD_PORT`.

## Option 2: Local Python

Needs **Python 3.12** and Git.

```bash
git clone https://github.com/priyankavasavada/purchase_intent_xai.git
cd purchase_intent_xai
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt -c constraints.txt
```

`constraints.txt` fixes the version of every package, so you get exactly what the project
was built and tested with.

### macOS only: OpenMP

XGBoost and LightGBM need the OpenMP library. Install it once:

```bash
brew install libomp
```

If you can't use Homebrew, point to the copy that comes with scikit-learn instead. Add
this line to the end of `.venv/bin/activate` and activate the environment again:

```bash
export DYLD_FALLBACK_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/sklearn/.dylibs"
```

### Check it works

```bash
pytest                                                # about 20 seconds
python -m src.models.predict examples/sample_session.json
```

### Run the services

```bash
uvicorn src.api.main:app --reload          # API: http://127.0.0.1:8000/docs
streamlit run dashboard/app.py             # dashboard: http://localhost:8501
```

The dashboard scores sessions itself by default, using the same saved model and code as the
API. To make it score through the running API instead (as it does in docker-compose), pass
the API's address when starting it:

```bash
API_URL=http://127.0.0.1:8000 streamlit run dashboard/app.py
```

The System box at the bottom of *Performance & threshold* shows which one is in use.

### Retrain from scratch (optional)

The saved model is already in `models/`. To rebuild everything:

```bash
python -m src.features.selection      # feature rankings       (~1 min)
python -m src.models.compare_models   # six-model comparison   (~1 min)
python -m src.models.tune_models      # tuning                 (~10 min)
python -m src.models.train_final      # final models + MLflow  (~1 min)
python -m src.explain.report          # SHAP importance for the dashboard
python -m src.models.tracking         # MLflow run log -> reports/
```

Training saves a new version (`purchase_intent_v2`, ...); the API and dashboard always use
the newest. Results are reproducible: the same data and seed give the same numbers.

## Troubleshooting

| Problem | Fix |
|---|---|
| `XGBoost Library (libxgboost.dylib) could not be loaded` | macOS OpenMP, see above |
| `No 'purchase_intent' model in .../models` | Run `python -m src.models.train_final` |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root, with the environment activated |
| Port already in use | Change `API_PORT` / `DASHBOARD_PORT` in `.env`, or stop the other program |
