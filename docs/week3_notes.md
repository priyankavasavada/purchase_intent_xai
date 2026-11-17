# Week 3 — Explainability & API

Status: complete. SHAP and LIME explanations, the XAI notebook, the model card, and a
tested FastAPI service.

## 1. Why explain a model at all?

A CRO team won't act on a number it doesn't understand. For every session we want to
answer: *which behaviours pushed it towards a purchase, and by how much?* Explanations
also catch problems: the comparison model's explanations show PageValues outweighing
everything else, which is what leaning on a leaky feature looks like.

## 2. SHAP (`src/explain/shap_explainer.py`)

SHAP splits the model's score for one session into a contribution per feature. It starts
from the **base value**, the score of an average session (log-odds −1.679, which is a
probability of 0.157, the buyer rate), and each feature pushes the score up or down from
there. The contributions add up exactly to the model's score; a test checks this.

- **Log-odds** is the model's internal score before it becomes a probability. Positive
  contributions push towards a purchase, negative ones away from it.
- **TreeExplainer** gives exact contributions for tree models and is fast (about 9 ms
  per API request, including the prediction).
- **Readable names:** the model sees 73 columns after one-hot encoding (`Month_Nov`,
  `Month_Dec`, ...). The explainer adds those back together, so the answer says "Month".

| Method | Gives |
|---|---|
| `top_contributors(session)` | The 5 features that moved one session's score most, with their values |
| `global_importance(sessions)` | Average size of each feature's contribution: what matters overall |

**What the model learned:** a high exit rate lowers the score strongly (it turns negative
above about 0.05). More product pages raise it, with a step at about 25–30 pages.

## 3. LIME (`src/explain/lime_explainer.py`)

LIME takes one session, makes thousands of small random changes to it, watches how the
model's probability reacts, and fits a simple linear model to those reactions. Its output
reads like rules: `ProductRelated <= 11.00` lowers the probability by 0.11.

Two details:
- **Categories as numbers:** LIME only understands numbers, so categories are coded as
  numbers and turned back before the model sees them. A test checks that this round
  trip gives exactly the same prediction.
- **Repeatable results:** LIME is random. It is re-seeded before every explanation so
  the same session always gets the same answer.

**SHAP vs LIME:** on 40 random test sessions, SHAP's top feature was among LIME's top
five, pointing the same way, every time. But LIME's local fit is usually weak (median
R² about 0.16), so its exact weights are rough. SHAP is used in the API; LIME is the
independent check.

## 4. Model card (section 6 of `docs/user_guide.md`)

A short document for anyone using the model: what it is for, what it must not be used for,
how it performs, how to choose a threshold, and its limitations.

## 5. FastAPI service (`src/api/`)

| File | Job |
|---|---|
| `schemas.py` | Pydantic models for the request and response |
| `scoring.py` | Probability, decision, confidence and explanation |
| `main.py` | The web app: endpoints, request logging, loading at startup |

**Endpoints**

| Endpoint | Does |
|---|---|
| `POST /predict` | Scores one session. Optional `?threshold=0.4` to use the team's own operating point. |
| `GET /health` | Says the model is loaded, and which version |
| `GET /docs` | Interactive documentation where you can try requests |

**Validation.** Pydantic checks every request against the schema. A negative page count,
an exit rate above 1 or a missing field gets a **422** error that names the problem,
before anything reaches the model. A test checks that the schema's fields match the input
columns in `config.yaml`, so the two cannot drift apart.

**Unknown categories.** A Month or Browser ID never seen in training is accepted; the
encoder (`handle_unknown="ignore"`) treats it as "none of the known values".

**PageValues** isn't part of the schema. If a client sends it anyway, it is ignored, and a
test checks the prediction doesn't change.

**Train/serve skew.** The API loads the exact pipeline saved by training, so feature
engineering, scaling and encoding are identical. Nothing is re-implemented by hand.

**Load once.** The model and SHAP explainer are loaded in FastAPI's `lifespan` function,
which runs once at startup, not on every request. A test checks this.

**Confidence** is `low` when the probability is within 0.1 of the threshold, `medium`
within 0.25, and `high` beyond that. It's a rule of thumb for how clear-cut the decision
is, not a statistical certainty.

## Running it

```bash
source .venv/bin/activate
python -m src.explain.report          # global SHAP importance -> reports/
uvicorn src.api.main:app --reload     # API at http://127.0.0.1:8000/docs
```

Example request:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"Administrative": 1, "Administrative_Duration": 14.65, "Informational": 0,
       "Informational_Duration": 0, "ProductRelated": 19, "ProductRelated_Duration": 283.88,
       "BounceRates": 0.008, "ExitRates": 0.042, "SpecialDay": 0, "Month": "June",
       "OperatingSystems": 1, "Browser": 2, "Region": 1, "TrafficType": 1,
       "VisitorType": "Returning_Visitor", "Weekend": false}'
```

The API needs a trained model in `models/`. Run `python -m src.models.train_final` first
on a fresh checkout.
