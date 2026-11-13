# API guide

The API takes one shopping session and answers three questions:

1. How likely is this visitor to buy? (`conversion_probability`)
2. Should we nudge them? (`will_convert`)
3. Why did the model think so? (`top_contributors`)

## Start it

```bash
source .venv/bin/activate
python -m src.models.train_final      # only needed once, on a fresh checkout
uvicorn src.api.main:app --reload
```

Open http://127.0.0.1:8000/docs to see every endpoint and send test requests from the browser.

## How a request flows

```
client ──JSON──▶ POST /predict
                    │
                    ▼
          schemas.py: Session         checks every field; bad input → 422 error
                    │
                    ▼
          scoring.py: Scorer.score()  model → probability → decision → SHAP reasons
                    │
                    ▼
          schemas.py: Prediction      checks the answer's shape
                    │
client ◀──JSON──────┘
```

The model and explainer are loaded **once**, when the server starts (`lifespan` in
`main.py`). Requests reuse them, which is why each one takes about 9 ms.

| File | What to look at it for |
|---|---|
| `src/api/main.py` | The endpoints and startup. Start here. |
| `src/api/schemas.py` | Which fields a request needs, and what the response contains |
| `src/api/scoring.py` | How a session becomes a probability, decision and reasons |

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Checks the server is up and which model is loaded |
| `POST` | `/predict` | Scores one session |
| `GET` | `/docs` | Interactive documentation |

### `GET /health`

```json
{"status": "ok", "model_version": "purchase_intent_v1"}
```

### `POST /predict`

**Request body**: one session, all 16 fields required.

| Field | Type | Rule |
|---|---|---|
| `Administrative`, `Informational`, `ProductRelated` | integer | pages visited, ≥ 0 |
| `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` | number | seconds, ≥ 0 |
| `BounceRates`, `ExitRates`, `SpecialDay` | number | between 0 and 1 |
| `Month` | text | e.g. `Nov`, `June` |
| `OperatingSystems`, `Browser`, `Region`, `TrafficType` | integer | ID numbers (categories) |
| `VisitorType` | text | `Returning_Visitor`, `New_Visitor` or `Other` |
| `Weekend` | true / false | |

Do not send `PageValues`: the model doesn't use it (see the model card). If you send it,
or any other extra field, it is ignored.

A category the model has never seen, such as `"Month": "Jan"` or `"Browser": 99`, is still
accepted and scored.

**Optional query parameter:** `threshold`, a number between 0 and 1. Without it, the model's
default (0.27) is used. Example: `POST /predict?threshold=0.4`.

**Example**

```bash
curl -X POST "localhost:8000/predict" -H "Content-Type: application/json" -d '{
  "Administrative": 1, "Administrative_Duration": 14.65,
  "Informational": 0, "Informational_Duration": 0,
  "ProductRelated": 19, "ProductRelated_Duration": 283.88,
  "BounceRates": 0.008, "ExitRates": 0.042, "SpecialDay": 0,
  "Month": "June", "OperatingSystems": 1, "Browser": 2, "Region": 1,
  "TrafficType": 1, "VisitorType": "Returning_Visitor", "Weekend": false
}'
```

**Response**

```json
{
  "model_version": "purchase_intent_v1",
  "conversion_probability": 0.1347,
  "will_convert": false,
  "threshold": 0.27,
  "confidence": "medium",
  "top_contributors": [
    {"feature": "ExitRates", "value": 0.042106, "contribution": 0.2532, "effect": "increases"},
    {"feature": "ProductRelated", "value": 19, "contribution": -0.2051, "effect": "decreases"},
    {"feature": "total_pages", "value": 20, "contribution": -0.1828, "effect": "decreases"},
    {"feature": "total_duration", "value": 298.53, "contribution": -0.0507, "effect": "decreases"},
    {"feature": "ProductRelated_Duration", "value": 283.88, "contribution": -0.0279, "effect": "decreases"}
  ]
}
```

**Reading it:** a 13% chance of buying, below the 0.27 threshold, so no nudge. The decision is
`medium` confidence: 0.13 is 0.14 away from the threshold. The low exit rate pushed the score
up, but only 19 product pages pulled it down more.

| Field | Meaning |
|---|---|
| `conversion_probability` | Chance the session ends in a purchase, from 0 to 1 |
| `will_convert` | `true` when the probability is at or above `threshold` |
| `threshold` | The cut-off used: the default, or the one you sent |
| `confidence` | `low` if the probability is within 0.1 of the threshold, `medium` within 0.25, `high` beyond. A rule of thumb for how clear-cut the decision is, not a statistical certainty. |
| `top_contributors` | The 5 behaviours that moved the score most, biggest first |
| `contribution` | SHAP value in log-odds. Positive pushes towards a purchase, negative away. Use it to compare features; it is not a probability. |

Some features are calculated by the model from the raw fields, such as `total_pages`
(all pages visited) and `total_duration`, so they can appear in `top_contributors` even
though you didn't send them.

## Errors

Invalid input gets HTTP **422** with a message naming the problem, for example:

```json
{"detail": [{"type": "less_than_equal", "loc": ["body", "ExitRates"],
             "msg": "Input should be less than or equal to 1", "input": 1.5,
             "ctx": {"le": 1.0}}]}
```

| Cause | Example |
|---|---|
| Missing field | no `ExitRates` |
| Wrong type | `"ProductRelated": "lots"` |
| Out of range | `"ExitRates": 1.5`, `"ProductRelated": -3` |
| Bad threshold | `?threshold=1.5` |

If the server can't find a trained model at startup, it stops with an error saying to run
`python -m src.models.train_final`.

## Logs

Every request is logged with its status and time, to the console and `logs/app.log`:

```
request method=POST path=/predict status=200 duration_ms=9.4
```
