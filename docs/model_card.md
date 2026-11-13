# Model card: purchase-intent

| | |
|---|---|
| Model | `purchase_intent_v1` (MLflow registry: `purchase-intent`, version 1) |
| Type | XGBoost classifier inside a scikit-learn / imbalanced-learn pipeline |
| Trained | 2026-11-13 |
| Output | Probability that a session ends in a purchase, plus SHAP explanations |
| Code | `src/models/train_final.py`, notebooks `03_modeling` and `04_xai` |

## Intended use

Score an online shopping session **mid-visit**, so a growth / CRO team can decide
whether to show an intervention such as a discount or a free-shipping banner. The team
sees a conversion probability per session and the behaviours behind it, and picks its own
operating threshold from the value of a reached buyer and the cost of a nudge.

## Not intended for

- **Decisions about individual people beyond a site nudge**, such as pricing, credit or
  eligibility. The model only sees anonymous browsing behaviour.
- **Linking to personal data.** Predictions must not be joined to names, emails or other
  identifying fields.
- **Measuring whether a nudge works.** The model predicts who will buy, not who a nudge
  would change. That needs an A/B test.
- **Scoring right at the start of a visit.** With very few pages viewed, there is little
  for the model to go on.

## Data

12,000 anonymised e-commerce sessions (`dataset/ecommerce_sessions.csv`), 15.7% of which
converted. Stratified split: 9,600 sessions for training, 2,400 for testing. The test set
was used once, after the model and threshold were chosen.

## Features

Used: page counts and durations, BounceRates, ExitRates, SpecialDay, Month, OperatingSystems,
Browser, Region, TrafficType, and six engineered features (total pages, total duration, time
per page, time per product page, product share of pages and of time).

Left out:

| Feature | Reason |
|---|---|
| PageValues | **Data leakage.** It mostly fills in on the way to checkout, so it is not known mid-session. No buyer has PageValues = 0. |
| Administrative, Informational | Weak in feature selection (still used inside `total_pages`) |
| VisitorType, Weekend | Weak in feature selection; removing them did not lower PR-AUC |

The integer-coded IDs (OperatingSystems, Browser, Region, TrafficType) are treated as
categories, not numbers. Unknown categories at prediction time are ignored safely.

## Performance

Test set, 2,400 sessions, default threshold 0.27:

| Metric | Value |
|---|---|
| PR-AUC | 0.559 (random guessing: 0.157) |
| Recall | 0.610 |
| Precision | 0.529 |
| ROC-AUC | 0.849 |
| Accuracy | 0.853 (always predicting "no purchase" scores 0.843) |

| | Predicted no | Predicted buy |
|---|---|---|
| Actual no | 1,818 | 205 |
| Actual buy | 147 | 230 |

Cross-validated PR-AUC on the training split was 0.538 ± 0.019, close to the test score.
Five of the six algorithms tried were within about 0.015 of each other; CatBoost tied with XGBoost.

For comparison only, the same model with PageValues reaches a test PR-AUC of 0.867. That
number is not reachable when scoring mid-session.

## Threshold

The default threshold of 0.27 gives the best F1 on out-of-fold training predictions. It is
a neutral starting point, not a business decision. `src/models/threshold.py` shows
precision, recall and the share of sessions flagged at every threshold, and picks the
threshold with the best net value once the team supplies a value per conversion and a
cost per nudge. Nudging a group of sessions pays only if its precision is above cost ÷ value.

## Explanations

- **SHAP** (TreeExplainer) gives an exact contribution per feature for every session,
  in log-odds. Used in the API and dashboard.
- **LIME** gives an independent local explanation. On 40 test sessions, SHAP's top feature
  was in LIME's top five, pointing the same way, every time. LIME's local fit is usually
  weak (median R² about 0.16), so its exact weights are rough.

What drives the model: a **high exit rate** lowers the score strongly; **more product pages**
raise it, with a step at about 25–30 pages; bounce rate and total time matter less.

## Limitations

- **Mid-session totals:** training used whole-session totals. Mid-visit, counts and durations
  are smaller, so live performance is likely lower than the test results.
- **Probabilities are not formally calibrated.** No class reweighting was used, so they are
  not inflated, but they have not been checked against a reliability curve.
- **One dataset, one time period:** the data has only ten months (no January or April) and
  no year information. Behaviour may drift; the model should be re-checked on new data.
- **Explanations describe the model, not cause and effect.** A high exit rate lowering the
  score does not prove that reducing exits would create purchases.

## Responsible use

The data is anonymised browsing behaviour, but the same techniques apply to real user
tracking. Keep predictions away from identifying data, use them only for low-stakes site
interventions, and tell users about personalised offers where the law requires it.
