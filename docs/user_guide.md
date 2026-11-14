# User guide: the dashboard

For the growth / CRO team. The dashboard shows how likely each shopping session is to end
in a purchase, why, and helps you decide who to nudge with an offer.

Open it at http://localhost:8501 (see `docs/install.md` to start it). Pick a section in the
left sidebar.

## Words used

| Term | Meaning |
|---|---|
| Session | One visit to the shop |
| Probability / chance of buying | The model's estimate that the session ends in a purchase |
| Threshold | Sessions at or above it get a nudge |
| Recall / buyers reached | Of the people who would buy, the share that gets a nudge |
| Precision / nudged who buy | Of the people nudged, the share who buy |
| PR-AUC | One number for how well the model ranks buyers above non-buyers, 0 to 1. Guessing scores 0.157. |

## 1. Overview

Headline numbers: 12,000 sessions, a 15.7% conversion rate, and how well the model does on
sessions it never saw during training. Read the blue note: the model does **not** use
PageValues, because it is only known late in a visit.

## 2. Live session scoring

Enter what a visitor has done so far and press **Score session**.

- **Chance of buying**: for example 13.6%.
- **Decision**: *Nudge* if the chance reaches your threshold, *Don't nudge* otherwise.
- **Confidence**: *low* means the chance is close to the threshold, so the decision could easily
  go the other way.
- **Why**: the behaviours that moved the score most. Orange bars push towards buying, blue bars
  away. Longer bars matter more.

Things to try: raise *Product pages* from 19 to 40, or raise *Exit rate*, and score again.

The expander at the bottom shows what a model *with* PageValues would say. It is for comparison
only: you would not know PageValues in time to act.

## 3. Funnel & cohorts

Analytics on all 12,000 sessions, not model predictions. Filter by month and visitor type at
the top.

- **Engagement funnel**: how many sessions view any product, 10+ and 30+ product pages, and how
  many purchase.
- **Conversion by engagement**: conversion rises sharply with product pages viewed.
- **Cohorts**: conversion by month, visitor type, traffic source, region, browser or operating
  system. Hover a bar to see how many sessions it is based on; small groups give unreliable rates.

## 4. Model comparison

- **Six models, tuned**: the algorithms that were tried. The top five are within about 0.015 of
  each other; XGBoost is used.
- **With and without PageValues**: the comparison model scores much higher (test PR-AUC 0.867
  vs 0.559), but only because it uses information that isn't available mid-session. The gap is
  the price of scoring early, not a better model you could use.

## 5. Explainability

- **What drives the model overall**: exit rate and product pages matter most. Switch to the
  comparison model to see PageValues dominate everything.
- **Why did this session get its score?** Pick a test session (most likely buyer, least likely,
  closest to the threshold, or any row). You see its score, whether it really bought, and two
  explanations:
  - **SHAP**: the exact contribution of each behaviour. This is what the API returns.
  - **LIME**: an independent, approximate check. If both point the same way, you can trust the
    direction. A low "local fit" means LIME's numbers are rough.

## 6. Performance & threshold

This is where you choose your operating point.

1. Move the **Threshold** slider. The numbers, the confusion matrix and the chart update:
   a lower threshold reaches more buyers but nudges more people who won't buy.
2. Enter **Value per reached buyer** and **Cost per nudge**. The page finds the threshold with
   the best net value on the 2,400 test sessions and draws the net value curve.

   Use the *extra* value a nudge brings, not the full order value: many people who get a nudge
   would have bought anyway. If no threshold makes money, the page says so.

   Rule of thumb: nudging a group only pays if its precision is above cost ÷ value. With a value
   of 20 and a cost of 2, that is 10%.
3. Use the chosen threshold in the API with `?threshold=...` (see `docs/api.md`).

The **System** box shows which model is loaded, when it was trained, and whether the dashboard
is connected to the API.

## Things to keep in mind

- The model predicts **who will buy**, not **who a nudge would change**. To measure what nudges
  actually achieve, run an A/B test.
- It was trained on whole-session totals. Early in a visit, counts are smaller, so live
  performance may be lower than shown here.
- Use scores only for low-stakes site offers, never for pricing, credit or other decisions about
  individuals. See `docs/model_card.md`.
