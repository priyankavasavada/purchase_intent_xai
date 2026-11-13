"""Explainability: global SHAP importance and local SHAP/LIME for any test session."""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.data import (
    BLUE,
    COMPARISON_MODEL,
    MAIN_MODEL,
    ORANGE,
    lime_explainer,
    report,
    scorer,
    test_probabilities,
    test_split,
)
from dashboard.sections.scoring import contributors_chart


def render() -> None:
    st.title("Explainability")

    st.subheader("What drives the model overall")
    importance = report("shap_global_importance.csv").set_index("feature")
    model = st.radio(
        "Model",
        [MAIN_MODEL, COMPARISON_MODEL],
        format_func=lambda name: (
            "Real model" if name == MAIN_MODEL else "Comparison model (with PageValues)"
        ),
        horizontal=True,
    )
    top = importance[model].dropna().sort_values().tail(10).reset_index()
    fig = px.bar(
        top,
        x=model,
        y="feature",
        orientation="h",
        color_discrete_sequence=[BLUE if model == MAIN_MODEL else ORANGE],
        labels={model: "Mean |SHAP contribution| on the test set", "feature": ""},
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    if model == MAIN_MODEL:
        st.caption("A high exit rate lowers the score; more product pages raise it.")
    else:
        st.caption("With PageValues, one leaky feature outweighs all the others combined.")

    st.subheader("Why did this session get its score?")
    X_test, y_test = test_split()
    probabilities = pd.Series(test_probabilities()[MAIN_MODEL], index=X_test.index)

    pick = st.selectbox(
        "Pick a test session",
        [
            "Most likely buyer",
            "Least likely buyer",
            "Closest to the threshold",
            "Choose by row number",
        ],
    )
    threshold = scorer().artifact.threshold
    if pick == "Most likely buyer":
        row = probabilities.idxmax()
    elif pick == "Least likely buyer":
        row = probabilities.idxmin()
    elif pick == "Closest to the threshold":
        row = (probabilities - threshold).abs().idxmin()
    else:
        position = st.number_input("Row in the test set", 0, len(X_test) - 1, 0)
        row = X_test.index[position]

    session = X_test.loc[[row]]
    col1, col2, col3 = st.columns(3)
    col1.metric("Chance of buying", f"{probabilities[row]:.1%}")
    col2.metric("Model decision", "Nudge" if probabilities[row] >= threshold else "Don't nudge")
    col3.metric("Actually bought", "Yes" if y_test[row] == 1 else "No")

    left, right = st.columns(2)
    with left:
        st.markdown("**SHAP** (exact, used in the API)")
        contributors = scorer().explainer.top_contributors(session, n=6)
        st.plotly_chart(contributors_chart(contributors), width="stretch")
    with right:
        st.markdown("**LIME** (independent check, approximate)")
        lime = lime_explainer().explain(session, n=6)
        st.dataframe(pd.DataFrame(lime["conditions"]), hide_index=True, width="stretch")
        st.caption(
            f"Local fit R² = {lime['local_fit_r2']:.2f}. LIME's simple model often fits loosely, "
            "so trust its directions more than its exact weights."
        )
    with st.expander("Session details"):
        # One column mixes text and numbers, so show it all as text.
        st.dataframe(session.T.rename(columns={row: "value"}).astype(str), width="stretch")
