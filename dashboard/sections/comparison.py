"""Model comparison: the six algorithms, and the effect of PageValues."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

from dashboard.data import (
    BLUE,
    COMPARISON_MODEL,
    GREY,
    MAIN_MODEL,
    ORANGE,
    final_models,
    report,
    sessions,
    test_probabilities,
    test_split,
)

LABELS = {
    MAIN_MODEL: "Without PageValues (real model)",
    COMPARISON_MODEL: "With PageValues (comparison)",
}


def render() -> None:
    st.title("Model comparison")

    st.subheader("Six models, tuned")
    st.caption("PR-AUC from 5-fold cross-validation on the training split, without PageValues.")
    tuning = report("tuning_results.csv").sort_values("pr_auc_mean")
    fig = go.Figure(
        go.Bar(
            x=tuning["pr_auc_mean"],
            y=tuning["model"],
            orientation="h",
            marker_color=BLUE,
            error_x=dict(type="data", array=tuning["pr_auc_std"], color=GREY),
        )
    )
    fig.add_vline(
        x=sessions()["Converted"].mean(),
        line_dash="dash",
        line_color=GREY,
        annotation_text="random guess",
    )
    fig.update_layout(
        xaxis_title="PR-AUC (error bar = 1 standard deviation)",
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, width="stretch")
    columns = [
        "model",
        "pr_auc_mean",
        "pr_auc_std",
        "recall_mean",
        "precision_mean",
        "roc_auc_mean",
        "accuracy_mean",
    ]
    st.dataframe(
        tuning.sort_values("pr_auc_mean", ascending=False)[columns].round(3), hide_index=True
    )
    st.caption(
        "XGBoost and CatBoost are tied; XGBoost was chosen by the rule set before tuning "
        "(highest mean PR-AUC). Recall and precision here use a 0.5 threshold."
    )

    st.subheader("With and without PageValues")
    comparison = report("model_comparison.csv")
    best = (
        comparison.sort_values("pr_auc_mean", ascending=False)
        .groupby(["feature_set", "model"])
        .head(1)
    )
    table = best.pivot(index="model", columns="feature_set", values="pr_auc_mean").sort_values(
        "without_page_values"
    )
    fig = go.Figure()
    fig.add_bar(
        y=table.index,
        x=table["without_page_values"],
        name=LABELS[MAIN_MODEL],
        orientation="h",
        marker_color=BLUE,
    )
    fig.add_bar(
        y=table.index,
        x=table["with_page_values"],
        name=LABELS[COMPARISON_MODEL],
        orientation="h",
        marker_color=ORANGE,
    )
    fig.update_layout(
        barmode="group",
        xaxis_title="Cross-validated PR-AUC (before tuning)",
        height=380,
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    variants = final_models()["variants"]
    rows = {
        LABELS[prefix]: {"threshold": v["threshold"], **v["test_metrics"]}
        for prefix, v in variants.items()
    }
    st.write("Final models on the test set:")
    st.dataframe(pd.DataFrame(rows).T, width="stretch")

    X_test, y_test = test_split()
    probabilities = test_probabilities()
    fig = go.Figure()
    for prefix, colour in [(MAIN_MODEL, BLUE), (COMPARISON_MODEL, ORANGE)]:
        precision, recall, _ = precision_recall_curve(y_test, probabilities[prefix])
        fig.add_scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=LABELS[prefix],
            line=dict(color=colour, width=2),
        )
    fig.add_hline(
        y=y_test.mean(), line_dash="dash", line_color=GREY, annotation_text="random guess"
    )
    fig.update_layout(
        xaxis_title="Recall (share of buyers reached)",
        yaxis_title="Precision (share of flagged who buy)",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.02],
        height=420,
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    st.warning(
        "PageValues adds about 0.3 PR-AUC, far more than the choice of algorithm. But it mostly "
        "fills in on the way to checkout, so it is not known when a session is scored mid-visit. "
        "The gap between the curves is the price of scoring early, not a model we could deploy."
    )
