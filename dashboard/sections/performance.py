"""Performance at any threshold, choosing an operating point, and system info."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import (
    API_URL,
    BLUE,
    GREY,
    MAIN_MODEL,
    ORANGE,
    api_health,
    scorer,
    test_probabilities,
    test_split,
)
from src.models.evaluate import compute_metrics
from src.models.threshold import add_expected_value, operating_points, value_threshold


def render() -> None:
    st.title("Performance & threshold")
    st.write(
        "Every threshold trades buyers reached against nudges wasted. The CRO team owns that "
        "choice; this page shows the trade-off on the 2,400 test sessions."
    )

    _, y_test = test_split()
    probabilities = test_probabilities()[MAIN_MODEL]
    artifact = scorer().artifact

    threshold = st.slider("Threshold", 0.01, 0.99, float(artifact.threshold), 0.01)
    metrics = compute_metrics(y_test, probabilities, threshold)
    flagged = probabilities >= threshold

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sessions nudged", f"{flagged.mean():.0%}")
    col2.metric("Buyers reached (recall)", f"{metrics['recall']:.0%}")
    col3.metric("Nudged who buy (precision)", f"{metrics['precision']:.0%}")
    col4.metric("PR-AUC (any threshold)", f"{metrics['pr_auc']:.3f}")

    left, right = st.columns([1, 2])
    with left:
        actual = y_test.to_numpy() == 1
        matrix = pd.DataFrame(
            [
                [(~flagged & ~actual).sum(), (flagged & ~actual).sum()],
                [(~flagged & actual).sum(), (flagged & actual).sum()],
            ],
            index=["Actual: no", "Actual: buy"],
            columns=["Not nudged", "Nudged"],
        )
        fig = px.imshow(matrix, text_auto=True, color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(
            height=300,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=30, b=10),
            title="Confusion matrix",
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.Series(metrics, name="value").to_frame(), width="stretch")
    with right:
        points = operating_points(y_test, probabilities)
        fig = go.Figure()
        fig.add_scatter(
            x=points["threshold"],
            y=points["recall"],
            name="Buyers reached (recall)",
            line=dict(color=BLUE, width=2),
        )
        anyone = points[points["tp"] + points["fp"] > 0]
        fig.add_scatter(
            x=anyone["threshold"],
            y=anyone["precision"],
            name="Nudged who buy (precision)",
            line=dict(color=ORANGE, width=2),
        )
        fig.add_scatter(
            x=points["threshold"],
            y=points["flagged_share"],
            name="Sessions nudged",
            line=dict(color=GREY, dash="dot"),
        )
        fig.add_vline(x=threshold, line_color=GREY)
        fig.update_layout(
            xaxis_title="Threshold",
            yaxis_title="Share",
            yaxis_range=[0, 1.05],
            height=380,
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Choose a threshold from value and cost")
    st.write(
        "Enter what a reached buyer is worth and what a nudge costs. Use the **extra** value a "
        "nudge brings, not the full order value: many flagged buyers would buy anyway."
    )
    col1, col2 = st.columns(2)
    value = col1.number_input("Value per reached buyer", 0.0, 10000.0, 20.0, 1.0)
    cost = col2.number_input("Cost per nudge", 0.0, 10000.0, 2.0, 0.5)

    valued = add_expected_value(points, value, cost)
    best = value_threshold(points, value, cost)
    if best is None:
        st.error("At these numbers no threshold makes money: nudging is not worth it.")
    else:
        row = valued.set_index("threshold").loc[best]
        nudged, reached = int(row["tp"] + row["fp"]), int(row["tp"])
        st.success(
            f"Best threshold: **{best:.2f}**. On the test set it nudges {nudged} sessions, "
            f"reaches {reached} buyers and gives a net value of {row['net_value']:,.0f}."
        )
    fig = px.line(
        valued, x="threshold", y="net_value", labels={"net_value": "Net value on the test set"}
    )
    fig.update_traces(line_color=BLUE)
    fig.add_hline(y=0, line_color=GREY)
    if best is not None:
        fig.add_vline(x=best, line_dash="dash", line_color=ORANGE, annotation_text="best")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    if value > 0:
        st.caption(
            "Rule of thumb: a group of sessions is worth nudging only if its precision is "
            f"above cost ÷ value ({cost / value:.0%})."
        )

    st.subheader("System")
    meta = artifact.metadata
    col1, col2 = st.columns(2)
    col1.write(
        f"**Model:** {meta['name']} ({meta['model_type']})  \n"
        f"**Trained:** {meta['created_at'][:10]}  \n"
        f"**Default threshold:** {meta['threshold']} ({meta['threshold_rule'].split('.')[0]})  \n"
        f"**Left out for leakage:** {', '.join(meta['leakage_excluded_features'])}  \n"
        f"**scikit-learn:** {meta['sklearn_version']}"
    )
    if API_URL:
        health = api_health()
        status = f"up, {health['model_version']}" if health else "not reachable"
        col2.write(f"**API:** {API_URL}  \n**Status:** {status}")
    else:
        col2.write("**Scoring:** local (no API_URL set)")
    with col2.expander("Model parameters"):
        st.json(meta["params"])
