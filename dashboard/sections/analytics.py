"""Funnel and cohort analytics on the raw sessions."""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.data import BLUE, GREY, MONTH_ORDER, sessions

COHORTS = {
    "Month": "Month",
    "Visitor type": "VisitorType",
    "Traffic source": "TrafficType",
    "Region": "Region",
    "Browser": "Browser",
    "Operating system": "OperatingSystems",
    "Weekend": "Weekend",
}


def render() -> None:
    st.title("Funnel & cohorts")
    data = sessions()

    col1, col2 = st.columns(2)
    months = col1.multiselect("Months", MONTH_ORDER, default=MONTH_ORDER)
    visitors = col2.multiselect(
        "Visitor types",
        sorted(data["VisitorType"].unique()),
        default=sorted(data["VisitorType"].unique()),
    )
    data = data[data["Month"].isin(months) & data["VisitorType"].isin(visitors)]
    if data.empty:
        st.warning("No sessions match these filters.")
        return
    st.caption(f"{len(data):,} sessions, {data['Converted'].mean():.1%} converted.")

    st.subheader("Engagement funnel")
    stages = pd.DataFrame(
        {
            "stage": [
                "All sessions",
                "Viewed a product page",
                "Viewed 10+ product pages",
                "Viewed 30+ product pages",
                "Purchased",
            ],
            "sessions": [
                len(data),
                (data["ProductRelated"] > 0).sum(),
                (data["ProductRelated"] >= 10).sum(),
                (data["ProductRelated"] >= 30).sum(),
                data["Converted"].sum(),
            ],
        }
    )
    fig = px.funnel(stages, x="sessions", y="stage", color_discrete_sequence=[BLUE])
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Purchases are not a subset of the 30+ stage: some buyers view fewer pages. The bar shows "
        "how many sessions in the selection ended in a purchase."
    )

    st.subheader("Conversion by engagement")
    bands = pd.cut(
        data["ProductRelated"],
        bins=[-1, 0, 9, 29, 59, float("inf")],
        labels=["0", "1–9", "10–29", "30–59", "60+"],
    )
    by_band = (
        data.groupby(bands, observed=True)["Converted"]
        .agg(rate="mean", sessions="size")
        .reset_index()
    )
    fig = px.bar(
        by_band,
        x="ProductRelated",
        y="rate",
        hover_data={"sessions": True, "rate": ":.1%"},
        labels={"ProductRelated": "Product pages viewed", "rate": "Conversion rate"},
        color_discrete_sequence=[BLUE],
    )
    fig.update_layout(yaxis_tickformat=".0%", height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Cohorts")
    label = st.selectbox("Compare conversion by", list(COHORTS))
    column = COHORTS[label]
    cohort = data.groupby(column)["Converted"].agg(rate="mean", sessions="size").reset_index()
    if column == "Month":
        cohort[column] = pd.Categorical(cohort[column], MONTH_ORDER, ordered=True)
        cohort = cohort.sort_values(column)
    cohort[column] = cohort[column].astype(str)

    fig = px.bar(
        cohort,
        x=column,
        y="rate",
        hover_data={"sessions": True, "rate": ":.1%"},
        labels={column: label, "rate": "Conversion rate"},
        color_discrete_sequence=[BLUE],
    )
    fig.add_hline(
        y=data["Converted"].mean(),
        line_dash="dash",
        line_color=GREY,
        annotation_text="selection average",
    )
    fig.update_layout(yaxis_tickformat=".0%", height=360, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Hover a bar for the number of sessions. Small cohorts give noisy rates.")
