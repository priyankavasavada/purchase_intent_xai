"""Live session scoring: enter a session, get a score and the reasons."""

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.data import BLUE, MONTH_ORDER, ORANGE, comparison_model, score_session, scorer
from src.api.schemas import EXAMPLE_SESSION


def contributors_chart(contributors: list[dict]):
    table = pd.DataFrame(contributors).iloc[::-1]
    table["label"] = [f"{row.feature} = {row.value}" for row in table.itertuples()]
    fig = px.bar(
        table,
        x="contribution",
        y="label",
        orientation="h",
        color="effect",
        color_discrete_map={"increases": ORANGE, "decreases": BLUE},
        labels={"contribution": "Effect on the score (SHAP, log-odds)", "label": ""},
    )
    fig.update_layout(height=320, legend_title_text="", margin=dict(l=10, r=10, t=10, b=10))
    return fig


def render() -> None:
    st.title("Live session scoring")
    st.write("Enter what a visitor has done so far. The model gives the chance they will buy.")

    default_threshold = scorer().artifact.threshold
    s = EXAMPLE_SESSION

    with st.form("session"):
        st.subheader("Pages and time")
        col1, col2, col3 = st.columns(3)
        product_pages = col1.number_input("Product pages", 0, 1000, s["ProductRelated"])
        product_time = col1.number_input(
            "Seconds on product pages", 0.0, 100000.0, s["ProductRelated_Duration"]
        )
        admin_pages = col2.number_input("Account pages", 0, 1000, s["Administrative"])
        admin_time = col2.number_input(
            "Seconds on account pages", 0.0, 100000.0, s["Administrative_Duration"]
        )
        info_pages = col3.number_input("Info pages", 0, 1000, s["Informational"])
        info_time = col3.number_input(
            "Seconds on info pages", 0.0, 100000.0, s["Informational_Duration"]
        )

        st.subheader("Behaviour and context")
        col1, col2, col3 = st.columns(3)
        exit_rate = col1.slider("Exit rate", 0.0, 0.2, s["ExitRates"], 0.001, format="%.3f")
        bounce_rate = col1.slider("Bounce rate", 0.0, 0.2, s["BounceRates"], 0.001, format="%.3f")
        special_day = col1.slider("Closeness to a special day", 0.0, 1.0, s["SpecialDay"], 0.2)
        month = col2.selectbox("Month", MONTH_ORDER, MONTH_ORDER.index(s["Month"]))
        visitor = col2.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
        weekend = col2.checkbox("Weekend", s["Weekend"])
        traffic = col3.number_input("Traffic type ID", 1, 50, s["TrafficType"])
        region = col3.number_input("Region ID", 1, 50, s["Region"])
        browser = col3.number_input("Browser ID", 1, 50, s["Browser"])
        system = col3.number_input("Operating system ID", 1, 50, s["OperatingSystems"])

        threshold = st.slider(
            "Threshold (nudge when the probability is at least this)",
            0.01,
            0.99,
            float(default_threshold),
            0.01,
            help=f"Default {default_threshold}: best F1. Pick your own in Performance & threshold.",
        )
        submitted = st.form_submit_button("Score session", type="primary")

    if not submitted:
        return

    session = {
        "Administrative": admin_pages,
        "Administrative_Duration": admin_time,
        "Informational": info_pages,
        "Informational_Duration": info_time,
        "ProductRelated": product_pages,
        "ProductRelated_Duration": product_time,
        "BounceRates": bounce_rate,
        "ExitRates": exit_rate,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": system,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic,
        "VisitorType": visitor,
        "Weekend": weekend,
    }
    result = score_session(session, threshold)

    col1, col2, col3 = st.columns(3)
    col1.metric("Chance of buying", f"{result['conversion_probability']:.1%}")
    col2.metric("Decision", "Nudge" if result["will_convert"] else "Don't nudge")
    col3.metric(
        "Confidence", result["confidence"].capitalize(), help="Distance from the threshold."
    )

    st.subheader("Why")
    st.caption(
        "Orange pushes the score towards a purchase, blue pushes it away. Longer bars matter more."
    )
    st.plotly_chart(contributors_chart(result["top_contributors"]), width="stretch")

    with st.expander("What if PageValues were known? (comparison model, not used for decisions)"):
        page_values = st.number_input("PageValues", 0.0, 1000.0, 0.0)
        reference = comparison_model()
        frame = pd.DataFrame([{**session, "PageValues": page_values}])[reference.feature_columns]
        with_page_values = reference.pipeline.predict_proba(frame)[0, 1]
        real = result["conversion_probability"]
        st.write(
            f"With PageValues = {page_values:g}, the comparison model gives "
            f"**{with_page_values:.1%}** (real model: {real:.1%}). PageValues is only known late "
            "in a visit, which is why the real model leaves it out."
        )
