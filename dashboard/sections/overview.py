"""Overview: what the project does and the headline numbers."""

import streamlit as st

from dashboard.data import final_models, scorer, sessions


def render() -> None:
    st.title("E-commerce purchase intent")
    st.write(
        "The model looks at what a visitor has done so far in a session (pages viewed, time "
        "spent, exit and bounce rates, month, traffic source) and estimates the chance the "
        "session ends in a purchase. It also shows which behaviours drove that estimate, so a "
        "growth or CRO team can decide who to nudge with an offer."
    )

    data = sessions()
    main = final_models()["variants"]["purchase_intent"]
    metrics = main["test_metrics"]
    conversion = data["Converted"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sessions", f"{len(data):,}")
    col2.metric("Conversion rate", f"{conversion:.1%}")
    col3.metric(
        "Model PR-AUC (test)",
        f"{metrics['pr_auc']:.3f}",
        help=f"Area under the precision-recall curve. Random guessing scores {conversion:.3f}.",
    )
    col4.metric(
        "Buyers reached at default threshold",
        f"{metrics['recall']:.0%}",
        help=f"Recall at threshold {main['threshold']}, with {metrics['precision']:.0%} precision.",
    )

    st.subheader("How to use this dashboard")
    st.markdown(
        "- **Live session scoring**: enter a session and see its score and the reasons.\n"
        "- **Funnel & cohorts**: how engagement and conversion differ by month, visitor type "
        "and traffic source.\n"
        "- **Model comparison**: the six algorithms tried, and the effect of PageValues.\n"
        "- **Explainability**: what drives the model overall, and why any test session got "
        "its score.\n"
        "- **Performance & threshold**: choose an operating point from your own conversion value "
        "and nudge cost."
    )

    st.info(
        "**PageValues is not used by the model.** It mostly fills in on the way to checkout, so "
        "it is not known when a session is scored mid-visit (data leakage). A comparison model "
        "with PageValues is shown in Model comparison, for reference only."
    )
    trained = scorer().artifact.metadata["created_at"][:10]
    st.caption(
        f"Model: {scorer().model_version}, trained {trained}. "
        "See docs/model_card.md for intended use and limitations."
    )
