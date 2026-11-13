"""The six dashboard sections, in the order they appear in the sidebar."""

from dashboard.sections import (
    analytics,
    comparison,
    explainability,
    overview,
    performance,
    scoring,
)

SECTIONS = {
    "Overview": overview,
    "Live session scoring": scoring,
    "Funnel & cohorts": analytics,
    "Model comparison": comparison,
    "Explainability": explainability,
    "Performance & threshold": performance,
}
