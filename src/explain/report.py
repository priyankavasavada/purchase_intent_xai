"""Save global SHAP importance for both final models, for the dashboard.

Importance is the average size of each feature's contribution over the test
split, so it describes how the model behaves on unseen sessions.

Run from the project root:
    python -m src.explain.report
"""

from dataclasses import replace

import pandas as pd

from src.config.settings import PROJECT_ROOT, get_settings
from src.data.loader import load_data, split_data
from src.explain.shap_explainer import ShapExplainer
from src.models.artifacts import latest_version_name, load_artifact
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

REPORT_FILE = PROJECT_ROOT / "reports" / "shap_global_importance.csv"


def main() -> None:
    settings = get_settings()
    setup_logging(settings.logs_dir)

    _, X_test, _, _ = split_data(load_data(settings=settings), settings)
    variants = {
        "purchase_intent": settings,
        "purchase_intent_with_page_values": replace(settings, leakage_excluded_features=[]),
    }

    columns = {}
    for prefix, variant_settings in variants.items():
        name = latest_version_name(settings.models_dir, prefix)
        artifact = load_artifact(name, settings.models_dir)
        explainer = ShapExplainer(artifact.pipeline, variant_settings)
        columns[prefix] = explainer.global_importance(X_test[artifact.feature_columns])

    importance = pd.DataFrame(columns).sort_values("purchase_intent", ascending=False)
    importance.round(4).to_csv(REPORT_FILE, index_label="feature")
    logger.info("Global SHAP importance:\n%s", importance.round(3).head(12).to_string())
    logger.info("Saved to %s", REPORT_FILE)


if __name__ == "__main__":
    main()
