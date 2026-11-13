"""Run the real dashboard without a browser and check every section works."""

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.sections import SECTIONS
from src.config.settings import PROJECT_ROOT

APP = str(PROJECT_ROOT / "dashboard" / "app.py")


def open_section(name: str) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    app.sidebar.radio[0].set_value(name).run()
    return app


@pytest.mark.parametrize("name", list(SECTIONS))
def test_section_loads_without_errors(name):
    app = open_section(name)
    assert not app.exception
    assert app.title[0].value


def test_scoring_form_returns_a_score():
    app = open_section("Live session scoring")
    app.button[0].click().run()

    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Chance of buying" in labels
    assert "Decision" in labels


def test_threshold_from_value_and_cost():
    app = open_section("Performance & threshold")
    app.number_input[1].set_value(50.0).run()  # very expensive nudges

    assert not app.exception
    assert app.error, "nudging should not pay when a nudge costs more than a buyer is worth"


def test_explainability_shows_shap_and_lime():
    app = open_section("Explainability")
    assert not app.exception
    assert any("LIME" in item.value for item in app.markdown)
    assert any("SHAP" in item.value for item in app.markdown)
