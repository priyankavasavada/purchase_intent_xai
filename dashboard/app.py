"""Streamlit dashboard for the purchase-intent model.

Run from the project root:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Streamlit only puts dashboard/ on the import path; add the project root so
# `src` and `dashboard` can be imported.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from dashboard.sections import SECTIONS  # noqa: E402

st.set_page_config(page_title="Purchase Intent", page_icon="🛒", layout="wide")

st.sidebar.title("Purchase Intent")
choice = st.sidebar.radio("Section", list(SECTIONS))
st.sidebar.caption("Scores shopping sessions mid-visit and explains why.")

SECTIONS[choice].render()
