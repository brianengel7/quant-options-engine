from pathlib import Path

import streamlit as st


STYLE_PATH = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "app.css"
)


def apply_app_style():
    if STYLE_PATH.exists():
        st.html(STYLE_PATH)