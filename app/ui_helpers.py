"""Streamlit UI helpers for the Auto MPG app."""

from collections.abc import Sequence
from typing import Any

import joblib
import pandas as pd
import streamlit as st

__all__ = [
    "number_input_widget",
    "page_layouts",
    "slider_widget",
    "widget_column_container",
    "widget_session_state"
]

Number = int | float


def _build_prediction_markup(prediction: float) -> str:
    """Return HTML markup for the predicted fuel economy."""
    return f"""
    <p style='margin: 0; text-align: center;'>
        Estimated fuel economy: <font color='red'>{prediction:.2f} MPG</font>
    </p>
    """


def page_layouts(
    css_inject: str,
    body_sidebar: str,
    width_custom: int,
    body_title: str,
    text_alignment_custom: str,
    spec_columns: int,
) -> tuple[Any, Any, Any, list[Any]]:
    """Create the page shell with custom CSS, a sidebar, title, and columns."""
    html_custom = st.html(css_inject)
    sidebar = st.sidebar.markdown(body_sidebar, width=width_custom)
    title = st.title(body_title, text_alignment=text_alignment_custom)
    widget_columns = st.columns(spec_columns)
    return html_custom, sidebar, title, widget_columns


def number_input_widget(
    description: str,
    min_value: Number,
    max_value: Number,
    key: str,
    step: Number = 1,
    format_custom: str | None = None,
) -> Number:
    """Render a numeric input widget and return the selected value."""
    return st.number_input(
        description,
        min_value,
        max_value,
        step=step,
        format=format_custom,
        key=key,
    )


def slider_widget(
    description: str,
    min_value: Number,
    max_value: Number,
    key: str,
    step: Number = 1,
    format_custom: str | None = None,
    integer: bool = False,
) -> Number:
    """Render a slider and return the selected value."""
    midpoint = (min_value + max_value) / 2

    if integer:
        return st.slider(
            description,
            int(min_value),
            int(max_value),
            value=int(midpoint),
            step=step,
            key=key,
        )

    return st.slider(
        description,
        min_value,
        max_value,
        value=midpoint,
        step=step,
        format=format_custom,
        key=key,
    )


def widget_session_state(column: str, value: Any) -> None:
    """Store a value in session state when it has not been set yet."""
    st.session_state.setdefault(column, value)


def widget_column_container(
    columns: Sequence[str],
    width_custom: str,
    model_path: str,
    index_first: int,
) -> Any:
    """Display the predicted fuel economy for the current input values."""
    user_input = {column: st.session_state[column] for column in columns}
    input_df = pd.DataFrame([user_input])
    container_key = "container"

    with st.container(border=True, key=container_key), st.spinner(width=width_custom):
        model = joblib.load(model_path)
        prediction = model.predict(input_df)[index_first]
        prediction_markup = _build_prediction_markup(prediction)
        return st.html(prediction_markup)