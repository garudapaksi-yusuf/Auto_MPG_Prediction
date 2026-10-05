# load dependencies
from pathlib import Path

import pandas as pd
from ui_helpers import (
    number_input_widget,
    page_layouts,
    slider_widget,
    widget_column_container,
    widget_session_state,
)

app_dir = Path(__file__).resolve().parent
model_path = app_dir.parent / "model" / "model.pkl"
X_train_path = app_dir.parent / "data" / "X_train.csv"
index_first, index_last = 0, -1
X_train = pd.read_csv(X_train_path, index_col=index_first)

# global app layout
body_sidebar = """
# About

This tool forecasts the fuel efficiency of a car in Miles Per Gallon (MPG) using Gradient Boosting model.
* Median Absolute Error (MedAE): ~1.5 MPG on test set.
* Adjust the specifications and result will be displayed automatically.

# Resources
* [UCI ML Repository: Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg)
* [Histogram-based Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [App Repository](https://github.com/garudapaksi-yusuf/auto_mpg)
"""
width_custom = "stretch"
body_title = "Auto MPG Prediction"
text_alignment_custom = "center"

spec_columns = 2
columns = [column for column in X_train.columns]
descriptions = [
    "Engine displacement *(cu. inches)*",
    "Number of cylinders",
    "Horsepower *(hp)*",
    "Curb weight *(lbs.)*",
    "Time to accelerate from 0 to 60 miles per hour *(sec.)*",
    "Model year *(19xx)*",
    "Origin of car *(1: American, 2: European, 3: Japanese)*"
]
step_displacement = 0.5
step_cylinders = 2
step_acceleration = 0.1
format_float = "%.1f"
css_inject = """
<style>
    # section[data-testid="stMain"] div[data-testid="stHeading"] h1 {
    #     font-size: 2rem;
    # }
    div[data-testid="stNumberInput"] p {
        font-size: 1rem;
        margin-bottom: 0.5rem
    }
    div[data-testid="stSlider"] p {
        font-size: 1rem;
        margin-bottom: 0.5rem
    }
    .st-key-container {
        margin-top: 0.5rem;
    }
    .st-key-container [data-testid="stHtml"] p {
        text-align: center;
    }
    div[data-testid="stSpinner"] > div {
        justify-content: center;
    }
</style>
"""

def main():
    _, _, _, widget_columns = page_layouts(css_inject, body_sidebar, width_custom, body_title, text_alignment_custom, spec_columns)

    for index, (column, description) in enumerate(zip(columns, descriptions)):
        min_value, max_value = X_train[column].min(), X_train[column].max()
        kwargs = {
            "description": description,
            "min_value": min_value,
            "max_value": max_value,
            "key": f"{column}"
        }
        with widget_columns[index % len(widget_columns)]:
            if column == "displacement":
                value = number_input_widget(**kwargs,
                                            step=step_displacement,
                                            format_custom=format_float)
                widget_session_state(column, value)
            elif column == "cylinders":
                value = slider_widget(**kwargs,
                                      step=step_cylinders,
                                      integer=True)
                widget_session_state(column, value)
            elif column == "weight":
                value = number_input_widget(**kwargs)
                widget_session_state(column, value)
            elif column == "acceleration":
                value = slider_widget(**kwargs,
                                      step=step_acceleration,
                                      format_custom=format_float)
                widget_session_state(column, value)
            else:
                value = slider_widget(**kwargs,
                                      integer=True)
                widget_session_state(column, value)

    with widget_columns[index_last]:
        widget_column_container(columns, width_custom, model_path, index_first)

if __name__ == "__main__":
    main()