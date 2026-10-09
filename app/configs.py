"""Streamlit configuration settings and constants for the Auto MPG app."""

from __future__ import annotations

# apply frozen (read-only) to featurespec 
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = PROJECT_DIR / "model" / "model.pkl"
TRAIN_DATA_PATH = PROJECT_DIR / "data" / "X_train.csv"
FIRST_INDEX = 0
LAST_INDEX = -1

X_TRAIN = pd.read_csv(TRAIN_DATA_PATH, index_col=FIRST_INDEX)
MODEL_COLUMNS = list(X_TRAIN.columns)

WIDTH_CUSTOM = "stretch"
BODY_TITLE = "Auto MPG Prediction"
TEXT_ALIGNMENT_CUSTOM = "center"
SPEC_COLUMNS = 2
FLOAT_FORMAT = "%.1f"


@dataclass(frozen=True)
class FeatureSpec:
    """Feature definition specification used in the model"""
    column: str
    label: str
    step: float
    format_custom: str | None = None
    widget_type: str = "slider"
    integer: bool = False


FEATURE_SPECS = (
    FeatureSpec("displacement", "Engine displacement *(cu. inches)*", 0.5, FLOAT_FORMAT, "number"),
    FeatureSpec("cylinders", "Number of cylinders", 2, integer=True),
    FeatureSpec("horsepower", "Horsepower *(hp)*", 1, integer=True),
    FeatureSpec("weight", "Curb weight *(lbs.)*", 1, widget_type="number"),
    FeatureSpec("acceleration", "Time to accelerate from 0 to 60 miles per hour *(sec.)*", 0.1, FLOAT_FORMAT),
    FeatureSpec("model_year", "Model year *(19xx)*", 1, integer=True),
    FeatureSpec("origin", "Origin of car *(1: American, 2: European, 3: Japanese)*", 1, integer=True),
)

BODY_SIDEBAR = """
# About

This tool forecasts fuel efficiency of the car in Miles Per Gallon (MPG) using Gradient Boosting model.
* Median Absolute Error (MedAE): ~1.5 MPG on test set.
* Adjust the specifications and result will be displayed in real-time.

# Resources
* [UCI ML Repository: Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg)
* [Histogram-based Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [App Repository](https://github.com/garudapaksi-yusuf/auto_mpg)
"""

CSS_CUSTOM = """
<style>
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
        justify-content: center; /* alignment for spinner icon and text */
    }
</style>
"""
