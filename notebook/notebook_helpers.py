from pathlib import Path

import numpy as np
import pandas as pd
from feature_engine.preprocessing import MatchCategories
from sklearn.base import BaseEstimator, TransformerMixin
from ucimlrepo import fetch_ucirepo

ID_AUTO_MPG = 9
INVALID_COLUMN = "origin"
AXIS_COLUMNS = 1
FIRST_COLUMN_INDEX = 0
NOTEBOOK_DIR = Path.cwd()
SEED = 42
RARE_COLUMN, RARE_VALUES = "cylinders", [3, 5]


class DropValues(BaseEstimator, TransformerMixin):
    """Remove rows whose specified column contains any of the given values."""

    def __init__(self, column, values):
        self.column = column
        self.values = values if isinstance(values, list) else [values]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = pd.DataFrame(X).copy()
        keep_rows = ~data[self.column].isin(self.values)
        return data.loc[keep_rows]


def data_raw_etl(dataset_id):
    """Fetch and combine features and target from a UCI dataset."""
    dataset = fetch_ucirepo(id=dataset_id)
    features = dataset.data.features
    target = dataset.data.targets

    return pd.concat([features, target], axis=AXIS_COLUMNS)


def data_inspect(data):
    """Summarize column types, unique values, ranges, duplicates, and missing data."""
    return pd.DataFrame(
        {
            "dtype": data.dtypes,
            "sample": [data[column].unique() for column in data.columns],
            "min": [data[column].min() for column in data.columns],
            "max": [data[column].max() for column in data.columns],
            "duplicated": data.duplicated().sum(),
            "missing": data.isna().sum(),
        }
    )


def match_transformer(column):
    """Create a transformer that normalizes categories in the given column."""
    return MatchCategories(
        column,
        ignore_format=True,
    )


def match_data(column, data):
    """Match categories in a column and return the transformed DataFrame."""
    transformer = match_transformer(column)
    return pd.DataFrame(
        transformer.fit_transform(data),
        columns=transformer.get_feature_names_out(),
    )


def data_cleaned_etl():
    """Load cleaned dataset saved by the initial analysis notebook."""
    data_dir = NOTEBOOK_DIR.parent / "data"
    data_path = data_dir / "data_cleaned.csv"
    return pd.read_csv(
        data_path,
        index_col=FIRST_COLUMN_INDEX,
    )


def random_number_generator(seed):
    """Create a reproducible NumPy random number generator."""
    return np.random.RandomState(seed)


def inspect_dimensionality(dataset, train, test):
    """Compare row counts, row proportions, and column counts across data splits."""
    row_count = len(dataset)
    return pd.DataFrame(
        {
            "row_n": [row_count, len(train), len(test)],
            "row_ratio": [1, len(train) / row_count, len(test) / row_count],
            "column": [dataset.shape[1], train.shape[1], test.shape[1]],
        },
        index=["dataset", "train", "test"],
    )