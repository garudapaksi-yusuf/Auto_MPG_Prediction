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
    """DropValues class for removing specified values from a given column in a DataFrame.
    
    This transformer is designed to be used with scikit-learn's pipeline 
    and allows for the removal of rows in a DataFrame where the specified column contains certain values.
    
    Attributes:
        column (str): The name of the column from which to drop values.
        values (list): A list of values to be removed from the specified column.
    
    Methods:
        fit(X, y=None):
            Fits the transformer to the data. This method is a no-op for this transformer.
    
        transform(X):
            Removes rows from the DataFrame where the specified column contains any of the values to be dropped.
    
    Args:
        column (str): The name of the column to process.
        values (str or list): The value(s) to drop from the specified column. Can be a single value or a list of values.
    """
    def __init__(self, column, values):
        self.column = column
        self.values = values if isinstance(values, list) else [values]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        mask = ~X[self.column].isin(self.values)
        return X.loc[mask]

def data_raw_etl(id):
    """Fetches raw dataset from the UCI repository and prepares it for analysis.
    
    Args:
        id (int): The identifier for the dataset to be fetched from the UCI repository.
    
    Returns:
        pd.DataFrame: A DataFrame containing the features and target variables of the dataset.
    """
    ucirepo_dotdict = fetch_ucirepo(id=id)

    features = ucirepo_dotdict.data.features
    target = ucirepo_dotdict.data.targets

    data_raw = pd.concat([features, target], axis=AXIS_COLUMNS)

    return data_raw

def data_inspect(data):
    inspection_summary = pd.DataFrame({
        "dtype": data.dtypes,
        "sample": [data[column].unique() for column in data.columns],
        "min": [data[column].min() for column in data.columns],
        "max": [data[column].max() for column in data.columns],
        "duplicated": data.duplicated().sum(),
        "missing": data.isna().sum()
    })

    return inspection_summary

def match_transformer(column):
    transformer = MatchCategories(
        column,
        ignore_format=True # allow transformer on variables with non-string data type (e.g., numeric)
    )

    return transformer

def data_cleaned_etl():
    data_dir = NOTEBOOK_DIR.parent / "data"
    data_path = data_dir / "data_cleaned.csv"
    data_cleaned = pd.read_csv(
        data_path,
        index_col=FIRST_COLUMN_INDEX # use column position `0` as row label
    )

    return data_cleaned

def random_number_generator(seed):
    return np.random.RandomState(seed)

def inspect_dimensionality(dataset, train, test):
    """Performs dimensionality analysis of the dataset and its train-test subsets.

    Args:
        dataset (pd.DataFrame): The original dataset.
        train (pd.DataFrame): The training subset of the dataset.
        test (pd.DataFrame): The testing subset of the dataset.
        
    Returns:
        pd.DataFrame: A DataFrame summarizing the number of rows, row ratio, 
        and number of columns for the dataset and its subsets.
    """
    row, column = 0, 1
    dimensionality_summary = pd.DataFrame({
        "row_n": [dataset.shape[row], train.shape[row], test.shape[row]],
        "row_ratio": [
            dataset.shape[row] / dataset.shape[row],
            train.shape[row] / dataset.shape[row],
            test.shape[row] / dataset.shape[row]
        ],
        "column": [dataset.shape[column], train.shape[column], test.shape[column]]
    }, index=["dataset", "train", "test"])
    
    return dimensionality_summary