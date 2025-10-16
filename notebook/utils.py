def convert_categorical_variable(categorical_variables, data):
    """Function to convert categorical variables in dataset

    Args:
        categorical_variable (List): List containing names of categorical variables to convert
        data (DataFrame): dataset containing names of categorical variables

    Returns:
        DataFrame: cleaned dataset with converted variables
    """    
    for variable in categorical_variables:
        data[variable] = data[variable].astype(dtype='object')

    return data

import pandas as pd

def data_etl(filepath_or_buffer, categorical_variables):
    data_cleaned = pd.read_csv(filepath_or_buffer=filepath_or_buffer)
    data_modeling = convert_categorical_variable(
        categorical_variables=categorical_variables,
        data=data_cleaned
    )

    return data_modeling

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class DropLabelsTransformer(BaseEstimator, TransformerMixin):
    """
    Drops rows where specified feature columns contain certain unwanted labels.
    """
    def __init__(self, columns=None, labels_to_drop=None):
        self.columns = columns
        self.labels_to_drop = labels_to_drop

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X, y=None):
        # Drop rows where any specified column has unwanted values
        mask = X[self.columns].isin(self.labels_to_drop)
        X_transformed = X.loc[~mask].reset_index(drop=True)

        if y is not None:
            y_transformed = y.loc[~mask].reset_index(drop=True)
            return X_transformed, y_transformed

        return X_transformed
    
    def fit_transform(self, X, y=None):
        result = self.fit(X).transform(X)

        if y is not None:
            result = self.fit(X, y).transform(X, y)
            return result
        
        return result
    
    def get_feature_names_out(self, input_features=None):
        input_features = getattr(self, "feature_names_in_")
        return np.array([feature for feature in input_features if feature not in self.labels_to_drop])
    
class RatioInteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Create ratio features between specified numeric column pairs and converts those to numeric dtypes.
    Automatically drops all original columns used in those ratios.
    """

    def __init__(self, ratios=None):
        """
        Parameters
        ----------
        ratios : list of tuple
            Each tuple defines (numerator, denominator).
        """
        self.ratios = ratios

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):

        used_cols = set()
        for num, den in self.ratios:
            ratio_name = f"{num}_div_{den}"
            X[ratio_name] = (X[num] / X[den]).apply(func=pd.to_numeric)
            used_cols.update([num, den])

        # Always drop original ratio source columns
        X = X.drop(columns=list(used_cols))

        self.feature_names_out_ = list(X.columns)
        return X

    def get_feature_names_out(self, input_features=None):
        return getattr(self, "feature_names_out_", self.feature_names_in_)